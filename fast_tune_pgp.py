import argparse
import csv
import os
import pprint
import random
import time
from math import ceil

import numpy as np
import torch
import torch.nn as nn

from Untils.calculation import privacy_accounting
from Untils.data_process import load_data, separate_data
from Untils.model import PGP
from Untils.perturbation import (
    Aggre_Perturb,
    Katz_Aggre_Perturb,
    perturb_adj_continuous,
    perturb_adj_elementwise_continuous,
    random_adj_same_edge_count,
)
from Untils.transform import index_to_dense
from main import clip_graphs_by_pagerank, graph_max_degree


def parse_float_list(value):
    return [float(item) for item in value.split(",") if item.strip()]


def csv_fieldnames_for_append(path, default_fieldnames):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "r", newline="", encoding="utf-8") as csv_file:
            reader = csv.reader(csv_file)
            header = next(reader, None)
            if header:
                return header
    return default_fieldnames


def normalize_features(x):
    row_norms = torch.norm(x, p=2, dim=1, keepdim=True)
    row_norms[row_norms == 0] = 1
    return x / row_norms


def is_numeric_value(value):
    return value not in ("", None)


def katz_elementwise_mu_for_propagation(args, privacy_report):
    if args.aggregation_method != "katz":
        return None
    if args.level == "node" or args.katz_mu_mode == "inverse_dmax":
        value = privacy_report.get("elementwise_mu", "")
        return float(value) if is_numeric_value(value) else None
    return None


def limit_graphs(graphs, max_graphs, max_nodes, seed):
    selected = list(graphs)
    if max_nodes > 0:
        selected = [graph for graph in selected if graph.node_features.shape[0] <= max_nodes]
    rng = random.Random(seed)
    rng.shuffle(selected)
    if max_graphs > 0:
        selected = selected[:max_graphs]
    return selected


def split_train_val(train_graphs, val_ratio, seed):
    graphs = list(train_graphs)
    rng = random.Random(seed)
    rng.shuffle(graphs)
    val_size = max(1, int(len(graphs) * val_ratio)) if len(graphs) > 1 else 0
    return graphs[val_size:], graphs[:val_size]


def build_private_inputs(graphs, args, device, noise_scale, epsilon_edge, privacy_report, is_train):
    aggre_list = []
    adj_list = []
    start = time.time()
    for idx, graph in enumerate(graphs):
        num_nodes = graph.node_features.shape[0]
        adj_dense = index_to_dense(graph.edge_mat, num_nodes=num_nodes).to(device)
        feature = normalize_features(graph.node_features).to(device).float()

        if args.aggregation_method == "katz":
            aggre = Katz_Aggre_Perturb(
                adj_dense,
                feature,
                noise_scale=noise_scale if is_train else 0,
                depth=args.katz_depth if args.katz_depth > 0 else args.Hop_1,
                beta=args.katz_beta,
                elementwise_mu=katz_elementwise_mu_for_propagation(args, privacy_report),
            )
        else:
            aggre = Aggre_Perturb(
                adj_dense,
                feature,
                noise_scale=noise_scale if is_train else 0,
                hop=args.Hop_1,
                multi_GNN_output_flag=False,
            )

        if is_train and args.adj_mode == "exact":
            if args.private_adj_mode == "elementwise":
                mu = privacy_report.get("elementwise_mu", "")
                if not is_numeric_value(mu):
                    raise ValueError("--private_adj_mode elementwise requires Katz elementwise_mu.")
                adj_for_model = perturb_adj_elementwise_continuous(
                    adj_dense,
                    epsilon=epsilon_edge,
                    noise_type="laplace",
                    noise_seed=args.seed + idx,
                    delta=args.delta,
                    sensitivity=privacy_report["private_adj_l1_sensitivity"],
                    elementwise_mu=float(mu),
                ).float().to(device)
            else:
                adj_for_model = perturb_adj_continuous(
                    adj_dense,
                    epsilon=epsilon_edge,
                    noise_type="laplace",
                    noise_seed=args.seed + idx,
                    delta=args.delta,
                    sensitivity=privacy_report["edge_l1_sensitivity"],
                ).float().to(device)
        elif is_train and args.adj_mode == "random":
            adj_for_model = random_adj_same_edge_count(
                adj_dense,
                seed=args.seed + idx,
            ).float().to(device)
        else:
            adj_for_model = adj_dense.float()

        aggre_list.append(aggre)
        adj_list.append(adj_for_model)

    return aggre_list, adj_list, time.time() - start


def run_epoch(model, graphs, aggre_list, adj_list, optimizer, criterion, device, batch_size, train):
    indices = list(range(len(graphs)))
    if train:
        random.shuffle(indices)
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    batches = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in batches:
            if train:
                optimizer.zero_grad()
            outs = []
            labels = []
            for graph_idx in batch:
                _, _, _, out, _ = model(aggre_list[graph_idx], adj_list[graph_idx])
                outs.append(out)
                labels.append(graphs[graph_idx].label)
            out_all = torch.stack(outs, dim=0).to(device)
            target = torch.tensor(labels, dtype=torch.long, device=device)
            loss = criterion(out_all, target)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(batch)
            total_correct += (out_all.argmax(dim=1) == target).sum().item()
            total_seen += len(batch)

    return total_loss / max(1, total_seen), total_correct / max(1, total_seen)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="PROTEINS")
    parser.add_argument("--level", default="edge", choices=["edge", "node"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--Fold", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--aggregation_method", default="katz", choices=["original", "katz"])
    parser.add_argument("--pooling_method", default="diffpool",
                        choices=["diffpool", "mean", "max", "sum", "topk", "sag"])
    parser.add_argument("--pool_ratio", type=float, default=0.5)
    parser.add_argument("--Hop_1", type=int, default=3)
    parser.add_argument("--katz_beta", type=float, default=0.5)
    parser.add_argument("--katz_depth", type=int, default=-1)
    parser.add_argument("--katz_mu_mode", default="default", choices=["default", "inverse_dmax"],
                        help="default keeps previous edge behavior; inverse_dmax uses mu=1/D_max for Katz accounting and propagation.")
    parser.add_argument("--D_max", type=int, default=4)
    parser.add_argument("--epsilon_list", default="2,4,8")
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--edge_epsilon_ratio", type=float, default=0.2)
    parser.add_argument("--epoch_2", type=int, default=10)
    parser.add_argument("--Batch_size", type=int, default=64)
    parser.add_argument("--hidden_dimension_2", type=int, default=32)
    parser.add_argument("--lr_2", type=float, default=0.01)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--max_train_graphs", type=int, default=0)
    parser.add_argument("--max_test_graphs", type=int, default=0)
    parser.add_argument("--max_nodes", type=int, default=0)
    parser.add_argument("--adj_mode", default="public", choices=["public", "exact", "random"],
                        help="public skips adjacency perturbation; exact uses private perturbation; random uses a random graph with the same edge count.")
    parser.add_argument("--private_adj_mode", default="raw", choices=["raw", "elementwise"],
                        help="raw perturbs binary A; elementwise perturbs mu*A for private pooling.")
    parser.add_argument("--result_file", default="")
    parser.add_argument("--degree_as_tag", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs("results", exist_ok=True)
    if not args.result_file:
        args.result_file = f"results/fast_tune_{args.dataset}_{args.level}_{args.aggregation_method}_{args.pooling_method}.csv"

    print("Fast tuning parameters:")
    pprint.pprint(vars(args))
    if args.adj_mode == "public":
        print("WARNING: --adj_mode public skips private adjacency perturbation for speed. Use main.py or --adj_mode exact for final DP runs.")
    if args.adj_mode == "random":
        print("WARNING: --adj_mode random replaces training adjacency with a random graph of the same edge count. Use only for ablation.")

    device = torch.device(args.device)
    graphs, num_classes = load_data(args.dataset, args.degree_as_tag)
    train_graphs, test_graphs = separate_data(graphs, args.seed, args.Fold)
    train_graphs = limit_graphs(train_graphs, args.max_train_graphs, args.max_nodes, args.seed)
    test_graphs = limit_graphs(test_graphs, args.max_test_graphs, args.max_nodes, args.seed + 1)
    train_graphs, val_graphs = split_train_val(train_graphs, args.val_ratio, args.seed)
    if not train_graphs or not test_graphs:
        raise ValueError("No graphs left after --max_train_graphs/--max_test_graphs/--max_nodes filtering.")

    original_d_max = max((graph_max_degree(graph) for graph in train_graphs + val_graphs + test_graphs), default=0)
    accounting_d_max = args.D_max if args.D_max > 0 else original_d_max
    clipped_d_max = original_d_max
    if args.level == "node":
        if args.D_max <= 0:
            raise ValueError("--level node requires positive --D_max")
        original_d_max, clipped_d_max = clip_graphs_by_pagerank(train_graphs + val_graphs + test_graphs, args.D_max)
        print(f"Node-level PageRank clipping: original max degree {original_d_max}, clipped max degree {clipped_d_max}, D_max {args.D_max}")
        if clipped_d_max > args.D_max:
            raise ValueError(
                f"PageRank clipping failed to enforce node-DP degree bound: "
                f"clipped max degree {clipped_d_max}, D_max={args.D_max}."
            )

    avg_nodes = int(np.mean([graph.node_features.shape[0] for graph in train_graphs]))
    input_dim = train_graphs[0].node_features.shape[-1]
    num_clusters = max(2, ceil(avg_nodes / 2))
    katz_depth = args.Hop_1 if args.katz_depth <= 0 else args.katz_depth
    epsilon_values = parse_float_list(args.epsilon_list)

    result_exists = os.path.exists(args.result_file)
    with open(args.result_file, "a", newline="", encoding="utf-8") as result_csv:
        default_fieldnames = [
            "dataset", "level", "method", "pooling_method", "seed", "epsilon", "adj_mode",
            "private_adj_mode",
            "hop", "d_max", "original_max_degree", "clipped_max_degree",
            "katz_beta", "katz_depth", "katz_mu_mode", "embedding_l2_sensitivity", "embedding_sigma",
            "edge_l1_sensitivity", "private_adj_l1_sensitivity", "elementwise_mu",
            "katz_structural_sensitivity", "katz_feature_sensitivity",
            "train_acc", "val_acc", "test_acc", "preprocess_seconds", "train_seconds",
            "num_train", "num_val", "num_test", "avg_nodes",
        ]
        writer = csv.DictWriter(
            result_csv,
            fieldnames=csv_fieldnames_for_append(args.result_file, default_fieldnames),
            extrasaction="ignore",
        )
        if not result_exists:
            writer.writeheader()

        for epsilon in epsilon_values:
            epsilon_edge = epsilon * args.edge_epsilon_ratio
            elementwise_mu_value = None
            if args.aggregation_method == "katz" and args.katz_mu_mode == "inverse_dmax":
                if accounting_d_max <= 0:
                    raise ValueError("--katz_mu_mode inverse_dmax requires a positive D_max or non-empty graph set.")
                elementwise_mu_value = 1.0 / float(accounting_d_max)
            privacy_report = privacy_accounting(
                epsilon,
                delta=args.delta,
                level=args.level,
                d_max=accounting_d_max,
                hop=args.Hop_1,
                epsilon_edge=epsilon_edge,
                aggregation=args.aggregation_method,
                beta=args.katz_beta,
                depth=katz_depth,
                elementwise_mu_value=elementwise_mu_value,
                private_adj_mode=args.private_adj_mode,
            )
            print(f"\n===== epsilon={epsilon} =====")
            pprint.pprint(privacy_report)
            train_x, train_adj, prep_train = build_private_inputs(
                train_graphs, args, device, privacy_report["embedding_sigma"],
                epsilon_edge, privacy_report, is_train=True
            )
            val_x, val_adj, prep_val = build_private_inputs(
                val_graphs, args, device, 0, epsilon_edge, privacy_report, is_train=False
            )
            test_x, test_adj, prep_test = build_private_inputs(
                test_graphs, args, device, 0, epsilon_edge, privacy_report, is_train=False
            )

            model = PGP(
                input_dim=input_dim,
                hidden_dim=args.hidden_dimension_2,
                output_dim=num_classes,
                num_clusters=num_clusters,
                device=device,
                merge_mode="ATT",
                pooling_method=args.pooling_method,
                pool_ratio=args.pool_ratio,
            ).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_2)

            train_start = time.time()
            best_state = None
            best_val_acc = -1.0
            last_train_acc = 0.0
            for epoch in range(args.epoch_2):
                train_loss, train_acc = run_epoch(
                    model, train_graphs, train_x, train_adj, optimizer, criterion,
                    device, args.Batch_size, train=True
                )
                val_loss, val_acc = run_epoch(
                    model, val_graphs, val_x, val_adj, optimizer, criterion,
                    device, args.Batch_size, train=False
                ) if val_graphs else (0.0, 0.0)
                last_train_acc = train_acc
                if val_acc >= best_val_acc:
                    best_val_acc = val_acc
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                print(f"epoch={epoch + 1:03d} train_acc={train_acc:.4f} val_acc={val_acc:.4f} train_loss={train_loss:.4f}")
            if best_state is not None:
                model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
            _, test_acc = run_epoch(
                model, test_graphs, test_x, test_adj, optimizer, criterion,
                device, args.Batch_size, train=False
            )
            train_seconds = time.time() - train_start
            preprocess_seconds = prep_train + prep_val + prep_test
            print(f"epsilon={epsilon} test_acc={test_acc:.4f} preprocess={preprocess_seconds:.1f}s train={train_seconds:.1f}s")

            writer.writerow({
                "dataset": args.dataset,
                "level": args.level,
                "method": args.aggregation_method,
                "pooling_method": args.pooling_method,
                "seed": args.seed,
                "epsilon": epsilon,
                "adj_mode": args.adj_mode,
                "private_adj_mode": args.private_adj_mode,
                "hop": args.Hop_1,
                "d_max": accounting_d_max,
                "original_max_degree": original_d_max,
                "clipped_max_degree": clipped_d_max,
                "katz_beta": args.katz_beta if args.aggregation_method == "katz" else "",
                "katz_depth": katz_depth if args.aggregation_method == "katz" else "",
                "katz_mu_mode": args.katz_mu_mode if args.aggregation_method == "katz" else "",
                "embedding_l2_sensitivity": privacy_report["embedding_l2_sensitivity"],
                "embedding_sigma": privacy_report["embedding_sigma"],
                "edge_l1_sensitivity": privacy_report["edge_l1_sensitivity"],
                "private_adj_l1_sensitivity": privacy_report["private_adj_l1_sensitivity"],
                "elementwise_mu": privacy_report.get("elementwise_mu", ""),
                "katz_structural_sensitivity": privacy_report.get("katz_structural_sensitivity", ""),
                "katz_feature_sensitivity": privacy_report.get("katz_feature_sensitivity", ""),
                "train_acc": round(last_train_acc, 6),
                "val_acc": round(best_val_acc, 6),
                "test_acc": round(test_acc, 6),
                "preprocess_seconds": round(preprocess_seconds, 3),
                "train_seconds": round(train_seconds, 3),
                "num_train": len(train_graphs),
                "num_val": len(val_graphs),
                "num_test": len(test_graphs),
                "avg_nodes": avg_nodes,
            })
            result_csv.flush()
    print(f"Saved fast tuning result to {args.result_file}")


if __name__ == "__main__":
    main()
