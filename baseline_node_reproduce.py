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

from Untils.calculation import calibrate_gaussian_sigma
from Untils.data_process import load_data, separate_data
from Untils.model import PGP
from Untils.perturbation import creat_noise, normalize_adjacency, perturb_adj_continuous, row_l2_normalize
from Untils.transform import index_to_dense
from main import graph_max_degree


def parse_float_list(value):
    return [float(item) for item in value.split(",") if item.strip()]


def parse_str_list(value):
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def uses_gradient_noise(method):
    return method.lower() in {"dpgnn", "sagd"}


def calibrate_composed_gaussian_multiplier(epsilon, delta, total_steps):
    """Calibrate sigma_T for T composed Gaussian gradient releases.

    If each step releases g + N(0, (S(g) * sigma_T)^2 I), the composed
    RDP term is T * alpha / (2 * sigma_T^2). This is equivalent to
    calibrating a Gaussian mechanism with sensitivity sqrt(T).
    """
    return calibrate_gaussian_sigma(epsilon, delta, np.sqrt(max(1, total_steps)))


def normalize_features(x):
    row_norms = torch.norm(x, p=2, dim=1, keepdim=True)
    row_norms[row_norms == 0] = 1
    return x / row_norms


def split_train_val(train_graphs, val_ratio, seed):
    graphs = list(train_graphs)
    rng = random.Random(seed)
    rng.shuffle(graphs)
    val_size = max(1, int(len(graphs) * val_ratio)) if len(graphs) > 1 else 0
    return graphs[val_size:], graphs[:val_size]


def limit_graphs(graphs, max_graphs, max_nodes, seed):
    selected = list(graphs)
    if max_nodes > 0:
        selected = [graph for graph in selected if graph.node_features.shape[0] <= max_nodes]
    rng = random.Random(seed)
    rng.shuffle(selected)
    if max_graphs > 0:
        selected = selected[:max_graphs]
    return selected


def finite_ppr_operator(adj_dense, depth, alpha, norm="sym"):
    if norm == "sym":
        prop = normalize_adjacency(adj_dense, add_self_loops=True)
    elif norm == "elementwise":
        prop = adj_dense.float()
    else:
        raise ValueError(f"Unsupported PPR norm: {norm}")

    n_nodes = adj_dense.size(0)
    eye = torch.eye(n_nodes, device=adj_dense.device, dtype=torch.float32)
    current = eye
    operator = torch.zeros_like(eye)
    for order in range(depth):
        operator = operator + alpha * ((1.0 - alpha) ** order) * current
        current = torch.mm(prop, current)
    operator = operator + ((1.0 - alpha) ** depth) * current
    return operator


def dpar_appr_operator(adj_dense, depth, alpha, c1, sigma, seed):
    operator = finite_ppr_operator(adj_dense, depth, alpha, norm="sym")
    row_norm = operator.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
    clip_factor = torch.clamp(c1 / row_norm, max=1.0)
    operator = operator * clip_factor
    generator = torch.Generator(device=operator.device)
    generator.manual_seed(seed)
    noise = torch.normal(0.0, sigma, size=operator.shape, generator=generator, device=operator.device)
    return operator + noise


def baseline_sensitivity(method, depth, d_max, args):
    method = method.lower()
    if method == "gap":
        sensitivity = np.sqrt(float(d_max) * float(depth))
        return sensitivity, f"sqrt(D_max * L) = sqrt({d_max} * {depth}) from GAP node-level RDP composition"

    if method == "dpgnn":
        if d_max == 1:
            receptive = depth + 1
        else:
            receptive = (float(d_max) ** (depth + 1) - 1.0) / (float(d_max) - 1.0)
        sensitivity = 2.0 * args.dpgnn_clip * receptive
        return sensitivity, f"gradient sensitivity 2C * (D_max^(L+1)-1)/(D_max-1), C={args.dpgnn_clip}"

    if method == "dpar":
        return args.dpar_c1, f"C1 APPR vector clipping, C1={args.dpar_c1}"

    if method == "sagd":
        mu = 1.0 / float(d_max)
        gamma = args.sagd_gamma
        sensitivity = 2.0 * mu * (1.0 - gamma) / gamma * (1.0 - (1.0 - gamma) ** depth)
        return sensitivity, f"propagated feature sensitivity S(Z)=2mu(1-gamma)/gamma*(1-(1-gamma)^L), mu=1/D_max, gamma={gamma}"

    raise ValueError(f"Unsupported baseline method: {method}")


def propagate_baseline(adj_dense, feature, method, depth, sigma, d_max, args, seed):
    method = method.lower()

    if method == "gap":
        x = feature.float()
        prop = adj_dense.float()
        for _ in range(depth):
            x = torch.mm(prop.t(), x)
            x = row_l2_normalize(x + creat_noise(x, sigma))
        return x

    if method == "dpgnn":
        x = feature.float()
        prop = normalize_adjacency(adj_dense, add_self_loops=True)
        for _ in range(depth):
            x = torch.mm(prop, x)
        return row_l2_normalize(x)

    if method == "dpar":
        ppr = dpar_appr_operator(adj_dense, depth, args.dpar_alpha, args.dpar_c1, sigma, seed)
        return row_l2_normalize(torch.mm(ppr, feature.float()))

    if method == "sagd":
        scaled_adj = adj_dense.float() / float(d_max)
        operator = finite_ppr_operator(scaled_adj, depth, args.sagd_gamma, norm="elementwise")
        x = torch.mm(operator, feature.float())
        return row_l2_normalize(x)

    raise ValueError(f"Unsupported baseline method: {method}")


def build_inputs(graphs, method, args, device, sigma, epsilon_pooling, d_max, is_train):
    aggre_list = []
    adj_list = []
    start = time.time()
    for idx, graph in enumerate(graphs):
        num_nodes = graph.node_features.shape[0]
        adj_dense = index_to_dense(graph.edge_mat, num_nodes=num_nodes).to(device).float()
        feature = normalize_features(graph.node_features).to(device).float()
        embedding_sigma = 0.0 if uses_gradient_noise(method) else sigma
        aggre = propagate_baseline(
            adj_dense, feature, method, args.depth, embedding_sigma if is_train else 0.0,
            d_max, args, args.seed + idx
        )

        if is_train and args.adj_mode == "exact":
            adj_for_model = perturb_adj_continuous(
                adj_dense,
                epsilon=epsilon_pooling,
                noise_type="laplace",
                noise_seed=args.seed + 100000 + idx,
                delta=args.delta,
                sensitivity=float(d_max),
            ).float().to(device)
        else:
            adj_for_model = adj_dense

        aggre_list.append(aggre)
        adj_list.append(adj_for_model)

    return aggre_list, adj_list, time.time() - start


def add_gaussian_noise_to_gradients(model, sigma):
    if sigma <= 0:
        return
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.add_(torch.normal(0.0, sigma, size=parameter.grad.shape, device=parameter.grad.device))


def spectral_norm(weight):
    return float(torch.linalg.matrix_norm(weight.detach(), ord=2).item())


def classifier_weight(model):
    if getattr(model, "pooling_method", "") == "diffpool":
        return model.DiffPool_network.fc.weight
    return model.readout_fc.weight


def sagd_gradient_sensitivity(model, feature_sensitivity, args):
    """SaGD Lemma-2 style two-layer sensitivity bound.

    The original paper derives the bound for a two-layer prediction
    network over propagated node features. In this graph-classification
    reproduction, fc1 is treated as the first layer and the final graph
    classifier as the second layer after pooling.
    """
    theta1 = spectral_norm(model.fc1.weight)
    theta2 = spectral_norm(classifier_weight(model))
    c1 = args.sagd_loss_lipschitz
    c2 = args.sagd_loss_derivative_lipschitz
    ch1 = args.sagd_activation_lipschitz
    ch2 = args.sagd_activation_derivative_lipschitz
    sz = feature_sensitivity

    new_theta1 = c1 * ch1 * theta2
    new_theta2 = c1 * theta1
    rest_theta1 = (c2 * ch1 * theta2 * theta1 + c1 * ch1 * theta2 + c1 * ch2 * theta2 * theta1) * sz
    rest_theta2 = (c2 * ch1 * theta2 * (theta1 ** 2) + c1 * ch1 * theta1) * sz
    return float(np.sqrt((new_theta1 + rest_theta1) ** 2 + (new_theta2 + rest_theta2) ** 2))


def run_epoch(model, graphs, aggre_list, adj_list, optimizer, criterion, device, batch_size,
              train, method="", gradient_noise_multiplier=0.0, static_gradient_sensitivity=0.0,
              sagd_feature_sensitivity=0.0, gradient_clip=0.0, stats=None, args=None):
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
                if gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                if gradient_noise_multiplier > 0:
                    if method == "sagd":
                        sensitivity = sagd_gradient_sensitivity(model, sagd_feature_sensitivity, args)
                    else:
                        sensitivity = static_gradient_sensitivity
                    averaged_sensitivity = sensitivity / max(1, len(batch))
                    gradient_sigma = averaged_sensitivity * gradient_noise_multiplier
                    add_gaussian_noise_to_gradients(model, gradient_sigma)
                    if stats is not None:
                        stats["last_gradient_l2_sensitivity"] = averaged_sensitivity
                        stats["last_gradient_sigma"] = gradient_sigma
                        stats["max_gradient_l2_sensitivity"] = max(stats.get("max_gradient_l2_sensitivity", 0.0), averaged_sensitivity)
                        stats["max_gradient_sigma"] = max(stats.get("max_gradient_sigma", 0.0), gradient_sigma)
                optimizer.step()
            total_loss += loss.item() * len(batch)
            total_correct += (out_all.argmax(dim=1) == target).sum().item()
            total_seen += len(batch)

    return total_loss / max(1, total_seen), total_correct / max(1, total_seen)


def write_header_if_needed(path, fieldnames):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    return not os.path.exists(path)


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce node-level graph-classification baselines under a shared node embedding + pooling + MLP framework."
    )
    parser.add_argument("--dataset", default="PROTEINS")
    parser.add_argument("--methods", default="gap,dpar,dpgnn,sagd")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--Fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epsilon_list", default="1,2,4,8")
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--D_max", type=int, default=0, help="0 uses the actual maximum degree of the selected split.")
    parser.add_argument("--pooling_method", default="diffpool", choices=["diffpool", "mean", "max", "sum", "topk", "sag"])
    parser.add_argument("--pool_ratio", type=float, default=0.5)
    parser.add_argument("--adj_mode", default="exact", choices=["exact", "public"])
    parser.add_argument("--pooling_epsilon_ratio", type=float, default=0.2)
    parser.add_argument("--epoch_2", type=int, default=20)
    parser.add_argument("--Batch_size", type=int, default=128)
    parser.add_argument("--hidden_dimension_2", type=int, default=32)
    parser.add_argument("--lr_2", type=float, default=0.01)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--max_train_graphs", type=int, default=0)
    parser.add_argument("--max_test_graphs", type=int, default=0)
    parser.add_argument("--max_nodes", type=int, default=0)
    parser.add_argument("--dpar_alpha", type=float, default=0.15)
    parser.add_argument("--dpar_c1", type=float, default=1.0)
    parser.add_argument("--dpgnn_clip", type=float, default=1.0)
    parser.add_argument("--sagd_clip", type=float, default=0.0, help="SaGD does not use clipping in the original algorithm; keep 0 for faithful runs.")
    parser.add_argument("--sagd_gamma", type=float, default=0.15)
    parser.add_argument("--sagd_loss_lipschitz", type=float, default=1.41421356237)
    parser.add_argument("--sagd_loss_derivative_lipschitz", type=float, default=1.0)
    parser.add_argument("--sagd_activation_lipschitz", type=float, default=1.0)
    parser.add_argument("--sagd_activation_derivative_lipschitz", type=float, default=1.0)
    parser.add_argument("--result_file", default="")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if not args.result_file:
        args.result_file = f"results/baseline_node_reproduce/{args.dataset}_node_baselines.csv"

    print("Node-level baseline reproduction parameters:")
    pprint.pprint(vars(args))

    device = torch.device(args.device)
    graphs, num_classes = load_data(args.dataset)
    train_graphs, test_graphs = separate_data(graphs, args.seed, args.Fold)
    train_graphs = limit_graphs(train_graphs, args.max_train_graphs, args.max_nodes, args.seed)
    test_graphs = limit_graphs(test_graphs, args.max_test_graphs, args.max_nodes, args.seed + 1)
    train_graphs, val_graphs = split_train_val(train_graphs, args.val_ratio, args.seed)
    if not train_graphs or not test_graphs:
        raise ValueError("No graphs left after graph limiting.")

    actual_d_max = max(graph_max_degree(graph) for graph in train_graphs + val_graphs + test_graphs)
    d_max = actual_d_max if args.D_max <= 0 else args.D_max
    if d_max < actual_d_max:
        raise ValueError(f"D_max={d_max} is smaller than actual max degree {actual_d_max}.")

    avg_nodes = int(np.mean([graph.node_features.shape[0] for graph in train_graphs]))
    input_dim = train_graphs[0].node_features.shape[-1]
    num_clusters = max(2, ceil(avg_nodes / 2))
    methods = parse_str_list(args.methods)
    epsilons = parse_float_list(args.epsilon_list)

    fieldnames = [
        "dataset", "level", "method", "seed", "fold", "epsilon", "delta",
        "depth", "d_max", "actual_max_degree", "pooling_method", "adj_mode",
        "epsilon_embedding", "epsilon_pooling", "embedding_l2_sensitivity",
        "embedding_sigma", "noise_target", "gradient_l2_sensitivity", "gradient_sigma",
        "sensitivity_formula", "dpar_alpha", "dpar_c1", "dpgnn_clip", "sagd_clip",
        "sagd_gamma", "sagd_loss_lipschitz", "sagd_loss_derivative_lipschitz",
        "sagd_activation_lipschitz", "sagd_activation_derivative_lipschitz",
        "total_gradient_steps", "gradient_noise_multiplier",
        "last_gradient_l2_sensitivity", "last_gradient_sigma",
        "max_gradient_l2_sensitivity", "max_gradient_sigma",
        "train_acc", "val_acc", "test_acc",
        "preprocess_seconds", "train_seconds", "num_train", "num_val",
        "num_test", "avg_nodes",
    ]
    should_write_header = write_header_if_needed(args.result_file, fieldnames)
    with open(args.result_file, "a", newline="", encoding="utf-8") as result_csv:
        writer = csv.DictWriter(result_csv, fieldnames=fieldnames)
        if should_write_header:
            writer.writeheader()

        for method in methods:
            sensitivity, formula = baseline_sensitivity(method, args.depth, d_max, args)
            for epsilon in epsilons:
                epsilon_pooling = epsilon * args.pooling_epsilon_ratio if args.adj_mode == "exact" else 0.0
                epsilon_embedding = epsilon - epsilon_pooling
                noise_target = "gradient" if uses_gradient_noise(method) else ("appr_operator" if method == "dpar" else "node_embedding")
                total_gradient_steps = int(np.ceil(len(train_graphs) / args.Batch_size) * args.epoch_2)
                gradient_noise_multiplier = (
                    calibrate_composed_gaussian_multiplier(epsilon_embedding, args.delta, total_gradient_steps)
                    if uses_gradient_noise(method) else 0.0
                )
                sigma = 0.0 if uses_gradient_noise(method) else calibrate_gaussian_sigma(epsilon_embedding, args.delta, sensitivity)
                embedding_sigma = 0.0 if uses_gradient_noise(method) else sigma
                gradient_clip = args.dpgnn_clip if method == "dpgnn" else (args.sagd_clip if method == "sagd" else 0.0)
                print(f"\n===== method={method} epsilon={epsilon} =====")
                print({
                    "epsilon_embedding": epsilon_embedding,
                    "epsilon_pooling": epsilon_pooling,
                    "noise_target": noise_target,
                    "feature_or_static_l2_sensitivity": sensitivity,
                    "embedding_sigma": embedding_sigma,
                    "total_gradient_steps": total_gradient_steps if uses_gradient_noise(method) else 0,
                    "gradient_noise_multiplier": gradient_noise_multiplier,
                    "sensitivity_formula": formula,
                })

                train_x, train_adj, prep_train = build_inputs(
                    train_graphs, method, args, device, embedding_sigma, epsilon_pooling, d_max, is_train=True
                )
                val_x, val_adj, prep_val = build_inputs(
                    val_graphs, method, args, device, 0.0, epsilon_pooling, d_max, is_train=False
                )
                test_x, test_adj, prep_test = build_inputs(
                    test_graphs, method, args, device, 0.0, epsilon_pooling, d_max, is_train=False
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

                best_state = None
                best_val_acc = -1.0
                last_train_acc = 0.0
                gradient_stats = {}
                train_start = time.time()
                for epoch in range(args.epoch_2):
                    train_loss, train_acc = run_epoch(
                        model, train_graphs, train_x, train_adj, optimizer, criterion,
                        device, args.Batch_size, train=True,
                        method=method,
                        gradient_noise_multiplier=gradient_noise_multiplier,
                        static_gradient_sensitivity=sensitivity if method == "dpgnn" else 0.0,
                        sagd_feature_sensitivity=sensitivity if method == "sagd" else 0.0,
                        gradient_clip=gradient_clip,
                        stats=gradient_stats,
                        args=args,
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
                print(f"method={method} epsilon={epsilon} test_acc={test_acc:.4f}")

                writer.writerow({
                    "dataset": args.dataset,
                    "level": "node",
                    "method": method,
                    "seed": args.seed,
                    "fold": args.Fold,
                    "epsilon": epsilon,
                    "delta": args.delta,
                    "depth": args.depth,
                    "d_max": d_max,
                    "actual_max_degree": actual_d_max,
                    "pooling_method": args.pooling_method,
                    "adj_mode": args.adj_mode,
                    "epsilon_embedding": epsilon_embedding,
                    "epsilon_pooling": epsilon_pooling,
                    "embedding_l2_sensitivity": 0.0 if uses_gradient_noise(method) else sensitivity,
                    "embedding_sigma": embedding_sigma,
                    "noise_target": noise_target,
                    "gradient_l2_sensitivity": gradient_stats.get("last_gradient_l2_sensitivity", "") if uses_gradient_noise(method) else "",
                    "gradient_sigma": gradient_stats.get("last_gradient_sigma", "") if uses_gradient_noise(method) else "",
                    "sensitivity_formula": formula,
                    "dpar_alpha": args.dpar_alpha if method == "dpar" else "",
                    "dpar_c1": args.dpar_c1 if method == "dpar" else "",
                    "dpgnn_clip": args.dpgnn_clip if method == "dpgnn" else "",
                    "sagd_clip": args.sagd_clip if method == "sagd" else "",
                    "sagd_gamma": args.sagd_gamma if method == "sagd" else "",
                    "sagd_loss_lipschitz": args.sagd_loss_lipschitz if method == "sagd" else "",
                    "sagd_loss_derivative_lipschitz": args.sagd_loss_derivative_lipschitz if method == "sagd" else "",
                    "sagd_activation_lipschitz": args.sagd_activation_lipschitz if method == "sagd" else "",
                    "sagd_activation_derivative_lipschitz": args.sagd_activation_derivative_lipschitz if method == "sagd" else "",
                    "total_gradient_steps": total_gradient_steps if uses_gradient_noise(method) else "",
                    "gradient_noise_multiplier": gradient_noise_multiplier if uses_gradient_noise(method) else "",
                    "last_gradient_l2_sensitivity": gradient_stats.get("last_gradient_l2_sensitivity", "") if uses_gradient_noise(method) else "",
                    "last_gradient_sigma": gradient_stats.get("last_gradient_sigma", "") if uses_gradient_noise(method) else "",
                    "max_gradient_l2_sensitivity": gradient_stats.get("max_gradient_l2_sensitivity", "") if uses_gradient_noise(method) else "",
                    "max_gradient_sigma": gradient_stats.get("max_gradient_sigma", "") if uses_gradient_noise(method) else "",
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

    print(f"Saved node-level baseline reproduction result to {args.result_file}")


if __name__ == "__main__":
    main()
