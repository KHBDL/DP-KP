import argparse
import csv
import os
import random
import time
from math import ceil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fast_tune_pgp import build_private_inputs, limit_graphs, split_train_val
from baseline_node_reproduce import (
    baseline_sensitivity,
    build_inputs as build_baseline_inputs,
    calibrate_composed_gaussian_multiplier,
    run_epoch as run_baseline_epoch,
    uses_gradient_noise,
)
from main import clip_graphs_by_pagerank, graph_max_degree
from Untils.calculation import calibrate_gaussian_sigma, privacy_accounting
from Untils.data_process import load_data, separate_data
from Untils.model import PGP


def parse_float_list(value):
    return [float(item) for item in value.split(",") if item.strip()]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def entropy_from_probs(probs):
    probs = np.clip(probs, 1e-12, 1.0)
    return float(-(probs * np.log(probs)).sum())


def margin_from_probs(probs):
    if len(probs) < 2:
        return 0.0
    top2 = np.sort(probs)[-2:]
    return float(top2[-1] - top2[-2])


def safe_auc(y_true, scores):
    if len(set(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def safe_ap(y_true, scores):
    if len(set(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, scores))


def make_attack_model(kind, seed):
    if kind == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
        )
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=200, max_depth=None, min_samples_leaf=2,
            class_weight="balanced", random_state=seed, n_jobs=4
        )
    if kind == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=seed),
        )
    raise ValueError(f"Unknown attack model: {kind}")


def evaluate_binary_attack(features, labels, attack_model, seed, test_size=0.4):
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    if len(labels) < 4 or len(set(labels.tolist())) < 2:
        return {"auc": float("nan"), "ap": float("nan"), "acc": float("nan"), "f1": float("nan"), "n": len(labels)}
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=seed, stratify=labels
    )
    attack_model.fit(x_train, y_train)
    if hasattr(attack_model, "predict_proba"):
        scores = attack_model.predict_proba(x_test)[:, 1]
    else:
        scores = attack_model.decision_function(x_test)
    pred = (scores >= 0.5).astype(np.int64)
    return {
        "auc": safe_auc(y_test, scores),
        "ap": safe_ap(y_test, scores),
        "acc": float(accuracy_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "n": int(len(labels)),
    }


def train_target_model(args, graphs, num_classes, device, epsilon):
    if getattr(args, "method_family", "katz") == "baseline":
        if args.level == "edge":
            if getattr(args, "baseline_method", "gap") != "gap":
                raise ValueError("Only GAP/original is currently implemented for edge-level baseline attacks.")
            edge_args = argparse.Namespace(**vars(args))
            edge_args.method_family = "katz"
            edge_args.aggregation_method = "original"
            edge_args.private_adj_mode = "raw"
            return train_katz_target_model(edge_args, graphs, num_classes, device, epsilon)
        return train_baseline_target_model(args, graphs, num_classes, device, epsilon)
    return train_katz_target_model(args, graphs, num_classes, device, epsilon)


def train_katz_target_model(args, graphs, num_classes, device, epsilon):
    train_graphs, test_graphs = separate_data(graphs, args.seed, args.Fold)
    train_graphs = limit_graphs(train_graphs, args.max_train_graphs, args.max_nodes, args.seed)
    test_graphs = limit_graphs(test_graphs, args.max_test_graphs, args.max_nodes, args.seed + 1)
    train_graphs, val_graphs = split_train_val(train_graphs, args.val_ratio, args.seed)

    if not train_graphs or not test_graphs:
        raise ValueError("No graphs left after graph filtering.")

    original_d_max = max((graph_max_degree(graph) for graph in train_graphs + val_graphs + test_graphs), default=0)
    accounting_d_max = args.D_max if args.D_max > 0 else original_d_max
    clipped_d_max = original_d_max
    if args.level == "node":
        if accounting_d_max <= 0:
            raise ValueError("node-level DP requires positive D_max.")
        original_d_max, clipped_d_max = clip_graphs_by_pagerank(
            train_graphs + val_graphs + test_graphs, accounting_d_max
        )
        if clipped_d_max > accounting_d_max:
            raise ValueError(
                f"PageRank clipping failed to enforce node-DP degree bound: "
                f"clipped max degree {clipped_d_max}, D_max={accounting_d_max}."
            )

    avg_nodes = int(np.mean([graph.node_features.shape[0] for graph in train_graphs]))
    input_dim = train_graphs[0].node_features.shape[-1]
    num_clusters = max(2, ceil(avg_nodes / 2))
    katz_depth = args.Hop_1 if args.katz_depth <= 0 else args.katz_depth

    if args.non_dp:
        epsilon_edge = 0.0
        privacy_report = {
            "embedding_sigma": 0.0,
            "epsilon_total": float("inf"),
            "epsilon_embedding": float("inf"),
            "epsilon_edge": 0.0,
            "embedding_l2_sensitivity": 0.0,
            "edge_l1_sensitivity": 0.0,
            "private_adj_l1_sensitivity": 0.0,
            "elementwise_mu": 1.0 / float(accounting_d_max) if accounting_d_max > 0 else "",
            "katz_structural_sensitivity": 0.0,
            "katz_feature_sensitivity": 0.0,
        }
        args_for_inputs = argparse.Namespace(**vars(args))
        args_for_inputs.adj_mode = "public"
    else:
        epsilon_edge = epsilon * args.edge_epsilon_ratio
        elementwise_mu_value = None
        if args.aggregation_method == "katz" and args.katz_mu_mode == "inverse_dmax":
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
        args_for_inputs = args

    train_x, train_adj, prep_train = build_private_inputs(
        train_graphs, args_for_inputs, device, privacy_report["embedding_sigma"],
        epsilon_edge, privacy_report, is_train=True
    )
    val_x, val_adj, prep_val = build_private_inputs(
        val_graphs, args_for_inputs, device, 0, epsilon_edge, privacy_report, is_train=False
    )
    test_x, test_adj, prep_test = build_private_inputs(
        test_graphs, args_for_inputs, device, 0, epsilon_edge, privacy_report, is_train=False
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
    train_start = time.time()
    for epoch in range(args.epoch_2):
        run_epoch_attack(model, train_graphs, train_x, train_adj, optimizer, criterion, device, args.Batch_size, True)
        _, val_acc = run_epoch_attack(
            model, val_graphs, val_x, val_adj, optimizer, criterion, device, args.Batch_size, False
        ) if val_graphs else (0.0, 0.0)
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"epoch={epoch+1:03d} val_acc={val_acc:.4f}", flush=True)

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    _, train_acc = run_epoch_attack(model, train_graphs, train_x, train_adj, optimizer, criterion, device, args.Batch_size, False)
    _, test_acc = run_epoch_attack(model, test_graphs, test_x, test_adj, optimizer, criterion, device, args.Batch_size, False)

    bundle = {
        "model": model,
        "criterion": criterion,
        "train_graphs": train_graphs,
        "val_graphs": val_graphs,
        "test_graphs": test_graphs,
        "train_x": train_x,
        "val_x": val_x,
        "test_x": test_x,
        "train_adj": train_adj,
        "val_adj": val_adj,
        "test_adj": test_adj,
        "privacy_report": privacy_report,
        "train_acc": train_acc,
        "val_acc": best_val_acc,
        "test_acc": test_acc,
        "train_seconds": time.time() - train_start,
        "preprocess_seconds": prep_train + prep_val + prep_test,
        "d_max": accounting_d_max,
        "original_max_degree": original_d_max,
        "clipped_max_degree": clipped_d_max,
    }
    return bundle


def train_baseline_target_model(args, graphs, num_classes, device, epsilon):
    train_graphs, test_graphs = separate_data(graphs, args.seed, args.Fold)
    train_graphs = limit_graphs(train_graphs, args.max_train_graphs, args.max_nodes, args.seed)
    test_graphs = limit_graphs(test_graphs, args.max_test_graphs, args.max_nodes, args.seed + 1)
    train_graphs, val_graphs = split_train_val(train_graphs, args.val_ratio, args.seed)
    if not train_graphs or not test_graphs:
        raise ValueError("No graphs left after graph filtering.")

    method = args.baseline_method.lower()
    depth = args.Hop_1
    setattr(args, "depth", depth)
    actual_d_max = max((graph_max_degree(graph) for graph in train_graphs + val_graphs + test_graphs), default=0)
    d_max = actual_d_max if args.D_max <= 0 else args.D_max
    if d_max < actual_d_max:
        raise ValueError(f"D_max={d_max} is smaller than actual max degree {actual_d_max}.")

    avg_nodes = int(np.mean([graph.node_features.shape[0] for graph in train_graphs]))
    input_dim = train_graphs[0].node_features.shape[-1]
    num_clusters = max(2, ceil(avg_nodes / 2))
    sensitivity, formula = baseline_sensitivity(method, depth, d_max, args)

    if args.non_dp:
        epsilon_pooling = 0.0
        epsilon_embedding = float("inf")
        embedding_sigma = 0.0
        gradient_noise_multiplier = 0.0
    else:
        epsilon_pooling = epsilon * getattr(args, "pooling_epsilon_ratio", args.edge_epsilon_ratio)
        epsilon_embedding = epsilon - epsilon_pooling
        gradient_noise_multiplier = (
            calibrate_composed_gaussian_multiplier(
                epsilon_embedding,
                args.delta,
                int(np.ceil(len(train_graphs) / args.Batch_size) * args.epoch_2),
            )
            if uses_gradient_noise(method) else 0.0
        )
        embedding_sigma = 0.0 if uses_gradient_noise(method) else calibrate_gaussian_sigma(
            epsilon_embedding, args.delta, sensitivity
        )

    train_x, train_adj, prep_train = build_baseline_inputs(
        train_graphs, method, args, device, embedding_sigma, epsilon_pooling, d_max, is_train=True
    )
    val_x, val_adj, prep_val = build_baseline_inputs(
        val_graphs, method, args, device, 0.0, epsilon_pooling, d_max, is_train=False
    )
    test_x, test_adj, prep_test = build_baseline_inputs(
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
    gradient_stats = {}
    train_start = time.time()
    total_gradient_steps = int(np.ceil(len(train_graphs) / args.Batch_size) * args.epoch_2)
    gradient_clip = args.dpgnn_clip if method == "dpgnn" else (args.sagd_clip if method == "sagd" else 0.0)
    for epoch in range(args.epoch_2):
        run_baseline_epoch(
            model, train_graphs, train_x, train_adj, optimizer, criterion, device, args.Batch_size,
            train=True, method=method, gradient_noise_multiplier=gradient_noise_multiplier,
            static_gradient_sensitivity=sensitivity if method == "dpgnn" else 0.0,
            sagd_feature_sensitivity=sensitivity if method == "sagd" else 0.0,
            gradient_clip=gradient_clip, stats=gradient_stats, args=args,
        )
        _, val_acc = run_baseline_epoch(
            model, val_graphs, val_x, val_adj, optimizer, criterion, device, args.Batch_size, train=False
        ) if val_graphs else (0.0, 0.0)
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"epoch={epoch+1:03d} val_acc={val_acc:.4f}", flush=True)

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    _, train_acc = run_baseline_epoch(
        model, train_graphs, train_x, train_adj, optimizer, criterion, device, args.Batch_size, train=False
    )
    _, test_acc = run_baseline_epoch(
        model, test_graphs, test_x, test_adj, optimizer, criterion, device, args.Batch_size, train=False
    )

    privacy_report = {
        "epsilon_total": float("inf") if args.non_dp else epsilon,
        "epsilon_embedding": epsilon_embedding,
        "epsilon_edge": epsilon_pooling,
        "embedding_l2_sensitivity": 0.0 if uses_gradient_noise(method) else sensitivity,
        "embedding_sigma": embedding_sigma,
        "edge_l1_sensitivity": d_max,
        "private_adj_l1_sensitivity": d_max if args.adj_mode == "exact" else 0.0,
        "elementwise_mu": "",
        "baseline_method": method,
        "baseline_sensitivity": sensitivity,
        "baseline_sensitivity_formula": formula,
        "noise_target": "gradient" if uses_gradient_noise(method) else ("appr_operator" if method == "dpar" else "node_embedding"),
        "gradient_noise_multiplier": gradient_noise_multiplier,
        "last_gradient_l2_sensitivity": gradient_stats.get("last_gradient_l2_sensitivity", ""),
        "last_gradient_sigma": gradient_stats.get("last_gradient_sigma", ""),
        "max_gradient_l2_sensitivity": gradient_stats.get("max_gradient_l2_sensitivity", ""),
        "max_gradient_sigma": gradient_stats.get("max_gradient_sigma", ""),
        "total_gradient_steps": total_gradient_steps if uses_gradient_noise(method) else 0,
    }

    return {
        "model": model,
        "criterion": criterion,
        "train_graphs": train_graphs,
        "val_graphs": val_graphs,
        "test_graphs": test_graphs,
        "train_x": train_x,
        "val_x": val_x,
        "test_x": test_x,
        "train_adj": train_adj,
        "val_adj": val_adj,
        "test_adj": test_adj,
        "privacy_report": privacy_report,
        "train_acc": train_acc,
        "val_acc": best_val_acc,
        "test_acc": test_acc,
        "train_seconds": time.time() - train_start,
        "preprocess_seconds": prep_train + prep_val + prep_test,
        "d_max": d_max,
        "original_max_degree": actual_d_max,
        "clipped_max_degree": actual_d_max,
    }


def run_epoch_attack(model, graphs, x_list, adj_list, optimizer, criterion, device, batch_size, train):
    indices = list(range(len(graphs)))
    if train:
        random.shuffle(indices)
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for start in range(0, len(indices), batch_size):
            batch = indices[start:start + batch_size]
            if train:
                optimizer.zero_grad()
            outs, labels = [], []
            for idx in batch:
                _, _, _, out, _ = model(x_list[idx], adj_list[idx])
                outs.append(out)
                labels.append(graphs[idx].label)
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


@torch.no_grad()
def graph_forward_features(model, x, adj, label):
    _, _, _, out, graph_emb = model(x, adj)
    logits = out.detach().cpu().numpy().reshape(-1)
    probs = F.softmax(out.detach(), dim=0).cpu().numpy().reshape(-1)
    loss = float(F.cross_entropy(out.reshape(1, -1), torch.tensor([label], device=out.device)).item())
    return logits, probs, graph_emb.detach()


def node_mia_features_for_graph(model, x, adj, label, node_indices):
    base_logits, base_probs, base_emb = graph_forward_features(model, x, adj, label)
    base_conf = float(base_probs.max())
    base_entropy = entropy_from_probs(base_probs)
    base_margin = margin_from_probs(base_probs)
    base_pred_correct = float(int(base_probs.argmax() == label))
    rows = []
    degrees = adj.detach().sum(dim=1).cpu().numpy()
    for node_idx in node_indices:
        x_mask = x.clone()
        adj_mask = adj.clone()
        x_mask[node_idx] = 0
        adj_mask[node_idx, :] = 0
        adj_mask[:, node_idx] = 0
        _, masked_probs, masked_emb = graph_forward_features(model, x_mask, adj_mask, label)
        delta_probs = base_probs - masked_probs
        emb_delta = torch.norm(base_emb - masked_emb).item()
        node_vec = x[node_idx].detach().cpu().numpy()
        feature = np.concatenate([
            base_probs,
            masked_probs,
            delta_probs,
            np.asarray([
                base_conf,
                base_entropy,
                base_margin,
                base_pred_correct,
                float(np.linalg.norm(delta_probs, ord=1)),
                float(np.linalg.norm(delta_probs, ord=2)),
                float(emb_delta),
                float(np.linalg.norm(node_vec, ord=2)),
                float(degrees[node_idx]),
            ], dtype=np.float32),
        ])
        rows.append(feature)
    return rows


def run_node_mia(bundle, args, seed):
    rng = random.Random(seed)
    model = bundle["model"]
    model.eval()
    features, labels = [], []

    for member_label, graphs, x_list, adj_list in [
        (1, bundle["train_graphs"], bundle["train_x"], bundle["train_adj"]),
        (0, bundle["test_graphs"], bundle["test_x"], bundle["test_adj"]),
    ]:
        graph_indices = list(range(len(graphs)))
        rng.shuffle(graph_indices)
        for graph_idx in graph_indices[:args.max_attack_graphs]:
            n = x_list[graph_idx].shape[0]
            nodes = list(range(n))
            rng.shuffle(nodes)
            nodes = nodes[:min(args.nodes_per_graph, n)]
            rows = node_mia_features_for_graph(
                model, x_list[graph_idx], adj_list[graph_idx], graphs[graph_idx].label, nodes
            )
            features.extend(rows)
            labels.extend([member_label] * len(rows))

    return evaluate_binary_attack(features, labels, make_attack_model(args.attack_model, seed), seed)


def sample_link_pairs(adj, max_pos, rng):
    adj_np = (adj.detach().cpu().numpy() > 0).astype(np.int8)
    n = adj_np.shape[0]
    pos = [(i, j) for i in range(n) for j in range(i + 1, n) if adj_np[i, j] > 0]
    rng.shuffle(pos)
    pos = pos[:max_pos]
    pos_set = set(pos)
    neg = []
    attempts = 0
    max_attempts = max(1000, max_pos * 100)
    while len(neg) < len(pos) and attempts < max_attempts:
        i = rng.randrange(n)
        j = rng.randrange(n)
        attempts += 1
        if i == j:
            continue
        if i > j:
            i, j = j, i
        if adj_np[i, j] == 0 and (i, j) not in pos_set and (i, j) not in neg:
            neg.append((i, j))
    return pos, neg


def link_feature(x, adj, i, j):
    zi = x[i].detach().cpu().numpy()
    zj = x[j].detach().cpu().numpy()
    dot = float(np.dot(zi, zj))
    norm_i = float(np.linalg.norm(zi))
    norm_j = float(np.linalg.norm(zj))
    cos = dot / max(1e-12, norm_i * norm_j)
    l2 = float(np.linalg.norm(zi - zj))
    degrees = adj.detach().sum(dim=1).cpu().numpy()
    return np.concatenate([
        zi * zj,
        np.abs(zi - zj),
        np.asarray([dot, cos, l2, norm_i, norm_j, float(degrees[i]), float(degrees[j])], dtype=np.float32),
    ])


def run_link_stealing(bundle, args, seed):
    rng = random.Random(seed + 17)
    features, labels = [], []
    graph_sources = [
        (bundle["train_graphs"], bundle["train_x"], bundle["train_adj"]),
        (bundle["test_graphs"], bundle["test_x"], bundle["test_adj"]),
    ]
    for graphs, x_list, adj_list in graph_sources:
        graph_indices = list(range(len(graphs)))
        rng.shuffle(graph_indices)
        for graph_idx in graph_indices[:args.max_attack_graphs]:
            pos, neg = sample_link_pairs(adj_list[graph_idx], args.links_per_graph, rng)
            for i, j in pos:
                features.append(link_feature(x_list[graph_idx], adj_list[graph_idx], i, j))
                labels.append(1)
            for i, j in neg:
                features.append(link_feature(x_list[graph_idx], adj_list[graph_idx], i, j))
                labels.append(0)

    return evaluate_binary_attack(features, labels, make_attack_model(args.attack_model, seed), seed)


def append_result(path, row):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="PROTEINS")
    parser.add_argument("--level", default="node", choices=["node", "edge"])
    parser.add_argument("--attack", default="both", choices=["node_mia", "link_stealing", "both"])
    parser.add_argument("--attack_model", default="logreg", choices=["logreg", "rf", "mlp"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--Fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epsilon_list", default="1,2,4,8")
    parser.add_argument("--non_dp", action="store_true")
    parser.add_argument("--method_family", default="katz", choices=["katz", "baseline"])
    parser.add_argument("--baseline_method", default="gap", choices=["gap", "dpar", "dpgnn", "sagd"])
    parser.add_argument("--aggregation_method", default="katz", choices=["original", "katz"])
    parser.add_argument("--pooling_method", default="diffpool", choices=["diffpool", "mean", "max", "sum", "topk", "sag"])
    parser.add_argument("--pool_ratio", type=float, default=0.5)
    parser.add_argument("--Hop_1", type=int, default=2)
    parser.add_argument("--katz_beta", type=float, default=0.15)
    parser.add_argument("--katz_depth", type=int, default=-1)
    parser.add_argument("--katz_mu_mode", default="inverse_dmax", choices=["default", "inverse_dmax"])
    parser.add_argument("--D_max", type=int, default=25)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--edge_epsilon_ratio", type=float, default=0.2)
    parser.add_argument("--pooling_epsilon_ratio", type=float, default=0.2)
    parser.add_argument("--adj_mode", default="exact", choices=["public", "exact"])
    parser.add_argument("--private_adj_mode", default="elementwise", choices=["raw", "elementwise"])
    parser.add_argument("--epoch_2", type=int, default=20)
    parser.add_argument("--Batch_size", type=int, default=128)
    parser.add_argument("--hidden_dimension_2", type=int, default=32)
    parser.add_argument("--lr_2", type=float, default=0.01)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--max_train_graphs", type=int, default=0)
    parser.add_argument("--max_test_graphs", type=int, default=0)
    parser.add_argument("--max_nodes", type=int, default=0)
    parser.add_argument("--max_attack_graphs", type=int, default=200)
    parser.add_argument("--nodes_per_graph", type=int, default=8)
    parser.add_argument("--links_per_graph", type=int, default=32)
    parser.add_argument("--degree_as_tag", action="store_true")
    parser.add_argument("--dpar_alpha", type=float, default=0.15)
    parser.add_argument("--dpar_c1", type=float, default=1.0)
    parser.add_argument("--dpgnn_clip", type=float, default=1.0)
    parser.add_argument("--sagd_clip", type=float, default=0.0)
    parser.add_argument("--sagd_gamma", type=float, default=0.15)
    parser.add_argument("--sagd_loss_lipschitz", type=float, default=1.41421356237)
    parser.add_argument("--sagd_loss_derivative_lipschitz", type=float, default=1.0)
    parser.add_argument("--sagd_activation_lipschitz", type=float, default=1.0)
    parser.add_argument("--sagd_activation_derivative_lipschitz", type=float, default=1.0)
    parser.add_argument("--result_file", default="results/privacy_attacks/privacy_attacks.csv")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    graphs, num_classes = load_data(args.dataset, args.degree_as_tag)
    eps_values = [float("inf")] if args.non_dp else parse_float_list(args.epsilon_list)

    for epsilon in eps_values:
        print(f"\n===== target dataset={args.dataset} level={args.level} epsilon={epsilon} =====", flush=True)
        bundle = train_target_model(args, graphs, num_classes, device, epsilon)

        attacks = []
        if args.attack in ("node_mia", "both"):
            print("Running node membership inference attack...", flush=True)
            attacks.append(("node_mia", run_node_mia(bundle, args, args.seed)))
        if args.attack in ("link_stealing", "both"):
            print("Running link stealing attack...", flush=True)
            attacks.append(("link_stealing", run_link_stealing(bundle, args, args.seed)))

        for attack_name, metrics in attacks:
            row = {
                "dataset": args.dataset,
                "level": args.level,
                "method_family": args.method_family,
                "method": (
                    ("non_dp_" if args.non_dp else "") +
                    (args.baseline_method if args.method_family == "baseline" else args.aggregation_method)
                ),
                "pooling_method": args.pooling_method,
                "attack": attack_name,
                "attack_model": args.attack_model,
                "seed": args.seed,
                "epsilon": "non_dp" if args.non_dp else epsilon,
                "delta": args.delta,
                "hop": args.Hop_1,
                "katz_beta": args.katz_beta if args.aggregation_method == "katz" else "",
                "katz_depth": args.Hop_1 if args.katz_depth <= 0 else args.katz_depth,
                "d_max": bundle["d_max"],
                "adj_mode": args.adj_mode if not args.non_dp else "public",
                "private_adj_mode": args.private_adj_mode if not args.non_dp else "none",
                "elementwise_mu": bundle["privacy_report"].get("elementwise_mu", ""),
                "embedding_l2_sensitivity": bundle["privacy_report"].get("embedding_l2_sensitivity", ""),
                "private_adj_l1_sensitivity": bundle["privacy_report"].get("private_adj_l1_sensitivity", ""),
                "baseline_sensitivity": bundle["privacy_report"].get("baseline_sensitivity", ""),
                "noise_target": bundle["privacy_report"].get("noise_target", ""),
                "gradient_noise_multiplier": bundle["privacy_report"].get("gradient_noise_multiplier", ""),
                "target_train_acc": round(bundle["train_acc"], 6),
                "target_val_acc": round(bundle["val_acc"], 6),
                "target_test_acc": round(bundle["test_acc"], 6),
                "attack_auc": round(metrics["auc"], 6) if not np.isnan(metrics["auc"]) else "",
                "attack_ap": round(metrics["ap"], 6) if not np.isnan(metrics["ap"]) else "",
                "attack_acc": round(metrics["acc"], 6) if not np.isnan(metrics["acc"]) else "",
                "attack_f1": round(metrics["f1"], 6) if not np.isnan(metrics["f1"]) else "",
                "attack_samples": metrics["n"],
                "max_attack_graphs": args.max_attack_graphs,
                "nodes_per_graph": args.nodes_per_graph,
                "links_per_graph": args.links_per_graph,
                "num_train": len(bundle["train_graphs"]),
                "num_val": len(bundle["val_graphs"]),
                "num_test": len(bundle["test_graphs"]),
                "preprocess_seconds": round(bundle["preprocess_seconds"], 3),
                "train_seconds": round(bundle["train_seconds"], 3),
            }
            append_result(args.result_file, row)
            print(row, flush=True)

    print(f"Saved attack results to {args.result_file}", flush=True)


if __name__ == "__main__":
    main()
