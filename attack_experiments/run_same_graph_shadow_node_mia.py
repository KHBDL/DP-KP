import argparse
import csv
import os
import random
import sys
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Untils.data_process import load_data
from run_privacy_attacks import (
    make_attack_model,
    node_mia_features_for_graph,
    parse_float_list,
    set_seed,
    train_target_model,
)


def eval_trained_attack(attack_model, x, y):
    if len(y) < 2 or len(set(y.tolist())) < 2:
        return {"auc": float("nan"), "ap": float("nan"), "acc": float("nan"), "f1": float("nan"), "n": int(len(y))}
    if hasattr(attack_model, "predict_proba"):
        scores = attack_model.predict_proba(x)[:, 1]
    else:
        scores = attack_model.decision_function(x)
    pred = (scores >= 0.5).astype(np.int64)
    return {
        "auc": float(roc_auc_score(y, scores)),
        "ap": float(average_precision_score(y, scores)),
        "acc": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "n": int(len(y)),
    }


def append_result(path, row):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def args_for_model(base_args, seed, fold, non_dp):
    values = vars(base_args).copy()
    values["seed"] = seed
    values["Fold"] = fold
    values["non_dp"] = non_dp
    return argparse.Namespace(**values)


def train_bundle_with_fresh_data(model_args, device, epsilon):
    graphs, num_classes = load_data(model_args.dataset)
    return train_target_model(model_args, graphs, num_classes, device, epsilon)


def build_donor_pool(x_list, max_graphs, nodes_per_graph, seed):
    rng = random.Random(seed)
    donors = []
    graph_indices = list(range(len(x_list)))
    rng.shuffle(graph_indices)
    for graph_idx in graph_indices[:max_graphs]:
        n_nodes = x_list[graph_idx].shape[0]
        node_indices = list(range(n_nodes))
        rng.shuffle(node_indices)
        for node_idx in node_indices[:min(nodes_per_graph, n_nodes)]:
            donors.append(x_list[graph_idx][node_idx].detach().clone())
    if not donors:
        raise ValueError("No donor nodes available for same-graph replacement attack.")
    return donors


def replace_node_feature(x, node_idx, donor, mode, rng):
    x_replaced = x.clone()
    if mode == "feature":
        x_replaced[node_idx] = donor.to(x.device)
    elif mode == "shuffle_feature":
        donor_vec = donor.detach().clone()
        perm = torch.randperm(donor_vec.numel(), device=donor_vec.device)
        x_replaced[node_idx] = donor_vec[perm].to(x.device)
    elif mode == "zero_feature":
        x_replaced[node_idx] = 0
    else:
        raise ValueError("replacement_mode must be feature, shuffle_feature, or zero_feature")
    return x_replaced


def collect_same_graph_replacement_features(bundle, max_attack_graphs, nodes_per_graph,
                                            donor_graphs, donor_nodes_per_graph,
                                            replacement_mode, seed):
    rng = random.Random(seed)
    model = bundle["model"]
    model.eval()
    features, labels = [], []

    # Donors come from holdout/test graphs, but both member and non-member
    # samples are evaluated in the same training graph context.
    donors = build_donor_pool(bundle["test_x"], donor_graphs, donor_nodes_per_graph, seed + 17)

    graph_indices = list(range(len(bundle["train_graphs"])))
    rng.shuffle(graph_indices)
    with torch.no_grad():
        for graph_idx in graph_indices[:max_attack_graphs]:
            x = bundle["train_x"][graph_idx]
            adj = bundle["train_adj"][graph_idx]
            label = bundle["train_graphs"][graph_idx].label
            n_nodes = x.shape[0]
            node_indices = list(range(n_nodes))
            rng.shuffle(node_indices)
            node_indices = node_indices[:min(nodes_per_graph, n_nodes)]

            for node_idx in node_indices:
                member_row = node_mia_features_for_graph(model, x, adj, label, [node_idx])[0]
                donor = donors[rng.randrange(len(donors))]
                x_replaced = replace_node_feature(x, node_idx, donor, replacement_mode, rng)
                nonmember_row = node_mia_features_for_graph(model, x_replaced, adj, label, [node_idx])[0]
                features.append(member_row)
                labels.append(1)
                features.append(nonmember_row)
                labels.append(0)

    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def run_one_epsilon(args, device, epsilon):
    target_non_dp = args.target_mode == "non_dp"
    shadow_non_dp = args.shadow_mode == "non_dp"

    print(
        f"\n===== Same-graph Shadow Node-MIA dataset={args.dataset} epsilon={epsilon} "
        f"target={args.target_mode} shadow={args.shadow_mode} replacement={args.replacement_mode} =====",
        flush=True,
    )

    target_args = args_for_model(args, args.seed, args.Fold, target_non_dp)
    target_bundle = train_bundle_with_fresh_data(target_args, device, epsilon)
    target_x, target_y = collect_same_graph_replacement_features(
        target_bundle,
        args.max_attack_graphs,
        args.nodes_per_graph,
        args.donor_graphs,
        args.donor_nodes_per_graph,
        args.replacement_mode,
        args.seed + 10000,
    )
    print(f"Target same-graph attack eval samples: {len(target_y)}", flush=True)

    shadow_x_parts, shadow_y_parts = [], []
    shadow_train_acc, shadow_test_acc = [], []
    shadow_feature_seconds = 0.0
    for shadow_idx in range(args.num_shadow_models):
        shadow_seed = args.shadow_seed_offset + args.seed * 100 + shadow_idx
        shadow_fold = (args.Fold + shadow_idx + 1) % 10
        print(f"Training shadow model {shadow_idx + 1}/{args.num_shadow_models}: seed={shadow_seed}, fold={shadow_fold}", flush=True)
        shadow_args = args_for_model(args, shadow_seed, shadow_fold, shadow_non_dp)
        shadow_bundle = train_bundle_with_fresh_data(shadow_args, device, epsilon)
        t0 = time.time()
        sx, sy = collect_same_graph_replacement_features(
            shadow_bundle,
            args.max_shadow_attack_graphs,
            args.nodes_per_graph,
            args.donor_graphs,
            args.donor_nodes_per_graph,
            args.replacement_mode,
            shadow_seed + 20000,
        )
        shadow_feature_seconds += time.time() - t0
        shadow_x_parts.append(sx)
        shadow_y_parts.append(sy)
        shadow_train_acc.append(shadow_bundle["train_acc"])
        shadow_test_acc.append(shadow_bundle["test_acc"])
        print(f"Shadow same-graph samples: {len(sy)}", flush=True)

    shadow_x = np.concatenate(shadow_x_parts, axis=0)
    shadow_y = np.concatenate(shadow_y_parts, axis=0)
    attack_model = make_attack_model(args.attack_model, args.seed)
    attack_model.fit(shadow_x, shadow_y)

    target_metrics = eval_trained_attack(attack_model, target_x, target_y)
    shadow_metrics = eval_trained_attack(attack_model, shadow_x, shadow_y)
    report = target_bundle["privacy_report"]

    row = {
        "dataset": args.dataset,
        "level": "node",
        "attack": "same_graph_shadow_node_mia",
        "method_family": args.method_family,
        "method": args.baseline_method if args.method_family == "baseline" else args.aggregation_method,
        "replacement_mode": args.replacement_mode,
        "attack_model": args.attack_model,
        "target_mode": args.target_mode,
        "shadow_mode": args.shadow_mode,
        "num_shadow_models": args.num_shadow_models,
        "seed": args.seed,
        "epsilon": "non_dp" if target_non_dp else epsilon,
        "shadow_epsilon": "non_dp" if shadow_non_dp else epsilon,
        "delta": args.delta,
        "hop": args.Hop_1,
        "katz_beta": args.katz_beta if args.aggregation_method == "katz" else "",
        "katz_depth": args.Hop_1 if args.katz_depth <= 0 else args.katz_depth,
        "d_max": target_bundle["d_max"],
        "adj_mode": args.adj_mode if not target_non_dp else "public",
        "private_adj_mode": args.private_adj_mode if not target_non_dp else "none",
        "elementwise_mu": report.get("elementwise_mu", ""),
        "embedding_l2_sensitivity": report.get("embedding_l2_sensitivity", ""),
        "private_adj_l1_sensitivity": report.get("private_adj_l1_sensitivity", ""),
        "baseline_sensitivity": report.get("baseline_sensitivity", ""),
        "noise_target": report.get("noise_target", ""),
        "gradient_noise_multiplier": report.get("gradient_noise_multiplier", ""),
        "target_train_acc": round(target_bundle["train_acc"], 6),
        "target_val_acc": round(target_bundle["val_acc"], 6),
        "target_test_acc": round(target_bundle["test_acc"], 6),
        "shadow_train_acc_mean": round(float(np.mean(shadow_train_acc)), 6),
        "shadow_test_acc_mean": round(float(np.mean(shadow_test_acc)), 6),
        "target_attack_auc": round(target_metrics["auc"], 6),
        "target_attack_ap": round(target_metrics["ap"], 6),
        "target_attack_acc": round(target_metrics["acc"], 6),
        "target_attack_f1": round(target_metrics["f1"], 6),
        "shadow_attack_auc": round(shadow_metrics["auc"], 6),
        "shadow_attack_ap": round(shadow_metrics["ap"], 6),
        "shadow_attack_acc": round(shadow_metrics["acc"], 6),
        "shadow_attack_f1": round(shadow_metrics["f1"], 6),
        "target_attack_samples": target_metrics["n"],
        "shadow_attack_samples": shadow_metrics["n"],
        "max_attack_graphs": args.max_attack_graphs,
        "max_shadow_attack_graphs": args.max_shadow_attack_graphs,
        "nodes_per_graph": args.nodes_per_graph,
        "donor_graphs": args.donor_graphs,
        "donor_nodes_per_graph": args.donor_nodes_per_graph,
        "num_train": len(target_bundle["train_graphs"]),
        "num_val": len(target_bundle["val_graphs"]),
        "num_test": len(target_bundle["test_graphs"]),
        "target_preprocess_seconds": round(target_bundle["preprocess_seconds"], 3),
        "target_train_seconds": round(target_bundle["train_seconds"], 3),
        "shadow_feature_seconds": round(shadow_feature_seconds, 3),
    }
    append_result(args.result_file, row)
    print(row, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="PROTEINS")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--Fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epsilon_list", default="1,2,4,8")
    parser.add_argument("--target_mode", default="dp", choices=["dp", "non_dp"])
    parser.add_argument("--shadow_mode", default="dp", choices=["dp", "non_dp"])
    parser.add_argument("--num_shadow_models", type=int, default=3)
    parser.add_argument("--shadow_seed_offset", type=int, default=1000)
    parser.add_argument("--replacement_mode", default="feature", choices=["feature", "shuffle_feature", "zero_feature"])
    parser.add_argument("--attack_model", default="logreg", choices=["logreg", "rf", "mlp"])
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
    parser.add_argument("--max_shadow_attack_graphs", type=int, default=200)
    parser.add_argument("--nodes_per_graph", type=int, default=8)
    parser.add_argument("--donor_graphs", type=int, default=200)
    parser.add_argument("--donor_nodes_per_graph", type=int, default=8)
    parser.add_argument("--dpar_alpha", type=float, default=0.15)
    parser.add_argument("--dpar_c1", type=float, default=1.0)
    parser.add_argument("--dpgnn_clip", type=float, default=1.0)
    parser.add_argument("--sagd_clip", type=float, default=0.0)
    parser.add_argument("--sagd_gamma", type=float, default=0.15)
    parser.add_argument("--sagd_loss_lipschitz", type=float, default=1.41421356237)
    parser.add_argument("--sagd_loss_derivative_lipschitz", type=float, default=1.0)
    parser.add_argument("--sagd_activation_lipschitz", type=float, default=1.0)
    parser.add_argument("--sagd_activation_derivative_lipschitz", type=float, default=1.0)
    parser.add_argument("--result_file", default="results/privacy_attacks/same_graph_shadow_node_mia.csv")
    args = parser.parse_args()

    args.level = "node"
    args.attack = "same_graph_shadow_node_mia"
    set_seed(args.seed)
    device = torch.device(args.device)
    eps_values = [float("inf")] if args.target_mode == "non_dp" else parse_float_list(args.epsilon_list)
    for epsilon in eps_values:
        run_one_epsilon(args, device, epsilon)
    print(f"Saved same-graph shadow Node-MIA results to {args.result_file}", flush=True)


if __name__ == "__main__":
    main()
