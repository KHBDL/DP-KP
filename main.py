import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,JumpingKnowledge
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from math import ceil
from torch_geometric.utils import to_dense_adj, to_dense_batch,unbatch_edge_index
import numpy as np
import networkx as nx
import random
from sklearn.model_selection import StratifiedKFold
import math
import argparse
import pprint
import warnings
import time
import os
import csv
import scipy.sparse as sp
from tqdm import tqdm
from scipy.sparse import coo_matrix
from sklearn.model_selection import KFold
from torch_geometric.loader import NeighborSampler
from tqdm import tqdm
import torch_geometric as tg
from torch_scatter import scatter
from Untils.data_process import load_data,separate_data
from Untils.perturbation import Aggre_Perturb, Katz_Aggre_Perturb, perturb_adj_continuous, perturb_adj_elementwise_continuous
from Untils.transform import index_to_dense,shuffle_and_group
from Untils.calculation import privacy_accounting
from Untils.model import PGP

def graph_max_degree(graph):
    edge_index = graph.edge_mat
    if torch.is_tensor(edge_index):
        src = edge_index[0].cpu().numpy()
    else:
        src = edge_index[0]
    if len(src) == 0:
        return 0
    return int(np.bincount(src.astype(int), minlength=graph.node_features.shape[0]).max())


def csv_fieldnames_for_append(path, default_fieldnames):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "r", newline="", encoding="utf-8") as csv_file:
            reader = csv.reader(csv_file)
            header = next(reader, None)
            if header:
                return header
    return default_fieldnames


def _refresh_graph_edges(graph, edges):
    clipped_g = nx.Graph()
    clipped_g.add_nodes_from(range(graph.node_features.shape[0]))
    clipped_g.add_edges_from(edges)
    graph.g = clipped_g
    graph.neighbors = [[] for _ in range(len(clipped_g))]
    for u, v in clipped_g.edges():
        graph.neighbors[u].append(v)
        graph.neighbors[v].append(u)
    graph.max_neighbor = max((len(items) for items in graph.neighbors), default=0)
    directed_edges = [list(pair) for pair in clipped_g.edges()]
    directed_edges.extend([[v, u] for u, v in directed_edges])
    if directed_edges:
        graph.edge_mat = torch.LongTensor(directed_edges).transpose(0, 1)
    else:
        graph.edge_mat = torch.empty((2, 0), dtype=torch.long)


def clip_graph_by_pagerank(graph, d_max, alpha=0.85):
    """Clip an undirected graph to max degree d_max using personalized PageRank preferences."""
    if d_max <= 0 or graph_max_degree(graph) <= d_max:
        return graph

    n_nodes = graph.node_features.shape[0]
    original_g = nx.Graph()
    original_g.add_nodes_from(range(n_nodes))
    original_g.add_edges_from((int(u), int(v)) for u, v in graph.g.edges())

    directed_scores = {}
    for src in range(n_nodes):
        neighbors = list(original_g.neighbors(src))
        if len(neighbors) <= d_max:
            for dst in neighbors:
                directed_scores[(src, dst)] = 1.0
            continue
        personalization = {node: 0.0 for node in range(n_nodes)}
        personalization[src] = 1.0
        scores = nx.pagerank(original_g, alpha=alpha, personalization=personalization)
        ranked_neighbors = sorted(neighbors, key=lambda dst: (-scores.get(dst, 0.0), dst))
        for rank, dst in enumerate(ranked_neighbors[:d_max]):
            directed_scores[(src, dst)] = scores.get(dst, 0.0) + 1.0 / (rank + 1)

    candidates = []
    for u, v in original_g.edges():
        score = directed_scores.get((u, v), 0.0) + directed_scores.get((v, u), 0.0)
        if score > 0:
            candidates.append((score, min(u, v), max(u, v)))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

    degrees = [0] * n_nodes
    kept_edges = []
    for _, u, v in candidates:
        if degrees[u] < d_max and degrees[v] < d_max:
            kept_edges.append((u, v))
            degrees[u] += 1
            degrees[v] += 1

    _refresh_graph_edges(graph, kept_edges)
    return graph


def clip_graphs_by_pagerank(graphs, d_max):
    before = max((graph_max_degree(graph) for graph in graphs), default=0)
    for graph in graphs:
        clip_graph_by_pagerank(graph, d_max)
    after = max((graph_max_degree(graph) for graph in graphs), default=0)
    return before, after


def is_numeric_value(value):
    return value not in ("", None)


def katz_elementwise_mu_for_propagation(args, privacy_report):
    if args.aggregation_method != "katz":
        return None
    if args.level == "node" or args.katz_mu_mode == "inverse_dmax":
        value = privacy_report.get("elementwise_mu", "")
        return float(value) if is_numeric_value(value) else None
    return None

warnings.filterwarnings("ignore", category=UserWarning)

# Reproducibility.
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def early_stopping(indicator_style, val_indicator, best_val_indicator, patience, patience_counter, model, model_path):
    """Update early-stopping state and save the best checkpoint."""
    if indicator_style == 'loss':
        if val_indicator < best_val_indicator:
            best_val_indicator = val_indicator
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
            return best_val_indicator, patience_counter, False
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                return best_val_indicator, patience_counter, True
            else:
                return best_val_indicator, patience_counter, False
    elif indicator_style == 'acc':
        if val_indicator > best_val_indicator:
            best_val_indicator = val_indicator
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
            return best_val_indicator, patience_counter, False
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                return best_val_indicator, patience_counter, True
            else:
                return best_val_indicator, patience_counter, False

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default="cuda:0", help="device_name")
    parser.add_argument('--dataset', type=str, default="NCI1", help="dataset")
    parser.add_argument('--level', type=str, default="node", help="dataset")
    parser.add_argument('--merge_mode', type=str, default="ATT", help="merge_mode")
    parser.add_argument('--pooling_method', type=str, default="diffpool",
                        choices=["diffpool", "mean", "max", "sum", "topk", "sag"],
                        help="Pooling/readout method after private preprocessing")
    parser.add_argument('--pool_ratio', type=float, default=0.5, help="Pooling ratio for TopKPool and SAGPool")

    parser.add_argument('--Fold', type=int, default=0, help="The index of train and test data")
    parser.add_argument('--Hop_1', type=int, default=5, help="The GNN hop of module 1")
    parser.add_argument('--hidden_dimension_2', type=int, default=16, help="The hidden_dimension of module 2")
    parser.add_argument('--Batch_size', type=int, default=32, help="Batch_size")
    parser.add_argument('--epoch_2', type=int, default=50, help="Epoch number of module 2")
    parser.add_argument('--D_max', type=int, default=-1, help="The number of max neighboring nodes")

    parser.add_argument('--lr_2', type=float, default=0.01, help="The learning rate of module 2")
    parser.add_argument('--entropy_mean_2', type=float, default=0.001, help="The entropy_mean of module 2")
    parser.add_argument('--Noise_Scale_adj_perturb', type=float, default=2.0, help="The noise scale of adj perturbation")
    parser.add_argument('--aggregation_method', type=str, default="original", choices=["original", "katz"], help="Private aggregation method")
    parser.add_argument('--katz_beta', type=float, default=0.15, help="Katz decay factor")
    parser.add_argument('--katz_depth', type=int, default=-1, help="Katz truncation depth; -1 uses Hop_1")
    parser.add_argument('--katz_mu_mode', type=str, default="default", choices=["default", "inverse_dmax"],
                        help="default keeps previous edge behavior; inverse_dmax uses mu=1/D_max for Katz accounting and propagation.")
    parser.add_argument('--private_adj_mode', type=str, default="raw", choices=["raw", "elementwise"],
                        help="raw perturbs binary A; elementwise perturbs mu*A for private pooling.")
    parser.add_argument('--epsilon_list', type=str, default="2,4,8,10,12,16", help="Comma-separated privacy budgets")
    parser.add_argument('--delta', type=float, default=1e-5, help="DP delta")
    parser.add_argument('--edge_epsilon_ratio', type=float, default=0.2, help="Fraction of total epsilon for edge perturbation when Noise_Scale_adj_perturb is invalid")
    parser.add_argument('--result_file', type=str, default="", help="CSV file for summarized results")

    parser.add_argument('--degree_as_tag', action='store_true', help="The flag of degree as tag")
    parser.add_argument('--Multi_GNN_output_flag_1', action='store_true', help="The flag that module 1 outputs multi GNN results")


    args = parser.parse_args()
    os.makedirs("./model_save/PGP_Graph_Classification", exist_ok=True)
    os.makedirs("./results", exist_ok=True)
    if not args.result_file:
        args.result_file = f"./results/{args.dataset}_{args.level}_{args.aggregation_method}_results.csv"
    if args.aggregation_method == 'katz' and args.Multi_GNN_output_flag_1:
        raise ValueError("Katz strict accounting releases one diffusion embedding; do not set --Multi_GNN_output_flag_1 with --aggregation_method katz.")
    print("The Parameters:")
    pprint.pprint(vars(args))

    device = torch.device(args.device)

    dataset = args.dataset
    degree_as_tag = args.degree_as_tag
    epsilon_list = [float(item) for item in args.epsilon_list.split(',') if item.strip()]
    for epsilon_item in epsilon_list:
        seed = 0
        fold_idx = 0
        graphs, num_classes = load_data(dataset, degree_as_tag)
        if dataset == 'ECG':
            train_graphs = graphs[:19634][:int(19634*0.4)]
            test_graphs = graphs[19634:][:int(2203*0.4)]
        else:
            train_graphs, test_graphs = separate_data(graphs, seed, args.Fold)
        original_d_max = max((graph_max_degree(graph) for graph in train_graphs + test_graphs), default=0)
        accounting_d_max = args.D_max if args.D_max > 0 else original_d_max
        clipped_d_max = original_d_max
        # For node-level DP, first clip the public graph topology to the declared
        # degree bound, then account privacy on the clipped graph.
        if args.level == 'node':
            if args.D_max <= 0:
                raise ValueError("Bounded-degree node-DP requires a public positive --D_max.")
            original_d_max, clipped_d_max = clip_graphs_by_pagerank(train_graphs + test_graphs, args.D_max)
            print(f'Node-level PageRank clipping: original max degree {original_d_max}, clipped max degree {clipped_d_max}, D_max {args.D_max}')
            if clipped_d_max > args.D_max:
                raise ValueError(
                    f"PageRank clipping failed to enforce bounded-degree node-DP: "
                    f"clipped max degree {clipped_d_max}, D_max={args.D_max}."
                )
        print('The num of Train graph:',len(train_graphs))
        print('The num of Test graph:',len(test_graphs))

        print('>>>>>>>>>>>>> Step 1. Aggregation & Adj Perturbation <<<<<<<<<<<<<<<<<<')


        Hop_1 = args.Hop_1
        katz_depth = Hop_1 if args.katz_depth <= 0 else args.katz_depth
        epsilon_edge = args.Noise_Scale_adj_perturb
        if epsilon_edge <= 0 or epsilon_edge >= epsilon_item:
            epsilon_edge = epsilon_item * args.edge_epsilon_ratio
        elementwise_mu_value = None
        if args.aggregation_method == 'katz' and args.katz_mu_mode == 'inverse_dmax':
            if accounting_d_max <= 0:
                raise ValueError("--katz_mu_mode inverse_dmax requires a positive D_max or non-empty graph set.")
            elementwise_mu_value = 1.0 / float(accounting_d_max)
        # Calibrate privacy parameters for the selected aggregation method.
        privacy_report = privacy_accounting(
            epsilon_item,
            delta=args.delta,
            level=args.level,
            d_max=accounting_d_max,
            hop=Hop_1,
            epsilon_edge=epsilon_edge,
            aggregation=args.aggregation_method,
            beta=args.katz_beta,
            depth=katz_depth,
            elementwise_mu_value=elementwise_mu_value,
            private_adj_mode=args.private_adj_mode,
        )
        Noise_Scale_1 = privacy_report['embedding_sigma']
        print('Privacy accounting:')
        pprint.pprint(privacy_report)
        Multi_GNN_output_flag_1 = args.Multi_GNN_output_flag_1

        # One-shot private preprocessing for training graphs.
        train_aggre_perturb_list = []
        adj_noise_list_train = []
        num_classes = 0
        average_nodes = 0
        average_edges = 0
        for index, data in tqdm(enumerate(train_graphs),total=len(train_graphs),leave = True):
            if (data.label + 1) > num_classes:
                num_classes = data.label+1
            average_nodes += data.node_features.shape[0]
            average_edges += (data.edge_mat.shape[1])/2
            num_nodes = data.node_features.shape[0]
            adj_dense = index_to_dense(data.edge_mat, num_nodes=num_nodes).to(device)
            row_norms = torch.norm(data.node_features, p=2, dim=1, keepdim=True)
            row_norms[row_norms == 0] = 1
            x_normalized = data.node_features / row_norms
            feature = torch.tensor(x_normalized).to(device)
            train_loader_D = NeighborSampler(data.edge_mat, node_idx=None,sizes=[args.D_max], batch_size=feature.shape[0], shuffle=False,num_workers=1)
            for batch_size, n_id, adjs in train_loader_D:
                adj_dense_D = index_to_dense(adjs.edge_index, num_nodes=num_nodes).to(device)
                if args.aggregation_method == 'katz':
                    Agg_result = Katz_Aggre_Perturb(
                        adj_dense_D,
                        feature,
                        noise_scale=Noise_Scale_1,
                        depth=katz_depth,
                        beta=args.katz_beta,
                        multi_GNN_output_flag=Multi_GNN_output_flag_1,
                        elementwise_mu=katz_elementwise_mu_for_propagation(args, privacy_report),
                    )
                else:
                    Agg_result = Aggre_Perturb(adj_dense_D, feature, noise_scale=Noise_Scale_1, hop=Hop_1, multi_GNN_output_flag=Multi_GNN_output_flag_1)
            train_aggre_perturb_list.append(Agg_result)
            if args.private_adj_mode == 'elementwise':
                mu = privacy_report.get('elementwise_mu', "")
                if mu == "":
                    raise ValueError("--private_adj_mode elementwise requires Katz elementwise_mu.")
                adj_noise = perturb_adj_elementwise_continuous(
                    adj_dense,
                    epsilon=epsilon_edge,
                    noise_type='laplace',
                    noise_seed=46,
                    delta=args.delta,
                    sensitivity=privacy_report['private_adj_l1_sensitivity'],
                    elementwise_mu=float(mu),
                ).float().to(device)
            else:
                adj_noise = perturb_adj_continuous(
                    adj_dense,
                    epsilon=epsilon_edge,
                    noise_type='laplace',
                    noise_seed=46,
                    delta=args.delta,
                    sensitivity=privacy_report['edge_l1_sensitivity'],
                ).float().to(device)
            adj_noise_list_train.append(adj_noise)
        average_nodes = int(average_nodes/len(train_graphs))
        average_edges = int(average_edges/len(train_graphs))
        print('The average node number is:',average_nodes)
        print('The average edge number is:',average_edges)
        # Build non-private test representations for evaluation.
        test_aggre_perturb_list = []
        adj_list_test = []
        for data in test_graphs:
            num_nodes = data.node_features.shape[0]
            adj_dense = index_to_dense(data.edge_mat, num_nodes=num_nodes).to(device)
            row_norms = torch.norm(data.node_features, p=2, dim=1, keepdim=True)
            row_norms[row_norms == 0] = 1
            x_normalized = data.node_features / row_norms
            feature = torch.tensor(x_normalized).to(device)
            if args.aggregation_method == 'katz':
                Agg_result = Katz_Aggre_Perturb(
                    adj_dense,
                    feature,
                    noise_scale=0,
                    depth=katz_depth,
                    beta=args.katz_beta,
                    multi_GNN_output_flag=Multi_GNN_output_flag_1,
                    elementwise_mu=katz_elementwise_mu_for_propagation(args, privacy_report),
                )
            else:
                Agg_result = Aggre_Perturb(adj_dense, feature, noise_scale=0, hop=Hop_1, multi_GNN_output_flag=Multi_GNN_output_flag_1)
            test_aggre_perturb_list.append(Agg_result)
            adj_list_test.append(adj_dense)
        print('>>>>>>>>>>>>> Step 1: Finished <<<<<<<<<<<<<<<<<<')

        print('>>>>>>>>>>>>> Step 2. PGP Training  <<<<<<<<<<<<<<<<<<')
        num_node_features = train_graphs[0].node_features.shape[-1]
        hidden_dimension_2 = args.hidden_dimension_2
        lr_2 = args.lr_2

        if Multi_GNN_output_flag_1:
            model = PGP(input_dim = num_node_features * (Hop_1), hidden_dim = int((hidden_dimension_2 * (Hop_1+1))/2),
                             output_dim = num_classes, num_clusters = ceil(average_nodes/2), device = device,
                             merge_mode = args.merge_mode, pooling_method=args.pooling_method,
                             pool_ratio=args.pool_ratio).to(device)
        else:
            model = PGP(input_dim = num_node_features, hidden_dim = hidden_dimension_2,
                             output_dim = num_classes, num_clusters = ceil(average_nodes/2), device = device,
                             merge_mode = args.merge_mode, pooling_method=args.pooling_method,
                             pool_ratio=args.pool_ratio).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_2)

        Batch_size = args.Batch_size
        entropy_mean_2 = args.entropy_mean_2
        epoch_2 = args.epoch_2
        train_graph_index_all = shuffle_and_group(len(train_graphs), Batch_size)

        # Cross-validation over grouped training graphs.
        test_acc_folder = []
        for K in range(len(train_graph_index_all)):
            print(f"Training fold {K+1}/{len(train_graph_index_all)}...")
            train_graph_index = train_graph_index_all[:K] + train_graph_index_all[K+1:]
            valid_graph_index = train_graph_index_all[K]

            patience = int(epoch_2/2)
            indicator_style = 'loss'
            if indicator_style == 'loss':
                best_val_indicator = float('inf')
            elif indicator_style == 'acc':
                best_val_indicator = 0
            patience_counter = 0
            model_path_fold = f"./model_save/PGP_Graph_Classification/best_model_fold{K}_"+args.dataset+".pth"
            for epoch in range(epoch_2):
                model.train()
                total_loss = 0
                correct = 0
                x_ass_list_train = []
                s_purturb_list_train = []
                iter_num = 0
                for i in range(len(train_graph_index)):
                    optimizer.zero_grad()
                    out_all = []
                    target_all = []
                    entropy_mean_all = 0
                    for j in range(len(train_graph_index[i])):
                        index_graph = train_graph_index[i][j]
                        x_ass, cluster_assignments, entropy_mean, out, graph_emb = model(train_aggre_perturb_list[index_graph],adj_noise_list_train[index_graph])
                        x_ass_list_train.append(x_ass)
                        s_purturb_list_train.append(cluster_assignments)
                        out_all.append(out)
                        target_all.append(train_graphs[index_graph].label)
                        entropy_mean_all += entropy_mean


                    out_all = torch.stack(out_all, dim=0).to(device)
                    target_all = torch.tensor(target_all).view(-1).to(device)
                    loss = criterion(out_all, target_all)
                    loss.backward()
                    optimizer.step()
                    iter_num += 1
                    total_loss += loss.item()
                    pred = out_all.argmax(dim=1)
                    correct += (pred == target_all).sum().item()
                train_loss = total_loss / iter_num
                train_acc = correct / len(train_graphs)

                model.eval()
                correct = 0
                val_loss = 0
                with torch.no_grad():
                    x_ass_list_test = []
                    s_purturb_list_test = []
                    out_all = []
                    target_all = []
                    for k in valid_graph_index:
                        x_ass, cluster_assignments, entropy_mean, out, graph_emb = model(train_aggre_perturb_list[k],adj_noise_list_train[k])
                        x_ass_list_test.append(x_ass)
                        s_purturb_list_test.append(cluster_assignments)
                        out_all.append(out)
                        target_all.append(train_graphs[k].label)
                    out_all = torch.stack(out_all, dim=0).to(device)
                    target_all = torch.tensor(target_all).view(-1).to(device)
                    loss = criterion(out_all, target_all)
                    val_loss += loss.item()
                    pred = out_all.argmax(dim=1)
                    correct += (pred == target_all).sum().item()
                valid_acc = correct / len(valid_graph_index)
                val_loss /= len(valid_graph_index)
                if indicator_style == 'loss':
                    val_indicator = val_loss
                elif indicator_style == 'acc':
                    val_indicator = valid_acc


                best_val_indicator, patience_counter, stop_training = early_stopping(indicator_style, val_indicator, best_val_indicator, patience, patience_counter, model, model_path_fold)

                if stop_training:
                    break



            model.eval()
            correct = 0
            with torch.no_grad():
                x_ass_list_test = []
                s_purturb_list_test = []
                out_all = []
                target_all = []
                for k in range(len(test_graphs)):
                    x_ass, cluster_assignments, entropy_mean, out, graph_emb = model(test_aggre_perturb_list[k],adj_list_test[k])
                    x_ass_list_test.append(x_ass)
                    s_purturb_list_test.append(cluster_assignments)
                    out_all.append(out)
                    target_all.append(test_graphs[k].label)
                out_all = torch.stack(out_all, dim=0).to(device)
                target_all = torch.tensor(target_all).view(-1).to(device)
                pred = out_all.argmax(dim=1)
                correct += (pred == target_all).sum().item()
                test_acc = correct / len(test_graphs)

            test_acc_folder.append(round(test_acc, 4))
        print(f'Method {args.aggregation_method}, Epsilon {epsilon_item}, Noise_scale {Noise_Scale_1}, The test acc in all {len(train_graph_index_all)}. Cross valid folds:')
        print(test_acc_folder)
        result_exists = os.path.exists(args.result_file)
        with open(args.result_file, "a", newline="", encoding="utf-8") as result_csv:
            default_fieldnames = [
                "dataset", "level", "method", "pooling_method", "epsilon", "delta",
                "private_adj_mode",
                "hop", "d_max", "original_max_degree", "clipped_max_degree", "katz_beta", "katz_depth", "katz_mu_mode",
                "epsilon_embedding", "epsilon_edge",
                "embedding_l2_sensitivity", "embedding_sigma",
                "edge_l1_sensitivity", "private_adj_l1_sensitivity", "elementwise_mu",
                "katz_structural_sensitivity", "katz_feature_sensitivity",
                "fold_acc", "mean_acc"
            ]
            writer = csv.DictWriter(
                result_csv,
                fieldnames=csv_fieldnames_for_append(args.result_file, default_fieldnames),
                extrasaction="ignore",
            )
            if not result_exists:
                writer.writeheader()
            writer.writerow({
                "dataset": args.dataset,
                "level": args.level,
                "method": args.aggregation_method,
                "pooling_method": args.pooling_method,
                "epsilon": epsilon_item,
                "delta": args.delta,
                "private_adj_mode": args.private_adj_mode,
                "hop": Hop_1,
                "d_max": accounting_d_max,
                "original_max_degree": original_d_max,
                "clipped_max_degree": clipped_d_max,
                "katz_beta": args.katz_beta if args.aggregation_method == 'katz' else "",
                "katz_depth": katz_depth if args.aggregation_method == 'katz' else "",
                "katz_mu_mode": args.katz_mu_mode if args.aggregation_method == 'katz' else "",
                "epsilon_embedding": privacy_report['epsilon_embedding'],
                "epsilon_edge": privacy_report['epsilon_edge'],
                "embedding_l2_sensitivity": privacy_report['embedding_l2_sensitivity'],
                "embedding_sigma": privacy_report['embedding_sigma'],
                "edge_l1_sensitivity": privacy_report['edge_l1_sensitivity'],
                "private_adj_l1_sensitivity": privacy_report['private_adj_l1_sensitivity'],
                "elementwise_mu": privacy_report.get('elementwise_mu', ""),
                "katz_structural_sensitivity": privacy_report.get('katz_structural_sensitivity', ""),
                "katz_feature_sensitivity": privacy_report.get('katz_feature_sensitivity', ""),
                "fold_acc": ";".join(str(x) for x in test_acc_folder),
                "mean_acc": round(float(np.mean(test_acc_folder)), 6) if test_acc_folder else "",
            })
        print(f"Saved summary result to {args.result_file}")
        print('>>>>>>>>>>>>> Step 2: Finished <<<<<<<<<<<<<<<<<<')


if __name__ == "__main__":
    main()
