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
from scipy.sparse import coo_matrix
import scipy.sparse as sp

def _rdp_epsilon_gaussian(sigma, sensitivity, delta, orders=None):
    if sensitivity <= 0:
        return 0.0
    if orders is None:
        orders = list(np.arange(1.25, 10, 0.25)) + list(range(10, 128))
    rdp = [(order * (sensitivity ** 2)) / (2 * (sigma ** 2)) for order in orders]
    eps = [rdp_i + np.log(1 / delta) / (order - 1) for rdp_i, order in zip(rdp, orders)]
    return float(min(eps))

def calibrate_gaussian_sigma(epsilon, delta, sensitivity, orders=None):
    if epsilon <= 0:
        raise ValueError("epsilon must be positive for Gaussian calibration")
    if sensitivity <= 0:
        return 0.0
    low = 1e-12
    high = max(1.0, sensitivity)
    while _rdp_epsilon_gaussian(high, sensitivity, delta, orders) > epsilon:
        high *= 2
    for _ in range(80):
        mid = (low + high) / 2
        if _rdp_epsilon_gaussian(mid, sensitivity, delta, orders) > epsilon:
            low = mid
        else:
            high = mid
    return float(high)

def katz_decay_sum(beta, depth):
    if depth <= 0:
        return 0.0
    if beta == 1:
        return depth * (depth + 1) / 2
    return beta * (1 - (depth + 1) * (beta ** depth) + depth * (beta ** (depth + 1))) / ((1 - beta) ** 2)

def katz_geometric_sum(beta, depth):
    if depth < 0:
        return 0.0
    if beta == 1:
        return depth + 1
    return (1 - beta ** (depth + 1)) / (1 - beta)

def elementwise_mu(level, d_max, default_mu=1.0):
    if level == 'node' and d_max is not None and d_max > 0:
        return 1.0 / float(d_max)
    return float(default_mu)

def graph_change_sensitivity(level, d_max):
    if level == 'edge':
        return np.sqrt(2.0), 1.0
    if level == 'node':
        if d_max is None or d_max <= 0:
            raise ValueError("Strict node-level accounting requires a positive public D_max")
        return np.sqrt(2.0 * d_max), float(d_max)
    raise ValueError("level must be 'edge' or 'node'")

def adjacency_l1_sensitivity(level, d_max, mode='raw', mu=None):
    """L1 sensitivity for the adjacency released to private pooling."""
    _, raw_sensitivity = graph_change_sensitivity(level, d_max)
    if mode == 'raw':
        return float(raw_sensitivity)
    if mode == 'elementwise':
        if mu is None:
            mu = elementwise_mu(level, d_max)
        return float(raw_sensitivity * float(mu))
    raise ValueError("mode must be 'raw' or 'elementwise'")

def katz_embedding_sensitivity(level, d_max, beta, depth, feature_norm_bound=1.0,
                               mu=None, include_node_feature_change=True):
    """Element-wise normalized Katz sensitivity.

    Edge structural term: 2 * mu * sum_{l=1}^L l beta^l.
    Node structural term after D_max clipping: sqrt(1 + mu) * sum_{l=1}^L l beta^l.
    Node feature term: sum_{l=0}^L beta^l, when node features may change.
    """
    if level == 'node' and (d_max is None or d_max <= 0):
        raise ValueError("Strict node-level accounting requires a positive public D_max")
    if mu is None:
        mu = elementwise_mu(level, d_max)
    if level == 'edge':
        structural = 2.0 * float(mu) * feature_norm_bound * katz_decay_sum(beta, depth)
    elif level == 'node':
        structural = np.sqrt(1.0 + float(mu)) * feature_norm_bound * katz_decay_sum(beta, depth)
    else:
        raise ValueError("level must be 'edge' or 'node'")
    feature = 0.0
    if level == 'node' and include_node_feature_change:
        feature = feature_norm_bound * katz_geometric_sum(beta, depth)
    return float(structural + feature)

def katz_structural_sensitivity(level, d_max, beta, depth, feature_norm_bound=1.0, mu=None):
    if level == 'node' and (d_max is None or d_max <= 0):
        raise ValueError("Strict node-level accounting requires a positive public D_max")
    if mu is None:
        mu = elementwise_mu(level, d_max)
    if level == 'edge':
        return float(2.0 * float(mu) * feature_norm_bound * katz_decay_sum(beta, depth))
    if level == 'node':
        return float(np.sqrt(1.0 + float(mu)) * feature_norm_bound * katz_decay_sum(beta, depth))
    raise ValueError("level must be 'edge' or 'node'")

def privacy_accounting(epsilon_total, delta, level, d_max, hop, epsilon_edge,
                       aggregation='original', beta=0.15, depth=None,
                       feature_norm_bound=1.0, elementwise_mu_value=None,
                       include_node_feature_change=True, private_adj_mode='raw'):
    if epsilon_total <= 0:
        raise ValueError("epsilon_total must be positive")
    if epsilon_edge < 0 or epsilon_edge >= epsilon_total:
        raise ValueError("epsilon_edge must be non-negative and smaller than epsilon_total")
    epsilon_embedding = epsilon_total - epsilon_edge
    if aggregation == 'katz':
        depth = hop if depth is None else depth
        mu = elementwise_mu_value
        if mu is None:
            mu = elementwise_mu(level, d_max)
        sensitivity_embedding = katz_embedding_sensitivity(
            level, d_max, beta, depth, feature_norm_bound=feature_norm_bound,
            mu=mu, include_node_feature_change=include_node_feature_change
        )
    else:
        mu = ""
        if level == 'node':
            if d_max is None or d_max <= 0:
                raise ValueError("Strict node-level accounting requires a positive public D_max")
            sensitivity_embedding = np.sqrt(d_max + d_max ** 2) * max(1, hop)
        elif level == 'edge':
            sensitivity_embedding = np.sqrt(2.0) * max(1, hop)
        else:
            raise ValueError("level must be 'edge' or 'node'")
    sigma = calibrate_gaussian_sigma(epsilon_embedding, delta, sensitivity_embedding)
    _, edge_l1_sensitivity = graph_change_sensitivity(level, d_max)
    private_adj_l1_sensitivity = adjacency_l1_sensitivity(
        level,
        d_max,
        mode=private_adj_mode,
        mu=(float(mu) if aggregation == 'katz' and mu != "" else None),
    )
    return {
        'epsilon_total': float(epsilon_total),
        'epsilon_embedding': float(epsilon_embedding),
        'epsilon_edge': float(epsilon_edge),
        'delta': float(delta),
        'level': level,
        'aggregation': aggregation,
        'embedding_l2_sensitivity': float(sensitivity_embedding),
        'embedding_sigma': float(sigma),
        'edge_l1_sensitivity': float(edge_l1_sensitivity),
        'private_adj_l1_sensitivity': float(private_adj_l1_sensitivity),
        'private_adj_mode': private_adj_mode,
        'elementwise_mu': float(mu) if aggregation == 'katz' else "",
        'katz_structural_sensitivity': (
            katz_structural_sensitivity(
                level, d_max, beta, depth, feature_norm_bound=feature_norm_bound, mu=mu
            )
            if aggregation == 'katz' else ""
        ),
        'katz_feature_sensitivity': (
            float(feature_norm_bound * katz_geometric_sum(beta, depth))
            if aggregation == 'katz' and level == 'node' and include_node_feature_change else
            (0.0 if aggregation == 'katz' else "")
        ),
    }

def cal_privacy_budget(Hop_num,Degree_max, sigma_noise, adj_perturb_epsilon, privacy_level, delta=1e-5):
    if privacy_level == 'node':
        Degree_max = np.sqrt(Degree_max**2+ Degree_max)
        epsilon_sum = Hop_num*Degree_max/(2*(sigma_noise**2))+np.sqrt(2*Hop_num*Degree_max*np.log(1/delta))/sigma_noise + adj_perturb_epsilon
    else:
        epsilon_sum = np.sqrt(2)*Hop_num/(2*(sigma_noise**2))+np.sqrt(2*np.sqrt(2)*Hop_num*np.log(1/delta))/sigma_noise + adj_perturb_epsilon
    return epsilon_sum

def cal_noise_scale(epsilon, delta, D_max, hop, epsilon_e, level):
    report = privacy_accounting(epsilon, delta, level, D_max, hop, epsilon_e, aggregation='original')
    return report['embedding_sigma']
