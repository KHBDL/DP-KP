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

def GNN_NAP(adj_matrix_dense, X, noise_scale):
    adj_matrix_dense = adj_matrix_dense + torch.eye(adj_matrix_dense.size(0)).to(adj_matrix_dense.device)
    aggre_result = torch.mm(adj_matrix_dense.T,X.to(torch.float32))
    aggre_result += creat_noise(aggre_result,noise_scale)  #鑱氬悎鎵板姩
    norm_vals = aggre_result.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
    aggre_result_nor = aggre_result/norm_vals  
    return aggre_result_nor

def creat_noise(array,noise_scale):
    # 鏋勫缓涓€涓笌浜岀淮鏁扮粍鍚岀淮搴︾殑楂樻柉鐧藉櫔澹扮煩闃?    # noise = torch.randn_like(array)  # 浣跨敤 randn_like 鐢熸垚涓?array 鐩稿悓缁村害鐨勬爣鍑嗘鎬佸垎甯冨櫔澹?    # # 濡傛灉闇€瑕佸彲浠ラ€氳繃涔樹互涓€涓父鏁版潵璋冩暣鍣０鐨勬爣鍑嗗樊锛堜緥濡傦紝璋冩暣鍣０寮哄害锛?    # std_dev = noise_scale # 璁剧疆鍣０鏍囧噯宸?    # scaled_noise = noise * std_dev  # 鎸夌収璁惧畾鐨勬爣鍑嗗樊缂╂斁鍣０
    noise = torch.normal(0, noise_scale, size=array.shape).to(array.device)
    return noise

def row_l2_normalize(x, eps=1e-12):
    norm_vals = x.norm(p=2, dim=1, keepdim=True).clamp_min(eps)
    return x / norm_vals

def normalize_adjacency(adj_dense, add_self_loops=True):
    adj = adj_dense.to(torch.float32)
    if add_self_loops:
        adj = adj + torch.eye(adj.size(0), device=adj.device)
    degree = adj.sum(dim=1).clamp_min(1.0)
    degree_inv_sqrt = torch.pow(degree, -0.5)
    return adj * degree_inv_sqrt.view(-1, 1) * degree_inv_sqrt.view(1, -1)
    
def Aggre_Perturb(adj_dense, x, noise_scale,hop,multi_GNN_output_flag):  
    # Graph Convolution Layer
    if multi_GNN_output_flag: #鏄惁閲囩敤GNN澶氬眰杈撳嚭
        out_all = []
        for i in range(hop):
            # if i == 0:  #灏嗗師濮嬬壒寰佸悜閲忎篃杩涜鍫嗗彔
            #     out_all.append(x)
            x = GNN_NAP(adj_dense, x, noise_scale)
            out_all.append(x)
        stacked_output = torch.cat(out_all, dim=1)  # 娌跨潃缁村害1鍫嗗彔
        return stacked_output
    else:
        for i in range(hop):
            x = GNN_NAP(adj_dense, x, noise_scale)
        return x

def Katz_Aggre_Perturb(adj_dense, x, noise_scale, depth, beta=0.15,
                       multi_GNN_output_flag=False, normalize_adj=True,
                       elementwise_mu=None):
    if elementwise_mu is not None:
        propagation_adj = adj_dense.to(torch.float32) * float(elementwise_mu)
    elif normalize_adj:
        propagation_adj = normalize_adjacency(adj_dense, add_self_loops=False)
    else:
        propagation_adj = adj_dense.to(torch.float32)

    z = x.to(torch.float32)
    current = x.to(torch.float32)
    out_all = []
    for order in range(1, depth + 1):
        current = torch.mm(propagation_adj, current)
        weighted = (beta ** order) * current
        if multi_GNN_output_flag:
            noisy_level = row_l2_normalize(weighted + creat_noise(weighted, noise_scale))
            out_all.append(noisy_level)
        z = z + weighted

    if multi_GNN_output_flag:
        return torch.cat(out_all, dim=1)

    z = z + creat_noise(z, noise_scale)
    return row_l2_normalize(z)

def add_noise_to_gradients(model, noise_multiplier, max_grad_norm):
    """
    Adds noise to gradients and performs gradient clipping.

    :param model: The model whose gradients are being modified.
    :param noise_multiplier: The factor by which noise is added to gradients.
    :param max_grad_norm: The maximum allowed gradient norm for clipping.
    """
    # First, apply gradient clipping
    for param in model.parameters():
        if param.grad is not None:
            # Clip gradients to max_grad_norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    # Now, add noise to the gradients
    for param in model.parameters():
        if param.grad is not None:
            noise = torch.randn_like(param.grad) * noise_multiplier
            param.grad += noise

def randomized_response(binary_tensor, epsilon):
    """
    搴旂敤闅忔満鍝嶅簲鏈哄埗鍒颁簩杩涘埗寮犻噺銆?
    :param binary_tensor: 浜岃繘鍒跺紶閲?(N, D)锛屽叾涓?N 鏄牱鏈暟锛孌 鏄壒寰佺淮搴?    :param epsilon: 闅愮鍙傛暟 (float)
    :return: 鎵板姩鍚庣殑浜岃繘鍒跺紶閲?    鍐冲畾浜嗘壈鍔ㄧ殑绋嬪害銆傝緝灏忕殑饾湒鎻愪緵鏇村己鐨勯殣绉佷繚鎶わ紝浣嗗彲鑳藉鑷存暟鎹疄鐢ㄦ€ч檷浣庯紱杈冨ぇ鐨勷潨栨彁渚涙洿楂樼殑鏁版嵁瀹炵敤鎬э紝浣嗛殣绉佷繚鎶よ緝寮便€?    鎵板姩姒傜巼p锛氫笌系鐩稿叧锛屾帶鍒朵簡姣忎釜浣嶈鐪熷疄鎶ュ憡鎴栬缈昏浆鐨勬鐜囥€?    """
    # 璁＄畻 p 鐨勫€?    p = math.exp(epsilon) / (1 + math.exp(epsilon))
    
    # 鐢熸垚涓?binary_tensor 褰㈢姸鐩稿悓鐨勯殢鏈哄紶閲忥紝鍊煎湪 [0, 1) 涔嬮棿
    random_tensor = torch.rand_like(binary_tensor, dtype=torch.float)
    
    # 鍒涘缓涓€涓帺鐮侊紝鍐冲畾鍝簺浣嶉渶瑕佺炕杞?    flip_mask = random_tensor > p
    
    # 鏍规嵁鎺╃爜缈昏浆浣?    perturbed = torch.where(flip_mask, 1 - binary_tensor, binary_tensor)
    
    return perturbed

def get_noise(noise_type, size, seed, eps=10, delta=1e-5, sensitivity=2):  # https://github.com/AI-secure/LinkTeller/blob/master/worker.py
    rng = np.random.default_rng(seed)
    return sample_noise(rng, noise_type, size, eps=eps, delta=delta, sensitivity=sensitivity)

def sample_noise(rng, noise_type, size, eps=10, delta=1e-5, sensitivity=2):
    if noise_type == 'laplace':
        return rng.laplace(0, sensitivity/eps, size)
    elif noise_type == 'gaussian':
        c = np.sqrt(2 * np.log(1.25 / delta))
        return rng.normal(0, c * sensitivity / eps, size)
    else:
        raise NotImplementedError(f'Noise type {noise_type} not implemented!')

def perturb_adj_continuous(adj_matrix, epsilon=10, noise_type='laplace', noise_seed=46, delta=1e-5, sensitivity=1.0):
    adj = coo_matrix(adj_matrix.cpu())
    n_nodes, n_edges = adj.shape[0], len(adj.data) // 2

    # Work on the lower-triangular adjacency vector. For an undirected graph,
    # one edge changes one lower-triangular entry under edge-DP and at most
    # D_max entries under bounded-degree node-DP.
    A = sp.tril(adj, k=-1).toarray()
    epsilon_1 = epsilon * 0.6
    epsilon_2 = epsilon - epsilon_1
    rng = np.random.default_rng(noise_seed)

    lower_rows, lower_cols = np.tril_indices(n_nodes, k=-1)
    lower_scores = A[lower_rows, lower_cols].astype(float)
    lower_scores += sample_noise(
        rng, noise_type, lower_scores.shape, eps=epsilon_2, delta=delta, sensitivity=sensitivity
    )

    # Use an independent draw from the same RNG stream for the private edge count.
    n_edges_keep = n_edges + int(sample_noise(
        rng, noise_type, (1,), eps=epsilon_1, delta=delta, sensitivity=sensitivity
    )[0])
    n_edges_keep = max(0, min(n_edges_keep, n_nodes * (n_nodes - 1) // 2))
    if n_edges_keep == 0:
        return torch.zeros((n_nodes, n_nodes), dtype=torch.int32)

    n_edges_keep = min(n_edges_keep, lower_scores.shape[0])
    selected = np.argpartition(lower_scores, -n_edges_keep)[-n_edges_keep:]
    row_idx = lower_rows[selected]
    col_idx = lower_cols[selected]

    data_idx = np.ones(n_edges_keep, dtype=np.int32)
    mat = sp.csr_matrix((data_idx, (row_idx, col_idx)), shape=(n_nodes, n_nodes))
    return torch.tensor((mat + mat.T).todense())

def perturb_adj_elementwise_continuous(adj_matrix, epsilon=10, noise_type='laplace',
                                       noise_seed=46, delta=1e-5, sensitivity=1.0,
                                       elementwise_mu=1.0):
    """Perturb the bounded weighted adjacency mu * A for private pooling.

    The released matrix is continuous and clipped to [0, mu]. DiffPool consumes
    adjacency through matrix multiplications, so weighted adjacency is valid.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive for private adjacency perturbation")
    mu = float(elementwise_mu)
    if mu <= 0:
        raise ValueError("elementwise_mu must be positive")

    adj = coo_matrix(adj_matrix.cpu())
    n_nodes = adj.shape[0]
    lower = sp.tril(adj, k=-1).toarray().astype(float) * mu
    lower_rows, lower_cols = np.tril_indices(n_nodes, k=-1)
    lower_scores = lower[lower_rows, lower_cols]
    rng = np.random.default_rng(noise_seed)
    lower_scores += sample_noise(
        rng, noise_type, lower_scores.shape, eps=epsilon, delta=delta, sensitivity=sensitivity
    )
    lower_scores = np.clip(lower_scores, 0.0, mu)

    mat = sp.csr_matrix((lower_scores, (lower_rows, lower_cols)), shape=(n_nodes, n_nodes))
    return torch.tensor((mat + mat.T).todense(), dtype=torch.float32)


def random_adj_same_edge_count(adj_matrix, seed=46, edge_prob=0.5):
    """Random adjacency for ablation without preserving the original edge count."""
    adj = coo_matrix(adj_matrix.cpu())
    n_nodes = adj.shape[0]
    rng = np.random.default_rng(seed)
    lower_rows, lower_cols = np.tril_indices(n_nodes, k=-1)
    lower_scores = (rng.random(len(lower_rows)) < float(edge_prob)).astype(np.float32)
    mat = sp.csr_matrix((lower_scores, (lower_rows, lower_cols)), shape=(n_nodes, n_nodes))
    return torch.tensor((mat + mat.T).todense(), dtype=torch.float32)

    
def Assignment_purturb(s_purturb, adj_dense_purturb, device):
    # adj_dense_purturb = randomized_response(adj_dense, epsilon=0.5)  ###########################epsilon涓洪噸瑕佸弬鏁?    adj_ass = torch.mm(torch.mm(s_purturb.T, adj_dense_purturb.float()),s_purturb)
    adj_ass = F.softmax(adj_ass, dim=1) 
    # adj_ass += creat_noise(adj_ass,noise_scale)
    # norm_vals = adj_ass.norm(p=2, dim=1, keepdim=True)  # 璁＄畻姣忎竴琛岀殑 L2 鑼冩暟
    # adj_ass = adj_ass/norm_vals  
    # adj_ass = F.softmax(adj_ass, dim=-1)   # 鍚﹀垯鏂伴偦鎺ョ煩闃典細闈炲父濂囨€?    return adj_ass
