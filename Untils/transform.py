import random

import torch


def index_to_dense(index_tensor, num_nodes=None):
    """Convert a 2 x E edge-index tensor to a dense adjacency matrix."""
    if num_nodes is None:
        if index_tensor.numel() == 0:
            raise ValueError("num_nodes is required when index_tensor is empty.")
        size = (index_tensor.max() + 1).item()
    else:
        size = int(num_nodes)

    dense_matrix = torch.zeros((size, size), dtype=torch.int32)
    if index_tensor.numel() > 0:
        dense_matrix[index_tensor[0], index_tensor[1]] = 1
    return dense_matrix


def shuffle_and_group(num_range, k):
    numbers = list(range(0, num_range))
    random.shuffle(numbers)
    return [numbers[i:i + k] for i in range(0, len(numbers), k)]
