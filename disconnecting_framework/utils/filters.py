import torch
import numpy as np
import random
from torch_geometric.data import Data

def random_filter(graph, p=0.2,seed=42):
    n_edges = graph.edge_index.shape[1]

    edge_keys = [key for key, value in graph.items() if isinstance(value, torch.Tensor) and value.dim() == 1 and value.shape[0] == n_edges]

    assert p >= 0 and p <= 1, 'p must be between 0 and 1'
    if p > 0.5:
        random.seed(seed)
        keep_edges = random.sample(range(0,graph.edge_index.shape[1]),int(graph.edge_index.shape[1]*(1-p)))
        graph.edge_index = torch.from_numpy(np.take(graph.edge_index.numpy(),keep_edges,axis=1))
        for key in edge_keys:
            graph[key] = torch.from_numpy(np.take(graph[key].numpy(),keep_edges,axis=0))

    else:
        random.seed(seed)
        remove_edges = random.sample(range(0,graph.edge_index.shape[1]),int(graph.edge_index.shape[1]*p))
        graph.edge_index = torch.from_numpy(np.delete(graph.edge_index.numpy(),remove_edges,axis=1))
        for key in edge_keys:
            graph[key] = torch.from_numpy(np.delete(graph[key].numpy(),remove_edges,axis=0))

    return graph