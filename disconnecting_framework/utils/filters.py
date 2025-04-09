import torch
import numpy as np
import random
from torch_geometric.data import Data

def random_filter(graph,p):
    '''
    Randomly removes p percent of edges from the graph
    Optimized for usage on GPU
    '''

    n_edges = graph.edge_index.shape[1]
    remove_m_edges = int(n_edges*p)
    if remove_m_edges == 0:
        return graph
    
    remove_indices = torch.randperm(n_edges,device=graph.edge_index.device)[:remove_m_edges]

    mask = torch.ones(n_edges, dtype=torch.bool, device=graph.edge_index.device)
    mask[remove_indices] = False

    graph.edge_index = graph.edge_index[:,mask]

    chi2 = random.random() #assign random 0<chi2<1 for testing purposes

    return graph, chi2