import torch
import numpy as np
from torch_geometric.data import Data

def random_filter(graph,p):
    '''
    Randomly removes p percent of edges from the graph
    Optimized for usage on GPU
    '''

    n_edges = graph.edge_index.shape[1]
    remove_m_edges = int(n_edges*p)
    if remove_m_edges == 0:
        tracklet_scores = torch.rand(graph.edge_index.shape[1]).tolist()
        if hasattr(graph,'scores') and len(tracklet_scores) > 0:
            graph.scores.append(tracklet_scores)
        elif hasattr(graph,'scores') and len(tracklet_scores) == 0:
            return graph
        else:
            graph.scores = [tracklet_scores]

        return graph
    
    remove_indices = torch.randperm(n_edges,device=graph.edge_index.device)[:remove_m_edges]

    mask = torch.ones(n_edges, dtype=torch.bool, device=graph.edge_index.device)
    mask[remove_indices] = False

    graph.edge_index = graph.edge_index[:,mask]

    #Randomly assign score to each tracklet
    #This is a placeholder, in the future this should be replaced
    tracklet_scores = torch.rand(graph.edge_index.shape[1]).tolist()

    if hasattr(graph,'scores') and graph.edge_index.shape[1] > 0:
        graph.scores.append(tracklet_scores)
    elif hasattr(graph,'scores') and graph.edge_index.shape[1] == 0:
        return graph
    else:  
        graph.scores = [tracklet_scores]
    return graph