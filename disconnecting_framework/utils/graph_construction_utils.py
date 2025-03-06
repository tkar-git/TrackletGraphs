import torch
from torch_geometric.data import Data

def add_metagraph(graph):
    '''
    Add a metagraph to the graph
    Needs to be updated after each linegraph iteration
    '''
    assert hasattr(graph, 'edge_index'), 'Edge index does not exist in graph'
    
    graph.metagraph = [{graph.edge_index[0,i].item(), graph.edge_index[1,i].item()} for i in range(graph.edge_index.shape[1])]
    return graph

def update_metagraph(graph):
    '''
    Updates the metagraph after each linegraph iteration
    '''
    assert hasattr(graph, 'metagraph'), 'Metagraph does not exist in graph'
    assert hasattr(graph, 'edge_index'), 'Edge index does not exist in graph'

    new_metagraph =  []
    for tracklet in graph.edge_index.T:
        trackletA = graph.metagraph[tracklet[0]]
        trackletB = graph.metagraph[tracklet[1]]
        new_tracklet = trackletA|trackletB
        new_metagraph.append(new_tracklet)
    graph.metagraph = new_metagraph
    return graph

def flip_edges(graph):
    '''
    Flips the direction of an edge if the receiving node is closer to the origin
    '''
    assert hasattr(graph, 'edge_index'), 'Edge index does not exist in graph'
    assert hasattr(graph, 'r') or hasattr(graph, 'z'), 'r or z coordinates do not exist in graph'

    r,z = graph.r, graph.z
    d = torch.sqrt(r**2 + z**2)

    d_source = d[graph.edge_index[0]]
    d_target = d[graph.edge_index[1]]
    
    swap_mask = d_target < d_source  # Boolean mask

    # Perform swapping using advanced indexing (efficient in-place swap)
    graph.edge_index[0, swap_mask], graph.edge_index[1, swap_mask] = graph.edge_index[1, swap_mask], graph.edge_index[0, swap_mask]
    graph.edge_index = torch.unique(graph.edge_index, dim=1)  # Remove duplicates
    
    return graph

def remove_edges_in_layer(graph):
    raise NotImplementedError

def apply_mask_to_graph(graph, mask, mask_type='node'):
    '''
    Removes nodes or edges from the graph based based on a mask and reindexes edge_index and track_edges if nodes are removed
    Input: graph - pytorch geometric data object
           mask - boolean mask for nodes or edge features
           mask_type - 'node' or 'edge'
    '''
    mask_len = mask.size(0)
    num_nodes = graph.num_nodes
    num_edges = graph.edge_index.size(1)

    assert (mask_type == 'node' and mask_len == num_nodes) or (mask_type == 'edge' and mask_len == num_edges), 'Mask length does not match number of nodes or edges'

    if mask_type == 'node':
        print('Filtering node features')
        return filter_node_feature(graph, mask)
    elif mask_type == 'edge':
        print('Filtering edge features')
        return filter_edge_feature(graph, mask)
    else:
        raise ValueError('Invalid mask type')

def filter_node_feature(graph, mask):
    #Create a mapping from old indices to new indices
    mapping = -torch.ones(graph.hit_id.shape[0], dtype=torch.long)  # Initialize mapping
    mapping[mask] = torch.arange(mask.sum())  # Assign new indices

    #Filter edges where both nodes are in the mask
    edge_mask = mask[graph.edge_index[0]] & mask[graph.edge_index[1]]
    filtered_edge_index = graph.edge_index[:, edge_mask]

    #Remap edges to new indices
    new_edge_index = mapping[filtered_edge_index]

    #Filter all node and edge features and create new graph
    node_feature_dict = {
        key: value[mask] for key, value in graph.items()
        if isinstance(value, torch.Tensor) and value.dim() == 1 and 
        value.shape[0] == int(graph.num_nodes)} # Filters only node features
 
    edge_feature_dict = {
        key: value[edge_mask] for key, value in graph.items()
        if isinstance(value, torch.Tensor) and value.dim() == 1 and
        value.shape[0] == graph.edge_index.shape[1]} # Filters only edge features
    
    #Filter track edges if they exist
    if 'track_edges' in graph.keys():
        te_mask = mask[graph.track_edges[0]] & mask[graph.track_edges[1]]
        filtered_track_edges = graph.track_edges[:, te_mask]
        new_track_edges = mapping[filtered_track_edges]

        track_edge_feature_dict = {
            key: value[te_mask] for key, value in graph.items() 
            if isinstance(value, torch.Tensor) and value.dim() == 1 and 
            value.shape[0] == graph.track_edges.shape[1]} # Filters only track edge features
        
    graph.update(node_feature_dict)
    graph.update(edge_feature_dict)
    graph.update(track_edge_feature_dict)
    graph.edge_index = new_edge_index
    graph.track_edges = new_track_edges
    graph.truth_map = build_truth_map(graph.edge_index, graph.track_edges)
    graph.num_nodes = mask.sum() 
    return graph

def filter_edge_feature(graph, mask):
    raise NotImplementedError

def build_truth_map(edge_index, track_edges):
    edge_index_exp = edge_index.T.unsqueeze(0)  # Shape (1, n, 2)
    track_edges_exp = track_edges.T.unsqueeze(1)  # Shape (m, 1, 2)

    # Compare all pairs at once
    matches = torch.all(edge_index_exp == track_edges_exp, dim=2)  # Shape (m, n), True where matches exist

    # Get the first matching index for each row in B, or -1 if no match
    truth_map = torch.where(matches.any(dim=1), matches.int().argmax(dim=1), torch.tensor(-1))
    return truth_map
