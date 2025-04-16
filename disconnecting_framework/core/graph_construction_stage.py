import yaml
import torch
import glob
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from functools import partial
from disconnecting_framework import utils

'''
Config needs:
    stage_dir: directory containing the initial graph pygs
    output _dir: directory to save the output
        only_metagraph: set to True if only the final metagraph is to be stored
    max_workers: number of workers to use for parallel processing
    preprocess: set to True if the graphs need to be preprocessed
        will initialize metagraph and flip edges to point outwards
    filter: user responsibility
        set to 'random' to use random filter after each linegraph iteration for testing
        set to 'None' to not use any filter
        ToDo: make it possible to choose filter for each level (doublet, triplet, etc)
'''

def main(config_file):
    print("Entered metagraph construction stage")
    
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)

    stage_dir = config['stage_dir'] #directory containing the graph pygs

    paths = glob.glob(stage_dir + '/*.pyg')
    
    max_workers = config['max_workers']

    if max_workers != 1:
        process_map(
            partial(build_metagraph),
            config, paths,
            max_workers=max_workers,
            chunksize=1,
            desc=f"Starting metagraph construction",)
    elif max_workers == 1 and config['test']:
        print("Test mode activated")
        build_metagraph(config, config['test_path'])
    else:
        for path in tqdm(paths, desc=f'Starting metagraph construction'):
            build_metagraph(config, path)

def build_metagraph(config, path):
    #Loads hit graph
    graph = torch.load(path)

    #Preprocesses if specified in config
    if config['preprocess']:
        graph = preprocess_graph(config, graph)

    graph = utils.graph_construction_utils.add_metagraph(graph)
    if 'scores' in graph: 
        del graph.scores
        
    while graph.edge_index.shape[1] > 0:
        graph = utils.graph_construction_utils.linegraph(graph)
        graph = filter_graph(config, graph)
        graph = utils.graph_construction_utils.update_metagraph(graph)
        
    #Save metagraph in hdf5 format
    torch.save(graph, config['output_dir'] + path.split('/')[-1].replace('.pyg','_metagraph.pyg'))

def preprocess_graph(config, graph):
    '''
    Preprocesses the graph by applying the specified preprocessing functions
    '''

    if 'flip_edges' in config['preprocessing_functions']:
        print("Flipping edges")
        graph = utils.graph_construction_utils.flip_edges(graph)

    if 'remove_edges_in_layer' in config['preprocessing_functions']:
        print("Removing edges in layer")
        graph = utils.graph_construction_utils.remove_edges_in_layer(graph)

    if 'barrel_only' in config['preprocessing_functions']:
        print("Filtering out non barrel hits")
        mask = torch.isin(graph.region, torch.tensor([3,4]))
        graph = utils.graph_construction_utils.filter_node_feature(graph, mask)

    return graph

def filter_graph(config, graph):
    '''
    Applies the specified edge filter to the line graph and return a chi2 value for every edge (tracklet)
    '''
    for f in config['filters']:
        if f['name'] == 'random':
            p = f['prob']
            assert p >= 0 or p <= 1, "Probability must be between 0 and 1"
            graph = utils.filters.random_filter(graph, p)

        ### Add own filters here ### +

    return graph