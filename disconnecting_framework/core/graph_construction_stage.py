import yaml
import torch
import glob
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from functools import partial
from disconnecting_framework import utils

def main(config_file):
    print("Entered disconnect stage")
    
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)

    stage_dir = config['stage_dir'] #directory containing the graph pygs
    output_dir = config['output_dir']

def preprocess_all(stage_dir, output_dir, config_file):
    max_workers = config_file['max_workers', 1]
    graph_paths = glob.glob(stage_dir + '/*.pyg')
    if max_workers != 1:
        process_map(
            partial(preprocessing, output_dir=output_dir),
            graph_paths,
            max_workers=max_workers,
            chunksize=1,
            desc=f"Preprocessing graphs",)
    else:
        for graph_path in tqdm(graph_paths, desc=f'Preprocessing graphs'):
            preprocessing(graph_path, output_dir)

def preprocessing(graph_path, output_dir):
    graph = torch.load(graph_path)

    graph = utils.graph_construction_utils.remove_edges_in_layer(graph)
    graph = utils.graph_construction_utils.flip_edges(graph)
    graph = utils.graph_construction_utils.add_metagraph(graph)

    torch.save(graph, output_dir + '/' + graph_path.split('/')[-1])