
import numpy as np
import torch
import yaml
import glob
import tqdm

from itertools import chain
from functools import partial
from tqdm.contrib.concurrent import process_map
from disconnecting_framework.utils.graph_construction_utils import load_from_hdf5, save_to_hdf5

def main(config_file):
    print("Entered disconnect stage")
    
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)

    stage_dir = config['stage_dir'] #directory containing the graph pygs

    paths = glob.glob(stage_dir + '/*.pyg')
    
    max_workers = config['max_workers']

    if max_workers != 1:
        process_map(
            partial(get_all_track_candidates),
            config, paths,
            max_workers=max_workers,
            chunksize=1,
            desc=f"Preprocessing metagraphs",)
    elif max_workers == 1 and config['test']:
        get_all_track_candidates(config, config['test_path'])
    else:
        for path in tqdm(paths, desc=f'Producing track candidate list'):
            get_all_track_candidates(config, path)

def get_all_track_candidates(config, path):
    #Check if metagraph is stored in hdf5 format and load as such, else get from pyg file
    if config['load_only_metagraph']:
        metagraph = load_from_hdf5(path)
    else:
        metagraph = torch.load(path).metagraph
        metagraph_path = path.replace('.pyg', '.h5')
        save_to_hdf5(metagraph, metagraph_path)

    #If metagraph is structured like [[],[],[]] -> flatten
    if any(isinstance(item, list) for item in metagraph): 
        flattened_metagraph = list(chain(*metagraph))
    else:
        flattened_metagraph = metagraph

    track_candidates = []

    while flattened_metagraph:
        largest_set, flattened_metagraph = produce_longest_track_candidate(flattened_metagraph)

        track_candidates.append(largest_set)

    save_to_hdf5(track_candidates, config['output_dir'] + 'track_candidates.h5')

def produce_longest_track_candidate(flattened_metagraph):
    longest_track_candidate = max(flattened_metagraph, key=len)
    flattened_metagraph = [s for s in flattened_metagraph if s != longest_track_candidate and s.isdisjoint(longest_track_candidate)]

    return longest_track_candidate, flattened_metagraph
