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
    graph = torch.load(path)
    metagraph = graph.metagraph
    scores = graph.scores

    #Metagraph and scores are structured like [[],[],[]] with longest tracklists sublist being the last

    track_candidates = []

    while metagraph:
        #Find longest tracklet with highest score
        best_tracklet_index = max(range(len(scores[-1])), key=scores[-1].__getitem__)
        best_tracklet = metagraph[-1][best_tracklet_index]
        track_candidates.append(best_tracklet)

        #Flatten first for easier tracklet removal
        flat_metagraph = list(chain(*metagraph))
        flat_scores = list(chain(*scores))
        print("Flat metagraph:",flat_metagraph)

        #Remove all tracklets with non-zero intersection and their scores from the metagraph
        mask = [not (best_tracklet & s) for s in flat_metagraph]
        new_metagraph = [s for s, m in zip(flat_metagraph, mask) if m]
        new_scores = [s for s, m in zip(flat_scores, mask) if m]
        print("New metagraph:",new_metagraph)

        #Restore original structure [[],[],...]
        metagraph, scores = regroup_by_set_size(new_metagraph, new_scores)
    print("Track candidates:",track_candidates)
    save_to_hdf5(track_candidates, config['output_dir'] + 'track_candidates.h5')

def regroup_by_set_size(sets, scores):
    # Create a list of sublists indexed by set size - 1
    size_to_sets = {}
    size_to_scores = {}

    for s, sc in zip(sets, scores):
        size = len(s)
        if size not in size_to_sets:
            size_to_sets[size] = []
            size_to_scores[size] = []
        size_to_sets[size].append(s)
        size_to_scores[size].append(sc)

    # Sort by set size to maintain original order (smallest to largest)
    sizes_sorted = sorted(size_to_sets)
    regrouped_sets = [size_to_sets[size] for size in sizes_sorted]
    regrouped_scores = [size_to_scores[size] for size in sizes_sorted]
    
    return regrouped_sets, regrouped_scores