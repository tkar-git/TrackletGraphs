import yaml
import torch
import glob
import h5py
from itertools import chain
import tqdm
from tqdm.contrib.concurrent import process_map
from functools import partial

###
#NEEDS TO BE TESTED
###

def main(config_file):
    print("Entered disconnect stage")
    
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)

    stage_dir = config['stage_dir'] #directory containing the graph pygs
    output_dir = config['output_dir']

    metagraph_paths = glob.glob(stage_dir + '/*.pyg')
    
    max_workers = config['max_workers', 1]
    if max_workers != 1:
        process_map(
            partial(get_all_track_candidates),
            metagraph_paths,
            max_workers=max_workers,
            chunksize=1,
            desc=f"Preprocessing metagraphs",)
    else:
        for metagraph_path in tqdm(metagraph_paths, desc=f'Producing track candidate list'):
            get_all_track_candidates(metagraph_path)

def produce_longest_track_candidate(flattened_metagraph):
    longest_track_candidate = max(flattened_metagraph, key=len)
    flattened_metagraph = [s for s in flattened_metagraph if s != longest_track_candidate and s.isdisjoint(longest_track_candidate)]

    return longest_track_candidate, flattened_metagraph

def get_all_track_candidates(metagraph_path):
    metagraph = load_metagraph_from_hdf5(metagraph_path)
    flattened_metagraph = list(chain(*metagraph))

    track_candidates = []

    while flattened_metagraph:
        largest_set, flattened_metagraph = produce_longest_track_candidate(flattened_metagraph)

        track_candidates.append(largest_set)

    #Save the track candidates
    save_data_to_hdf5(track_candidates, metagraph_path.replace('.h5', '_track_candidates.h5'))

def save_data_to_hdf5(metagraph, file_name):
    with h5py.File(file_name, 'w') as f:
        for i, sublist in enumerate(metagraph):
            # Convert each set to a tuple (or list) to store in HDF5
            sublist_data = [tuple(s) for s in sublist]
            f.create_dataset(f'sublist_{i}', data=sublist_data)

def load_metagraph_from_hdf5(file_path):
    metagraph = []
    with h5py.File(file_path, 'r') as f:
        for key in f.keys():
            sublist_data = f[key][:]
            # Convert each tuple back to a set
            sublist = [set(s) for s in sublist_data]
            metagraph.append(sublist)
    return metagraph