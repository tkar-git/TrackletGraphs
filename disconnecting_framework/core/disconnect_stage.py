
import numpy as np
import torch
import yaml
import glob
import tqdm
import h5py

from itertools import chain
from functools import partial
from tqdm.contrib.concurrent import process_map

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

def save_to_hdf5(sets_list, filename="sets_efficient.h5"):
    """
    Save a list of sets to a hdf5 file using a flat array with index offsets.
    """
    
    # Convert sets to sorted lists (ensures consistency in representation)
    sets_list = [sorted(s) for s in sets_list]

    # Flatten all sets into a single 1D array
    flat_data = np.concatenate(sets_list).astype(np.int64)

    # Store index offsets to reconstruct sets later
    indices = np.cumsum([0] + [len(s) for s in sets_list])

    # Save to HDF5
    with h5py.File(filename, "w") as f:
        f.create_dataset("data", data=flat_data, compression="gzip")   # Store elements
        f.create_dataset("indices", data=indices, compression="gzip")  # Store offsets

def load_from_hdf5(filename="sets_efficient.h5"):
    """
    Load a list of sets stored efficiently in an HDF5 file.
    """
    
    with h5py.File(filename, "r") as f:
        flat_data = f["data"][:]
        indices = f["indices"][:]
    
    # Reconstruct sets using index offsets
    sets_list = [set(flat_data[indices[i]:indices[i+1]]) for i in range(len(indices) - 1)]
    
    return sets_list
