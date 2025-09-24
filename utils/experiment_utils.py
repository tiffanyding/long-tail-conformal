import json
import os
# from pathlib import Path

# Read once
folder_path = "folders.json" 
try:
    with open(folder_path) as f:
        cfg = json.load(f)
except FileNotFoundError:
    try:
        with open(os.path.join("..", folder_path)) as f:
            cfg = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError("Could not find 'folders.json' in the current or parent directory.")

def get_plantnet_folder():
    return cfg['plantnet_data']


def get_inaturalist_folder():
    return cfg['inaturalist_data']
    
def get_inputs_folder():
    return cfg['scores_and_labels']

def get_outputs_folder():
    return cfg['conformal_results']

def get_figs_folder():
    return cfg['figs']


# data_dir    = Path(cfg["data_dir"])
# figures_dir = Path(cfg["figures_dir"])
# results_dir = Path(cfg["results_dir"])
# cache_dir   = Path(cfg["cache_dir"])
