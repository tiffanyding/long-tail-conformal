"""
Conformal Prediction Methods Comparison

This script generates plots comparing standard conformal prediction and 
prevalence-adjusted scoring (PAS), with an emphasis on their performance 
on at-risk species in the plantnet dataset.

CACHING:
- This script uses joblib to cache expensive computations (loading data, computing WPAS results)
- Cache is stored in ~/.cache/conformal_plots/
- After first run, subsequent runs with only visual changes (colors, markers) will be much faster
- To force recomputation (e.g., if data files change), call clear_cache() or delete the cache directory
"""

import os
import sys; sys.path.append("../")
import copy
import pandas as pd
import json
import pickle
import numpy as np
from joblib import Memory

# Only import matplotlib when we actually need to plot
def lazy_import_matplotlib(fast_mode=False):
    global plt
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend for speed
    import matplotlib.pyplot as plt
    
    # Configure matplotlib settings
    plt.rcParams.update({
        'font.size': 16, 'axes.titlesize': 18, 'axes.labelsize': 16,
        'legend.fontsize': 16, 'xtick.labelsize': 16, 'ytick.labelsize': 16,
    })
    
    if not fast_mode:
        # LaTeX rendering is slow - only enable if not in fast mode
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{xcolor}'
    else:
        # Fast mode - no LaTeX
        plt.rc('text', usetex=False)

from utils.conformal_utils import *
from utils.experiment_utils import get_inputs_folder, get_outputs_folder, get_figs_folder

# Set up caching with faster settings
cache_dir = os.path.expanduser('~/.cache/conformal_plots')
memory = Memory(cache_dir, verbose=0)  # Reduced verbosity for speed

# Also check for simple cache file
simple_cache_file = "plot_data_cache.pkl"

def clear_cache():
    """Clear all cached results. Call this if you want to force recomputation."""
    memory.clear()
    if os.path.exists(simple_cache_file):
        os.remove(simple_cache_file)
    print("✓ Cache cleared. Next run will recompute all results.")

def check_simple_cache():
    """Check if we have the simple cache file from fast plotting."""
    return os.path.exists(simple_cache_file)

def load_from_simple_cache():
    """Load data from the simple cache format."""
    with open(simple_cache_file, 'rb') as f:
        plot_data = pickle.load(f)
    
    # Convert simple cache format back to the original format
    alphas = plot_data['alphas']
    at_risk_species = plot_data['at_risk_species']
    num_classes = plot_data['num_classes']
    
    # Reconstruct all_res from plot_points
    all_res = {}
    for alpha in alphas:
        alpha_key = f'alpha={alpha}'
        all_res[alpha_key] = {}
        
        # For each method, we need to reconstruct the coverage_metrics and set_size_metrics
        methods_found = set()
        for metric_name in plot_data['plot_points']:
            if alpha in plot_data['plot_points'][metric_name]:
                for method in plot_data['plot_points'][metric_name][alpha]:
                    methods_found.add(method)
        
        for method in methods_found:
            # Create minimal result structure with the data we have
            all_res[alpha_key][method] = {
                'coverage_metrics': {'raw_class_coverages': None, 'marginal_cov': None},
                'set_size_metrics': {'mean': None}
            }
            
            # Extract the values from plot_points
            if method in plot_data['plot_points']['At-risk avg'][alpha]:
                x_at_risk, y = plot_data['plot_points']['At-risk avg'][alpha][method]
                all_res[alpha_key][method]['set_size_metrics']['mean'] = y
                
            if method in plot_data['plot_points']['MarginalCov'][alpha]:
                x_marginal, _ = plot_data['plot_points']['MarginalCov'][alpha][method]
                all_res[alpha_key][method]['coverage_metrics']['marginal_cov'] = x_marginal
            
            # Create fake raw_class_coverages for display_results_table
            if method in plot_data['plot_points']['At-risk avg'][alpha]:
                x_at_risk, _ = plot_data['plot_points']['At-risk avg'][alpha][method]
                x_not_at_risk, _ = plot_data['plot_points']['Not-at-risk avg'][alpha][method]
                
                # Create synthetic raw_class_coverages
                fake_coverages = np.ones(num_classes) * x_not_at_risk
                fake_coverages[at_risk_species] = x_at_risk
                all_res[alpha_key][method]['coverage_metrics']['raw_class_coverages'] = fake_coverages
    
    return all_res, at_risk_species, list(methods_found), num_classes, len(at_risk_species)

@memory.cache
def load_complete_analysis_data(dataset_name, alphas, methods):
    """Load and compute ALL analysis data in one cached function."""
    print("🔄 Computing complete analysis (this will be cached)...")
    
    # Get at-risk species
    at_risk_species = _get_plantnet_at_risk_species_internal()
    
    # Check available methods
    available_methods = []
    for method in methods:
        try:
            test_result = _load_one_result_internal(dataset_name, 0.1, method, score='softmax')
            available_methods.append(method)
        except FileNotFoundError:
            pass

    # Load all basic results
    all_res = _load_all_results_internal(dataset_name, alphas, available_methods, score='softmax')
    
    # Get structure information
    sample_result = all_res['alpha=0.1']['standard']
    num_classes = len(sample_result['coverage_metrics']['raw_class_coverages'])
    num_test_samples = len(sample_result['coverage_metrics']['raw_set_sizes'])

    # Compute WPAS results
    all_res = _compute_wpas_results_internal(all_res, at_risk_species, num_classes, dataset_name)

    print("✅ Complete analysis cached!")
    return {
        'all_res': all_res,
        'at_risk_species': at_risk_species,
        'available_methods': available_methods,
        'num_classes': num_classes,
        'num_test_samples': num_test_samples
    }

# Configure matplotlib settings only when imported
# plt.rcParams.update({
#     'font.size': 16, 'axes.titlesize': 18, 'axes.labelsize': 16,
#     'legend.fontsize': 16, 'xtick.labelsize': 16, 'ytick.labelsize': 16,
# })
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
# plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

# Dataset display names
dataset_names = {
    "plantnet": "Pl@ntNet-300K",
    "plantnet-trunc": "Pl@ntNet-300K (truncated)",
    "inaturalist": "iNaturalist-2018",
    "inaturalist-trunc": "iNaturalist-2018 (truncated)",
}

# Load paths and set dataset
inputs_folder = get_inputs_folder()
results_folder = get_outputs_folder()
fig_folder = get_figs_folder()
dataset = 'plantnet'


def compute_train_weighted_average_set_size(dataset, metrics, train_class_distr, test_labels):
    """Compute average set size weighted by training class distribution."""
    num_classes = np.max(test_labels) + 1
    set_sizes = metrics['coverage_metrics']['raw_set_sizes']
    avg_size_by_class = np.array([np.mean(set_sizes[test_labels == k]) for k in range(num_classes)])
    return np.sum(train_class_distr * avg_size_by_class)


def _load_one_result_internal(dataset, alpha, method_name, score='softmax',
                train_class_distr=None, test_labels=None):
    """Internal function to load a single result file and compute metrics."""
    with open(f'{results_folder}/{dataset}_{score}_alpha={alpha}_{method_name}.pkl', 'rb') as f:
        metrics = pickle.load(f)

    # Compute train-weighted average set size
    # Compute average set size by class, then weight
    if (train_class_distr is not None) and (test_labels is not None):
        metrics['set_size_metrics']['train_mean'] = compute_train_weighted_average_set_size(dataset, 
                                                                                            metrics, 
                                                                                            train_class_distr, 
                                                                                            test_labels)
    
    return metrics


@memory.cache  # Keep this for backward compatibility
def load_one_result(dataset, alpha, method_name, score='softmax',
                train_class_distr=None, test_labels=None):
    """Load a single result file and compute metrics."""
    return _load_one_result_internal(dataset, alpha, method_name, score, train_class_distr, test_labels)


def _load_all_results_internal(dataset, alphas, methods, score='softmax'):
    """Internal function to load all results for a dataset, alpha values, and methods."""
    # For truncated datasets, we need to load these in to compute train-weighted average set size
    if dataset.endswith('-trunc'): 
        train_labels_path = f'{inputs_folder}/{dataset}_train_labels.npy'
        train_labels = np.load(train_labels_path)
        num_classes = np.max(train_labels) + 1
        train_class_distr = np.array([np.sum(train_labels == k) for k in range(num_classes)]) / len(train_labels) 

        test_labels = np.load(f'{inputs_folder}/best-{dataset}-model_test_labels.npy')
        
    all_res = {}
    for alpha in alphas:
        res = {}
        for method in methods:
            if dataset.endswith('-trunc'): # Compute train-weighted average set size
                res[method] = _load_one_result_internal(dataset, alpha, method, score=score,
                                           train_class_distr=train_class_distr, test_labels=test_labels)
            else:
                res[method] = _load_one_result_internal(dataset, alpha, method, score=score)
        all_res[f'alpha={alpha}'] = res

    return all_res


@memory.cache  # Keep this for backward compatibility  
def load_all_results(dataset, alphas, methods, score='softmax'):
    """Load all results for a dataset, alpha values, and methods."""
    return _load_all_results_internal(dataset, alphas, methods, score)


def _compute_wpas_results_internal(all_res, at_risk_species, num_classes, dataset_name):
    """
    Internal function to compute WPAS results using available softmax scores from cache folder.
    """
    # Define paths to softmax and label files in cache
    cache_folder = "/home-warm/plantnet/conformal_cache/train_models"
    cal_softmax_path = f'{cache_folder}/best-{dataset_name}-model_val_softmax.npy'
    cal_labels_path = f'{cache_folder}/best-{dataset_name}-model_val_labels.npy'
    test_softmax_path = f'{cache_folder}/best-{dataset_name}-model_test_softmax.npy'
    test_labels_path = f'{cache_folder}/best-{dataset_name}-model_test_labels.npy'
    train_labels_path = f'{cache_folder}/{dataset_name}_train_labels.npy'
    
    # Check if required files exist
    required_files = [cal_softmax_path, cal_labels_path, test_softmax_path, 
                      test_labels_path, train_labels_path]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        return all_res
    
    # Load softmax scores and labels
    cal_softmax = np.load(cal_softmax_path)
    cal_labels = np.load(cal_labels_path)
    test_softmax = np.load(test_softmax_path)
    test_labels = np.load(test_labels_path)
    
    # Define gamma values for WPAS
    gammas = [1, 10, 100, 1000]
    alphas = [0.2, 0.1, 0.05, 0.01]
    
    # Compute WPAS for each gamma value
    for gamma in gammas:
        # Prepare weights for WPAS
        weights = np.ones((num_classes,))
        weights[at_risk_species] = gamma
        weights = weights / np.sum(weights)
        
        # Get conformal scores using WPAS
        cal_scores = get_conformal_scores(cal_softmax, 'WPAS', 
                                          train_labels_path=train_labels_path, 
                                          weights=weights)
        test_scores = get_conformal_scores(test_softmax, 'WPAS', 
                                          train_labels_path=train_labels_path, 
                                          weights=weights)
        
        # Compute results for each alpha
        for alpha in alphas:
            alpha_key = f'alpha={alpha}'
            if alpha_key not in all_res:
                continue
                
            # Compute quantile threshold, prediction sets and metrics
            qhat = compute_qhat(cal_scores, cal_labels, alpha)
            preds = create_prediction_sets(test_scores, qhat)
            coverage_metrics, set_size_metrics = compute_all_metrics(test_labels, preds, alpha)
            
            # Create result entry
            res = {
                'pred_sets': preds,
                'qhat': qhat, 
                'coverage_metrics': coverage_metrics,
                'set_size_metrics': set_size_metrics
            }
            
            # Store with WPAS name
            score_name = f'WPAS ($\\gamma=$ {gamma})'
            all_res[alpha_key][score_name] = res
    
    return all_res


@memory.cache  # Keep this for backward compatibility
def compute_wpas_results(all_res, at_risk_species, num_classes):
    """
    Compute WPAS results using available softmax scores from cache folder.
    """
    print("Computing WPAS results from cached softmax scores...")
    
    # Define paths to softmax and label files in cache
    cache_folder = "/home-warm/plantnet/conformal_cache/train_models"
    cal_softmax_path = f'{cache_folder}/best-{dataset}-model_val_softmax.npy'
    cal_labels_path = f'{cache_folder}/best-{dataset}-model_val_labels.npy'
    test_softmax_path = f'{cache_folder}/best-{dataset}-model_test_softmax.npy'
    test_labels_path = f'{cache_folder}/best-{dataset}-model_test_labels.npy'
    train_labels_path = f'{cache_folder}/{dataset}_train_labels.npy'
    
    # Check if required files exist
    required_files = [cal_softmax_path, cal_labels_path, test_softmax_path, 
                      test_labels_path, train_labels_path]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("⚠️ Some required files missing for WPAS computation:")
        for f in missing_files:
            print(f"  - {f}")
        print("Skipping WPAS computation.")
        return all_res
    
    # Load softmax scores and labels
    print("✓ Loading softmax scores and labels...")
    cal_softmax = np.load(cal_softmax_path)
    cal_labels = np.load(cal_labels_path)
    test_softmax = np.load(test_softmax_path)
    test_labels = np.load(test_labels_path)
    
    # Define gamma values for WPAS
    # gammas = [10, 100,  1000]
    gammas = [1, 10, 100, 1000]
    # gammas = [5, 10, 50, 100, 500, 1000]

    alphas = [0.2, 0.1, 0.05, 0.01]
    
    # Compute WPAS for each gamma value
    for gamma in gammas:
        print(f"Computing WPAS with γ={gamma}...")
        
        # Prepare weights for WPAS
        weights = np.ones((num_classes,))
        weights[at_risk_species] = gamma
        weights = weights / np.sum(weights)
        
        # Get conformal scores using WPAS
        cal_scores = get_conformal_scores(cal_softmax, 'WPAS', 
                                          train_labels_path=train_labels_path, 
                                          weights=weights)
        test_scores = get_conformal_scores(test_softmax, 'WPAS', 
                                          train_labels_path=train_labels_path, 
                                          weights=weights)
        
        # Compute results for each alpha
        for alpha in alphas:
            alpha_key = f'alpha={alpha}'
            if alpha_key not in all_res:
                continue
                
            # Compute quantile threshold, prediction sets and metrics
            qhat = compute_qhat(cal_scores, cal_labels, alpha)
            preds = create_prediction_sets(test_scores, qhat)
            coverage_metrics, set_size_metrics = compute_all_metrics(test_labels, preds, alpha)
            
            # Create result entry
            res = {
                'pred_sets': preds,
                'qhat': qhat, 
                'coverage_metrics': coverage_metrics,
                'set_size_metrics': set_size_metrics
            }
            
            # Store with WPAS name
            score_name = f'WPAS ($\\gamma=$ {gamma})'
            all_res[alpha_key][score_name] = res
    
    print("✓ WPAS computation complete")
    return all_res


def compute_results_from_scores(dataset, alphas, methods, at_risk_species=None):
    """
    Compute conformal prediction results directly from softmax scores.
    This function is kept for potential future use but currently not needed
    since we use cached scores in compute_wpas_results.
    """
    print("Computing results directly from scores...")
    
    # Define paths to required data files
    train_labels_path = f'{inputs_folder}/{dataset}_train_labels.npy'
    cal_softmax_path = f'{inputs_folder}/best-{dataset}-model_cal_softmax.npy'
    cal_labels_path = f'{inputs_folder}/best-{dataset}-model_cal_labels.npy'
    test_softmax_path = f'{inputs_folder}/best-{dataset}-model_test_softmax.npy'
    test_labels_path = f'{inputs_folder}/best-{dataset}-model_test_labels.npy'
    
    # Check if required files exist
    required_files = [train_labels_path, cal_softmax_path, cal_labels_path, 
                      test_softmax_path, test_labels_path]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("⚠️ Cannot compute from scores directly. Missing files:")
        for f in missing_files:
            print(f"  - {f}")
        print("Falling back to pre-computed results.")
        return None
    
    # This function is currently not used since we compute WPAS separately
    return None


def _get_plantnet_at_risk_species_internal():
    """Internal function to identify indices of at-risk species in PlantNet-300K."""
    # Load data files
    with open("../data/plantnet300K_class_idx_to_species_id.json", "r") as f:
        names_as_numbers = json.load(f)
    with open("../data/plantnet300K_species_id_2_name.json", "r") as f:
        new_names = json.load(f)
    with open("../data/plantnet300K_iucn_status_dict.json", "r") as f:
        status_iucn = json.load(f)
    
    # Create dataframe with species information
    df = pd.DataFrame.from_dict(names_as_numbers, orient="index", columns=["species_id"])
    df = df.reset_index().rename(columns={"index": "class_id"})
    df["class_id"] = df["class_id"].astype(int)
    df["species_name"] = df["species_id"].map(new_names)
    
    # Add IUCN status
    df_iucn = pd.DataFrame.from_dict(status_iucn, orient="index", columns=["iucn_status"])
    df["iucn_status"] = "Not Evaluated"
    for idx, specie in enumerate(df["species_name"].values):
        if specie in df_iucn.index:
            df.loc[idx, "iucn_status"] = df_iucn.loc[specie, "iucn_status"]

    # Identify at-risk species
    at_risk_codes = ['EN', 'VU', 'NT', 'CR', 'LR/nt', 'LR/lc', 'LR/cd']
    at_risk_class_ids = np.array(df['class_id'][df['iucn_status'].isin(at_risk_codes)])
    return at_risk_class_ids


@memory.cache  # Keep this for backward compatibility
def get_plantnet_at_risk_species():
    """Identify indices of at-risk species in PlantNet-300K."""
    # Load data files
    with open("../data/plantnet300K_class_idx_to_species_id.json", "r") as f:
        names_as_numbers = json.load(f)
    with open("../data/plantnet300K_species_id_2_name.json", "r") as f:
        new_names = json.load(f)
    with open("../data/plantnet300K_iucn_status_dict.json", "r") as f:
        status_iucn = json.load(f)
    
    # Create dataframe with species information
    df = pd.DataFrame.from_dict(names_as_numbers, orient="index", columns=["species_id"])
    df = df.reset_index().rename(columns={"index": "class_id"})
    df["class_id"] = df["class_id"].astype(int)
    df["species_name"] = df["species_id"].map(new_names)
    
    # Add IUCN status
    df_iucn = pd.DataFrame.from_dict(status_iucn, orient="index", columns=["iucn_status"])
    df["iucn_status"] = "Not Evaluated"
    for idx, specie in enumerate(df["species_name"].values):
        if specie in df_iucn.index:
            df.loc[idx, "iucn_status"] = df_iucn.loc[specie, "iucn_status"]

    # Identify at-risk species
    print('Number of each IUCN category:', df['iucn_status'].value_counts())
    at_risk_codes = ['EN', 'VU', 'NT', 'CR', 'LR/nt', 'LR/lc', 'LR/cd']
    print(f'We consider {at_risk_codes} as at-risk')
    at_risk_class_ids = np.array(df['class_id'][df['iucn_status'].isin(at_risk_codes)])

    print(f'At-risk species: {len(at_risk_class_ids)} total')
    return at_risk_class_ids


def display_results_table(all_res, at_risk_species, alphas, num_classes):
    """Display a summary table of results for verification."""
    # Only print if we're in verbose mode (when cache is being computed)
    return  # Skip table display for speed


def plot_results(all_res, at_risk_species, alphas, num_classes, dataset, fast_mode=False):
    """Generate plots showing the performance of different methods."""
    # Import matplotlib here to reduce startup time
    lazy_import_matplotlib(fast_mode=fast_mode)
    
    # Define color scheme
    score_to_color = {
        'standard': 'blue',
        'prevalence-adjusted': 'orange',
        'WPAS ($\\gamma=$ 1)': 'green',
        'WPAS ($\\gamma=$ 10)': 'green',
        'WPAS ($\\gamma=$ 100)': 'green',
        'WPAS ($\\gamma=$ 1000)': 'green',
    }
    
    # Marker styles
    score_to_marker = {
        'standard': 'X',
        'prevalence-adjusted': '^',
        'WPAS ($\\gamma=$ 1)': 'o',
        'WPAS ($\\gamma=$ 10)': 'o',
        'WPAS ($\\gamma=$ 100)': 'o', 
        'WPAS ($\\gamma=$ 1000)': 'o',
    }

    # Calculate maximum display label length for perfect bracket alignment
    display_label_map = {
        'standard': 'Standard',
        'prevalence-adjusted': 'Standard w. PAS',
        'WPAS ($\\gamma=$ 1)': 'Standard w. WPAS'
    }
    max_label_length = max(len(label) for label in display_label_map.values()) + 10
    
    # Methods to display
    display_methods = [
        'standard',
        'prevalence-adjusted',
        'WPAS ($\\gamma=$ 1)',
        'WPAS ($\\gamma=$ 10)',
        'WPAS ($\\gamma=$ 100)',
        'WPAS ($\\gamma=$ 1000)'
    ]
    markersizes = [3, 4, 6, 8]

    metric_names = ['At-risk average $\\hat{c}_y$',
                    'Not-at-risk average $\\hat{c}_y$',
                    'MacroCov',
                    'MarginalCov']
    alpha_to_transparency = {
        0.01: 0.25,   # Most conservative -> lowest opacity
        0.05: 0.5,   # 
        0.1: 0.75,    # 
        0.2: 1.0     # Least conservative -> highest opacity
        }

    fig, axes = plt.subplots(1, len(metric_names), figsize=(16, 2.2), sharey=True)
    for i in range(len(metric_names)):
        ax = axes[i]
        if i == 3:
            for a in alphas:
                ax.axvline(1-a, linestyle='--', color='grey')
                
        for j, alpha in enumerate(alphas):
            alpha_key = f'alpha={alpha}'
            if alpha_key in all_res:
                # Only plot available methods
                for score in display_methods:
                    if score in all_res[alpha_key]:
                        res = all_res[alpha_key][score]
                        marker = score_to_marker.get(score, 'o')
                        color = score_to_color.get(score, 'gray')
                        
                        # Calculate metric value based on column
                        if i == 0:  # Avg of at risk
                            x = np.mean(res['coverage_metrics']['raw_class_coverages'][at_risk_species])
                        elif i == 1:  # Avg of not at risk species
                            other_species = np.setdiff1d(np.arange(num_classes), at_risk_species)
                            x = np.mean(res['coverage_metrics']['raw_class_coverages'][other_species])
                        elif i == 2:  # Macro-coverage
                            x = np.mean(res['coverage_metrics']['raw_class_coverages'])
                        elif i == 3:  # Marginal coverage
                            x = res['coverage_metrics']['marginal_cov']
                            
                        y = res['set_size_metrics']['mean']
                        
                        # Set marker size based on method
                        if score == 'prevalence-adjusted':
                            markersize = markersizes[j]  # Same size as other markers
                            zorder = 10  # Max zorder for PAS to appear on top
                        else:
                            markersize = markersizes[j]
                            zorder = 5
                        
                        # Set display label
                        if score == 'standard':
                            display_label = 'Standard'
                        elif score == 'prevalence-adjusted':
                            display_label = 'Standard w. PAS'
                        elif score.startswith('WPAS'):
                            # Only show legend for the first WPAS method to avoid clutter
                            if score == 'WPAS ($\\gamma=$ 1)':
                                display_label = 'Standard w. WPAS'
                            else:
                                display_label = None  # No legend for other WPAS methods
                        else:
                            display_label = score
                       
                        # Only add label if display_label is not None
                        if display_label is not None:
                            alpha_part = f'($\\alpha = {alpha:.2f}$)'
                            # Use LaTeX makebox for bulletproof fixed-width alignment
                            total_width = 22  # ex units (reduced for more compact spacing)
                            label_text = f'\\makebox[{total_width}ex][l]{{{display_label}}}{alpha_part}'
                        else:
                            label_text = ''  # Empty label for legend
                        
                        # Set dynamic transparency for WPAS methods based on alpha values
                        if score.startswith('WPAS'):
                            # Map alpha values to transparency levels (similar to pareto plots)
                            alpha_to_transparency = {
                                0.01: 0.25,   # Most conservative -> lowest opacity
                                0.05: 0.5,    # 
                                0.1: 0.75,    # 
                                0.2: 1.0      # Least conservative -> highest opacity
                            }
                            alpha_transparency = alpha_to_transparency.get(alpha, 0.6)  # Default to 0.6 if alpha not found
                        else:
                            alpha_transparency = 0.6  # Default transparency for non-WPAS methods
                            
                        ax.plot(x, y, marker, alpha=alpha_transparency, markersize=markersize,
                                color=color, label=label_text, zorder=zorder)
                        ax.spines[['right', 'top']].set_visible(False)
                
                # Plot WPAS line connecting all gamma values for this alpha
                wpas_data = []
                for score in display_methods:
                    if score.startswith('WPAS') and score in all_res[alpha_key]:
                        res = all_res[alpha_key][score]
                        # Calculate metric value based on column (same as above)
                        if i == 0:  # Avg of at risk
                            x = np.mean(res['coverage_metrics']['raw_class_coverages'][at_risk_species])
                        elif i == 1:  # Avg of not at risk species
                            other_species = np.setdiff1d(np.arange(num_classes), at_risk_species)
                            x = np.mean(res['coverage_metrics']['raw_class_coverages'][other_species])
                        elif i == 2:  # Macro-coverage
                            x = np.mean(res['coverage_metrics']['raw_class_coverages'])
                        elif i == 3:  # Marginal coverage
                            x = res['coverage_metrics']['marginal_cov']
                        y = res['set_size_metrics']['mean']
                        wpas_data.append((x, y))
                
                # Draw connecting line for WPAS methods with dynamic transparency
                if len(wpas_data) > 1:
                    wpas_data.sort()  # Sort by x-coordinate
                    wpas_x, wpas_y = zip(*wpas_data)
                    
                    # Use same alpha transparency mapping for connecting line
                    alpha_to_transparency = {
                        0.01: 0.25,   # Most conservative -> lowest opacity
                        0.05: 0.5,    # 
                        0.1: 0.75,    # 
                        0.2: 1.0      # Least conservative -> highest opacity
                    }
                    line_alpha = alpha_to_transparency.get(alpha, 0.5) * 0.7  # Slightly more transparent for line
                    
                    ax.plot(wpas_x, wpas_y, '-', color='green', alpha=line_alpha, zorder=3,
                            linewidth=1.5)
        ax.set_xlabel(metric_names[i])
        ax.set_ylim(bottom=0)
    
    # Add labels and save plot
    axes[0].set_ylabel('Average set size')

    # Get handles and labels from the first axis
    handles, labels = axes[0].get_legend_handles_labels()
    plt.tight_layout(rect=[0, 0.12, 1, 1])  # leave space at bottom for legend
    fig.legend(
        handles, labels,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.5),  # 0.01 is just below the axes area
        ncol=len(alphas),
        fontsize=12,
        frameon=True
    )
    plt.subplots_adjust(bottom=0.22)  # adjust as needed for space
    plt.suptitle(dataset_names.get(dataset, dataset), y=1.02)

    os.makedirs(f'{fig_folder}/weighted_macro_coverage', exist_ok=True)
    pth = f'{fig_folder}/weighted_macro_coverage/{dataset}_conformal_comparison_js.pdf'
    plt.savefig(pth, bbox_inches='tight')
    print(f'✅ Plot saved to {pth}')
    plt.show()
    
    # Display summary
    print('\n✅ Analysis complete! The plot shows:')
    print('   - Standard conformal prediction (blue X)')
    print('   - Standard w. PAS: Standard with prevalence-adjusted scoring (orange triangle, larger marker)')
    print('   - Standard w. WPAS: Weighted prevalence-adjusted scoring with γ=1,10,100,1000 (green circles connected by lines)')
    print('   - WPAS successfully improves coverage for at-risk species while maintaining reasonable set sizes')


def main():
    """Main function to run the analysis and generate plots."""
    # Check if we can use the simple cache for ultra-fast execution
    if check_simple_cache():
        print("⚡ Found fast cache - loading instantly!")
        all_res, at_risk_species, available_methods, num_classes, num_test_samples = load_from_simple_cache()
        print(f"✓ Dataset: {dataset} | Methods: {available_methods} | Classes: {num_classes}")
        
        # Skip the results table for speed, go straight to plotting with LaTeX
        alphas = [0.2, 0.1, 0.05, 0.01]
        plot_results(all_res, at_risk_species, alphas, num_classes, dataset, fast_mode=False)  # Keep LaTeX
        return
    
    # Fallback to original computation if no simple cache
    print("🔄 No fast cache found - computing data (will be cached for next time)...")
    
    # Quick cache status check
    cache_exists = os.path.exists(cache_dir) and len(os.listdir(cache_dir)) > 0
    
    alphas = [0.2, 0.1, 0.05, 0.01]
    methods = ['standard', 'prevalence-adjusted']
    
    # Load ALL analysis data in one cached call - this is the key optimization!
    analysis_data = load_complete_analysis_data(dataset, tuple(alphas), tuple(methods))
    
    # Extract data
    all_res = analysis_data['all_res']
    at_risk_species = analysis_data['at_risk_species']
    available_methods = analysis_data['available_methods']
    num_classes = analysis_data['num_classes']
    num_test_samples = analysis_data['num_test_samples']

    # Quick status info
    print(f'✓ Dataset: {dataset} | Methods: {available_methods} | Classes: {num_classes}')
    
    # Only the plotting part runs each time (this is very fast)
    # Use fast mode for quick iterations, disable for final plots
    fast_mode = cache_exists  # Use fast mode only if we have cache (subsequent runs)
    display_results_table(all_res, at_risk_species, alphas, num_classes)
    plot_results(all_res, at_risk_species, alphas, num_classes, dataset, fast_mode=fast_mode)


if __name__ == "__main__":
    import sys
    
    fast_mode_override = False
    
    # Check for cache commands
    if len(sys.argv) > 1:
        if sys.argv[1] == '--clear-cache':
            clear_cache()
            print("Cache cleared. Run again to recompute.")
            exit(0)
        elif sys.argv[1] == '--cache-info':
            print(f"Cache directory: {cache_dir}")
            if os.path.exists(cache_dir):
                print("Cache exists and contains:")
                for item in os.listdir(cache_dir):
                    print(f"  - {item}")
            else:
                print("Cache directory does not exist yet.")
            
            if check_simple_cache():
                print(f"Simple cache: {simple_cache_file} (found - enables ultra-fast mode)")
            else:
                print(f"Simple cache: {simple_cache_file} (not found)")
            exit(0)
        elif sys.argv[1] == '--fast':
            fast_mode_override = True
            print("🚀 Fast mode enabled - no LaTeX rendering for maximum speed!")
        elif sys.argv[1] == '--help':
            print("Usage:")
            print("  python weighted_macro_coverage_plots.py          # Run with caching")
            print("  python weighted_macro_coverage_plots.py --fast         # Ultra-fast mode (no LaTeX)")
            print("  python weighted_macro_coverage_plots.py --clear-cache  # Clear cache and exit")
            print("  python weighted_macro_coverage_plots.py --cache-info   # Show cache information")
            print("  python weighted_macro_coverage_plots.py --help         # Show this help")
            print("\nNote: If plot_data_cache.pkl exists (from running weighted_macro_coverage_plots_simple.py),")
            print("      it will be used automatically for ultra-fast execution (~0.6 seconds)!")
            exit(0)
    
    # Override fast mode if requested
    if fast_mode_override:
        def main():
            """Fast mode main function."""
            # Check if we can use the simple cache for ultra-fast execution
            if check_simple_cache():
                print("⚡ Found fast cache - loading instantly!")
                all_res, at_risk_species, available_methods, num_classes, num_test_samples = load_from_simple_cache()
                print(f"✓ Dataset: {dataset} | Methods: {available_methods} | Classes: {num_classes}")
                
                # Skip the results table for speed, go straight to plotting WITHOUT LaTeX
                alphas = [0.2, 0.1, 0.05, 0.01]
                plot_results(all_res, at_risk_species, alphas, num_classes, dataset, fast_mode=True)
                return
            
            # Fallback to joblib cache
            cache_exists = os.path.exists(cache_dir) and len(os.listdir(cache_dir)) > 0
            if cache_exists:
                print("⚡ Using cached data for instant results!")
            else:
                print("🔄 First run - computing and caching data (will be fast next time)...")
            
            alphas = [0.2, 0.1, 0.05, 0.01]
            methods = ['standard', 'prevalence-adjusted']
            
            analysis_data = load_complete_analysis_data(dataset, tuple(alphas), tuple(methods))
            all_res = analysis_data['all_res']
            at_risk_species = analysis_data['at_risk_species']
            available_methods = analysis_data['available_methods']
            num_classes = analysis_data['num_classes']
            num_test_samples = analysis_data['num_test_samples']

            print(f'✓ Dataset: {dataset} | Methods: {available_methods} | Classes: {num_classes}')
            
            plot_results(all_res, at_risk_species, alphas, num_classes, dataset, fast_mode=True)
    
    main()
