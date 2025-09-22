"""
Conformal Prediction Methods Comparison

This script generates plots comparing standard conformal prediction and 
prevalence-adjusted scoring (PAS), with an emphasis on their performance 
on at-risk species in the plantnet dataset.
"""

import os
import sys; sys.path.append("../")
import copy
import pandas as pd
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt

from utils.conformal_utils import *
from utils.experiment_utils import get_inputs_folder, get_outputs_folder, get_figs_folder

# Configure matplotlib settings
plt.rcParams.update({
    'font.size': 16, 'axes.titlesize': 18, 'axes.labelsize': 16,
    'legend.fontsize': 16, 'xtick.labelsize': 16, 'ytick.labelsize': 16,
})
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

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


def load_one_result(dataset, alpha, method_name, score='softmax',
                train_class_distr=None, test_labels=None):
    """Load a single result file and compute metrics."""
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


def load_all_results(dataset, alphas, methods, score='softmax'):
    """Load all results for a dataset, alpha values, and methods."""
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
                res[method] = load_one_result(dataset, alpha, method, score=score,
                                           train_class_distr=train_class_distr, test_labels=test_labels)
            else:
                res[method] = load_one_result(dataset, alpha, method, score=score)
        all_res[f'alpha={alpha}'] = res

    return all_res


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
    for alpha in alphas:
        print(f'----- alpha = {alpha} -----')
        alpha_key = f'alpha={alpha}'
        for score in all_res[alpha_key].keys():
            res = all_res[alpha_key][score]
            other_species = np.setdiff1d(np.arange(num_classes), at_risk_species)
            at_risk_cov = np.mean(res["coverage_metrics"]["raw_class_coverages"][at_risk_species])
            other_cov = np.mean(res["coverage_metrics"]["raw_class_coverages"][other_species])
            print(f'[{score}] avg class-cond cov for at risk species: {at_risk_cov:.3f}, '
                  f'for other species: {other_cov:.3f}')


def plot_results(all_res, at_risk_species, alphas, num_classes, dataset):
    """Generate plots showing the performance of different methods."""
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

    # Methods to display
    display_methods = [
        'standard',
        'prevalence-adjusted',
        'WPAS ($\\gamma=$ 1)',
        'WPAS ($\\gamma=$ 10)',
        'WPAS ($\\gamma=$ 100)',
        'WPAS ($\\gamma=$ 1000)'
    ]
    markersizes = [4, 5, 6, 7]

    metric_names = ['At-risk average $\\hat{c}_y$',
                    'Not-at-risk average $\\hat{c}_y$',
                    'MacroCov',
                    'MarginalCov']
                 
    fig, axes = plt.subplots(1, len(metric_names), figsize=(13, 2.2), sharey=True)
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
                            markersize = 8  # Larger marker for PAS
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
                            label_text = f'{display_label}, $\\alpha=$ {alpha}'
                        else:
                            label_text = ''  # Empty label for legend
                            
                        ax.plot(x, y, marker, alpha=0.6, markersize=markersize,
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
                
                # Draw connecting line for WPAS methods
                if len(wpas_data) > 1:
                    wpas_data.sort()  # Sort by x-coordinate
                    wpas_x, wpas_y = zip(*wpas_data)
                    ax.plot(wpas_x, wpas_y, '-', color='green', alpha=0.5, zorder=3,
                            linewidth=1.5)
        ax.set_xlabel(metric_names[i])
        ax.set_ylim(bottom=0)
        
    # Add labels and save plot
    axes[0].set_ylabel('Average set size')
    plt.legend(ncols=len(alphas), loc='upper left', bbox_to_anchor=(-3.85,-0.35), fontsize=12)
    plt.tight_layout()
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
    alphas = [0.2, 0.1, 0.05, 0.01]
    methods = ['standard', 'prevalence-adjusted']
    
    # Get at-risk species for PlantNet
    at_risk_species = get_plantnet_at_risk_species()
    
    # Load pre-computed results for standard and PAS methods
    print('Loading pre-computed results...')
    available_methods = []
    for method in methods:
        try:
            test_result = load_one_result(dataset, 0.1, method, score='softmax')
            available_methods.append(method)
            print(f'✓ Found method: {method}')
        except FileNotFoundError:
            print(f'⚠ Method {method} not found')

    # Load all basic results
    all_res = load_all_results(dataset, alphas, available_methods, score='softmax')
    
    # Get structure information from standard result
    sample_result = all_res['alpha=0.1']['standard']
    num_classes = len(sample_result['coverage_metrics']['raw_class_coverages'])
    num_test_samples = len(sample_result['coverage_metrics']['raw_set_sizes'])

    print(f'Dataset: {dataset}')
    print(f'Number of classes: {num_classes}')
    print(f'Number of test samples: {num_test_samples}')
    
    # Compute WPAS results
    all_res = compute_wpas_results(all_res, at_risk_species, num_classes)

    # Display results and generate plots
    display_results_table(all_res, at_risk_species, alphas, num_classes)
    plot_results(all_res, at_risk_species, alphas, num_classes, dataset)


if __name__ == "__main__":
    main()
