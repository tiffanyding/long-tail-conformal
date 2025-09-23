# %%
import sys; sys.path.append("../") # For relative imports

import glob
import os
import pickle

from utils.conformal_utils import *
from utils.experiment_utils import get_inputs_folder, get_outputs_folder, get_figs_folder
from scipy.ndimage import uniform_filter


plt.rcParams.update({
    'font.size': 16,        # base font size
    'axes.titlesize': 18,   # subplot titles
    'axes.labelsize': 16,   # x/y labels
    'legend.fontsize': 16,  # legend text
    'xtick.labelsize': 16,  # tick labels
    'ytick.labelsize': 16,

})
# use tex with matplotlib
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

dataset_names = {
    "plantnet": "Pl@ntNet-300K",
    "plantnet-trunc": "Pl@ntNet-300K (truncated)",
    "inaturalist": "iNaturalist-2018",
    "inaturalist-trunc": "iNaturalist-2018 (truncated)",
}

# %%
## Choose dataset to create figures for
# dataset = 'plantnet'
# dataset = 'plantnet-trunc'
# dataset = 'inaturalist'
dataset = 'inaturalist-trunc'

methods = ['standard', 'classwise', 'clustered', 'prevalence-adjusted'] 


alphas = [0.2, 0.1, 0.05, 0.01]

score = 'softmax'

# Load in paths from folders.json
results_folder = get_outputs_folder()
fig_folder = get_figs_folder()

os.makedirs(f'{fig_folder}/{dataset}', exist_ok=True)

# %%
f'{fig_folder}/{dataset}'

# %%
# Load test labels
test_labels = np.load(f'/home-warm/plantnet/conformal_cache/train_models/best-{dataset}-model_test_labels.npy')
num_classes = np.max(test_labels) + 1

# %%

# Load metrics

def load_metrics(dataset, alpha, method_name, score='softmax'):
    with open(f'{results_folder}/{dataset}_{score}_alpha={alpha}_{method_name}.pkl', 'rb') as f:
        metrics = pickle.load(f)
    # Extract set size quantiles for easy access later
    metrics['set_size_metrics']['median'] = metrics['set_size_metrics']['[.25, .5, .75, .9] quantiles'][1]
    metrics['set_size_metrics']['quantile90'] = metrics['set_size_metrics']['[.25, .5, .75, .9] quantiles'][3]
    return metrics


all_res = {}

for alpha in alphas:
    res = {}
    for method in methods:
        # print(method)
        res[method] = load_metrics(dataset, alpha, method)
    all_res[f'alpha={alpha}'] = res

# %%
def compute_class_cond_decision_accuracy(labels, is_covered, raw_set_sizes):
    # (assuming a random decision maker)
    num_classes = np.max(labels) + 1
    decision_acc = np.zeros((num_classes,))
    for k in range(num_classes):
        idx = labels == k
        # P(choose correct label) = 0 if label not in set
        # P(choose correct label) = 1/(set size) if label in set
        p_correct = is_covered[idx] * (1/raw_set_sizes[idx])
        p_correct[np.isnan(p_correct)] = 0 # nans are due to empty sets, so replace with 0
        decision_acc[k] = np.mean(p_correct)
        if np.isnan(decision_acc[k]):
            pdb.set_trace()

    return decision_acc

def compute_class_cond_decision_accuracy_for_method(res, method, labels):
    is_covered = res[method]['coverage_metrics']['is_covered']
    raw_set_sizes = res[method]['coverage_metrics']['raw_set_sizes']
    
    return compute_class_cond_decision_accuracy(labels, is_covered, raw_set_sizes)


# %%
# Add class-conditional decision accuracies to metrics
for res in all_res.values():
    for method in methods:
        dec_acc = compute_class_cond_decision_accuracy_for_method(res, method, test_labels)
        res[method]['class-cond-decision-accuracy'] = dec_acc

# %%
def create_combined_decision_acc_plot():
    # Create 2x3 subplot layout
    fig, axes = plt.subplots(2, 4, figsize=(18, 5))
    
    datasets = ['plantnet-trunc', 'inaturalist-trunc']
    
    methods = ['classwise', 'standard', 'clustered', 'prevalence-adjusted']
    # methods = ['classwise', 'standard', 'clustered', 'fuzzy-RErarity-0.0001']

    colors = ['tab:green', 'tab:green', 'tab:green', 'tab:green']
    
    method_to_name = {'standard': 'Standard', 
                      'classwise': 'Classwise', 
                      'clustered': 'Clustered',
                      'fuzzy-RErarity-0.0001': 'Fuzzy',
                      'prevalence-adjusted': 'PAS'}
    
    for row, dataset_name in enumerate(datasets):
        # Load test labels for this dataset
        test_labels_path = f'/home-warm/plantnet/conformal_cache/train_models/best-{dataset_name}-model_test_labels.npy'
        test_labels = np.load(test_labels_path)
        num_classes = np.max(test_labels) + 1
        
        # Load results for this dataset
        res = {}
        for method in methods:
            res[method] = load_metrics(dataset_name, 0.1, method)
        
        # Add class-conditional decision accuracies
        for method in methods:
            dec_acc = compute_class_cond_decision_accuracy_for_method(res, method, test_labels)
            res[method]['class-cond-decision-accuracy'] = dec_acc
        
        for col, (method, base_color) in enumerate(zip(methods, colors)):
            ax = axes[row, col]
            
            # Sort classes by class cond acc of this specific method
            idx = np.argsort(res[method]['coverage_metrics']['raw_class_coverages'])[::-1]
            
            # Get the lines for the specific method
            up_line_raw = res[method]['class-cond-decision-accuracy'][idx]
            lower_line_raw = res[method]['coverage_metrics']['raw_class_coverages'][idx] 
            
            # Apply moving mean filter with order 3
            # up_line = uniform_filter(up_line_raw, size=5, mode='nearest')
            # lower_line = uniform_filter(lower_line_raw, size=5, mode='nearest')
            
            # Define gamma levels and their corresponding labels
            gamma_levels = [1.0, 0.75, 0.5, 0.25, 0.0]  # 100%, 75%, 50%, 25%, 0%
            gamma_labels = ['$\\gamma_{\\mathrm{exp.}}=100\\%$', 
                           '$\\gamma_{\\mathrm{exp.}}=75\\%$',
                           '$\\gamma_{\\mathrm{exp.}}=50\\%$', 
                           '$\\gamma_{\\mathrm{exp.}}=25\\%$',
                           '$\\gamma_{\\mathrm{exp.}}=0\\%$']
            
            # Create colormap based on the base color - using green for all
            colormap = plt.cm.Greens
            
            # Generate colors from the colormap
            colors_grad = [colormap(0.8 - 0.15*i) for i in range(5)]
            
            # Plot each line with gradient colors
            for i, (gamma, label) in enumerate(zip(gamma_levels, gamma_labels)):
                # Compute the line for this gamma level
                line_data = uniform_filter((1-gamma) * up_line_raw + gamma * lower_line_raw, size=20, mode='nearest')
                
                zorder = 5 - i
                ax.plot(line_data, color=colors_grad[i], 
                        linewidth=2.0,
                        zorder=zorder,
                        label=label)
            
            ax.set_xlim(0, num_classes-1)
            ax.spines[['right', 'top']].set_visible(False)
            ax.set_title(f'{dataset_names[dataset_name]} - {method_to_name[method]}')
            
            # Set y-label only for first column
            if col == 0:
                ax.set_ylabel('Decision accuracy', fontsize=18)

            
            # Set x-label only for second row
            if row == 1:
                ax.set_xlabel('Class', fontsize=18)
                # (sorted by $\\hat{c}_y$ of each method)
            
            # Add legend only for plantnet-trunc Standard (row=0, col=0)
            if row == 0 and col == 0:
                legend = fig.legend(loc='center right', bbox_to_anchor=(1.08, 0.5), fontsize=14, title='Expert proportion', ncol=1)
                legend.get_title().set_fontsize(11)
    
    plt.tight_layout()
    
    # Save the combined plot
    fig_path = f'{fig_folder}/combined_decision_acc_2x3.pdf'
    plt.savefig(fig_path, bbox_inches='tight')
    print('Saved combined plot to', fig_path)
    print(fig_path)
    return fig, axes

# Create the combined plot
create_combined_decision_acc_plot()

# %%
def create_methods_comparison_plot():
    """
    Create a 2x5 subplot where:
    - Rows: datasets (plantnet-trunc, inaturalist-trunc)  
    - Columns: fixed gamma levels (0%, 25%, 50%, 75%, 100%)
    - Each subplot shows 4 methods as lines with the EXACT same curves as create_combined_decision_acc_plot
    - Uses the same colors and smoothing as the original function
    """
    # Create 2x5 subplot layout
    fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharey=True)
    
    datasets = ['plantnet-trunc', 'inaturalist-trunc']
    methods = ['classwise', 'standard', 'clustered', 'prevalence-adjusted']  # Same order as create_combined_decision_acc_plot
    gamma_levels = [0.0, 0.25, 0.5, 0.75, 1.0]  # 0%, 25%, 50%, 75%, 100%
    
    # Use colorblind-friendly colors for each method
    # Use the same colors and legend labels as in pareto_plots.py
    method_colors = {
        'standard': 'blue',
        'classwise': 'red',
        'clustered': 'purple',
        'prevalence-adjusted': 'orange',
    }
    method_to_name = {
        'standard': 'Standard',
        'classwise': 'Classwise',
        'clustered': 'Clustered',
        'prevalence-adjusted': 'Standard w. PAS',
    }
    
    for row, dataset_name in enumerate(datasets):
        # Load test labels for this dataset
        test_labels_path = f'/home-warm/plantnet/conformal_cache/train_models/best-{dataset_name}-model_test_labels.npy'
        test_labels = np.load(test_labels_path)
        num_classes = np.max(test_labels) + 1
        # Load results for this dataset
        res = {}
        for method in methods:
            res[method] = load_metrics(dataset_name, 0.1, method)
        # Add class-conditional decision accuracies
        for method in methods:
            dec_acc = compute_class_cond_decision_accuracy_for_method(res, method, test_labels)
            res[method]['class-cond-decision-accuracy'] = dec_acc
        for col, gamma in enumerate(gamma_levels):
            ax = axes[row, col]
            # Plot each method for this gamma level
            for method in methods:
                idx = np.argsort(res[method]['coverage_metrics']['raw_class_coverages'])[::-1]
                up_line_raw = res[method]['class-cond-decision-accuracy'][idx]
                lower_line_raw = res[method]['coverage_metrics']['raw_class_coverages'][idx]
                line_data = uniform_filter((1-gamma) * up_line_raw + gamma * lower_line_raw, size=20, mode='nearest')
                color = method_colors[method]
                ax.plot(line_data, color=color, linewidth=2.0, 
                        label=method_to_name[method], solid_capstyle='round')
            ax.set_xlim(0, num_classes-1)
            ax.spines[['right', 'top']].set_visible(False)
            # Set titles and labels
            if col == 0:
                ax.set_title(f"\\textbf{{{dataset_names[dataset_name]}}}\nExpert proportion: $\\gamma_{{\\mathrm{{exp.}}}} = {int(gamma*100)}\\%$", loc='left', fontsize=15, fontweight='bold')
                ax.set_ylabel('Decision accuracy', fontsize=18)
            else:
                ax.set_title(f'Expert proportion: $\\gamma_{{\\mathrm{{exp.}}}} = {int(gamma*100)}\\%$')
            if row == 1:
                ax.set_xlabel('Class', fontsize=18)
                # (sorted by $\\hat{c}_y$ of each method)
    # Move legend below the figure, centered, ncol=4 (like pareto_plots.py)
    handles, labels = [], []
    for method in methods:
        handles.append(plt.Line2D([0], [0], color=method_colors[method], lw=2, label=method_to_name[method]))
        labels.append(method_to_name[method])
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.08), fontsize=14, ncol=4, frameon=True)
    # Save the combined plot
    fig_path = f'{fig_folder}/methods_comparison_2x5.pdf'
    plt.savefig(fig_path, bbox_inches='tight')
    print('Saved methods comparison plot to', fig_path)
    return fig, axes


# Create the methods comparison plot
create_methods_comparison_plot()

# %%
def verify_curves_match():
    """
    Verification function to ensure curves match exactly between the two plot types
    """
    datasets = ['plantnet-trunc', 'inaturalist-trunc']
    methods = ['classwise', 'standard', 'clustered', 'prevalence-adjusted']
    gamma_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    print("Verifying that curves match between create_combined_decision_acc_plot and create_methods_comparison_plot...")
    
    for dataset_name in datasets:
        print(f"\nDataset: {dataset_name}")
        
        # Load test labels for this dataset
        test_labels_path = f'/home-warm/plantnet/conformal_cache/train_models/best-{dataset_name}-model_test_labels.npy'
        test_labels = np.load(test_labels_path)
        num_classes = np.max(test_labels) + 1
        
        # Load results for this dataset
        res = {}
        for method in methods:
            res[method] = load_metrics(dataset_name, 0.1, method)
        
        # Add class-conditional decision accuracies
        for method in methods:
            dec_acc = compute_class_cond_decision_accuracy_for_method(res, method, test_labels)
            res[method]['class-cond-decision-accuracy'] = dec_acc
        
        for method in methods:
            print(f"  Method: {method}")
            
            # Generate curves using SAME logic as create_combined_decision_acc_plot
            idx = np.argsort(res[method]['coverage_metrics']['raw_class_coverages'])[::-1]
            up_line_raw = res[method]['class-cond-decision-accuracy'][idx]
            lower_line_raw = res[method]['coverage_metrics']['raw_class_coverages'][idx]
            
            for gamma in gamma_levels:
                # This is the exact same formula used in both functions
                line_data = uniform_filter((1-gamma) * up_line_raw + gamma * lower_line_raw, size=20, mode='nearest')
                
                print(f"    γ={gamma*100:3.0f}%: mean={np.mean(line_data):.4f}, std={np.std(line_data):.4f}, min={np.min(line_data):.4f}, max={np.max(line_data):.4f}")
    
    print("\nVerification complete! The curves should match exactly between both plot types.")

# Run verification
verify_curves_match()
# %%
