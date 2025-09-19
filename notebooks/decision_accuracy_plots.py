# %%
import sys; sys.path.append("../") # For relative imports

import glob
import os
import pickle

from utils.conformal_utils import *
from utils.experiment_utils import get_inputs_folder, get_outputs_folder, get_figs_folder

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

methods = ['standard', 'classwise', 'classwise-exact', 'clustered', 'prevalence-adjusted'] + \
            [f'fuzzy-rarity-{bw}' for bw in [1e-16, 1e-12, 1e-8, 1e-6, 0.0001, 0.001, 0.01, .1 , 10, 1000]] +\
            [f'fuzzy-RErarity-{bw}' for bw in [1e-16, 1e-12, 1e-8, 1e-6, 0.0001, 0.001, 0.01, .1 , 10, 1000]] +\
            [f'fuzzy-READDrarity-{bw}' for bw in [1e-16, 1e-12, 1e-8, 1e-6, 0.0001, 0.001, 0.01, .1 , 10, 1000]] +\
            [f'cvx-cw_weight={w}' for w in [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975, 0.99 , 0.999, 1]] +\
            [f'monotonic-cvx-cw_weight={w}' for w in 1 - np.array([0, .001, .01, .025, .05, .1, .15, .2, .4, .6, .8, 1])]


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
    fig, axes = plt.subplots(2, 3, figsize=(15, 5))
    
    datasets = ['plantnet-trunc', 'inaturalist-trunc']
    methods = ['classwise', 'standard', 'fuzzy-RErarity-0.0001']
    colors = ['tab:green', 'tab:green', 'tab:green']
    
    method_to_name = {'standard': 'Standard', 
                      'classwise': 'Classwise', 
                      'fuzzy-RErarity-0.0001': 'Fuzzy'}
    
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
        
        # Sort classes by class cond acc of Standard CP
        idx = np.argsort(res['standard']['coverage_metrics']['raw_class_coverages'])[::-1]
        
        for col, (method, base_color) in enumerate(zip(methods, colors)):
            ax = axes[row, col]
            
            # Get the lines for the specific method
            up_line = res[method]['class-cond-decision-accuracy'][idx]
            lower_line = res[method]['coverage_metrics']['raw_class_coverages'][idx] 
            
            # Define the five lines and their corresponding labels
            lines_data = [
                (lower_line, '$\\gamma_{\\mathrm{exp.}}=100\\%$'),
                (0.25*up_line + 0.75*lower_line, '$\\gamma_{\\mathrm{exp.}}=75\\%$'),
                (0.5*up_line + 0.5*lower_line, '$\\gamma_{\\mathrm{exp.}}=50\\%$'),
                (0.75*up_line + 0.25*lower_line, '$\\gamma_{\\mathrm{exp.}}=25\\%$'),
                (up_line, '$\\gamma_{\\mathrm{exp.}}=0\\%$'),
            ]
            
            # Create colormap based on the base color - using green for all
            colormap = plt.cm.Greens
            
            # Generate colors from the colormap
            colors_grad = [colormap(0.8 - 0.15*i) for i in range(5)]
            
            # Plot each line with gradient colors
            for i, (line_data, label) in enumerate(lines_data):
                zorder = 5 - i
                ax.plot(line_data, color=colors_grad[i], 
                        linewidth=2.0,
                        zorder=zorder,
                        label=label)
            
            ax.set_xlim(0, num_classes-1)
            ax.set_title(f'{dataset_names[dataset_name]} - {method_to_name[method]}')
            
            # Set y-label only for first column
            if col == 0:
                ax.set_ylabel('Decision accuracy')
            
            # Set x-label only for second row
            if row == 1:
                ax.set_xlabel('Class (sorted by $\\hat{c}_y$ of Stand. CP)')
            
            # Add legend only for plantnet-trunc Standard (row=0, col=0)
            if row == 0 and col == 0:
                legend = fig.legend(loc='center right', bbox_to_anchor=(1.08, 0.5), fontsize=10, title='Expert proportion', ncol=1)
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
