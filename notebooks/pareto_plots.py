# %% [markdown]
# This notebook generates the Pareto plots in the paper using the saved results obtained by running `run_get_results.sh`

# %%
import sys; sys.path.append("../") # For relative imports

import glob
import os
import pandas as pd
import pickle

from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.ticker import LogFormatter, LogLocator, MultipleLocator

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
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{xcolor}'

dataset_names = {
    "plantnet": "Pl@ntNet-300K",
    # "plantnet-trunc": "Pl@ntNet-300K (truncated)",
    "inaturalist": "iNaturalist-2018",
    # "inaturalist-trunc": "iNaturalist-2018 (truncated)",
}

def format_legend_label(method_label, alpha, label_prefix='', total_width=22):
    """
    Format legend label with perfect alignment using LaTeX makebox.
    
    Args:
        method_label (str): Method name (e.g., 'Standard', 'Standard w. PAS')
        alpha (float): Alpha value
        label_prefix (str): Optional prefix
        total_width (int): Fixed width in ex units for alignment
    
    Returns:
        str: Formatted label with perfect alignment
    """
    if label_prefix:
        # If there's a prefix, include it before the alpha
        alpha_part = f'{label_prefix}, $\\alpha={alpha}$'
    else:
        alpha_part = f'($\\alpha={alpha}$)'
    
    # Use LaTeX makebox for bulletproof fixed-width alignment
    return f'\\makebox[{total_width}ex][l]{{{method_label}}}{alpha_part}'

# %%
# Default paths (can be overridden in generate_all_pareto_plots)
inputs_folder = get_inputs_folder()
# Set default folders for backward compatibility with existing code
results_folder = get_outputs_folder()
fig_folder = get_figs_folder()

# %%
def compute_train_weighted_average_set_size(dataset, metrics, train_class_distr, test_labels):
    num_classes = np.max(test_labels) + 1
    
    # Get average set size by class
    set_sizes = metrics['coverage_metrics']['raw_set_sizes']
    avg_size_by_class = np.array([np.mean(set_sizes[test_labels == k]) for k in range(num_classes)])

    return np.sum(train_class_distr * avg_size_by_class)

def load_one_result(dataset, alpha, method_name, score='softmax',
                train_class_distr=None, test_labels=None, results_folder=None):
    
    if results_folder is None:
        results_folder = get_outputs_folder()
    
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

def load_all_results(dataset, alphas, methods, score='softmax', results_folder=None):
    if results_folder is None:
        results_folder = get_outputs_folder()
        
    # For truncated datasets, we need to load these in to compute train-weighted average set size
    if dataset.endswith('-trunc'): 
        train_labels_path = f'{inputs_folder}/{dataset}_train_labels.npy'
        train_labels = np.load(train_labels_path)
        num_classes = np.max(train_labels) + 1
        train_class_distr = np.array([np.sum(train_labels == k) for k in range(num_classes)]) / len(train_labels) 

        test_labels = test_labels = np.load(f'{inputs_folder}/best-{dataset}-model_test_labels.npy')
        
    all_res = {}
    for alpha in alphas:
        res = {}
        for method in methods:
            if dataset.endswith('-trunc'): # Compute train-weighted average set size
                res[method] = load_one_result(dataset, alpha, method, score=score,
                                           train_class_distr=train_class_distr, test_labels=test_labels, 
                                           results_folder=results_folder)
            else:
                res[method] = load_one_result(dataset, alpha, method, score=score, results_folder=results_folder)
        all_res[f'alpha={alpha}'] = res

    return all_res

def plot_set_size_vs_cov_metric(
    all_res,
    coverage_metric,
    alphas=None,
    set_size_metric='mean',
    ax=None,
    show_legend=False,
    markersizes=None,
    label_prefix='',
    add_inset=True,
    inset_loc='upper right',
    inset_lims=(0.1, 0.3, 1, 15),
    inset_width="50%",
    inset_pad=0,
    dataset=None
):
    # --- prepare main axis ---
    if ax is None:
        fig, ax = plt.subplots(figsize=(3.9, 2.3))

    # --- determine alphas ---
    if alphas is None:
        alphas = sorted(
            float(k.split('=')[1]) for k in all_res if k.startswith('alpha=')
        )

    # --- markersizes ---
    n = len(alphas)
    if markersizes is None:
        base_markersizes = [5, 6, 7, 8]

    else:
        base_markersizes = markersizes

    # Function to get marker size for a specific method
    def get_marker_size(base_size, method_key):
        if method_key == 'cvx':
            return base_size - 2
        else:
            return base_size

    # --- fixed definitions ---
    cw_weights = [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975, 0.99 , 0.999, 1]
    rarity_bandwidths = [1e-30, 1e-15, 1e-10, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 10, 1000]
    random_bandwidths = [1e-30, 1e-15, 1e-10, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 10, 1000]
    quantile_bandwidths = [1e-30, 1e-15, 1e-10, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 10, 1000]
    core_methods = [
        ('standard',             'Standard',       'blue',      'X'),
        ('classwise',            'Classwise',      'red',       'X'),
        ('classwise-exact',      'Exact Classwise','magenta',  'x'),
        ('clustered',            'Clustered',      'purple', 'P'),
        ('prevalence-adjusted',  'Standard w. PAS','orange',    '^'),
        ('standard-softmax',     'Standard w. softmax', 'blue',  'x'),
    ]

    # --- plot main curves ---
    for alpha, base_ms in zip(alphas, base_markersizes):
        res = all_res.get(f'alpha={alpha}', {})

        # core scatter methods (only if present)
        for key, label, color, mk in core_methods:
            if key in res:
                # Get method-specific marker size
                ms = get_marker_size(base_ms, key)
                # Set higher zorder for prevalence-adjusted to put it on top
                plot_zorder = 20 if key == 'prevalence-adjusted' else 10
                
                # Format label with perfect alignment
                formatted_label = format_legend_label(label, alpha, label_prefix)
                
                ax.plot(
                    res[key]['coverage_metrics'][coverage_metric],
                    res[key]['set_size_metrics'][set_size_metric],
                    linestyle='', marker=mk, color=color,
                    markersize=ms, alpha=0.8, markeredgewidth=0,
                    label=formatted_label,
                    zorder=plot_zorder
                )

        # convex interpolation (only weights that exist)
        valid_w = [w for w in cw_weights if f'cvx-cw_weight={w}' in res]
        if valid_w:
            ms = get_marker_size(base_ms, 'cvx')  # Regular size for cvx methods
            x_cvx = []
            y_cvx = []
            
            for w in valid_w:
                # For tau=1 (cw_weight=1), use classwise results instead due to numerical issues
                # if w == 1 and 'classwise' in res:
                #     x_val = res['classwise']['coverage_metrics'][coverage_metric]
                #     y_val = res['classwise']['set_size_metrics'][set_size_metric]
                # else:
                x_val = res[f'cvx-cw_weight={w}']['coverage_metrics'][coverage_metric]
                y_val = res[f'cvx-cw_weight={w}']['set_size_metrics'][set_size_metric]
                
                x_cvx.append(x_val)
                y_cvx.append(y_val)
            
            # Format label with perfect alignment
            formatted_label = format_legend_label('Interp-Q', alpha, label_prefix)
            
            # Test fixed transparency values: 1, 0.8, 0.6, 0.4
            # Map alpha values to fixed transparency levels
            alpha_to_transparency = {
                0.01: 0.25,   # Most conservative -> lowest opacity
                0.05: 0.5,   # 
                0.1: 0.75,    # 
                0.2: 1.0     # Least conservative -> highest opacity
            }
            alpha_transparency = alpha_to_transparency.get(alpha, 0.8)  # Default to 0.8 if alpha not found
            
            ax.plot(
                x_cvx, y_cvx,
                '-o', color='dodgerblue', markersize=ms, markeredgewidth=0,
                alpha=alpha_transparency,
                label=formatted_label
            )

        # fuzzy rarity projection 
        for tag, mk_sym, alpha_val, label in [
            ## Uncomment to use short name for Fuzzy
            ('fuzzy-rarity',   'D', 0.3, 'Raw Fuzzy'),
            ('fuzzy-RErarity', 'v', 0.5, 'Fuzzy'),
            # # Uncomment to use long name for Fuzzy
            # ('fuzzy-rarity',   'o', 0.3, 'Raw Fuzzy-$\\Pi_{\\mathrm{prevalence}}$'),
            # ('fuzzy-RErarity', '^', 0.5, 'Fuzzy-$\\Pi_{\\mathrm{prevalence}}$'),
        ]:
            valid_b = [b for b in rarity_bandwidths if f'{tag}-{b}' in res]
            if valid_b:
                ms = get_marker_size(base_ms, tag)  # Regular size for fuzzy methods
                xs = [res[f'{tag}-{b}']['coverage_metrics'][coverage_metric] for b in valid_b]
                ys = [res[f'{tag}-{b}']['set_size_metrics'][set_size_metric]    for b in valid_b]
                
                # Format label with perfect alignment
                formatted_label = format_legend_label(label, alpha, label_prefix)
                
                ax.plot(
                    xs, ys,
                    '-' + mk_sym, color='salmon', markersize=ms, markeredgewidth=0,
                    alpha=alpha_val,
                    label=formatted_label
                )

        # fuzzy random projection 
        for tag, mk_sym, alpha_val, label in [
            ('fuzzy-random',   'o', 0.3, 'Raw Fuzzy-$\\Pi_{\\mathrm{random}}$'),
            ('fuzzy-RErandom', '^', 0.5, 'Fuzzy-$\\Pi_{\\mathrm{random}}$'),
        ]:
            valid_b = [b for b in rarity_bandwidths if f'{tag}-{b}' in res]
            if valid_b:
                ms = get_marker_size(base_ms, tag)  # Regular size for fuzzy methods
                xs = [res[f'{tag}-{b}']['coverage_metrics'][coverage_metric] for b in valid_b]
                ys = [res[f'{tag}-{b}']['set_size_metrics'][set_size_metric]    for b in valid_b]
                
                # Format label with perfect alignment
                formatted_label = format_legend_label(label, alpha, label_prefix)
                
                ax.plot(
                    xs, ys,
                    '-' + mk_sym, color='tab:purple', markersize=ms,
                    alpha=alpha_val,
                    label=formatted_label
                )
                
        # fuzzy quantile projection 
        for tag, mk_sym, alpha_val, label in [
            ('fuzzy-quantile',   'o', 0.3, 'Raw Fuzzy-$\\Pi_{\\mathrm{quantile}}$'),
            ('fuzzy-REquantile', '^', 0.5, 'Fuzzy-$\\Pi_{\\mathrm{quantile}}$'),
        ]:
            valid_b = [b for b in rarity_bandwidths if f'{tag}-{b}' in res]
            if valid_b:
                ms = get_marker_size(base_ms, tag)  # Regular size for fuzzy methods
                xs = [res[f'{tag}-{b}']['coverage_metrics'][coverage_metric] for b in valid_b]
                ys = [res[f'{tag}-{b}']['set_size_metrics'][set_size_metric]    for b in valid_b]
                
                # Format label with perfect alignment
                formatted_label = format_legend_label(label, alpha, label_prefix)
                
                ax.plot(
                    xs, ys,
                    '-' + mk_sym, color='gold', markersize=ms, markeredgewidth=0,
                    alpha=alpha_val,
                    label=formatted_label
                )


    # --- style main axis ---
    ax.set_yscale('log')
    
    # Set up y-axis ticks with both major and minor ticks
    # Major ticks at powers of 10
    major_locator = LogLocator(base=10.0, numticks=12)
    ax.yaxis.set_major_locator(major_locator)
    
    # Minor ticks at intermediate values
    minor_locator = LogLocator(base=10.0, subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), numticks=12)
    ax.yaxis.set_minor_locator(minor_locator)
    
    # Set custom y-axis tick labels to show numeric values instead of powers of 10
    class CustomLogFormatter(LogFormatter):
        def __call__(self, x, pos=None):
            # Return the numeric value instead of scientific notation
            if x >= 1:
                return f'{int(x):,}'
            else:
                return f'{x:.3f}'
    
    ax.yaxis.set_major_formatter(CustomLogFormatter())

    ax.spines[['right', 'top']].set_visible(False)

    # --- inset ---
    if add_inset:
        xmin, xmax, ymin, ymax = inset_lims
        axins = inset_axes(
            ax, width=inset_width, height="40%", loc=inset_loc, borderpad=inset_pad
        )
        axins.patch.set_alpha(0.7)

        # replot all alphas for all methods, with the same presence checks
        for alpha, base_ms in zip(alphas, 0.5 * np.array(base_markersizes)):
            res = all_res.get(f'alpha={alpha}', {})

            # core
            for key, _, color, mk in core_methods:
                if key in res:
                    # Get method-specific marker size for inset
                    ms = get_marker_size(base_ms, key)
                    # Set higher zorder for prevalence-adjusted to put it on top
                    plot_zorder = 20 if key == 'prevalence-adjusted' else 10
                    axins.plot(
                        res[key]['coverage_metrics'][coverage_metric],
                        res[key]['set_size_metrics'][set_size_metric],
                        linestyle='', marker=mk, color=color,
                        markersize=ms, alpha=0.8, zorder=plot_zorder,markeredgewidth=0,
                    )

            # convex
            valid_w = [w for w in cw_weights if f'cvx-cw_weight={w}' in res]
            if valid_w:
                ms = get_marker_size(base_ms, 'cvx')
                x_cvx_inset = []
                y_cvx_inset = []
                
                for w in valid_w:
                    # For tau=1 (cw_weight=1), use classwise results instead due to numerical issues
                    if w == 1 and 'classwise' in res:
                        x_val = res['classwise']['coverage_metrics'][coverage_metric]
                        y_val = res['classwise']['set_size_metrics'][set_size_metric]
                    else:
                        x_val = res[f'cvx-cw_weight={w}']['coverage_metrics'][coverage_metric]
                        y_val = res[f'cvx-cw_weight={w}']['set_size_metrics'][set_size_metric]
                    
                    x_cvx_inset.append(x_val)
                    y_cvx_inset.append(y_val)
                
                axins.plot(x_cvx_inset, y_cvx_inset, '-o', color='dodgerblue', markersize=ms, markeredgewidth=0, alpha=0.5)

            # fuzzy
            for tag, mk_sym, alpha_val in [('fuzzy-rarity','o',0.3),('fuzzy-RErarity','^',0.5)]:
                valid_b = [b for b in rarity_bandwidths if f'{tag}-{b}' in res]
                if valid_b:
                    ms = get_marker_size(base_ms, tag)
                    axins.plot(
                        [res[f'{tag}-{b}']['coverage_metrics'][coverage_metric] for b in valid_b],
                        [res[f'{tag}-{b}']['set_size_metrics'][set_size_metric]    for b in valid_b],
                        '-' + mk_sym, color='salmon', markersize=ms, alpha=alpha_val,markeredgewidth=0,
                    )

        # zoom & ticks
        axins.set_xlim(xmin, xmax)
        axins.set_ylim(ymin, ymax)
        
        # Customize y-tick positions and labels based on dataset
        if dataset and 'inaturalist' in dataset.lower():
            # For iNaturalist, only show ticks and labels at multiples of 10
            yticks_with_labels = np.arange(int(np.ceil(ymin/20)*20), int(ymax) + 1, 20)
            axins.set_yticks(yticks_with_labels)
            axins.set_yticklabels([str(int(y)) for y in yticks_with_labels])
        else:
            # Default behavior: show ticks at all positions, labels at even numbers
            yticks = np.arange(int(np.ceil(ymin)), int(ymax) + 1)
            axins.set_yticks(yticks)
            axins.set_yticklabels([str(int(y)) if (int(y) % 2 == 0) else '' for y in yticks])
            
        for spine in axins.spines.values():
            spine.set_edgecolor('grey')
        axins.tick_params(axis='both', colors='grey', labelsize=8, pad=1)

        if inset_loc == 'upper left':
            mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec=(0.1,0.1,0.1,0.2))
        else:
            mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec=(0.1,0.1,0.1,0.2))

    # --- legend & layout ---
    if show_legend:
        ax.legend(fontsize=6, bbox_to_anchor=(1, 1))
    # Use consistent layout - no tight_layout to avoid width changes

# Specify where to zoom in 
def get_inset_lims(dataset, coverage_metric):
    xlims = {'plantnet': {'cov_below50': (0.1, 0.3), 
                          'undercov_gap': (0.1, 0.38),
                          'macro_cov': (0.6, 0.9)},
             'plantnet-trunc': {'cov_below50': (0.01, 0.17), 
                                'undercov_gap': (0.02, 0.2),
                                'macro_cov': (0.77, 0.87)},
             'inaturalist': {'cov_below50': (0.005, 0.04), 
                            'undercov_gap': (0.01, 0.1),
                            'macro_cov': (0.85, 0.95)},
             'inaturalist-trunc': {'cov_below50': (0.01, 0.07), 
                                   'undercov_gap': (0.03, 0.11),
                                   'macro_cov': (0.75, 0.92)}
            }
    ylims = {'plantnet': (0.9, 8), 
             'plantnet-trunc': (0.9, 5),
             'inaturalist': (5, 90),
             'inaturalist-trunc': (1.1, 13)}

    return *xlims[dataset][coverage_metric], *ylims[dataset]


def generate_all_pareto_plots(dataset, score, alphas, methods, show_grid=False,
                              save_suffix='', show_inset=None, legendfontsize=14, use_focal_loss=False):
    """
    Generate comprehensive Pareto plots for conformal prediction methods.
    
    Parameters:
    -----------
    dataset : str
        Dataset name ('plantnet', 'inaturalist', etc.)
    score : str  
        Score type ('softmax', 'PAS', etc.)
    alphas : list
        List of alpha values for coverage
    methods : list
        List of method names to include
    show_grid : bool, default=False
        Whether to show grid on plots
    save_suffix : str, default=''
        Additional suffix for filename
    show_inset : bool, default=None
        Whether to show inset zoom plots
    legendfontsize : int, default=14
        Font size for legend
    use_focal_loss : bool, default=False
        If True, uses focal_loss subfolder for results and adds '_Focal_Loss' to filenames.
        Also adds ' (Focal Loss)' to plot title.
        If False, uses standard results folder and adds ' (Cross-Entropy)' to title.
    """
    
    # Set up folder paths based on focal_loss option
    if use_focal_loss:
        results_folder = get_outputs_folder() + '/focal_loss'
        fig_folder = get_figs_folder() + '/focal_loss'
        loss_suffix = '_Focal_Loss'
    else:
        results_folder = get_outputs_folder()
        fig_folder = get_figs_folder()
        loss_suffix = ''
    
    os.makedirs(fig_folder, exist_ok=True)
    
    # Load pre-computed metrics
    all_res = load_all_results(dataset, alphas, methods, score=score, results_folder=results_folder)

    if score == 'PAS':
        std_with_softmax = load_all_results(dataset, alphas, ['standard'], score='softmax', results_folder=results_folder)
        for alpha in alphas:
            all_res[f'{alpha=}']['standard-softmax'] = std_with_softmax[f'{alpha=}']['standard']

    # Make plots
    set_size_metric = 'train_mean' if dataset.endswith('-trunc') else 'mean'
    print(f'{set_size_metric=}')

    cov_metrics = ['cov_below50', 'undercov_gap', 'macro_cov',
               'train_marginal_cov']
    cov_metric_names = ['FracBelow50$\\%$', 
                       'UnderCovGap',
                        'MacroCov',
                       'MarginalCov']
    fig, axes = plt.subplots(1, len(cov_metrics), figsize=(13, 3), sharey=True)
    axes = axes.flatten()

    for i, (coverage_metric, xlabel) in enumerate(zip(cov_metrics, cov_metric_names)):
        ax = axes[i]

        if (not show_inset) or (coverage_metric == 'train_marginal_cov') or (score == 'PAS'):
            add_inset = False
        else:
            add_inset = True 
   
        if add_inset:
            inset_lims = get_inset_lims(dataset, coverage_metric)
        else:
            inset_lims = None
    
        if coverage_metric == 'macro_cov':
            inset_loc = 'upper left'
            inset_pad = 1
        else:
            inset_loc = 'upper right'
            inset_pad = 1
    
        if coverage_metric == 'train_marginal_cov':
            for a in alphas:
                ax.axvline(1-a, linestyle='--', color='grey')
                
        plot_set_size_vs_cov_metric(
            all_res,
            coverage_metric=coverage_metric,
            set_size_metric=set_size_metric,
            alphas=alphas,
            ax=ax,
            add_inset=add_inset,
            inset_loc=inset_loc,
            inset_lims=inset_lims,
            inset_width="50%",
            inset_pad=inset_pad,
            show_legend=False,
            dataset=dataset
        )
        
        ax.set_xlabel(xlabel)

    # Optionally plot transparent grid
    if show_grid:
        for ax in axes:
            ax.grid(True, alpha=0.2)

    if len(alphas) == 1:
        axes[3].set_xlim(1-alphas[0]-.05, 1-alphas[0]+.05)
    
    axes[0].set_ylabel('Average set size')
    
    # Add loss type to title
    loss_type = " (Focal Loss)" if use_focal_loss else " (Cross-Entropy)"
    plt.suptitle(dataset_names[dataset] + loss_type, y=0.9)  # Reduced from 0.98 to minimize gap
    
    # Use tight layout for clean, properly sized plots
    plt.tight_layout()
    
    # Add INSET to filename if show_inset is True
    inset_suffix = '_INSET' if show_inset else ''
    
    fig_path = f'{fig_folder}/{dataset}/ALL_metrics_{dataset}_{score}{save_suffix}_pareto{inset_suffix}_NO_LEGEND_js{loss_suffix}.pdf'
    os.makedirs(f'{fig_folder}/{dataset}', exist_ok=True)
    
    # Save clean version without legend
    plt.savefig(fig_path, bbox_inches='tight')
    
    # Create legend for standalone legend generation (but don't save WITH_LEGEND version)
    if len(alphas) > 1:
        legend = axes[0].legend(ncols=len(alphas), loc='upper center', bbox_to_anchor=(2.0, -0.08), fontsize=8)
    else:
        legend = axes[0].legend(ncols=3, loc='upper center', bbox_to_anchor=(2.0, -0.08), fontsize=8)
    
    # Create standalone legend PDF
    legend_fig = plt.figure(figsize=(12, 2))  # Wide figure for horizontal legend
    legend_fig.patch.set_visible(False)  # Make figure background transparent
    
    # Get legend handles and labels from the main plot
    handles, labels = axes[0].get_legend_handles_labels()
    
    # Create standalone legend
    if len(alphas) > 1:
        legend_standalone = legend_fig.legend(handles, labels, ncol=len(alphas), 
                                            loc='center', fontsize=8, frameon=True)
    else:
        legend_standalone = legend_fig.legend(handles, labels, ncol=3, 
                                            loc='center', fontsize=8, frameon=True)
    
    # Remove axes from legend figure
    legend_fig.gca().set_axis_off()
    
    # Save standalone legend
    legend_path = fig_path.replace(f'NO_LEGEND_js{loss_suffix}.pdf', f'LEGEND_ONLY_js{loss_suffix}.pdf')
    legend_fig.savefig(legend_path, bbox_inches='tight', transparent=True)
    
    print(f'✅ Saved two versions:')
    print(f'   - Main plot: {fig_path}')
    print(f'   - Legend only: {legend_path}')
    
    # Close the legend figure to free memory
    plt.close(legend_fig)
    
    plt.show()

        


# %%
# %% [markdown]
# ## Example Usage of Focal Loss Option
# 
# The `generate_all_pareto_plots` function now supports a `use_focal_loss` parameter:
# 
# - `use_focal_loss=False` (default): Uses standard results from main output folder, title shows " (Cross-Entropy)"
# - `use_focal_loss=True`: Uses focal loss results from 'focal_loss' subfolder, adds '_Focal_Loss' suffix to filenames, and title shows " (Focal Loss)"

# %%
# Example: Generate plots with standard results (default behavior)
# Title will be: "Pl@ntNet-300K (Cross-Entropy)"
# Filename: ALL_metrics_plantnet_softmax_pareto_NO_LEGEND_js.pdf
# generate_all_pareto_plots('plantnet', 'softmax', [0.1], ['standard', 'classwise'])

# Example: Generate plots with focal loss results  
# Title will be: "Pl@ntNet-300K (Focal Loss)"
# Filename: ALL_metrics_plantnet_softmax_pareto_NO_LEGEND_js_Focal_Loss.pdf
# generate_all_pareto_plots('plantnet', 'softmax', [0.1], ['standard', 'classwise'], use_focal_loss=True)

## Main Pareto plots

### (a) Simple version for main paper
# %%
alphas = [0.2, 0.1, 0.05, 0.01]
score = 'softmax'

rarity_bandwidths = [1e-15, 1e-10, 1e-5, 0.0001, 0.001, 0.01, .1 , 10, 1000]
cw_weights = [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975, 0.99 , 0.999, 1] # CHANGED!
# cw_weights = 1 - np.array([0, .001, .01, .025, .05, .1, .2, .4, .8, 1])


methods = ['standard', 'classwise', 'clustered', 'prevalence-adjusted'] + \
            [f'cvx-cw_weight={w}' for w in cw_weights
            # [f'fuzzy-rarity-{bw}' for bw in rarity_bandwidths] +\
            # [f'fuzzy-RErarity-{bw}' for bw in rarity_bandwidths] +\
             ] 

for dataset in ['plantnet', 'inaturalist']:
    # Generate standard plots (using default softmax results)
    generate_all_pareto_plots(dataset, score, alphas, methods, save_suffix='', legendfontsize=14, show_grid=True, use_focal_loss=False, show_inset=False)
    
    # Generate focal loss plots (uncomment to use focal loss results)
    # generate_all_pareto_plots(dataset, score, alphas, methods, save_suffix='_alpha=0.1', legendfontsize=14, show_grid=True, use_focal_loss=True)

# %% [markdown]
# ### (b) Full version for Appendix

# %%
alphas = [0.2, 0.1, 0.05, 0.01]
score = 'softmax'

# rarity_bandwidths = [1e-15, 1e-10, 1e-5, 0.0001, 0.001, 0.01, .1 , 10, 1000]
# cw_weights = [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975, 0.99 , 0.999, 1] # CHANGED!
# cw_weights = 1 - np.array([0, .001, .01, .025, .05, .1, .2, .4, .8, 1])

# methods = ['standard', 'classwise', 'clustered', 'prevalence-adjusted'] + \
#             [f'cvx-cw_weight={w}' for w in cw_weights] 

# APPENDIX:
methods = ['standard', 'classwise', 'classwise-exact', 'clustered', 'prevalence-adjusted'] + \
            [f'cvx-cw_weight={w}' for w in cw_weights] +\
            [f'fuzzy-rarity-{bw}' for bw in rarity_bandwidths] +\
            [f'fuzzy-RErarity-{bw}' for bw in rarity_bandwidths] 

for dataset in dataset_names.keys():
# for dataset in ['plantnet-trunc']:
    # Generate standard plots (using default softmax results)
    generate_all_pareto_plots(dataset, score, alphas, methods, show_inset=True)
    
    # Generate focal loss plots (uncomment to use focal loss results) 
    # generate_all_pareto_plots(dataset, score, alphas, methods, show_inset=True, use_focal_loss=True)

# %%
# x = np.load("/home/tding/code/clean/long-tail-conformal/train_models/data/plantnet-trunc_train_labels.npy")
# x.max()

# %% [markdown]
# ## Generate csv's of metrics

# %%
# Extract metrics into csv

def extract_metric_table(all_res, coverage_metrics, set_size_metric='mean'):
    records = []
    for alpha_key, methods in all_res.items():
        alpha_val = float(alpha_key.split('=')[1])
        for method, result in methods.items():
            row = {'alpha': alpha_val, 'method': method}
            for cov_metric in coverage_metrics:
                cov_val = result['coverage_metrics'].get(cov_metric, None)
                row[f'{cov_metric}'] = cov_val
            size_val = result['set_size_metrics'].get(set_size_metric, None)
            row[f'set_size_{set_size_metric}'] = size_val
            records.append(row)
    return pd.DataFrame(records)


score = 'softmax'
alphas = [0.2, 0.1, 0.05, 0.01]

for dataset in dataset_names.keys():
    set_size_metric = 'train_mean' if dataset.endswith('-trunc') else 'mean'
    
    # Load the results
    all_res = load_all_results(dataset, alphas, methods, score=score, results_folder=results_folder)
    
    # Extract into DataFrame
    df = extract_metric_table(
        all_res,
        coverage_metrics=['cov_below50', 'undercov_gap', 'macro_cov', 'train_marginal_cov'],
        set_size_metric=set_size_metric
    )
    
    # Save full CSV
    pth = f'{fig_folder}/{dataset}/{dataset}_{score}_metrics.csv'
    df.to_csv(pth, index=False)
    print(f'Saved csv of metrics to {pth}')

    ## --- Clean csv and save that ---
    selected_alpha = 0.1
    selected_methods = {'standard': 'Standard',
                       'classwise': 'Classwise',
                       'clustered': 'Clustered',
                       'prevalence-adjusted': 'Standard w. PAS',
                       'cvx-cw_weight=0.999': 'Interp-Q ($\\tau=0.999$)',
                       'cvx-cw_weight=0.99': 'Interp-Q ($\\tau=0.99$)',
                       'cvx-cw_weight=0.9': 'Interp-Q ($\\tau=0.9$)',
                       'fuzzy-rarity-1e-05': 'Raw Fuzzy ($\\sigma=0.00001$)',
                       'fuzzy-rarity-0.01': 'Raw Fuzzy ($\\sigma=0.01$)',
                       'fuzzy-rarity-0.1': 'Raw Fuzzy ($\\sigma=0.1$)',
                       'fuzzy-RErarity-1e-05': 'Fuzzy ($\\sigma=0.00001$)',
                       'fuzzy-RErarity-0.01': 'Fuzzy ($\\sigma=0.01$)',
                       'fuzzy-RErarity-0.1': 'Fuzzy ($\\sigma=0.1$)'}
    # Filter and rename methods
    df = df[df['alpha'] == selected_alpha] # Select alpha = 0.1 only
    df = df[df["method"].isin(selected_methods.keys())].copy()
    df["method"] = df["method"].map(selected_methods)

    # Remove alpha column
    df = df.drop('alpha', axis=1)

    # Rename columns
    df.rename(columns={'method': 'Method', 
                       'cov_below50': 'FracBelow50% $\\downarrow$',
                       'undercov_gap': 'UnderCovGap $\\downarrow$', 
                       'macro_cov': 'MacroCov $\\uparrow$',
                       'train_marginal_cov': 'MarginalCov (desired = 0.9)', 
                       'set_size_train_mean': 'Avg. set size $\\downarrow$',
                       'set_size_mean': 'Avg. set size'}, inplace=True)

    
    # Round numeric values to 3 significant digits
    def round_sig(x, sig=3):
        if isinstance(x, (int, float, np.float64)):
            return float(f"{x:.{sig}g}")
        return x
    df = df.map(round_sig)

    # Save
    pth = f'{fig_folder}/{dataset}/{dataset}_{score}_alpha={selected_alpha}_metrics_CLEANED.csv'
    df.to_csv(pth, index=False)
    print(f'Saved cleaned csv of metrics for alpha={selected_alpha} to {pth}')


# %% [markdown]
# ## Appendix plots
# 
# ### Fuzzy and Interp-Q with PAS

# %%
score = 'softmax'
alphas = [0.1, 0.2, 0.05, 0.01]

# 'classwise-exact'
methods = ['standard', 'classwise', 'clustered'] + \
            [f'fuzzy-rarity-{bw}' for bw in rarity_bandwidths] +\
            [f'fuzzy-RErarity-{bw}' for bw in rarity_bandwidths] +\
            [f'cvx-cw_weight={w}' for w in cw_weights] 


for dataset in dataset_names.keys():
    # Generate standard PAS plots 
    generate_all_pareto_plots(dataset, score, alphas, methods, legendfontsize=15, show_inset=True)
    
    # Generate focal loss PAS plots (uncomment to use focal loss results)
    # generate_all_pareto_plots(dataset, score, alphas, methods, legendfontsize=15, use_focal_loss=True)

# %% [markdown]
# ### Fuzzy with Random and Quantile projections

# %%
bandwidths = rarity_bandwidths
score = 'softmax'
methods = ['standard', 'classwise'] + \
        [f'fuzzy-rarity-{bw}' for bw in bandwidths] + \
        [f'fuzzy-RErarity-{bw}' for bw in bandwidths] + \
        [f'fuzzy-random-{bw}' for bw in bandwidths] + \
        [f'fuzzy-RErandom-{bw}' for bw in bandwidths] + \
        [f'fuzzy-quantile-{bw}' for bw in bandwidths] + \
        [f'fuzzy-REquantile-{bw}' for bw in bandwidths] 

for dataset in dataset_names.keys():
    # Generate standard plots with extra projections
    generate_all_pareto_plots(dataset, score, alphas, methods, save_suffix='_extra_projections', show_inset=True)
    
    # Generate focal loss plots with extra projections (uncomment to use focal loss results)
    # generate_all_pareto_plots(dataset, score, alphas, methods, save_suffix='_extra_projections', show_inset=False, use_focal_loss=True)

# %%



