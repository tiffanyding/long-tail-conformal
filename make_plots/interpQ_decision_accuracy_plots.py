import numpy as np
import pdb
from decision_accuracy_plots import get_outputs_folder, get_figs_folder

# %%
# %%
import sys

sys.path.append("../")  # For relative imports

import glob
import os
import pickle

dataset_names = {
    "plantnet": "Pl@ntNet-300K",
    "plantnet-trunc": "Pl@ntNet-300K-truncated",
    "inaturalist": "iNaturalist-2018",
    "inaturalist-trunc": "iNaturalist-2018-truncated",
}

# Import shared functions from decision_accuracy_plots.py
from decision_accuracy_plots import (
    load_metrics,
    compute_class_cond_decision_accuracy_for_method,
)

# %%
methods = [
    f"cvx-cw_weight={tau}"
    for tau in [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975, 0.99, 0.999, 1]
]
print(methods)


alphas = [0.2, 0.1, 0.05, 0.01]
score = "softmax"


# Load in paths from folders.json
results_folder = get_outputs_folder()
fig_folder = get_figs_folder()


# %%
def create_methods_comparison_plots_separate():
    """
    Create two separate figures:
    1. plantnet-trunc (top row)
    2. inaturalist-trunc (bottom row)
    3. Legend only
    """
    import matplotlib.pyplot as plt
    from scipy.ndimage import uniform_filter

    datasets = ["plantnet-trunc", "inaturalist-trunc"]
    # methods = ['classwise', 'standard', 'prevalence-adjusted']
    gamma_levels = [0.0, 0.25, 0.5, 0.75, 1.0]

    # ------- FOR INTERP-Q SPECIFICALLY ------
    method_colors = {}
    method_to_name = {}
    M = len(methods)
    print(methods)
    for i, item in enumerate(methods):
        # Extract weight as float
        w = float(item.split("=")[1])
        method_colors[item] = (i / M, 0, 1 - i / M)
        method_to_name[item] = f"$\\tau=$ {w}"
    # --------------------------------------

    fig_folder = get_figs_folder()
    for dataset_name in datasets:
        fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
        test_labels_path = f"/home-warm/plantnet/conformal_cache/train_models/best-{dataset_name}-model_test_labels.npy"
        test_labels = np.load(test_labels_path)
        num_classes = np.max(test_labels) + 1
        res = {}
        for method in methods:
            res[method] = load_metrics(dataset_name, 0.1, method)
        for method in methods:
            dec_acc = compute_class_cond_decision_accuracy_for_method(
                res, method, test_labels
            )
            res[method]["class-cond-decision-accuracy"] = dec_acc
        for col, gamma in enumerate(gamma_levels):
            ax = axes[col]
            for method in methods:
                idx = np.argsort(
                    res[method]["coverage_metrics"]["raw_class_coverages"]
                )[::-1]
                up_line_raw = res[method]["class-cond-decision-accuracy"][idx]
                lower_line_raw = res[method]["coverage_metrics"]["raw_class_coverages"][
                    idx
                ]
                line_data = uniform_filter(
                    (1 - gamma) * up_line_raw + gamma * lower_line_raw,
                    size=20,
                    mode="nearest",
                )
                color = method_colors[method]
                ax.plot(
                    line_data,
                    color=color,
                    linewidth=4.0,
                    linestyle="-",
                    label=method_to_name[method],
                    solid_capstyle="round",
                )
            ax.set_xlim(0, num_classes - 1)
            ax.spines[["right", "top"]].set_visible(False)
            ax.set_title(
                f"$\\gamma_{{\\mathrm{{exp.}}}} = {int(gamma*100)}\\%$", fontsize=30
            )
            if col == 0:
                ax.set_ylabel("Decision accuracy", fontsize=25)
            ax.set_xlabel("Class", fontsize=25)
        fig.suptitle(dataset_names[dataset_name], y=0.95, fontsize=30)
        plt.tight_layout()
        # plt.tight_layout(rect=[0, 0, 1, 0.93])
        fig_path = f"{fig_folder}/interpQ_decision_accuracy_{dataset_name}.pdf"
        fig.savefig(fig_path, bbox_inches="tight")
        print(f"Saved plot for {dataset_name} to", fig_path)
        plt.close(fig)
    # --- Legend only ---
    handles, labels = [], []
    for method in methods:
        handles.append(
            plt.Line2D(
                [0],
                [0],
                color=method_colors[method],
                lw=5,
                label=method_to_name[method],
            )
        )
        labels.append(method_to_name[method])
    legend_fig = plt.figure(figsize=(12, 2))
    legend_fig.patch.set_visible(False)
    legend = legend_fig.legend(
        handles, labels, loc="center", fontsize=25, ncol=5, frameon=True
    )
    legend_fig.gca().set_axis_off()
    legend_path = f"{fig_folder}/interpQ_decision_accuracy_LEGEND_ONLY.pdf"
    legend_fig.savefig(legend_path, bbox_inches="tight", transparent=True)
    print("Saved legend only to", legend_path)
    plt.close(legend_fig)


# Create the separate plots
create_methods_comparison_plots_separate()
