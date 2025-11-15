# %%
import sys

sys.path.append("../")  # For relative imports

import os
import pickle
import matplotlib.pyplot as plt

from utils.conformal_utils import *
from utils.experiment_utils import (
    get_inputs_folder,
    get_outputs_folder,
    get_figs_folder,
)

import seaborn as sns

sns.set_context("paper")

# Set up cache folder to avoid issues with get_datasets
import train_models.train as train_module

train_module.cache_folder = get_inputs_folder()
if not train_module.cache_folder.endswith("/"):
    train_module.cache_folder += "/"

from scipy.ndimage import uniform_filter


plt.rcParams.update(
    {
        "font.size": 16,  # base font size
        "axes.titlesize": 18,  # subplot titles
        "axes.labelsize": 16,  # x/y labels
        "legend.fontsize": 16,  # legend text
        "xtick.labelsize": 22,  # tick labels
        "ytick.labelsize": 22,
    }
)
# use tex with matplotlib
plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

dataset_names = {
    "plantnet": "Pl@ntNet-300K",
    "plantnet-trunc": "Pl@ntNet-300K-truncated",
    "inaturalist": "iNaturalist-2018",
    "inaturalist-trunc": "iNaturalist-2018-truncated",
}

methods = ["standard", "classwise", "prevalence-adjusted"]

alphas = [0.2, 0.1, 0.05, 0.01]
score = "softmax"


# Load in paths from folders.json
inputs_folder = get_inputs_folder()
# Set default folders for backward compatibility with existing code
results_folder = get_outputs_folder()
fig_folder = get_figs_folder()

for dataset in dataset_names:
    os.makedirs(f"{fig_folder}/{dataset}", exist_ok=True)
    print(f"{fig_folder}/{dataset}")


def load_metrics(dataset, alpha, method_name, score="softmax"):
    with open(
        f"{results_folder}/{dataset}_{score}_alpha={alpha}_{method_name}.pkl", "rb"
    ) as f:
        metrics = pickle.load(f)
    # Extract set size quantiles for easy access later
    metrics["set_size_metrics"]["median"] = metrics["set_size_metrics"][
        "[.25, .5, .75, .9] quantiles"
    ][1]
    metrics["set_size_metrics"]["quantile90"] = metrics["set_size_metrics"][
        "[.25, .5, .75, .9] quantiles"
    ][3]
    return metrics


# %%
def compute_class_cond_decision_accuracy(labels, is_covered, raw_set_sizes):
    # (assuming a random decision maker)
    num_classes = np.max(labels) + 1
    decision_acc = np.zeros((num_classes,))
    for k in range(num_classes):
        idx = labels == k
        # P(choose correct label) = 0 if label not in set
        # P(choose correct label) = 1/(set size) if label in set
        p_correct = is_covered[idx] * (1 / raw_set_sizes[idx])
        p_correct[np.isnan(p_correct)] = (
            0  # nans are due to empty sets, so replace with 0
        )
        decision_acc[k] = np.mean(p_correct)
        # if np.isnan(decision_acc[k]):
        #     pdb.set_trace()

    return decision_acc


def compute_class_cond_decision_accuracy_for_method(res, method, labels):
    is_covered = res[method]["coverage_metrics"]["is_covered"]
    raw_set_sizes = res[method]["coverage_metrics"]["raw_set_sizes"]

    return compute_class_cond_decision_accuracy(labels, is_covered, raw_set_sizes)


# %%


datasets = ["plantnet", "inaturalist", "plantnet-trunc", "inaturalist-trunc"]
# datasets = ["plantnet-trunc", "inaturalist-trunc"]
methods = ["classwise", "standard", "prevalence-adjusted"]
gamma_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
method_colors = {
    "standard": "blue",
    "classwise": "red",
    "clustered": "purple",
    "prevalence-adjusted": "orange",
}
method_style = {
    "standard": "-.",
    "classwise": ":",
    "clustered": "x",
    "prevalence-adjusted": "-",
}
method_to_name = {
    "standard": "Standard",
    "classwise": "Classwise",
    "clustered": "Clustered",
    "prevalence-adjusted": "Standard w. PAS",
}
fig_folder = get_figs_folder()
for dataset_name in datasets:
    print(dataset_name)
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)

    test_labels = np.load(f"{inputs_folder}/best-{dataset_name}-model_test_labels.npy")

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
            idx = np.argsort(res[method]["coverage_metrics"]["raw_class_coverages"])[
                ::-1
            ]
            up_line_raw = res[method]["class-cond-decision-accuracy"][idx]
            lower_line_raw = res[method]["coverage_metrics"]["raw_class_coverages"][idx]
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
                linestyle=method_style[method],
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
    fig_path = f"{fig_folder}/methods_comparison_{dataset_name}.pdf"
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
            linestyle=method_style[method],
            label=method_to_name[method],
        )
    )
    labels.append(method_to_name[method])
legend_fig = plt.figure(figsize=(12, 2))
legend_fig.patch.set_visible(False)
legend = legend_fig.legend(
    handles, labels, loc="center", fontsize=25, ncol=4, frameon=True
)
legend_fig.gca().set_axis_off()
legend_path = f"{fig_folder}/methods_comparison_LEGEND_ONLY.pdf"
legend_fig.savefig(legend_path, bbox_inches="tight", transparent=True)
print("Saved legend only to", legend_path)
plt.close(legend_fig)
