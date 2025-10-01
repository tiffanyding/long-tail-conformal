import sys
import os

# Add the parent directory to Python path to find train_models and utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from train_models.train import PlantNet, truncate_and_resplit_dataset, get_datasets
from utils.experiment_utils import get_inputs_folder
import seaborn as sns

sns.set_context("paper")

# Set up cache folder to avoid issues with get_datasets
import train_models.train as train_module

train_module.cache_folder = get_inputs_folder()
if not train_module.cache_folder.endswith("/"):
    train_module.cache_folder += "/"


# Setup plotting parameters
fig_folder = "../figs"
import os

os.makedirs(fig_folder, exist_ok=True)  # Create figs directory if it doesn't exist

plt.rcParams.update(
    {
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "legend.fontsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "text.usetex": True,
        "font.family": "serif",
        "text.latex.preamble": r"\usepackage{amsmath}",
    }
)


# Dataset configuration
dataset_names = {
    "plantnet": "Pl@ntNet-300K",
    "plantnet-trunc": "Pl@ntNet-300K-truncated",
    "inaturalist": "iNaturalist-2018",
    "inaturalist-trunc": "iNaturalist-2018-truncated",
}


def plot_class_distributions(
    train_labels,
    val_labels,
    test_labels,
    title=None,
    save_to=None,
    ax=None,
    show_legend=True,
):
    """Plot class distributions for train/val/test splits."""
    train_ctr = dict(Counter(train_labels))
    val_ctr = dict(Counter(val_labels))
    test_ctr = dict(Counter(test_labels))

    # Get counts for all classes (assume train has all classes)
    classes = train_ctr.keys()
    train_counts = np.array([train_ctr[k] for k in classes])
    val_counts = np.array([val_ctr.get(k, 0) for k in classes])
    test_counts = np.array([test_ctr.get(k, 0) for k in classes])
    total_counts = train_counts + val_counts + test_counts

    # Sort by frequency
    num_classes = len(classes)
    train_sorted = np.sort(train_counts)[::-1]
    val_sorted = np.sort(val_counts)[::-1]
    test_sorted = np.sort(test_counts)[::-1]
    total_sorted = np.sort(total_counts)[::-1]

    # Create plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 2.2))

    ax.plot(
        np.arange(num_classes),
        train_sorted,
        label=r"\texttt{train}",
        color="dodgerblue",
        linewidth=5,
    )
    ax.plot(
        np.arange(num_classes),
        val_sorted,
        label=r"\texttt{val}",
        color="tab:green",
        linewidth=5,
    )
    ax.plot(
        np.arange(num_classes),
        test_sorted,
        label=r"\texttt{test}",
        color="tab:pink",
        linewidth=5,
        linestyle="dotted",
    )
    ax.plot(
        np.arange(num_classes),
        total_sorted,
        label=r"\texttt{All}",
        color="black",
        linewidth=5,
    )

    # Format plot
    ax.set_yscale("log")
    ax.set_xlim(0, num_classes)
    ax.set_xlabel("Class")
    ax.set_ylabel(r"$\#$ of examples")
    ax.set_title(title)

    # Style
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_major_locator(plt.LogLocator(base=10.0, numticks=10))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(True, which="major", axis="y", linestyle="--", linewidth=0.5)

    if show_legend:
        ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    if save_to is not None:
        plt.savefig(save_to)
        print(f"Saved plot to {save_to}")

    return train_counts, val_counts, test_counts, total_counts


# Load Pl@ntNet-300K datasets
_, _, _, pn_train_labels, pn_val_labels, pn_test_labels = get_datasets(
    "plantnet", return_labels=True
)
_, _, _, pn_train_labels2, pn_val_labels2, pn_test_labels2 = get_datasets(
    "plantnet", return_labels=True, truncate=True
)
# Load iNaturalist datasets
_, _, _, inat_train_labels, inat_val_labels, inat_test_labels = get_datasets(
    "inaturalist", return_labels=True
)
_, _, _, inat_train_labels2, inat_val_labels2, inat_test_labels2 = get_datasets(
    "inaturalist", return_labels=True, truncate=True
)


# Create combined plot
fig, axes = plt.subplots(1, 4, figsize=(16, 3.2), sharey=True)

# Plot all four datasets
datasets = [
    (pn_train_labels, pn_val_labels, pn_test_labels, "plantnet"),
    (pn_train_labels2, pn_val_labels2, pn_test_labels2, "plantnet-trunc"),
    (inat_train_labels, inat_val_labels, inat_test_labels, "inaturalist"),
    (inat_train_labels2, inat_val_labels2, inat_test_labels2, "inaturalist-trunc"),
]

for i, (train_lab, val_lab, test_lab, name) in enumerate(datasets):
    plot_class_distributions(
        train_lab,
        val_lab,
        test_lab,
        ax=axes[i],
        title=dataset_names[name],
        show_legend=False,
    )
    if i > 0:
        axes[i].set_ylabel(None)

# Add legend to last subplot
handles, labels = axes[0].get_legend_handles_labels()
axes[3].legend(
    handles,
    labels,
    loc="lower center",
    ncol=1,
    handlelength=3,
    bbox_to_anchor=(1.27, 0.5),
    fontsize=10,
    frameon=False,
)

plt.tight_layout(pad=1.2, w_pad=0.3, rect=[0, 0.05, 1, 1])
plt.savefig(f"{fig_folder}/all_class_distributions.pdf", bbox_inches="tight")
plt.show()
