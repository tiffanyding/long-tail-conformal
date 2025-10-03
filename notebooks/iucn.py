# %%
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import json
import sys
import os
from collections import Counter

# Add the parent directory to Python path to find train_models and utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_models.train import PlantNet
import seaborn as sns

sns.set_context("paper")


# %%
fig_folder = "figs"
# plt.rcParams["font.family"] = "Monospace"
plt.rcParams.update(
    {
        "font.size": 16,  # base font size
        "axes.titlesize": 18,  # subplot titles
        "axes.labelsize": 16,  # x/y labels
        "legend.fontsize": 16,  # legend text
        "xtick.labelsize": 16,  # tick labels
        "ytick.labelsize": 16,
    }
)
# use tex with matplotlib
plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

# %% [markdown]
# ## PlantNet


def load_metadata(data_dir=None):
    """Load PlantNet metadata and return DataFrame with species & IUCN status."""
    if data_dir is None:
        # Try multiple possible locations
        candidates = [
            os.environ.get("PLANTNET_DATA"),
            "./data",
            "../data",
            (
                os.path.join(os.path.dirname(__file__), "..", "data")
                if "__file__" in globals()
                else None
            ),
        ]
        candidates = [c for c in candidates if c is not None]

        data_dir = None
        for candidate in candidates:
            test_file = os.path.join(
                candidate, "plantnet300K_class_idx_to_species_id.json"
            )
            if os.path.exists(test_file):
                data_dir = candidate
                break

        if data_dir is None:
            raise FileNotFoundError(
                f"Could not find metadata files. Tried: {candidates}"
            )

    files = {
        "class_idx": os.path.join(
            data_dir, "plantnet300K_class_idx_to_species_id.json"
        ),
        "names": os.path.join(data_dir, "plantnet300K_species_id_2_name.json"),
        "iucn": os.path.join(data_dir, "plantnet300K_iucn_status_dict.json"),
    }

    # Load all JSON files
    data = {}
    for key, path in files.items():
        try:
            with open(path, "r") as f:
                data[key] = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Missing metadata file: {path}")

    # Build metadata DataFrame
    df = (
        pd.DataFrame.from_dict(
            data["class_idx"], orient="index", columns=["species_id"]
        )
        .reset_index()
        .rename(columns={"index": "class_id"})
    )
    df["class_id"] = df["class_id"].astype(int)
    df["species_name"] = df["species_id"].map(data["names"])
    df["iucn_status"] = df["species_name"].map(
        lambda n: data["iucn"].get(n, "Not Evaluated")
    )

    return df


# %%
# Load PlantNet datasets
root = "/home-warm/plantnet/plantnet_300K/images"
train_dataset = PlantNet(root, "train")
val_dataset = PlantNet(root, "val")
test_dataset = PlantNet(root, "test")

# Extract labels and compute counts
train_labels = np.array(train_dataset.labels)
val_labels = np.array(val_dataset.labels)
test_labels = np.array(test_dataset.labels)

num_classes = max(train_labels.max(), val_labels.max(), test_labels.max()) + 1
train_counts = np.bincount(train_labels, minlength=num_classes)
val_counts = np.bincount(val_labels, minlength=num_classes)
test_counts = np.bincount(test_labels, minlength=num_classes)

# %%
# Load metadata and create plotting dataframe
df_meta = load_metadata()  # Will auto-detect the correct path

# Add counts to metadata (ensure alignment)
df_meta = df_meta.sort_values("class_id").reset_index(drop=True)
n_classes = min(len(df_meta), len(train_counts))
df_meta = df_meta.iloc[:n_classes].copy()
df_meta["counts"] = train_counts[:n_classes]
df_meta["counts_val"] = val_counts[:n_classes]
df_meta["counts_test"] = test_counts[:n_classes]

# Sort by prevalence for plotting
df_to_plot = df_meta.sort_values("counts", ascending=False).reset_index(drop=True)


# %%
# Define global constants for threatened species identification
THREATENED_CODES = {"CR", "EN", "VU"}


def get_threatened_status(df):
    """Get boolean array indicating which species are threatened."""
    return df["iucn_status"].isin(THREATENED_CODES)


def get_threatened_indices(df):
    """Get array indices of threatened species."""
    return np.where(get_threatened_status(df))[0]


# Plot IUCN threatened vs non-threatened species
is_threatened = get_threatened_status(df_to_plot)

# fig, ax = plt.subplots(figsize=(10, 3))
# x_idx = np.arange(len(df_to_plot))
# counts = df_to_plot["counts"].values

# # Define alpha values for different IUCN statuses
# alphas = {
#     "LC": 0.1,
#     "LR/lc": 0.8,
#     "NT": 0.8,
#     "LR/nt": 0.8,
#     "DD": 0.1,
#     "VU": 0.8,
#     "LR/cd": 0.8,
#     "EN": 0.8,
#     "CR": 0.8,
#     "Not Evaluated": 0.1,
# }

# # Plot each species with alpha based on IUCN status
# for i, (x, count, status, threat_status) in enumerate(
#     zip(x_idx, counts, df_to_plot["iucn_status"], is_threatened)
# ):
#     color = "#D80027" if threat_status else "black"
#     alpha_val = alphas.get(status, 0.6)  # Default alpha if status not found
#     size = 36 if threat_status else 28
#     zorder_val = 2 if threat_status else 1  # Higher zorder for threatened species

#     ax.scatter(x, count, c=color, s=size, alpha=alpha_val, zorder=zorder_val)

# # Add legend manually since we're plotting individual points
# ax.scatter([], [], c="gray", s=28, alpha=1.0, label="Non-threatened")
# ax.scatter([], [], c="#D80027", s=36, alpha=1.0, label="Threatened (CR/EN/VU)")

# # Add vertical lines at threatened species positions
# threatened_idx = np.where(is_threatened)[0]  # Get indices of threatened species
# for i in threatened_idx:
#     ax.axvline(i, color="#db2b39", linestyle="-", alpha=0.4, linewidth=0.5)

# ax.set_xlabel("Class (sorted by prevalence)")
# ax.set_ylabel(r"$\#$ examples")
# ax.set_title("Pl@ntNet-300K: Threatened vs Other Species")
# ax.legend(framealpha=0.9, fontsize=9, loc="upper right")

# # Remove top and right spines
# ax.spines["top"].set_visible(False)
# ax.spines["right"].set_visible(False)

# plt.tight_layout()
# plt.savefig(
#     f"{fig_folder}/plantnet_class_distribution_iucn_status.pdf",
#     bbox_inches="tight",
#     dpi=300,
# )

# # Print summary statistics
# print("IUCN status distribution:")
# print(df_to_plot["iucn_status"].value_counts().to_dict())
# print(f"\nThreatened species: {is_threatened.sum()}/{len(df_to_plot)} classes")

# %%
# Import utilities for conformal prediction results
from utils.experiment_utils import (
    get_inputs_folder,
    get_outputs_folder,
    get_figs_folder,
)
import pickle

# Default paths
inputs_folder = get_inputs_folder()
results_folder = get_outputs_folder()


def compute_train_weighted_average_set_size(
    dataset, metrics, train_class_distr, test_labels
):
    """Compute average set size weighted by training class distribution."""
    num_classes = np.max(test_labels) + 1
    set_sizes = metrics["coverage_metrics"]["raw_set_sizes"]
    avg_size_by_class = np.array(
        [np.mean(set_sizes[test_labels == k]) for k in range(num_classes)]
    )
    return np.sum(train_class_distr * avg_size_by_class)


def load_conformal_result(
    dataset="plantnet", alpha=0.1, method="standard", score="softmax"
):
    """Load conformal prediction results using the same pattern as pareto_plots.py"""
    file_path = f"{results_folder}/{dataset}_{score}_alpha={alpha}_{method}.pkl"
    try:
        with open(file_path, "rb") as f:
            metrics = pickle.load(f)
        print(f"✓ Loaded conformal results from: {file_path}")
        return metrics
    except FileNotFoundError:
        print(f"⚠️ Could not find conformal prediction results at: {file_path}")
        return None


def compute_avg_set_sizes_per_class(
    dataset="plantnet", alpha=0.1, method="standard", score="softmax"
):
    """Compute average set size per class from conformal prediction results."""
    # Load conformal results
    metrics = load_conformal_result(dataset, alpha, method, score)
    if metrics is None:
        return None

    # Load test labels
    test_labels_path = f"{inputs_folder}/best-{dataset}-model_test_labels.npy"
    try:
        test_labels = np.load(test_labels_path)
    except FileNotFoundError:
        print(f"⚠️ Could not load test labels from: {test_labels_path}")
        return None

    # Compute average set size per class
    num_classes = np.max(test_labels) + 1
    set_sizes = metrics["coverage_metrics"]["raw_set_sizes"]
    avg_size_by_class = np.array(
        [
            np.mean(set_sizes[test_labels == k]) if np.sum(test_labels == k) > 0 else 0
            for k in range(num_classes)
        ]
    )

    print(f"✓ Computed average set sizes for {num_classes} classes")
    return avg_size_by_class


def compute_avg_coverage_per_class(
    dataset="plantnet", alpha=0.1, method="standard", score="softmax"
):
    """Compute average coverage per class from conformal prediction results."""
    # Load conformal results
    metrics = load_conformal_result(dataset, alpha, method, score)
    if metrics is None:
        return None

    # Load test labels
    test_labels_path = f"{inputs_folder}/best-{dataset}-model_test_labels.npy"
    try:
        test_labels = np.load(test_labels_path)
    except FileNotFoundError:
        print(f"⚠️ Could not load test labels from: {test_labels_path}")
        return None

    # Get coverage per class - either from precomputed or compute from prediction sets
    if "raw_class_coverages" in metrics["coverage_metrics"]:
        # Use precomputed class coverages
        avg_coverage_by_class = metrics["coverage_metrics"]["raw_class_coverages"]
        print(
            f"✓ Loaded precomputed class coverages for {len(avg_coverage_by_class)} classes"
        )
    else:
        # Compute from prediction sets if available
        if "pred_sets" in metrics:
            pred_sets = metrics["pred_sets"]
            num_classes = np.max(test_labels) + 1
            avg_coverage_by_class = np.array(
                [
                    (
                        np.mean(
                            [
                                test_labels[i] in pred_sets[i]
                                for i in np.where(test_labels == k)[0]
                            ]
                        )
                        if np.sum(test_labels == k) > 0
                        else 0
                    )
                    for k in range(num_classes)
                ]
            )
            print(
                f"✓ Computed class coverages from prediction sets for {num_classes} classes"
            )
        else:
            print(
                "⚠️ No coverage data available (neither raw_class_coverages nor pred_sets)"
            )
            return None

    return avg_coverage_by_class


def plot_combined_analysis(df_sorted, alpha=0.1, methods=None, scores=None):
    """Create a combined plot with 3 subplots: Coverage, Set Size, and # Examples."""
    if methods is None:
        methods = ["standard", "classwise", "prevalence-adjusted"]
    if scores is None:
        scores = ["softmax", "softmax", "softmax"]

    # Create figure with 3 subplots arranged vertically
    fig, axes = plt.subplots(3, 1, figsize=(12, 6))

    # Use colors from core_methods
    core_methods = [
        ("standard", "Standard", "blue"),
        ("classwise", "Classwise", "red"),
        ("prevalence-adjusted", "Standard w. PAS", "orange"),
    ]

    # Define colors and markers using core_methods
    colors = {method: color for method, _, color in core_methods}

    markers = {
        "standard": "2",  # Tri-up marker
        "classwise": "2",  # Tri-up marker
        "prevalence-adjusted": "1",  # Tri-down marker
    }

    method_names = {
        "standard": "Standard",
        "classwise": "Classwise",
        "prevalence-adjusted": "Standard w. PAS",
    }

    # Determine the consistent data size for all subplots
    # Find the minimum data size across all methods to ensure alignment
    min_data_size = len(df_sorted)
    for method, score in zip(methods, scores):
        coverage_data = compute_avg_coverage_per_class("plantnet", alpha, method, score)
        if coverage_data is not None:
            min_data_size = min(min_data_size, len(coverage_data))

        size_data = compute_avg_set_sizes_per_class("plantnet", alpha, method, score)
        if size_data is not None:
            min_data_size = min(min_data_size, len(size_data))

    # Use consistent dataframe for all subplots
    df_consistent = df_sorted.iloc[:min_data_size].copy()
    x_idx = np.arange(len(df_consistent))
    is_threatened = get_threatened_status(df_consistent)
    threatened_idx = get_threatened_indices(df_consistent)

    # Define alpha values for IUCN statuses - high alpha for threatened, low for others
    def get_alpha_for_status(status):
        """Get alpha value based on IUCN status - high for threatened species."""
        if status in THREATENED_CODES:  # CR, EN, VU
            return 1.0  # High visibility for threatened species
        else:  # LC, NT, DD, Not Evaluated, etc.
            return 0.1  # Low visibility for non-threatened species

    # Plot 1: Class Coverage (top)
    ax_cov = axes[0]
    for i, (method, score) in enumerate(zip(methods, scores)):
        avg_coverage = compute_avg_coverage_per_class("plantnet", alpha, method, score)
        if avg_coverage is None:
            continue

        # Use consistent dataframe
        class_ids = df_consistent["class_id"].values
        reordered_coverage = np.zeros(len(df_consistent))

        for j, class_id in enumerate(class_ids):
            if class_id < len(avg_coverage):
                reordered_coverage[j] = avg_coverage[class_id]

        # Plot points with IUCN-based alpha
        for j, (x, coverage_val, status) in enumerate(
            zip(x_idx, reordered_coverage, df_consistent["iucn_status"])
        ):
            alpha_val = get_alpha_for_status(status)
            ax_cov.scatter(
                x,
                coverage_val,
                color=colors.get(method, f"C{i}"),
                marker=markers.get(method, "o"),
                s=300,
                alpha=alpha_val,
                zorder=i,
                linewidths=1,
            )

        # Legend entry
        ax_cov.scatter(
            [],
            [],
            color=colors.get(method, f"C{i}"),
            marker=markers.get(method, "o"),
            s=300,
            alpha=1.0,
            linewidths=1,
            label=method_names.get(method, method),
        )

    # Add vertical lines for threatened species (consistent across all subplots)
    for i in threatened_idx:
        ax_cov.axvline(i, color="#db2b39", linestyle="-", alpha=0.4, linewidth=0.5)

    ax_cov.set_ylabel("Class coverage")
    ax_cov.set_title(f"Pl@ntNet-300K: Combined Analysis ($\\alpha={alpha}$)")
    ax_cov.set_ylim(-0.05, 1.05)
    # Remove individual legend - will be added as figure legend
    ax_cov.spines["top"].set_visible(False)
    ax_cov.spines["right"].set_visible(False)

    # Plot 2: Average Set Size (middle)
    ax_size = axes[1]
    for i, (method, score) in enumerate(zip(methods, scores)):
        avg_set_sizes = compute_avg_set_sizes_per_class(
            "plantnet", alpha, method, score
        )
        if avg_set_sizes is None:
            continue

        # Use consistent dataframe
        class_ids = df_consistent["class_id"].values
        reordered_sizes = np.zeros(len(df_consistent))

        for j, class_id in enumerate(class_ids):
            if class_id < len(avg_set_sizes):
                reordered_sizes[j] = avg_set_sizes[class_id]

        # Plot points with IUCN-based alpha
        for j, (x, size_val, status) in enumerate(
            zip(x_idx, reordered_sizes, df_consistent["iucn_status"])
        ):
            alpha_val = get_alpha_for_status(status)
            ax_size.scatter(
                x,
                size_val,
                color=colors.get(method, f"C{i}"),
                marker=markers.get(method, "o"),
                s=300,
                alpha=alpha_val,
                zorder=i,
                linewidths=1,
            )

        # Legend entry
        ax_size.scatter(
            [],
            [],
            color=colors.get(method, f"C{i}"),
            marker=markers.get(method, "o"),
            s=300,
            alpha=1.0,
            linewidths=1,
            label=method_names.get(method, method),
        )

    # Add vertical lines for threatened species (same positions as subplot 1)
    for i in threatened_idx:
        ax_size.axvline(i, color="#db2b39", linestyle="-", alpha=0.4, linewidth=0.5)

    ax_size.set_ylabel("Av. set size")
    ax_size.set_yscale("log")
    num_classes = len(df_consistent)
    ax_size.set_ylim(1 / 4, num_classes)
    ax_size.yaxis.set_major_locator(plt.LogLocator(base=10.0, numticks=10))
    ax_size.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax_size.grid(True, which="major", axis="y", linestyle="--", alpha=0.7)
    # Remove individual legend - will be added as figure legend
    ax_size.spines["top"].set_visible(False)
    ax_size.spines["right"].set_visible(False)

    # Plot 3: # Examples (bottom)
    ax_examples = axes[2]
    counts = df_consistent["counts"].values

    # Plot each species with color and alpha based on threatened status
    for i, (x, count, status, threat_status) in enumerate(
        zip(x_idx, counts, df_consistent["iucn_status"], is_threatened)
    ):
        # Color: red for threatened, black for non-threatened
        color = "#D80027" if threat_status else "black"
        # Alpha: high visibility for threatened, moderate for non-threatened
        alpha_val = 1.0 if threat_status else 0.8
        # Size: larger for threatened species
        size = 70 if threat_status else 28
        zorder_val = 2 if threat_status else 1

        ax_examples.scatter(
            x, count, c=color, s=size, marker="*", alpha=alpha_val, zorder=zorder_val
        )

    # Add legend for examples plot
    ax_examples.scatter([], [], c="black", s=28, alpha=1.0, label="Non-threatened")
    ax_examples.scatter(
        [], [], c="#D80027", s=70, alpha=1.0, marker="*", label="Threatened (CR/EN/VU)"
    )

    # Add vertical lines for threatened species (same positions as other subplots)
    for i in threatened_idx:
        ax_examples.axvline(i, color="#db2b39", linestyle="-", alpha=0.4, linewidth=0.5)

    ax_examples.set_xlabel("Class (sorted by prevalence)")
    ax_examples.set_ylabel(r"$\#$ examples")
    ax_examples.legend(framealpha=0.9, fontsize=14, loc="upper right")
    ax_examples.spines["top"].set_visible(False)
    ax_examples.spines["right"].set_visible(False)

    # Create figure legend with methods in specified order: Classwise, Standard, Standard w. PAS
    # Use the same style as the first plot (coverage) with endangered species alpha
    legend_order = ["classwise", "standard", "prevalence-adjusted"]
    legend_handles = []
    legend_labels = []

    for method in legend_order:
        if method in colors:
            # Create scatter plot handle that matches the first plot appearance
            handle = ax_cov.scatter(
                [],
                [],
                color=colors[method],
                marker=markers.get(method, "o"),
                s=300,
                alpha=1.0,  # Use high alpha like endangered species
                linewidths=1,
            )
            legend_handles.append(handle)
            legend_labels.append(method_names.get(method, method))

    # Adjust spacing and add legend higher up
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # Add more space between title and first subplot

    fig.legend(
        legend_handles,
        legend_labels,
        fontsize=16,
        borderaxespad=0.1,
        bbox_to_anchor=(0.5, 1.06),  # Moved higher up
        loc="upper center",
        ncol=3,  # Three methods in one row
        markerscale=1,
        handlelength=1,
        handletextpad=0.4,
        columnspacing=1.5,
    )

    plt.savefig(
        f"{fig_folder}/plantnet_combined_analysis_alpha{alpha}.pdf",
        bbox_inches="tight",
        dpi=300,
    )
    return fig, axes


# Create combined plot with all three analyses
plot_combined_analysis(
    df_to_plot,
    alpha=0.1,
    methods=["standard", "classwise", "prevalence-adjusted"],
    scores=["softmax", "softmax", "softmax"],
)

# %%
