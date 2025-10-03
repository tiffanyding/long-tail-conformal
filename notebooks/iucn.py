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
# Plot IUCN threatened vs non-threatened species
threatened_codes = {"CR", "EN", "VU"}
is_threatened = df_to_plot["iucn_status"].isin(threatened_codes)

fig, ax = plt.subplots(figsize=(10, 3))
x_idx = np.arange(len(df_to_plot))
counts = df_to_plot["counts"].values

# Define alpha values for different IUCN statuses
alphas = {
    "LC": 0.1,
    "LR/lc": 0.8,
    "NT": 0.8,
    "LR/nt": 0.8,
    "DD": 0.1,
    "VU": 0.8,
    "LR/cd": 0.8,
    "EN": 0.8,
    "CR": 0.8,
    "Not Evaluated": 0.1,
}

# Plot each species with alpha based on IUCN status
for i, (x, count, status, threat_status) in enumerate(
    zip(x_idx, counts, df_to_plot["iucn_status"], is_threatened)
):
    color = "#D80027" if threat_status else "black"
    alpha_val = alphas.get(status, 0.6)  # Default alpha if status not found
    size = 36 if threat_status else 28
    zorder_val = 2 if threat_status else 1  # Higher zorder for threatened species

    ax.scatter(x, count, c=color, s=size, alpha=alpha_val, zorder=zorder_val)

# Add legend manually since we're plotting individual points
ax.scatter([], [], c="black", s=28, alpha=1.0, label="Non-threatened")
ax.scatter([], [], c="#D80027", s=36, alpha=1.0, label="Threatened (CR/EN/VU)")

# Add vertical lines at threatened species positions
threatened_idx = np.where(is_threatened)[0]  # Get indices of threatened species
for i in threatened_idx:
    ax.axvline(i, color="#db2b39", linestyle="-", alpha=0.4, linewidth=0.5)

ax.set_xlabel("Class (sorted by prevalence)")
ax.set_ylabel(r"$\#$ examples")
ax.set_title("Pl@ntNet-300K: Threatened vs Other Species")
ax.legend(framealpha=0.9, fontsize=9, loc="upper right")

# Remove top and right spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(
    f"{fig_folder}/plantnet_class_distribution_iucn_status.pdf",
    bbox_inches="tight",
    dpi=300,
)

# Print summary statistics
print("IUCN status distribution:")
print(df_to_plot["iucn_status"].value_counts().to_dict())
print(f"\nThreatened species: {is_threatened.sum()}/{len(df_to_plot)} classes")

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


def plot_average_set_sizes_comparison(df_sorted, alpha=0.1, methods=None, scores=None):
    """Plot average predicted set size comparison for multiple methods."""
    if methods is None:
        methods = ["standard", "classwise", "prevalence-adjusted"]
    if scores is None:
        scores = ["softmax", "softmax", "softmax"]

    # Define colors and markers as in the reference code
    colors = {
        "standard": "#41D719",  # Green for standard (marginal)
        "classwise": "#FFB41D",  # Orange for classwise
        "prevalence-adjusted": "#8619A9",  # Purple for prevalence
        "prevalence": "#8619A9",  # Alternative name mapping
        "marginal": "#41D719",
    }

    markers = {
        "standard": "2",  # Tri-up marker
        "classwise": "2",  # Tri-up marker
        "prevalence-adjusted": "1",  # Tri-down marker
        "prevalence": "1",
        "marginal": "2",
    }

    # Method display names
    method_names = {
        "standard": "Standard",
        "classwise": "Classwise",
        "prevalence-adjusted": "Standard w. PAS",
    }

    fig, ax = plt.subplots(figsize=(10, 3))

    # Load and plot data for each method
    all_results = {}
    for i, (method, score) in enumerate(zip(methods, scores)):
        # Compute set sizes per class for this method
        avg_set_sizes = compute_avg_set_sizes_per_class(
            "plantnet", alpha, method, score
        )
        if avg_set_sizes is None:
            print(f"⚠️ Skipping {method} - no data available")
            continue

        # Ensure alignment between metadata and set sizes
        n_classes = min(len(df_sorted), len(avg_set_sizes))
        df_plot = df_sorted.iloc[:n_classes].copy()

        # Reorder set sizes according to prevalence order in df_sorted
        class_ids = df_plot["class_id"].values
        reordered_sizes = np.zeros(n_classes)
        for j, class_id in enumerate(class_ids):
            if class_id < len(avg_set_sizes):
                reordered_sizes[j] = avg_set_sizes[class_id]

        all_results[method] = reordered_sizes

        # Define alpha values for different IUCN statuses
        alphas_iucn = {
            "LC": 0.05,
            "LR/lc": 1,
            "NT": 1,
            "LR/nt": 1,
            "DD": 0.05,
            "VU": 1,
            "LR/cd": 1,
            "EN": 1,
            "CR": 1,
            "Not Evaluated": 0.05,
        }

        # Plot all points for this method with varying alpha based on IUCN status
        x_idx = np.arange(len(df_plot))
        for j, (x, size_val, status) in enumerate(
            zip(x_idx, reordered_sizes, df_plot["iucn_status"])
        ):
            alpha_val = alphas_iucn.get(
                status, 0.065
            )  # Default alpha if status not found
            ax.scatter(
                x,
                size_val,
                color=colors.get(method, f"C{i}"),
                marker=markers.get(method, "o"),
                s=300,
                alpha=alpha_val,
                zorder=i,
                linewidths=1,
            )

        # Add single legend entry for this method
        ax.scatter(
            [],
            [],
            color=colors.get(method, f"C{i}"),
            marker=markers.get(method, "o"),
            s=300,
            alpha=1.0,
            linewidths=1,
            label=method_names.get(method, method),
        )

    # Add vertical lines at threatened species positions
    threatened_codes = {"CR", "EN", "VU"}
    is_threatened = df_plot["iucn_status"].isin(threatened_codes)
    threatened_idx = np.where(is_threatened)[0]  # Get indices of threatened species

    # Plot vertical lines at threatened species positions
    for i in threatened_idx:
        ax.axvline(i, color="#db2b39", linestyle="-", alpha=0.4, linewidth=0.5)

    ax.set_xlabel("Class (sorted by prevalence)")
    ax.set_ylabel("Av. set size")
    ax.set_title(f"Pl@ntNet-300K: Set Size Comparison ($\\alpha={alpha}$)")
    ax.set_yscale("log")

    # Set y-axis limits and formatting
    num_classes = len(df_plot)
    ax.set_ylim(1 / 4, num_classes)
    # Add major locator for y axis at 1, 10, 100, 1000
    ax.yaxis.set_major_locator(plt.LogLocator(base=10.0, numticks=10))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # Add horizontal grid lines at major ticks
    ax.grid(True, which="major", axis="y", linestyle="--", alpha=0.7)

    ax.legend(framealpha=0.9, fontsize=9, loc="upper right")

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(
        f"{fig_folder}/plantnet_avg_set_sizes_comparison_alpha{alpha}.pdf",
        bbox_inches="tight",
        dpi=300,
    )

    # Print summary statistics for all methods
    threatened_codes = {"CR", "EN", "VU"}
    is_threatened = df_plot["iucn_status"].isin(threatened_codes)

    print(f"\nSet size comparison (alpha={alpha}):")
    for method in all_results:
        sizes = all_results[method]
        threatened_avg = np.mean(sizes[is_threatened]) if np.any(is_threatened) else 0
        non_threatened_avg = (
            np.mean(sizes[~is_threatened]) if np.any(~is_threatened) else 0
        )
        overall_avg = np.mean(sizes[sizes > 0])

        print(f"  {method_names.get(method, method)}:")
        print(
            f"    Threatened: {threatened_avg:.3f}, Non-threatened: {non_threatened_avg:.3f}, Overall: {overall_avg:.3f}"
        )

    return fig, ax


def plot_average_coverage_comparison(df_sorted, alpha=0.1, methods=None, scores=None):
    """Plot average class coverage comparison for multiple methods."""
    if methods is None:
        methods = ["standard", "classwise", "prevalence-adjusted"]
    if scores is None:
        scores = ["softmax", "softmax", "softmax"]

    # Define colors and markers as in the reference code
    colors = {
        "standard": "#41D719",  # Green for standard (marginal)
        "classwise": "#FFB41D",  # Orange for classwise
        "prevalence-adjusted": "#8619A9",  # Purple for prevalence
        "prevalence": "#8619A9",  # Alternative name mapping
        "marginal": "#41D719",
    }

    markers = {
        "standard": "2",  # Tri-up marker
        "classwise": "2",  # Tri-up marker
        "prevalence-adjusted": "1",  # Tri-down marker
        "prevalence": "1",
        "marginal": "2",
    }

    # Method display names
    method_names = {
        "standard": "Standard",
        "classwise": "Classwise",
        "prevalence-adjusted": "Standard w. PAS",
    }

    fig, ax = plt.subplots(figsize=(10, 3))

    # Load and plot data for each method
    all_results = {}
    for i, (method, score) in enumerate(zip(methods, scores)):
        # Compute coverage per class for this method
        avg_coverage = compute_avg_coverage_per_class("plantnet", alpha, method, score)
        if avg_coverage is None:
            print(f"⚠️ Skipping {method} - no coverage data available")
            continue

        # Ensure alignment between metadata and coverage
        n_classes = min(len(df_sorted), len(avg_coverage))
        df_plot = df_sorted.iloc[:n_classes].copy()

        # Reorder coverage according to prevalence order in df_sorted
        class_ids = df_plot["class_id"].values
        reordered_coverage = np.zeros(n_classes)
        for j, class_id in enumerate(class_ids):
            if class_id < len(avg_coverage):
                reordered_coverage[j] = avg_coverage[class_id]

        all_results[method] = reordered_coverage

        # Define alpha values for different IUCN statuses
        alphas_iucn = {
            "LC": 0.05,
            "LR/lc": 1,
            "NT": 1,
            "LR/nt": 1,
            "DD": 0.05,
            "VU": 1,
            "LR/cd": 1,
            "EN": 1,
            "CR": 1,
            "Not Evaluated": 0.05,
        }

        # Plot all points for this method with varying alpha based on IUCN status
        x_idx = np.arange(len(df_plot))
        for j, (x, coverage_val, status) in enumerate(
            zip(x_idx, reordered_coverage, df_plot["iucn_status"])
        ):
            alpha_val = alphas_iucn.get(
                status, 0.065
            )  # Default alpha if status not found
            ax.scatter(
                x,
                coverage_val,
                color=colors.get(method, f"C{i}"),
                marker=markers.get(method, "o"),
                s=300,
                alpha=alpha_val,
                zorder=i,
                linewidths=1,
            )

        # Add single legend entry for this method
        ax.scatter(
            [],
            [],
            color=colors.get(method, f"C{i}"),
            marker=markers.get(method, "o"),
            s=300,
            alpha=1.0,
            linewidths=1,
            label=method_names.get(method, method),
        )

    # Add vertical lines at threatened species positions
    threatened_codes = {"CR", "EN", "VU"}
    is_threatened = df_plot["iucn_status"].isin(threatened_codes)
    threatened_idx = np.where(is_threatened)[0]  # Get indices of threatened species

    # Plot vertical lines at threatened species positions
    for i in threatened_idx:
        ax.axvline(i, color="#db2b39", linestyle="-", alpha=0.4, linewidth=0.5)

    ax.set_xlabel("Class (sorted by prevalence)")
    ax.set_ylabel("Class coverage")
    ax.set_title(f"Pl@ntNet-300K: Coverage Comparison ($\\alpha={alpha}$)")
    ax.set_ylim(-0.05, 1.05)  # Spread y-axis to better see values at 0 and 1
    ax.legend(framealpha=0.9, fontsize=9, loc="lower right")

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(
        f"{fig_folder}/plantnet_avg_coverage_comparison_alpha{alpha}.pdf",
        bbox_inches="tight",
        dpi=300,
    )

    # Print summary statistics for all methods
    print(f"\nCoverage comparison (alpha={alpha}):")
    for method in all_results:
        coverage = all_results[method]
        threatened_avg = (
            np.mean(coverage[is_threatened]) if np.any(is_threatened) else 0
        )
        non_threatened_avg = (
            np.mean(coverage[~is_threatened]) if np.any(~is_threatened) else 0
        )
        overall_avg = np.mean(coverage[coverage > 0])

        print(f"  {method_names.get(method, method)}:")
        print(
            f"    Threatened: {threatened_avg:.3f}, Non-threatened: {non_threatened_avg:.3f}, Overall: {overall_avg:.3f}"
        )

    return fig, ax


def plot_average_set_sizes(df_sorted, alpha=0.1, method="standard", score="softmax"):
    """Plot average predicted set size for each species in the same order as IUCN plot."""
    # Compute set sizes per class
    avg_set_sizes = compute_avg_set_sizes_per_class("plantnet", alpha, method, score)
    if avg_set_sizes is None:
        print("Cannot plot set sizes - no data available")
        return

    # Ensure alignment between metadata and set sizes
    n_classes = min(len(df_sorted), len(avg_set_sizes))
    df_plot = df_sorted.iloc[:n_classes].copy()

    # Reorder set sizes according to prevalence order in df_sorted
    class_ids = df_plot["class_id"].values
    reordered_sizes = np.zeros(n_classes)
    for i, class_id in enumerate(class_ids):
        if class_id < len(avg_set_sizes):
            reordered_sizes[i] = avg_set_sizes[class_id]

    threatened_codes = {"CR", "EN", "VU"}
    is_threatened = df_plot["iucn_status"].isin(threatened_codes)

    fig, ax = plt.subplots(figsize=(10, 3))
    x_idx = np.arange(len(df_plot))

    # Plot non-threatened (black) and threatened (red)
    ax.scatter(
        x_idx[~is_threatened],
        reordered_sizes[~is_threatened],
        c="black",
        s=28,
        alpha=0.6,
        label="Non-threatened",
    )
    ax.scatter(
        x_idx[is_threatened],
        reordered_sizes[is_threatened],
        c="#D80027",
        s=36,
        alpha=0.85,
        label="Threatened (CR/EN/VU)",
    )

    ax.set_xlabel("Class (sorted by prevalence)")
    ax.set_ylabel("Average prediction set size")
    ax.set_title(f"Pl@ntNet-300K: Average Set Size ({method}, $\\alpha={alpha}$)")
    ax.legend(framealpha=0.9, fontsize=9, loc="upper right")

    # Add horizontal line for overall average
    overall_avg = np.mean(reordered_sizes[reordered_sizes > 0])  # Exclude zeros
    ax.axhline(
        overall_avg,
        color="gray",
        linestyle="--",
        alpha=0.7,
        label=f"Overall avg: {overall_avg:.2f}",
    )

    plt.tight_layout()
    plt.savefig(
        f"{fig_folder}/plantnet_avg_set_sizes_{method}_alpha{alpha}.pdf",
        bbox_inches="tight",
        dpi=300,
    )

    # Print summary statistics
    threatened_avg = (
        np.mean(reordered_sizes[is_threatened]) if np.any(is_threatened) else 0
    )
    non_threatened_avg = (
        np.mean(reordered_sizes[~is_threatened]) if np.any(~is_threatened) else 0
    )

    print(f"\nAverage set sizes ({method}, alpha={alpha}):")
    print(f"  Threatened species: {threatened_avg:.3f}")
    print(f"  Non-threatened species: {non_threatened_avg:.3f}")
    print(f"  Overall average: {overall_avg:.3f}")

    return fig, ax


# Plot average set sizes comparison for multiple methods
plot_average_set_sizes_comparison(
    df_to_plot,
    alpha=0.1,
    methods=["standard", "classwise", "prevalence-adjusted"],
    scores=["softmax", "softmax", "softmax"],
)

# Plot average coverage comparison for multiple methods
plot_average_coverage_comparison(
    df_to_plot,
    alpha=0.1,
    methods=["standard", "classwise", "prevalence-adjusted"],
    scores=["softmax", "softmax", "softmax"],
)

# Plot individual method (optional - single method view)
# plot_average_set_sizes(df_to_plot, alpha=0.1, method="standard", score="softmax")

# %%
