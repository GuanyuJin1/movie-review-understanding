import json
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.movie_review_understanding.config.settings import FIGURES_DIR, METRICS_DIR
from src.movie_review_understanding.evaluation.metrics import build_confusion_matrix, build_model_comparison, evaluate_clustering
from src.movie_review_understanding.models.classifiers import ClassificationResult
from src.movie_review_understanding.models.clustering import ClusteringResult

sns.set_theme(style="whitegrid")


def _ensure_output_dirs() -> Tuple[Path, Path]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR, METRICS_DIR


def save_model_comparison_figure(results: List[ClassificationResult]) -> Path:
    """Save a bar chart comparing classifier accuracy and F1."""
    figures_dir, _ = _ensure_output_dirs()
    summary = build_model_comparison(results)

    model_names = [item["model_name"] for item in summary]
    accuracies = [item["accuracy"] for item in summary]
    f1_scores = [item["f1"] for item in summary]

    fig, ax = plt.subplots(figsize=(8, 5))
    x_positions = range(len(model_names))
    width = 0.35
    ax.bar([x - width / 2 for x in x_positions], accuracies, width=width, label="Accuracy")
    ax.bar([x + width / 2 for x in x_positions], f1_scores, width=width, label="F1 Score")
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Traditional Classifier Performance")
    ax.legend()
    fig.tight_layout()

    output_path = figures_dir / "model_comparison.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_confusion_matrix_figure(result: ClassificationResult) -> Path:
    """Save a confusion matrix heatmap for one classifier."""
    figures_dir, _ = _ensure_output_dirs()
    matrix, labels = build_confusion_matrix(result)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(f"Confusion Matrix: {result.model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()

    safe_name = result.model_name.lower().replace(" ", "_").replace("-", "_")
    output_path = figures_dir / f"confusion_matrix_{safe_name}.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_cluster_size_figure(result: ClusteringResult) -> Path:
    """Save a bar chart showing cluster membership counts."""
    figures_dir, _ = _ensure_output_dirs()
    cluster_ids = list(result.cluster_sizes.keys())
    cluster_counts = list(result.cluster_sizes.values())

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([str(cluster_id) for cluster_id in cluster_ids], cluster_counts, color="#4C78A8")
    ax.set_title("K-Means Cluster Sizes")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Reviews")
    fig.tight_layout()

    output_path = figures_dir / "cluster_sizes.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_cluster_projection_figure(result: ClusteringResult) -> Path:
    """Save a 2D projection of clustered review samples."""
    figures_dir, _ = _ensure_output_dirs()
    coordinates = pd.DataFrame(result.reduced_coordinates, columns=["component_1", "component_2"])
    coordinates["cluster"] = [str(label) for label in result.labels]

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(
        data=coordinates,
        x="component_1",
        y="component_2",
        hue="cluster",
        palette="tab10",
        alpha=0.75,
        s=30,
        ax=ax,
    )
    ax.set_title("K-Means Cluster Projection (TruncatedSVD)")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    fig.tight_layout()

    output_path = figures_dir / "cluster_projection.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_metric_tables(
    classification_results: List[ClassificationResult],
    clustering_result: ClusteringResult,
) -> List[Path]:
    """Save structured metric summaries for later reporting."""
    _, metrics_dir = _ensure_output_dirs()
    classification_summary = pd.DataFrame(build_model_comparison(classification_results))
    classification_path = metrics_dir / "classification_summary.csv"
    classification_summary.to_csv(classification_path, index=False)

    clustering_summary = evaluate_clustering(clustering_result)
    clustering_path = metrics_dir / "clustering_summary.json"
    clustering_path.write_text(json.dumps(clustering_summary, indent=2), encoding="utf-8")

    return [classification_path, clustering_path]


def save_visualizations(
    classification_results: List[ClassificationResult],
    clustering_result: ClusteringResult,
) -> List[Path]:
    """Generate and save the main project figures and summary tables."""
    saved_paths = [
        save_model_comparison_figure(classification_results),
        save_cluster_size_figure(clustering_result),
        save_cluster_projection_figure(clustering_result),
    ]
    for result in classification_results:
        saved_paths.append(save_confusion_matrix_figure(result))
    saved_paths.extend(save_metric_tables(classification_results, clustering_result))
    return saved_paths
