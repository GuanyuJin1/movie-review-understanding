from typing import List

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

from src.movie_review_understanding.models.classifiers import ClassificationResult
from src.movie_review_understanding.models.clustering import ClusteringResult


def evaluate_predictions(result: ClassificationResult) -> dict:
    """Compute core sentiment-classification metrics for a model result."""
    accuracy = accuracy_score(result.true_labels, result.predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        result.true_labels,
        result.predictions,
        average="weighted",
        zero_division=0,
    )

    return {
        "model_name": result.model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_clustering(result: ClusteringResult) -> dict:
    """Return summary metrics for a clustering result."""
    dominant_sentiment_per_cluster = {}
    for cluster_id, sentiment_counts in result.cluster_sentiment_mix.items():
        dominant_sentiment_per_cluster[cluster_id] = max(sentiment_counts, key=sentiment_counts.get)

    return {
        "algorithm_name": result.algorithm_name,
        "num_clusters": result.num_clusters,
        "sample_size": result.sample_size,
        "silhouette_score": result.silhouette_score,
        "cluster_sizes": result.cluster_sizes,
        "cluster_sentiment_mix": result.cluster_sentiment_mix,
        "dominant_sentiment_per_cluster": dominant_sentiment_per_cluster,
    }


def build_confusion_matrix(result: ClassificationResult):
    """Build a confusion matrix for a classification result."""
    labels = sorted(result.true_labels.unique())
    return confusion_matrix(result.true_labels, result.predictions, labels=labels), labels


def build_classification_report(result: ClassificationResult) -> str:
    """Build a text classification report for terminal display."""
    return classification_report(result.true_labels, result.predictions, zero_division=0)


def build_model_comparison(results: List[ClassificationResult]) -> List[dict]:
    """Create a compact metrics summary for multiple classifiers."""
    summary = [evaluate_predictions(result) for result in results]
    return sorted(summary, key=lambda item: item["accuracy"], reverse=True)


def extract_misclassified_examples(
    result: ClassificationResult,
    texts: List[str],
    max_examples: int = 5,
) -> List[dict]:
    """Return a small set of misclassified text examples for error analysis."""
    examples = []
    for text, true_label, predicted_label in zip(texts, result.true_labels.tolist(), result.predictions.tolist()):
        if true_label != predicted_label:
            examples.append(
                {
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "text": text,
                }
            )
        if len(examples) >= max_examples:
            break
    return examples
