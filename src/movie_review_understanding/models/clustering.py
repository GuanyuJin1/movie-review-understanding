from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score

from src.movie_review_understanding.config.settings import (
    DEFAULT_CLUSTER_REDUCTION_COMPONENTS,
    DEFAULT_CLUSTER_SAMPLE_SIZE,
    DEFAULT_NUM_CLUSTERS,
    DEFAULT_TOP_TERMS_PER_CLUSTER,
    RANDOM_STATE,
)
from src.movie_review_understanding.features.tfidf import TfidfDatasetSplit


@dataclass
class ClusteringResult:
    labels: List[int]
    algorithm_name: str
    num_clusters: int
    sample_size: int
    cluster_sizes: Dict[int, int]
    top_terms: Dict[int, List[str]]
    silhouette_score: float
    sample_texts: List[str]
    sample_true_labels: List[str]
    cluster_sentiment_mix: Dict[int, Dict[str, int]]
    reduced_coordinates: List[List[float]]


def _build_cluster_sentiment_mix(labels: np.ndarray, sentiments: List[str]) -> Dict[int, Dict[str, int]]:
    mix = {}
    for cluster_id in sorted(set(labels.tolist())):
        mix[int(cluster_id)] = {"positive": 0, "negative": 0}

    for cluster_id, sentiment in zip(labels.tolist(), sentiments):
        if sentiment not in mix[int(cluster_id)]:
            mix[int(cluster_id)][sentiment] = 0
        mix[int(cluster_id)][sentiment] += 1
    return mix


def run_kmeans_clustering(
    dataset: TfidfDatasetSplit,
    num_clusters: int = DEFAULT_NUM_CLUSTERS,
    sample_size: int = DEFAULT_CLUSTER_SAMPLE_SIZE,
    top_n_terms: int = DEFAULT_TOP_TERMS_PER_CLUSTER,
    random_state: int = RANDOM_STATE,
) -> ClusteringResult:
    """Run K-Means clustering on a sampled subset of TF-IDF training features."""
    available_samples = dataset.X_train.shape[0]
    if available_samples == 0:
        raise ValueError("Training feature matrix is empty.")

    effective_sample_size = min(sample_size, available_samples)
    rng = np.random.RandomState(random_state)
    sample_indices = np.sort(rng.choice(available_samples, size=effective_sample_size, replace=False))
    sampled_matrix = dataset.X_train[sample_indices]
    sample_texts = [dataset.train_texts[index] for index in sample_indices.tolist()]
    sample_true_labels = dataset.y_train.iloc[sample_indices].reset_index(drop=True).astype(str).tolist()

    model = KMeans(n_clusters=num_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(sampled_matrix)

    unique_labels, counts = np.unique(labels, return_counts=True)
    cluster_sizes = {int(label): int(count) for label, count in zip(unique_labels, counts)}

    feature_names = dataset.vectorizer.get_feature_names_out()
    top_terms = {}
    for cluster_idx, center in enumerate(model.cluster_centers_):
        top_indices = center.argsort()[-top_n_terms:][::-1]
        top_terms[cluster_idx] = [str(feature_names[index]) for index in top_indices]

    score = 0.0
    if effective_sample_size > num_clusters:
        score = float(silhouette_score(sampled_matrix, labels))

    reducer = TruncatedSVD(n_components=DEFAULT_CLUSTER_REDUCTION_COMPONENTS, random_state=random_state)
    reduced_coordinates = reducer.fit_transform(sampled_matrix).tolist()
    cluster_sentiment_mix = _build_cluster_sentiment_mix(labels, sample_true_labels)

    return ClusteringResult(
        labels=labels.tolist(),
        algorithm_name="K-Means",
        num_clusters=num_clusters,
        sample_size=effective_sample_size,
        cluster_sizes=cluster_sizes,
        top_terms=top_terms,
        silhouette_score=score,
        sample_texts=sample_texts,
        sample_true_labels=sample_true_labels,
        cluster_sentiment_mix=cluster_sentiment_mix,
        reduced_coordinates=reduced_coordinates,
    )


def run_clustering(dataset: TfidfDatasetSplit) -> ClusteringResult:
    """Run the clustering workflow used in the project."""
    return run_kmeans_clustering(dataset)
