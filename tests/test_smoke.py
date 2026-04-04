import pandas as pd

from src.movie_review_understanding.evaluation.metrics import (
    build_model_comparison,
    evaluate_clustering,
    evaluate_predictions,
    extract_misclassified_examples,
)
from src.movie_review_understanding.evaluation.visualization import save_visualizations
from src.movie_review_understanding.features.tfidf import (
    build_tfidf_vectorizer,
    prepare_tfidf_splits,
)
from src.movie_review_understanding.models.classifiers import train_classifiers
from src.movie_review_understanding.models.clustering import run_kmeans_clustering
from src.movie_review_understanding.models.llm_classifier import (
    LLMConfigurationError,
    build_few_shot_prompt,
    build_zero_shot_prompt,
    classify_with_llm,
    parse_sentiment_label,
)


def test_tfidf_vectorizer_can_be_created():
    vectorizer = build_tfidf_vectorizer()
    assert vectorizer is not None


def test_prepare_tfidf_splits_returns_non_empty_features():
    dataframe = pd.DataFrame(
        {
            "review": [
                "I loved this movie and the acting was great.",
                "Terrible plot and boring scenes.",
                "A wonderful and emotional film.",
                "I would not recommend this movie at all.",
                "Fantastic soundtrack and strong performances.",
                "This was dull, slow, and disappointing.",
            ],
            "sentiment": [
                "positive",
                "negative",
                "positive",
                "negative",
                "positive",
                "negative",
            ],
        }
    )

    dataset = prepare_tfidf_splits(dataframe, test_size=0.33, random_state=42)

    assert dataset.X_train.shape[0] > 0
    assert dataset.X_test.shape[0] > 0
    assert dataset.X_train.shape[1] > 0


def test_traditional_baselines_return_metrics():
    dataframe = pd.DataFrame(
        {
            "review": [
                "I loved this movie and the acting was great.",
                "Terrible plot and boring scenes.",
                "A wonderful and emotional film.",
                "I would not recommend this movie at all.",
                "Fantastic soundtrack and strong performances.",
                "This was dull, slow, and disappointing.",
                "Brilliant storytelling and excellent performances.",
                "Waste of time and badly written.",
                "An inspiring and beautifully made film.",
                "Awful dialogue and forgettable acting.",
            ],
            "sentiment": [
                "positive",
                "negative",
                "positive",
                "negative",
                "positive",
                "negative",
                "positive",
                "negative",
                "positive",
                "negative",
            ],
        }
    )

    dataset = prepare_tfidf_splits(dataframe, test_size=0.3, random_state=42)
    results = train_classifiers(dataset)
    comparison = build_model_comparison(results)
    examples = extract_misclassified_examples(results[0], dataset.test_texts, max_examples=2)

    assert len(results) == 3
    assert len(comparison) == 3
    assert len(examples) <= 2
    assert comparison[0]["accuracy"] >= comparison[-1]["accuracy"]

    for result in results:
        metrics = evaluate_predictions(result)
        assert len(result.predictions) == len(result.true_labels)
        assert 0.0 <= metrics["accuracy"] <= 1.0


def test_kmeans_clustering_returns_summary():
    dataframe = pd.DataFrame(
        {
            "review": [
                "I loved this movie and the acting was great.",
                "Terrible plot and boring scenes.",
                "A wonderful and emotional film.",
                "I would not recommend this movie at all.",
                "Fantastic soundtrack and strong performances.",
                "This was dull, slow, and disappointing.",
                "Brilliant storytelling and excellent performances.",
                "Waste of time and badly written.",
                "An inspiring and beautifully made film.",
                "Awful dialogue and forgettable acting.",
                "Heartwarming story with memorable characters.",
                "Poor editing and a weak screenplay.",
            ],
            "sentiment": [
                "positive",
                "negative",
                "positive",
                "negative",
                "positive",
                "negative",
                "positive",
                "negative",
                "positive",
                "negative",
                "positive",
                "negative",
            ],
        }
    )

    dataset = prepare_tfidf_splits(dataframe, test_size=0.25, random_state=42)
    result = run_kmeans_clustering(dataset, num_clusters=2, sample_size=6, top_n_terms=3, random_state=42)
    summary = evaluate_clustering(result)

    assert result.algorithm_name == "K-Means"
    assert summary["num_clusters"] == 2
    assert summary["sample_size"] == 6
    assert len(result.top_terms) == 2
    assert len(result.reduced_coordinates) == 6
    assert len(result.cluster_sentiment_mix) == 2


def test_visualizations_are_saved():
    dataframe = pd.DataFrame(
        {
            "review": [
                "I loved this movie and the acting was great.",
                "Terrible plot and boring scenes.",
                "A wonderful and emotional film.",
                "I would not recommend this movie at all.",
                "Fantastic soundtrack and strong performances.",
                "This was dull, slow, and disappointing.",
                "Brilliant storytelling and excellent performances.",
                "Waste of time and badly written.",
                "An inspiring and beautifully made film.",
                "Awful dialogue and forgettable acting.",
                "Heartwarming story with memorable characters.",
                "Poor editing and a weak screenplay.",
            ],
            "sentiment": [
                "positive",
                "negative",
                "positive",
                "negative",
                "positive",
                "negative",
                "positive",
                "negative",
                "positive",
                "negative",
                "positive",
                "negative",
            ],
        }
    )

    dataset = prepare_tfidf_splits(dataframe, test_size=0.25, random_state=42)
    classification_results = train_classifiers(dataset)
    clustering_result = run_kmeans_clustering(dataset, num_clusters=2, sample_size=6, top_n_terms=3, random_state=42)
    saved_paths = save_visualizations(classification_results, clustering_result)

    assert len(saved_paths) == 8
    for path in saved_paths:
        assert path.exists()


def test_llm_prompt_builders_include_review_text():
    review_text = "This movie was excellent."
    assert review_text in build_zero_shot_prompt(review_text)
    assert review_text in build_few_shot_prompt(review_text)


def test_parse_sentiment_label_normalizes_basic_outputs():
    assert parse_sentiment_label("positive") == "positive"
    assert parse_sentiment_label("Negative") == "negative"


def test_openai_backend_requires_api_key():
    dataframe = pd.DataFrame(
        {
            "review": [
                "Great movie.",
                "Terrible movie.",
                "Amazing acting.",
                "Very boring film.",
            ],
            "sentiment": ["positive", "negative", "positive", "negative"],
        }
    )

    dataset = prepare_tfidf_splits(dataframe, test_size=0.5, random_state=42)

    try:
        classify_with_llm(dataset, prompt_style="zero_shot", backend="openai", api_key=None, sample_size=2)
        raised = False
    except LLMConfigurationError:
        raised = True

    assert raised is True
