from typing import Optional

from openai import OpenAIError

from dotenv import load_dotenv

from src.movie_review_understanding.config.settings import (
    DEFAULT_ERROR_ANALYSIS_SAMPLES,
    DEFAULT_LLM_BACKEND,
    DEFAULT_LLM_SAMPLE_SIZE,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OPENAI_MODEL,
    PROJECT_ROOT,
)
from src.movie_review_understanding.data.loader import load_reviews
from src.movie_review_understanding.evaluation.metrics import (
    build_classification_report,
    build_confusion_matrix,
    build_model_comparison,
    evaluate_clustering,
    evaluate_predictions,
    extract_misclassified_examples,
)
from src.movie_review_understanding.evaluation.visualization import save_visualizations
from src.movie_review_understanding.features.tfidf import prepare_tfidf_splits
from src.movie_review_understanding.models.classifiers import train_classifiers
from src.movie_review_understanding.models.clustering import run_clustering
from src.movie_review_understanding.models.llm_classifier import (
    LLMConfigurationError,
    run_llm_experiments,
)

load_dotenv(PROJECT_ROOT / ".env", override=True)


def run_demo(skip_llm: bool = False, llm_sample_size: Optional[int] = None) -> None:
    """Run a terminal demo for preprocessing, TF-IDF, clustering, classification, and evaluation."""
    print("Intelligent Movie Review Understanding Demo")
    print("Stage: preprocessing + TF-IDF + clustering + classification + evaluation + LLM")

    try:
        reviews = load_reviews()
        tfidf_data = prepare_tfidf_splits(reviews)
    except FileNotFoundError as exc:
        print(f"Dataset status: {exc}")
        print("Next step: add an IMDb-style dataset file to data/raw.")
        print("Expected columns: review, sentiment")
        return

    clustering_result = run_clustering(tfidf_data)
    clustering_metrics = evaluate_clustering(clustering_result)
    classification_results = train_classifiers(tfidf_data)
    comparison = build_model_comparison(classification_results)
    saved_outputs = save_visualizations(classification_results, clustering_result)

    print(f"Loaded reviews: {len(reviews)}")
    print(f"Training samples: {len(tfidf_data.y_train)}")
    print(f"Test samples: {len(tfidf_data.y_test)}")
    print(f"TF-IDF feature count: {len(tfidf_data.vectorizer.get_feature_names_out())}")

    print()
    print(f"Clustering Algorithm: {clustering_metrics['algorithm_name']}")
    print(f"Cluster Count: {clustering_metrics['num_clusters']}")
    print(f"Cluster Sample Size: {clustering_metrics['sample_size']}")
    print(f"Silhouette Score: {clustering_metrics['silhouette_score']:.4f}")
    print(f"Cluster Sizes: {clustering_metrics['cluster_sizes']}")
    print("Cluster Sentiment Mix:")
    for cluster_id, sentiment_mix in clustering_metrics["cluster_sentiment_mix"].items():
        dominant = clustering_metrics["dominant_sentiment_per_cluster"][cluster_id]
        print(
            f"  Cluster {cluster_id}: {sentiment_mix} | dominant={dominant} | "
            f"top_terms={', '.join(clustering_result.top_terms[cluster_id])}"
        )

    print()
    print("Traditional Model Ranking:")
    for index, item in enumerate(comparison, start=1):
        print(
            f"  {index}. {item['model_name']} | accuracy={item['accuracy']:.4f} | "
            f"f1={item['f1']:.4f}"
        )

    for result in classification_results:
        metrics = evaluate_predictions(result)
        matrix, labels = build_confusion_matrix(result)

        print()
        print(f"Model: {metrics['model_name']}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"Confusion Matrix Labels: {labels}")
        print(matrix)
        print("Classification Report:")
        print(build_classification_report(result))

    best_model = comparison[0]
    best_result = next(result for result in classification_results if result.model_name == best_model["model_name"])
    misclassified_examples = extract_misclassified_examples(
        best_result,
        tfidf_data.test_texts,
        max_examples=DEFAULT_ERROR_ANALYSIS_SAMPLES,
    )

    print()
    print(f"Best Model: {best_model['model_name']} (Accuracy: {best_model['accuracy']:.4f})")
    print("Error Analysis Samples:")
    if misclassified_examples:
        for index, example in enumerate(misclassified_examples, start=1):
            snippet = example["text"][:160].replace("\n", " ")
            print(
                f"  {index}. true={example['true_label']}, predicted={example['predicted_label']}, "
                f"text='{snippet}...'"
            )
    else:
        print("  No misclassified examples found in the sampled output.")

    print("Saved Outputs:")
    for path in saved_outputs:
        print(f"  {path}")

    print()
    print("LLM Classification:")
    if skip_llm:
        print("  Skipped by --skip-llm.")
    else:
        try:
            llm_results = run_llm_experiments(
                tfidf_data,
                backend=DEFAULT_LLM_BACKEND,
                sample_size=llm_sample_size or DEFAULT_LLM_SAMPLE_SIZE,
            )
            for experiment in llm_results:
                metrics = evaluate_predictions(experiment.classification_result)
                print(
                    f"  Backend: {experiment.backend}, Prompt Style: {experiment.prompt_style}, "
                    f"Model: {experiment.model_name}, Sample Size: {experiment.sample_size}, "
                    f"Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}"
                )
        except LLMConfigurationError as exc:
            print(f"  Skipped: {exc}")
            print(
                f"  Best local option: Ollama with model {DEFAULT_OLLAMA_MODEL}. "
                f"Optional cloud option: OpenAI with model {DEFAULT_OPENAI_MODEL}."
            )
        except OpenAIError as exc:
            print("  Skipped: the configured LLM backend returned an API error.")
            print(f"  Error type: {type(exc).__name__}")
            print(f"  Error detail: {exc}")
            print("  Tip: check the API key, quota/billing, model name, or use --skip-llm.")
        except Exception as exc:
            print("  Skipped: the configured LLM backend failed unexpectedly.")
            print(f"  Error type: {type(exc).__name__}")
            print(f"  Error detail: {exc}")
            print("  Tip: use --skip-llm for the non-LLM demo path, or configure Ollama/OpenAI again.")

    print("Status: clustering, traditional ML, evaluation, visualization, demo flow, and LLM workflow are ready.")



