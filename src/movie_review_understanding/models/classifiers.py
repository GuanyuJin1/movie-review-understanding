from dataclasses import dataclass
from typing import List

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from src.movie_review_understanding.features.tfidf import TfidfDatasetSplit


@dataclass
class ClassificationResult:
    model_name: str
    predictions: pd.Series
    true_labels: pd.Series


def train_naive_bayes_baseline(dataset: TfidfDatasetSplit) -> ClassificationResult:
    """Train and evaluate a Multinomial Naive Bayes baseline."""
    model = MultinomialNB()
    model.fit(dataset.X_train, dataset.y_train)
    predictions = pd.Series(model.predict(dataset.X_test), name="prediction")

    return ClassificationResult(
        model_name="Multinomial Naive Bayes",
        predictions=predictions,
        true_labels=dataset.y_test.copy(),
    )


def train_logistic_regression_baseline(dataset: TfidfDatasetSplit) -> ClassificationResult:
    """Train and evaluate a Logistic Regression baseline."""
    model = LogisticRegression(max_iter=1000)
    model.fit(dataset.X_train, dataset.y_train)
    predictions = pd.Series(model.predict(dataset.X_test), name="prediction")

    return ClassificationResult(
        model_name="Logistic Regression",
        predictions=predictions,
        true_labels=dataset.y_test.copy(),
    )


def train_linear_svm_baseline(dataset: TfidfDatasetSplit) -> ClassificationResult:
    """Train and evaluate a Linear SVM baseline."""
    model = LinearSVC()
    model.fit(dataset.X_train, dataset.y_train)
    predictions = pd.Series(model.predict(dataset.X_test), name="prediction")

    return ClassificationResult(
        model_name="Linear SVM",
        predictions=predictions,
        true_labels=dataset.y_test.copy(),
    )


def train_classifiers(dataset: TfidfDatasetSplit) -> List[ClassificationResult]:
    """Train the traditional classifier baselines used in the project."""
    return [
        train_naive_bayes_baseline(dataset),
        train_logistic_regression_baseline(dataset),
        train_linear_svm_baseline(dataset),
    ]
