from dataclasses import dataclass

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from src.movie_review_understanding.config.settings import (
    DEFAULT_LABEL_COLUMN,
    DEFAULT_MAX_FEATURES,
    DEFAULT_NGRAM_RANGE,
    DEFAULT_TEST_SIZE,
    DEFAULT_TEXT_COLUMN,
    RANDOM_STATE,
)
from src.movie_review_understanding.data.preprocessing import batch_clean_text


@dataclass
class TfidfDatasetSplit:
    X_train: csr_matrix
    X_test: csr_matrix
    y_train: pd.Series
    y_test: pd.Series
    train_texts: list[str]
    test_texts: list[str]
    vectorizer: TfidfVectorizer


def build_tfidf_vectorizer(
    max_features: int = DEFAULT_MAX_FEATURES,
    ngram_range: tuple[int, int] = DEFAULT_NGRAM_RANGE,
) -> TfidfVectorizer:
    """Return a default TF-IDF vectorizer for review text."""
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words="english",
    )


def prepare_tfidf_splits(
    dataframe: pd.DataFrame,
    *,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> TfidfDatasetSplit:
    """Clean text, split the dataset, and create TF-IDF features."""
    if dataframe.empty:
        raise ValueError("The input dataframe is empty.")

    texts = batch_clean_text(dataframe[DEFAULT_TEXT_COLUMN].tolist())
    labels = dataframe[DEFAULT_LABEL_COLUMN].copy()

    train_texts, test_texts, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    vectorizer = build_tfidf_vectorizer()
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    return TfidfDatasetSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        train_texts=train_texts,
        test_texts=test_texts,
        vectorizer=vectorizer,
    )
