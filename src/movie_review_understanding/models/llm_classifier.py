import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd
from openai import OpenAI

from src.movie_review_understanding.config.settings import (
    DEFAULT_LLM_BACKEND,
    DEFAULT_LLM_MAX_OUTPUT_TOKENS,
    DEFAULT_LLM_PROMPT_STYLES,
    DEFAULT_LLM_SAMPLE_SIZE,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OPENAI_MODEL,
    RANDOM_STATE,
)
from src.movie_review_understanding.features.tfidf import TfidfDatasetSplit
from src.movie_review_understanding.models.classifiers import ClassificationResult


@dataclass
class LLMExperimentResult:
    prompt_style: str
    classification_result: ClassificationResult
    sample_size: int
    model_name: str
    backend: str


class LLMConfigurationError(ValueError):
    """Raised when the LLM module is not configured to call the API."""


def build_zero_shot_prompt(review_text: str) -> str:
    """Create a zero-shot sentiment classification prompt."""
    return (
        "You are a sentiment classifier for movie reviews. "
        "Return exactly one label: positive or negative.\n\n"
        f"Review:\n{review_text}\n\n"
        "Label:"
    )


def build_few_shot_prompt(review_text: str) -> str:
    """Create a few-shot sentiment classification prompt."""
    return (
        "You are a sentiment classifier for movie reviews. "
        "Return exactly one label: positive or negative.\n\n"
        "Examples:\n"
        "Review: This movie was touching, beautifully acted, and deeply memorable.\n"
        "Label: positive\n\n"
        "Review: The plot was dull, the acting was weak, and it felt far too long.\n"
        "Label: negative\n\n"
        "Review: Brilliant performances and a satisfying ending made this a joy to watch.\n"
        "Label: positive\n\n"
        "Review: I regret watching it because the script was messy and boring.\n"
        "Label: negative\n\n"
        f"Review: {review_text}\n"
        "Label:"
    )


def parse_sentiment_label(raw_text: str) -> str:
    """Normalize a raw model response to the project labels."""
    normalized = raw_text.strip().lower()
    if "positive" in normalized and "negative" not in normalized:
        return "positive"
    if "negative" in normalized and "positive" not in normalized:
        return "negative"
    if normalized in {"pos", "positive."}:
        return "positive"
    if normalized in {"neg", "negative."}:
        return "negative"
    raise ValueError(f"Could not parse sentiment label from response: {raw_text!r}")


def _extract_response_text(response) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text).strip()

    output_items = getattr(response, "output", [])
    for item in output_items:
        for content in getattr(item, "content", []):
            text_value = getattr(content, "text", None)
            if text_value:
                return str(text_value).strip()
    raise ValueError("No text content found in the API response.")


def _select_llm_subset(
    dataset: TfidfDatasetSplit,
    sample_size: int = DEFAULT_LLM_SAMPLE_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[List[str], pd.Series]:
    total_samples = len(dataset.test_texts)
    effective_size = min(sample_size, total_samples)
    sampled_texts = pd.Series(dataset.test_texts).sample(n=effective_size, random_state=random_state)
    sampled_labels = dataset.y_test.loc[sampled_texts.index].reset_index(drop=True)
    return sampled_texts.tolist(), sampled_labels


def _build_prompt(review_text: str, prompt_style: str) -> str:
    if prompt_style == "zero_shot":
        return build_zero_shot_prompt(review_text)
    if prompt_style == "few_shot":
        return build_few_shot_prompt(review_text)
    raise ValueError(f"Unsupported prompt style: {prompt_style}")


def _can_use_openai(api_key: Optional[str]) -> bool:
    return bool(api_key or os.getenv("OPENAI_API_KEY"))


def _can_use_ollama(base_url: str = DEFAULT_OLLAMA_BASE_URL) -> bool:
    try:
        client = OpenAI(base_url=base_url, api_key="ollama")
        client.models.list()
        return True
    except Exception:
        return False


def _resolve_backend(
    backend: str,
    api_key: Optional[str],
    ollama_base_url: str,
) -> str:
    if backend == "openai":
        if not _can_use_openai(api_key):
            raise LLMConfigurationError("OPENAI_API_KEY is not set.")
        return "openai"

    if backend == "ollama":
        if not _can_use_ollama(ollama_base_url):
            raise LLMConfigurationError("Ollama backend is not available.")
        return "ollama"

    if backend == "auto":
        if _can_use_ollama(ollama_base_url):
            return "ollama"
        if _can_use_openai(api_key):
            return "openai"
        raise LLMConfigurationError("No LLM backend is configured.")

    raise ValueError(f"Unsupported LLM backend: {backend}")


def _build_client(resolved_backend: str, api_key: Optional[str], ollama_base_url: str) -> Tuple[OpenAI, str]:
    if resolved_backend == "ollama":
        return OpenAI(base_url=ollama_base_url, api_key="ollama"), DEFAULT_OLLAMA_MODEL
    return OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY")), DEFAULT_OPENAI_MODEL


def classify_with_llm(
    dataset: TfidfDatasetSplit,
    prompt_style: str,
    api_key: Optional[str] = None,
    backend: str = DEFAULT_LLM_BACKEND,
    model_name: Optional[str] = None,
    sample_size: int = DEFAULT_LLM_SAMPLE_SIZE,
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL,
) -> LLMExperimentResult:
    """Classify a sampled subset with a configured LLM backend."""
    resolved_backend = _resolve_backend(backend, api_key, ollama_base_url)
    client, default_model = _build_client(resolved_backend, api_key, ollama_base_url)
    resolved_model_name = model_name or default_model

    texts, true_labels = _select_llm_subset(dataset, sample_size=sample_size)
    predictions = []

    for review_text in texts:
        prompt = _build_prompt(review_text, prompt_style)
        response = client.responses.create(
            model=resolved_model_name,
            input=prompt,
            max_output_tokens=DEFAULT_LLM_MAX_OUTPUT_TOKENS,
        )
        raw_output = _extract_response_text(response)
        predictions.append(parse_sentiment_label(raw_output))

    classification_result = ClassificationResult(
        model_name=f"LLM {prompt_style.replace('_', '-')} ({resolved_model_name})",
        predictions=pd.Series(predictions, name="prediction"),
        true_labels=true_labels,
    )
    return LLMExperimentResult(
        prompt_style=prompt_style,
        classification_result=classification_result,
        sample_size=len(texts),
        model_name=resolved_model_name,
        backend=resolved_backend,
    )


def run_llm_experiments(
    dataset: TfidfDatasetSplit,
    prompt_styles: Optional[List[str]] = None,
    api_key: Optional[str] = None,
    backend: str = DEFAULT_LLM_BACKEND,
    model_name: Optional[str] = None,
    sample_size: int = DEFAULT_LLM_SAMPLE_SIZE,
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL,
) -> List[LLMExperimentResult]:
    """Run the configured LLM prompt styles for sentiment classification."""
    styles = prompt_styles or list(DEFAULT_LLM_PROMPT_STYLES)
    return [
        classify_with_llm(
            dataset,
            prompt_style=style,
            api_key=api_key,
            backend=backend,
            model_name=model_name,
            sample_size=sample_size,
            ollama_base_url=ollama_base_url,
        )
        for style in styles
    ]
