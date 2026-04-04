# Intelligent Movie Review Understanding

Course-project-level Python demo for movie review understanding and conversational recommendation support.

## Project Focus

This repository is structured for a Data Science final project with:

- text preprocessing
- TF-IDF feature engineering
- clustering
- sentiment classification with multiple algorithms
- LLM-based classification through API prompting
- evaluation, error analysis, and visualization
- a simple terminal demo

## Recommended Structure

- `data/`
  - `raw/`: original datasets such as IMDb
  - `processed/`: cleaned splits and intermediate artifacts
- `notebooks/`: optional exploration notebooks
- `reports/`
  - `figures/`: generated plots and charts
  - `metrics/`: saved evaluation summaries for reporting
- `scripts/`: runnable entry points for demo and future experiments
- `src/movie_review_understanding/`: reusable project code
  - `config/`: project settings and constants
  - `data/`: loading and preprocessing utilities
  - `features/`: TF-IDF and other text features
  - `models/`: clustering, ML classifiers, and LLM classifier modules
  - `evaluation/`: metrics, error analysis, and visualization helpers
  - `demo/`: terminal demo orchestration
- `tests/`: unit tests for reusable logic

## Best Way To Run

The best demo experience is:

1. Use the included IMDb dataset in `data/raw/imdb_reviews.csv`.
2. Run the terminal demo with Python 3.9+:

```powershell
py scripts/run_demo.py
```

3. For the LLM step, the preferred order is:
   - local `Ollama` with `qwen2.5:7b`
   - `OpenAI` API as an optional fallback
   - automatic skip if neither backend is available

This means the full project can still run for your instructor even if no LLM backend is configured.

## LLM Options

### Option A: Local Ollama (Recommended for demo)

Install and run Ollama, then pull a model such as:

```powershell
ollama pull qwen2.5:7b
```

The code will automatically try Ollama first.

### Option B: OpenAI API (Optional)

Create a local `.env` file from `.env.example` and set:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

If Ollama is not available, the project can fall back to OpenAI.

## Current Status

The project now includes a working baseline workflow for:

- IMDb dataset loading
- text preprocessing
- TF-IDF feature engineering
- K-Means clustering
- three traditional sentiment classification baselines
- evaluation metrics, error analysis, and visualizations
- optional LLM classification with graceful fallback

## Notes For Demo

- If no LLM backend is available, the demo will skip the LLM section and continue normally.
- Main outputs are saved under `reports/figures/` and `reports/metrics/`.
- This project is designed as a course demo, not a production application.
