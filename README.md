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
- `scripts/`: runnable entry points for demo and dataset preparation
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

1. Prepare the IMDb dataset locally.
2. Run the terminal demo with Python 3.9+:

```powershell
py scripts/run_demo.py
```

3. For the LLM step, the preferred order is:
   - local `Ollama` with `qwen2.5:7b`
   - `OpenAI` API as an optional fallback
   - automatic skip if neither backend is available

This means the full project can still run for your instructor or teammates even if no LLM backend is configured.

## Dataset Setup

The full raw IMDb dataset is not included in this repository because of size.

### Option A: Reuse an existing prepared CSV

If you already have a prepared dataset file, place it at:

```text
data/raw/imdb_reviews.csv
```

The project expects at least these columns:

- `review`
- `sentiment`

### Option B: Prepare the dataset from the original Stanford IMDb release

1. Download the Stanford IMDb Large Movie Review Dataset (`aclImdb`).
2. Extract it so the folder structure looks like:

```text
data/raw/aclImdb/
```

3. Run the preparation script:

```powershell
py scripts/prepare_imdb_dataset.py
```

This will generate:

```text
data/raw/imdb_reviews.csv
```

### If you are a teammate cloning this repository

After cloning the repository, the raw dataset will still be missing by design. To run the project locally:

1. Download and extract the original `aclImdb` dataset into `data/raw/aclImdb/`
2. Run:

```powershell
py scripts/prepare_imdb_dataset.py
```

3. Then run:

```powershell
py -m pytest
py scripts/run_demo.py
```

## Recommended Run Order

After the dataset is ready, use the following order:

```powershell
py -m pytest
py scripts/run_demo.py
```

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

## What The Demo Produces

Running the demo will generate:

- clustering summary in terminal output
- classifier metrics in terminal output
- error analysis samples in terminal output
- figures under `reports/figures/`
- summary tables under `reports/metrics/`

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

- The repository includes generated figures and metric summaries, but not the full raw dataset.
- If no LLM backend is available, the demo will skip the LLM section and continue normally.
- This project is designed as a course demo, not a production application.
