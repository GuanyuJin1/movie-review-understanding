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

## Best Way To Run

For teammates, instructors, or Codex agents, use this order.

1. Install dependencies:

```powershell
py -m pip install -r requirements.txt
```

2. Prepare the IMDb dataset locally. See `Dataset Setup` below.

3. Run the non-LLM demo first. This is the safest command for CPU-only machines:

```powershell
py scripts/run_demo.py --skip-llm
```

4. If the machine has local Ollama with a usable GPU, run a small LLM demo:

```powershell
py scripts/run_demo.py --llm-sample-size 10
```

5. For the full local demo used in this project, run:

```powershell
py scripts/run_demo.py --llm-sample-size 100
```

The LLM step is optional. If no LLM backend is available, the project can still run the traditional ML, clustering, evaluation, and visualization workflow.

## Codex Agent Instruction

If using Codex to run this repository on a teammate's machine, give it this instruction:

```text
Read README.md. Prepare the dataset if needed. First run `py -m pytest`, then run `py scripts/run_demo.py --skip-llm`. Only run the LLM step with `--llm-sample-size 10` if Ollama is installed and GPU acceleration is available.
```

This avoids long CPU-only Ollama runs.

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

### If You Are A Teammate Cloning This Repository

After cloning the repository, the raw dataset will still be missing by design. To run the project locally:

1. Download and extract the original `aclImdb` dataset into `data/raw/aclImdb/`
2. Run:

```powershell
py scripts/prepare_imdb_dataset.py
```

3. Run the fast validation path:

```powershell
py -m pytest
py scripts/run_demo.py --skip-llm
```

4. Optional LLM check, only if Ollama is ready:

```powershell
ollama pull qwen2.5:7b
py scripts/run_demo.py --llm-sample-size 10
```

## LLM Options

### Option A: Local Ollama (Recommended For Demo)

Install and run Ollama, then pull a model such as:

```powershell
ollama pull qwen2.5:7b
```

The code will automatically try Ollama first. If the machine is CPU-only, use `--skip-llm` or a small sample size such as `--llm-sample-size 5`.

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
