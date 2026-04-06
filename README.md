# Intelligent Movie Review Understanding

Course-project-level Python demo for movie review understanding and conversational recommendation support.

## Project Focus

This repository is structured for a Data Science final project with:

- text preprocessing
- TF-IDF feature engineering
- clustering
- sentiment classification with multiple algorithms
- LLM-based classification through prompt/API-style calling
- evaluation, error analysis, and visualization
- a simple terminal demo

## Best Way To Run

For teammates, instructors, or Codex agents, use this order.

1. Install dependencies:

```powershell
py -m pip install -r requirements.txt
```

2. Prepare the IMDb dataset locally. See `Dataset Setup` below.

3. Run tests:

```powershell
py -m pytest
```

4. Run the fast validation path without LLM. This verifies preprocessing, TF-IDF, clustering, traditional ML, evaluation, and visualization:

```powershell
py scripts/run_demo.py --skip-llm
```

5. Run the full project demo with LLM only after one LLM backend is configured. See `Full LLM Demo Requirement` below:

```powershell
py scripts/run_demo.py --llm-sample-size 100
```

## Full LLM Demo Requirement

The project includes an LLM-based classification component, and the full project demo should run it. However, LLM execution requires one configured backend. The default `LLM_BACKEND=auto` mode uses this order:

1. Try local Ollama first.
2. If Ollama is not available, try OpenAI using `OPENAI_API_KEY`.
3. If neither backend is available, skip LLM gracefully and continue the non-LLM workflow.

This fallback prevents the demo from crashing on machines without LLM setup. It does not mean the LLM component is removed. For final verification of the LLM requirement, use Option A or Option B below.

Runtime note: `--llm-sample-size 100` runs both prompt styles (`zero_shot` and `few_shot`), so it can make about 200 LLM calls. Local Ollama on a GPU is usually much faster. Cloud API mode uses remote requests and can take noticeably longer; wait for it to finish or use `--llm-sample-size 10` for a quick check.

### Option A: Local Ollama Backend

Install Ollama and pull the local model used in this project:

```powershell
ollama pull qwen2.5:7b
```

Then run:

```powershell
py scripts/run_demo.py --llm-sample-size 100
```

For a quick LLM smoke test on slower machines, use a smaller sample:

```powershell
py scripts/run_demo.py --llm-sample-size 10
```

### Option B: Cloud API Backend

Create a private local `.env` file in the project root. Do not commit this file to GitHub.

```env
OPENAI_API_KEY=your_openai_api_key_here
```

If `LLM_BACKEND=auto` and Ollama is not available, the demo will automatically fall back to a configured cloud API key. In this project, the cloud backend is implemented with the OpenAI-compatible SDK path and uses `LLM_BACKEND=openai`. If you set `LLM_BACKEND=openai`, it will skip Ollama detection and use the API key directly:

```powershell
py scripts/run_demo.py --llm-sample-size 100
```

If you want to force the OpenAI backend, add this to your private `.env` file:

```env
LLM_BACKEND=openai
```

A real API key should never be hardcoded in Python files, committed to GitHub, or placed in `.env.example`. If another OpenAI-compatible provider or model is used, update the local `.env` settings and code configuration accordingly.

### Fallback: Skip LLM

If the machine has neither Ollama nor a configured API key, use:

```powershell
py scripts/run_demo.py --skip-llm
```

This is useful for teammates or graders who only want to validate the non-LLM pipeline. It is not the full LLM demo.

## Codex Agent Instruction

If using Codex to run this repository on a teammate's machine, give it this instruction:

```text
Read README.md. Prepare the dataset if needed. First run `py -m pytest`, then run `py scripts/run_demo.py --skip-llm`. Do not install Ollama automatically unless the user explicitly asks for it. Only run the LLM step with `--llm-sample-size 10` if Ollama is already installed or a compatible API key is already configured.
```

This avoids long CPU-only or accidental LLM setup runs.

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

1. Download and extract the original `aclImdb` dataset into `data/raw/aclImdb/`.
2. Run:

```powershell
py scripts/prepare_imdb_dataset.py
```

3. Run the fast validation path:

```powershell
py -m pytest
py scripts/run_demo.py --skip-llm
```

4. Optional full LLM path, only after an LLM backend is configured:

```powershell
py scripts/run_demo.py --llm-sample-size 100
```

## What The Demo Produces

Running the demo will generate:

- clustering summary in terminal output
- classifier metrics in terminal output
- error analysis samples in terminal output
- optional LLM classification metrics when an LLM backend is configured
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
- LLM classification with Ollama or OpenAI backend support
- graceful fallback when no LLM backend is configured

## Notes For Demo

- The repository includes generated figures and metric summaries, but not the full raw dataset.
- The full LLM demo requires local Ollama or a configured API key.
- `--skip-llm` is a compatibility fallback, not the full LLM demo.
- This project is designed as a course demo, not a production application.




