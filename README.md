# Week 3 — ML Baseline System

This repository implements a minimal, reproducible machine learning baseline system using a CLI-first workflow.  
It covers data generation, model training, inference, evaluation, and reporting.

---

## Setup

### 1. Create and activate virtual environment
```powershell
uv venv .venv
.venv\Scripts\activate

### 2 instail requiermint.txt
   pip install -r requirements.txt

### 3
uv sync
uv run ml-baseline --help
uv run pytest

### 4  sanity commands
uv run ml-baseline make-sample-data
uv run ml-baseline train --target is_high_value
uv run ml-baseline predict --run latest --input data/processed/features.csv --output outputs/preds.csv
uv run pytest



----------------------------------------------------------------------------------------------------------------------------------------------------------
### Colab Notebook
https://colab.research.google.com/drive/14pa9RNxukMnFPhn5fI1qO7Z0UnBzT_aD
  
