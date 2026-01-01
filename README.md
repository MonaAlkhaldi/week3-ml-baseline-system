# Week 3 — ML Baseline System

This repository implements a minimal, reproducible machine learning baseline system using a CLI-first workflow.  
It covers data generation, model training, inference, evaluation, and reporting.

---

## Setup
NOTE: Make sure you are in the `week3-ml-baseline-system` folder 

### 1. Create and activate virtual environment
uv venv .venv  
.venv\Scripts\activate

### Start running
uv sync -----> Note: **This installs the exact dependency versions defined for the project**  
uv run ml-baseline --help -----> Note: **Shows all available commands and options**  
uv run pytest ------> Note: **This runs all automated tests using pytest**

### Sanity commands
uv run ml-baseline make-sample-data  
uv run ml-baseline train --target is_high_value  

**CAUTION**  
In this run, you will face an assertion about the target. Do not panic—this is only to show that the assertion works (I hope :))  :

uv run ml-baseline predict --run latest --input data/processed/features.csv --output outputs/preds.csv  

**SOLUTION**  
Run this instead:  
uv run ml-baseline predict --run latest --input data/processed/features_infer.csv --output outputs/preds.csv  

**If you are wondering why this error happened**  
- `data/processed/features.csv` is the **training feature table** and includes the target column (`is_high_value`).
- For inference, the system enforces schema validation and **rejects inputs containing the target** to prevent data leakage.
- Therefore, predictions are run using `data/processed/features_infer.csv`, which is identical to the training features **minus the target column**.


uv run pytest 



----------------------------------------------------------------------------------------------------------------------------------------------------------
### Colab Notebook
https://colab.research.google.com/drive/14pa9RNxukMnFPhn5fI1qO7Z0UnBzT_aD
  
