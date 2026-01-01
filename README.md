# Week 3 — ML Baseline System

This repository implements a minimal, reproducible machine learning baseline system using a CLI-first workflow.  
It covers data generation, model training, inference, evaluation, and reporting.

---

## Setup
NOTE: Make sure you are in the `week3-ml-baseline-system` folder 

### 1. Create and activate virtual environment
```
uv venv .venv  
.venv\Scripts\activate
```
### Start running
```uv sync``` -----> Note: **This installs the exact dependency versions defined for the project**  
```uv run ml-baseline --help``` -----> Note: **Shows all available commands and options**  
```uv run pytest``` ------> Note: **This runs all automated tests using pytest**

### Sanity commands
``` uv run ml-baseline make-sample-data```  
```uv run ml-baseline train --target is_high_value```  

**Heads up:**  
In this run ```uv run ml-baseline predict --run latest --input data/processed/features.csv --output outputs/preds.csv  ```, you will face an assertion about the target. Do not assume that the code is wrong — I have the solution:


**SOLUTION**  
First, run this :)  
```
python -c "import pandas as pd; df = pd.read_csv('data/processed/features.csv'); df.drop(columns=['is_high_value']).to_csv('data/processed/features_infer.csv', index=False); print('Created data/processed/features_infer.csv')"
```

Then run this: 
```
uv run ml-baseline predict --run latest --input data/processed/features_infer.csv --output outputs/preds.csv  
```

**If you are wondering why this error happened**  
- `data/processed/features.csv` is the **training feature table** and includes the target column (`is_high_value`).
- For inference, the system enforces schema validation and **rejects inputs containing the target** to prevent data leakage.
- Therefore, predictions are run using `data/processed/features_infer.csv`, which is identical to the training features **minus the target column**.

**Now we can move forward**

```uv run pytest ``` 
  
