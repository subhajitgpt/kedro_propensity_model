## Kedro-like Propensity Pipeline (Pure Python)

This repo contains a single-file, Kedro-inspired ML workflow implemented in plain Python.

It provides a minimal version of:
- **Node**: a function with declared inputs/outputs
- **Pipeline**: an ordered list of nodes
- **Catalog**: a dict of named datasets/objects shared between nodes
- **Runner**: executes nodes and materializes outputs back into the catalog

The main script builds a **synthetic customer dataset**, creates a binary target (`applied_cc`), trains models, evaluates **AUC**, performs **decile analysis**, and exports a **leads CSV**.

## Files

- `kedro_like_propensity_pipeline.py` — the pipeline framework + the end-to-end propensity workflow + CLI
- `kedro_like_propensity_pipeline_spark.py` — Spark / PySpark version of the same Kedro-style flow
- `requirements.txt` — runtime dependencies
- `requirements.spark.txt` — extra dependencies for the Spark version
- `pyproject.toml` — project metadata and dependency ranges
- `main.py` — placeholder “hello world” entrypoint (not used by the pipeline)

## Setup

### Python

`pyproject.toml` currently specifies `requires-python = ">=3.13"`.

### Install dependencies

Using `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

Or install from `pyproject.toml`:

```bash
python -m pip install .
```

Note: the script also contains a small helper that will attempt to auto-install `catboost` at runtime if it’s missing.

## Run

### List available pipelines

```bash
python kedro_like_propensity_pipeline.py --list-pipelines
```

### Run the full pipeline (default)

```bash
python kedro_like_propensity_pipeline.py
```

### Run a specific pipeline

```bash
python kedro_like_propensity_pipeline.py --pipeline baseline
python kedro_like_propensity_pipeline.py --pipeline train
python kedro_like_propensity_pipeline.py --pipeline full
```

### Reduce node-by-node logging

```bash
python kedro_like_propensity_pipeline.py --quiet
```

## Spark / PySpark version

This repo also includes a Spark-native pipeline that mirrors the same end-to-end nodes and pipeline names:

- `baseline`: baseline LogisticRegression on a validation split
- `train`: adds a tuned GBTClassifier + feature importance + top-k selection
- `full`: scores test, adds deciles, exports leads + metrics, and saves the Spark ML model

### Install Spark deps

```bash
python -m pip install -r requirements.spark.txt
```

You also need Java installed (so `java` works in your terminal).

### Run

```bash
python kedro_like_propensity_pipeline_spark.py --list-pipelines
python kedro_like_propensity_pipeline_spark.py --pipeline full
```

Defaults run locally with `--spark-master local[*]`. You can override:

```bash
python kedro_like_propensity_pipeline_spark.py --pipeline full --spark-master local[4]
```

### Spark outputs

Spark writes CSV outputs as *folders* (containing a `part-*.csv`):

- `kedro_style_leads_spark_csv/`
- `kedro_style_performance_metrics_spark_csv/`

And the model is saved as a Spark MLlib folder:

- `kedro_style_model_spark/`

## Pipelines

The script registers three runnable pipelines:

- `baseline`
	- Generates synthetic customers + target
	- Splits train/validation
	- Trains a **scaled LogisticRegression** model with:
		- numeric features scaled via `StandardScaler`
		- categorical features (`region`, `channel`) one-hot encoded
	- Prints validation AUC

- `train`
	- Runs everything in `baseline`
	- Trains an initial **CatBoostClassifier**
	- Computes feature importance
	- Selects top-*k* features (default `k=8`)
	- Runs a lightweight random search tuning loop (default `10` trials)
	- Prints best validation AUC and top trials
	- Does **not** score test or export leads

- `full`
	- Runs everything in `train`
	- Scores the held-out test split
	- Adds deciles based on predicted probability
	- Produces a decile lift table
	- Exports “leads” (customers in the top deciles)

## Outputs

When you run `--pipeline full`, you should see:

- Baseline AUC (LogisticRegression) on validation
- Best tuned CatBoost params + best validation AUC
- Test AUC (CatBoost on selected features)
- Feature importance (top rows)
- Decile table (customers, responders, response rate, lift)
- A sample of the top leads
- A timing snapshot for each node

### Leads CSV

By default the script writes a file next to the script:

- `kedro_style_leads.csv`

It contains:

- `customer_id`, `email`, `phone`
- the **selected top-k feature columns** used by the final tuned model
- `p_apply` (predicted propensity)
- `decile` (1–10)

It also includes **row-wise explainability** columns computed using CatBoost SHAP values:

- `exp_expected_value`
- `exp_1_feature`, `exp_1_contribution`, `exp_1_value`
- ... up to `exp_5_*` by default (configurable)

Additionally, it includes one SHAP contribution column per selected feature:

- `shap__<feature>` for each selected feature (e.g., `shap__annual_income`)

Leads are selected as customers with `decile >= 9` by default.

### Model pickle

When you run `--pipeline full`, the tuned CatBoost model (trained on selected features) is also pickled next to the script by default:

- `kedro_style_model.pkl`

### SHAP importance plot

When you run `--pipeline full`, the script also saves an overall SHAP feature importance plot (mean |SHAP| per feature) next to the script:

- `kedro_style_shap_importance.png`

### Performance metrics CSV

When you run `--pipeline full`, the script also writes a compact model performance report as a CSV next to the script:

- `kedro_style_performance_metrics.csv`

It includes metrics such as ROC AUC, PR AUC, log loss, accuracy, precision, recall, F1, and the confusion matrix counts.

By default the report includes:

- Multiple threshold rows (`0.3`, `0.5`, `0.7`) for both baseline validation and final test
- One additional row per split with the threshold chosen to maximize F1 (`*_best_f1`)

## Default parameters

Defaults are defined in `build_default_catalog()` inside `kedro_like_propensity_pipeline.py`:

- `n_customers`: 6000
- `seed`: 42
- `valid_size`: 0.2
- `top_k`: 8
- `tune_trials`: 10
- `tune_verbose`: True
- `min_decile_for_leads`: 9
- `leads_csv_path`: resolved to `kedro_style_leads.csv` next to the script
- `model_pickle_path`: resolved to `kedro_style_model.pkl` next to the script
- `explain_top_n`: 5
- `shap_importance_png_path`: resolved to `kedro_style_shap_importance.png` next to the script
- `decision_threshold`: 0.5
- `metrics_csv_path`: resolved to `kedro_style_performance_metrics.csv` next to the script
- `report_thresholds`: `[0.3, 0.5, 0.7]`
- `f1_threshold_grid_points`: 101

## Notes

- The dataset is **synthetic** (generated in `create_customers()`), intended for learning / interview-style practice.
- CatBoost uses categorical features by index; the script normalizes and string-casts categorical columns before building `Pool` objects.

## Flask UI

This repo includes a simple Flask UI to run the full pipeline and watch logs/artifacts live.

### Run (PowerShell + Anaconda)

If `python` is not found in your terminal, prepend your Anaconda paths first, then start the server:

```powershell
$env:Path = "C:\ProgramData\anaconda3;C:\ProgramData\anaconda3\Scripts;C:\ProgramData\anaconda3\Library\bin;$env:Path"
python flask_ui.py
```

Then open:

- http://127.0.0.1:5000/

If the **Run end-to-end pipeline** button feels “unresponsive”, check the **Logs** panel; the pipeline can take a bit of time (especially the CatBoost training/tuning), but the `/run` request should return immediately and logs should start streaming.

