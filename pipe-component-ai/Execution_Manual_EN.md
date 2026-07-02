# Pipe Component AI Recognition Module — Execution Manual

> **pipe-component-ai** : A LightGBM-based AI module that automatically classifies
> piping components (valves / fittings / flanges, etc.) from the geometry, connector,
> and property features of BIM (Revit) family JSON data.

---

## 1. Overview

This module takes BIM JSON data extracted from Revit families as input and performs classification in two stages.

1. **Component group classification** — 15 groups such as VALVE, FITTING, FLANGE, GASKET, SUPPORT, INSTRUMENT, etc.
2. **Valve type classification** — when classified as VALVE, further into detailed types: BALL, GATE, GLOBE, BUTTERFLY, CHECK, etc.

The training pipeline is provided in two versions.

| Version | Label source | Data scope | Cross-validation | Output folder |
|---------|--------------|------------|------------------|---------------|
| **v1** (`run_pipeline.py`) | `학습용레이블_정답.xlsx` + filename fallback | Single folder (55 samples) | LeaveOneOut-CV | `models/` |
| **v2** (`run_pipeline_v2.py`) | `전문가_정의_csv/` CSV labels | TrainingModel multi-folder (181 models) | Stratified 5-Fold | `models_v2/` |

> **The currently recommended version is v2.** It is based on expert-defined CSV labels, has a larger dataset, and achieves higher accuracy (about 0.95).

---

## 2. Prerequisites

### 2.1 Runtime Environment

- **OS** : Windows (graphs are generated based on the Korean font `Malgun Gothic`)
- **Python** : 3.12 (based on the `cpython-312` cache in the code)
- For integration with C# / DevExpress WinForm, run this module as a FastAPI server and call it via REST API

### 2.2 Package Installation

Run the following in the `pipe-component-ai` folder.

```bat
cd pipe-component-ai
pip install -r requirements.txt
```

Key packages in `requirements.txt`:

```
lightgbm>=4.0.0      # Classification model
scikit-learn>=1.3.0  # Cross-validation, scaling, evaluation
pandas>=2.0.0        # Data processing
numpy>=1.24.0
fastapi>=0.110.0     # Inference API server
uvicorn[standard]    # ASGI server
pydantic>=2.0.0      # I/O schemas
joblib>=1.3.0        # Model serialization
matplotlib>=3.7.0    # Visualization
seaborn>=0.12.0
```

### 2.3 Data Folder Structure

The module uses `BIM모델_데이터/` in the parent folder as its data source.

```
배관ISO 논문 작업/
├── BIM모델_데이터/
│   ├── TrainingModel/            # v2 training folders per model (JSON bundles per model)
│   │   ├── {model1}/
│   │   │   ├── {component}.json
│   │   │   └── ...
│   │   └── {model2}/ ...
│   ├── 전문가_정의_csv/           # v2 label CSVs (model_name.csv)
│   │   └── {model}.csv           # Columns: FamilyAndTypeName, ItemKey, RevitId
│   ├── 학습용레이블_정답.xlsx     # v1 labels (ProductCode, SKY columns)
│   └── symbol.xml                # SKEY → component group / valve type mapping definitions
└── pipe-component-ai/            # This module
```

> **CSV label rules (v2)**
> - `ItemKey` is the SKEY code
> - `NC**` → undetermined item, excluded from training (recorded in `nc_pending_labels.csv`)
> - `_P**` → pipe type, excluded from training

---

## 3. How to Run

### 3.1 v2 Training Pipeline (Recommended)

The simplest way to run is the batch file.

```bat
cd pipe-component-ai
run_v2.bat
```

Or run directly:

```bat
python run_pipeline_v2.py
```

**Stages performed (automatic)**

1. Load `TrainingModel/` + `전문가_정의_csv/` → match features and labels
2. Separate and save NC** items (`models_v2/nc_pending_labels.csv`)
3. Build the feature matrix
4. Train with Stratified 5-Fold cross-validation
5. Train the final model on the full dataset and save
6. Generate evaluation reports (classification report, confusion matrix, feature importance)

**Outputs** (`models_v2/`)

| File | Description |
|------|-------------|
| `group_classifier.pkl` | Trained LightGBM group classification model |
| `group_scaler.pkl` | StandardScaler |
| `feature_cols.pkl` | List of feature columns used |
| `int_to_group.pkl` | Integer label → group name mapping |
| `dataset.csv` | Final training dataset |
| `classification_report.csv` | Per-class precision/recall/F1 |
| `confusion_matrix.csv` | Confusion matrix |
| `feature_importance.csv` | Feature importance |
| `fold_accuracies.csv` | Per-fold accuracy |
| `nc_pending_labels.csv` | List of undetermined (NC) items |

> Reference measured value: 5-Fold accuracy approximately **0.94 ~ 0.96** (mean ≈ 0.95)

### 3.2 Generating Visualization Graphs

After training, generate publication-ready graphs (PNG).

```bat
run_plots.bat
```

Or:

```bat
python generate_plots.py
```

Generated files (`models_v2/`): `confusion_matrix.png`, `feature_importance.png`, `fold_accuracy.png`, `classification_report.png`, `class_distribution.png`, `version_comparison.png`

### 3.3 v1 Training Pipeline (Reference)

```bat
python run_pipeline.py --data_dir ../BIM모델_데이터
```

**Main options**

| Option | Description |
|--------|-------------|
| `--data_dir` | (Required) BIM JSON folder path |
| `--model_dir` | Model/result output folder (default `models`) |
| `--ablation` | Perform per-feature-group ablation study |
| `--infer` | Run full batch inference on `data_dir` after training |
| `--output_csv` | Path to save batch inference results CSV |

Example:

```bat
python run_pipeline.py --data_dir ../BIM모델_데이터 --ablation --infer
```

> v1 measured values: group classification LOO-CV accuracy about 0.82, valve type about 0.38 (small dataset)

---

## 4. Running Inference (Prediction)

### 4.1 Batch Inference (CLI)

Predicts all JSON files in a folder at once.

```bat
python -m src.predict --json_dir <JSON_folder_path> --model_dir models --output_csv result.csv
```

| Option | Description |
|--------|-------------|
| `--json_dir` | (Required) Folder of JSON files to predict |
| `--model_dir` | Model folder (default `models`) |
| `--output_csv` | Path to save results CSV |

**Output fields** : `component_group`, `group_confidence`, `valve_type`, `type_confidence`, `review_required`, `review_reason`

**Confidence rules**

| Confidence | Status | Meaning |
|------------|--------|---------|
| ≥ 0.90 | auto | Automatic processing |
| 0.70 ~ 0.90 | review | Review recommended |
| < 0.70 | unknown | Low confidence |

### 4.2 FastAPI Inference Server

Use this when integrating with external programs such as C#/DevExpress WinForm.

```bat
cd pipe-component-ai
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

After starting, access the **Swagger UI** in a browser:

```
http://localhost:8000/docs
```

> The default model folder is `models/`. It can be changed with the `MODEL_DIR` environment variable
> (e.g., run after `set MODEL_DIR=models_v2`)

**API endpoints**

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Server status / whether model is loaded |
| POST | `/predict` | Feature vector (JSON) → classification result |
| POST | `/predict/file` | Upload Revit BIM JSON file → classification result |
| POST | `/predict/batch` | Upload multiple JSON files → batch results |
| GET | `/model/info` | Loaded model information |

**Example call (feature vector)**

```bat
curl -X POST http://localhost:8000/predict ^
     -H "Content-Type: application/json" ^
     -d @request_body.json
```

`/predict` is for passing the 24 numeric features extracted by a C# Revit Add-in directly,
while `/predict/file` extracts features on the server when the original BIM JSON is uploaded.

### 4.3 Validation Tool Server (Browser UI)

A tool to verify the trained v2 model directly in a browser.

```bat
run_ai_validator.bat
```

Or:

```bat
python ai_validator_server.py
```

Access in browser:

```
http://localhost:8765/ai_validator.html
```

**Provided APIs**

| Path | Description |
|------|-------------|
| `GET /api/status` | Server status |
| `GET /api/models` | List of subfolders under TrainingModel |
| `POST /api/predict/<model>` | Batch prediction over an entire model folder |
| `POST /api/predict_single` | Single JSON data prediction |

---

## 5. Typical Workflow

```
1. Install packages       →  pip install -r requirements.txt
2. Verify data prepared   →  BIM모델_데이터/TrainingModel, 전문가_정의_csv
3. v2 training            →  run_v2.bat
4. Generate graphs        →  run_plots.bat
5. Verify results         →  run_ai_validator.bat  (check in browser)
6. Production/integration →  uvicorn api.main:app  or  python -m src.predict
```

---

## 6. Frequently Encountered Issues

| Symptom | Cause / Solution |
|---------|------------------|
| `Model not loaded (503)` | Run training first, or verify that `MODEL_DIR` points to the correct model folder |
| `Group classifier not found` | No `*.pkl` model files in `models/` (or `models_v2/`) → re-run training |
| `JSON files not found` | Check the `--data_dir` / `--json_dir` path |
| `No valid training data` | CSV labels failed to match JSON FamilyName → check CSV column names/path |
| Korean text broken in graphs | Verify a Korean font such as `Malgun Gothic` is installed |
| v2 model not reflected in FastAPI server | Restart the server after `set MODEL_DIR=models_v2` |

---

*Document date: 2026-06-30*
