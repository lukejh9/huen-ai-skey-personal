# Pipe Component AI Recognition Module — Source Code Structure Guide

> A document summarizing the directory layout, role of each module, data flow,
> and core algorithms of **pipe-component-ai**.

---

## 1. Overall Directory Structure

```
pipe-component-ai/
├── requirements.txt              # Dependencies
│
├── src/                          # Core library (feature extraction · labeling · training · inference)
│   ├── __init__.py
│   ├── feature_extractor.py      # BIM JSON → 24 numeric features
│   ├── skey_mapper.py            # symbol.xml SKEY → group/valve type mapping
│   ├── label_extractor.py        # Filename-based label rules (fallback)
│   ├── label_loader.py           # xlsx ground-truth label loader (v1)
│   ├── data_loader.py            # JSON+xlsx → DataFrame (v1)
│   ├── data_loader_csv.py        # TrainingModel+CSV → DataFrame (v2)
│   ├── train_group.py            # Component group classifier training (v1)
│   ├── train_type.py             # Valve type classifier training (v1)
│   ├── predict.py                # Trained-model inference engine (PipeComponentPredictor)
│   └── evaluate.py               # Evaluation & visualization utilities
│
├── api/                          # Inference REST API
│   ├── __init__.py
│   └── main.py                   # FastAPI server
│
├── models/                       # v1 training artifacts (.pkl, reports, graphs)
├── models_v2/                    # v2 training artifacts (.pkl, reports, graphs)
│
├── run_pipeline.py               # v1 full pipeline entry point
├── run_pipeline_v2.py            # v2 full pipeline entry point (recommended)
├── generate_plots.py             # v2 results visualization script
├── ai_validator_server.py        # Browser validation tool server
│
├── run_v2.bat                    # v2 training launch batch
├── run_plots.bat                 # Graph generation batch
├── run_ai_validator.bat          # Validation server launch batch
│
├── make_ppt.py                   # Explanatory PPT generation
├── AI모델_설명자료.pptx
└── (update_paper_v16.py etc.)    # Auxiliary scripts for paper manuscript processing
```

> **Layer separation**
> - `src/` : Reusable library (domain logic)
> - Root `run_*.py` : Pipeline orchestration (entry points)
> - `api/` : External integration (REST)
> - `models*/` : Training result artifacts

---

## 2. Core Module Details

### 2.1 `src/feature_extractor.py` — Feature Extraction

Module that converts BIM (Revit) family JSON into machine learning input vectors. This is the starting point of the domain logic.

**Main functions**

- `extract_features(data: dict) -> dict` : JSON dict → feature dictionary
- `extract_features_from_file(json_path) -> dict` : File path version
- `NUMERIC_FEATURE_COLS` : List of 24 numeric feature columns used for training
- `FEATURE_GROUPS` : Feature groups for ablation study (geometry/connector/metadata)

**Extracted features (3 families, 24 total)**

| Family | Features | Description |
|--------|----------|-------------|
| **Metadata (2)** | `meta_max_size_mm`, `meta_min_size_mm` | Max/min bore diameter (mm) |
| **Geometry (12)** | `geo_surface_area`, `geo_volume`, `geo_solid_count`, `geo_face_count`, `geo_bbox_x/y/z`, `geo_bbox_diag`, `geo_aspect_xy/xz/yz`, `geo_compactness` | Surface area · volume · solid/face counts · bounding box · aspect ratios · compactness |
| **Connector (10)** | `conn_count`, `conn_radius_mean/std/max/min/range`, `conn_distance`, `conn_angle_deg`, `conn_same_axis`, `conn_is_inline` | Connector count · radius statistics · distance/angle between connectors · same-axis/inline flags |

**Unit conversion** : Revit internal units (feet) → metric
- Length × 0.3048, area × 0.3048², volume × 0.3048³

**Core domain logic**
- The bounding box is computed by iterating over Face → EdgeLoop → Tessellate points of all solids and taking min/max
- When there are 2+ connectors, the angle between their direction vectors determines `same_axis` (parallel, <10°/>170°) and `is_inline` (opposite direction, >170°) → a key cue for distinguishing inline components vs. elbows

---

### 2.2 `src/skey_mapper.py` — SKEY Label Mapping

A rule engine that converts the SKEY (symbol key) from `symbol.xml` into training labels (component group · valve type).

**Class system**

- `GROUP_CLASSES` (15) : VALVE, FITTING, FLANGE, GASKET, SLEEVE, SUPPORT, INSTRUMENT, FLEXIBLE_JOINT, ACCESSORY, OLET, CAP, COUPLING, TRAP, PIPE, UNKNOWN
- `VALVE_TYPE_CLASSES` (22) : BALL, GATE, GLOBE, BUTTERFLY, CHECK, CONTROL, SAFETY, PRESSURE_REDUCING, DIAPHRAGM, COCK, NEEDLE, PLUG, SLIDE, THREE_WAY, FOUR_WAY, KNIFE_GATE, BASIC, BREATHER, ANGLE, AUTO, UNKNOWN, NON_VALVE
- `GROUP_TO_INT`/`INT_TO_GROUP`, `VTYPE_TO_INT`/`INT_TO_VTYPE` : Integer encoding mappings

**Mapping priority (`_derive_labels`)**

1. **SKEY override** (`_SKEY_OVERRIDE`) — prefix exact/start matching (e.g., EX→FLEXIBLE_JOINT, SLVE→SLEEVE)
2. **PCF_Identification based** (`_PCF_TO_GROUP`) — VALVE/FLANGE/ELBOW/TEE, etc.
   - Ambiguous items such as MISC-COMPONENT are refined via `_refine_misc()` based on Description
3. **Category text fallback** — string containment check
4. If it is a valve group, `_derive_valve_type()` maps via `_SKEY_TO_VALVE_TYPE` (VB→BALL, VT→GATE, etc.)

**Main functions**
- `parse_symbol_xml(xml_path) -> {skey: SkeyInfo}` : parse symbol.xml
- `get_labels_from_skey(skey_value, skey_map) -> dict` : SKEY value → label (exact → prefix → rule fallback)

---

### 2.3 `src/label_extractor.py` — Filename-based Labels (fallback)

A regex rule engine that estimates labels from the **text patterns of FamilyName/filename** when xlsx/CSV labels are unavailable.

- `_GROUP_RULES` : "BALL VALVE"→VALVE, "GASKET"→GASKET, "PIPE SHOE"→SUPPORT, etc. (more specific patterns first)
- `_VALVE_TYPE_RULES` : valve detail type regex
- `extract_component_group(name)`, `extract_valve_type(name)`, `extract_labels(name)`

> The class system in this module is simpler than `skey_mapper` (10 groups · 11 valve types, an early version). It serves as the last safety net for v1.

---

### 2.4 Data Loaders — `data_loader.py` (v1) / `data_loader_csv.py` (v2)

| | v1 `data_loader.py` | v2 `data_loader_csv.py` |
|---|---|---|
| Input data | `*.json` in a single folder | `TrainingModel/{model}/*.json` |
| Label source | `학습용레이블_정답.xlsx` (+ filename fallback) | `전문가_정의_csv/{model}.csv` |
| Label mapping | `LabelRegistry` → SKEY → group | `_find_skey` → SKEY → group |
| Exclusion handling | — | Excludes `NC**` (undetermined), `_P**` (pipe) |
| Return | `DataFrame` | `(DataFrame, nc_samples)` |

**Common output columns** : `source_file`, `meta_*`, 24 numeric features, `skey`, `component_group`, `valve_type`, `is_valve`, `group_label` (int), `vtype_label` (int)

**v2 matching logic (`_find_skey`)** : Finds the SKEY in the order "FamilyName: TypeName" combination → exact match → normalized (whitespace/special chars removed) match → partial match.

---

### 2.5 `src/label_loader.py` — xlsx Ground-truth Label Loader (v1)

The `LabelRegistry` class that reads `학습용레이블_정답.xlsx` and builds a (FamilyName, TypeName) → label lookup.

- Automatic column detection (`ProductCode`/`SKY` candidate sets, case- and Korean-insensitive)
- `_parse_product_code` : splits "FamilyName: TypeName"
- `get_labels(family, type)` : multi-stage lookup of exact → partial → family match
- `get_labels_from_filename(stem)` : decomposes family/type from filename then looks up
- `extract_labels_with_fallback(...)` : registry first, falls back to `label_extractor` on failure

---

### 2.6 Training Modules — `train_group.py` / `train_type.py` (v1)

**`train_group_classifier(df, ...)`**
- Model : LightGBM (`LGBM_PARAMS`, optimized for small data — num_leaves=8, max_depth=4, strong regularization, class_weight=balanced)
- Cross-validation : samples < 30 or `loo` → LeaveOneOut, otherwise StratifiedKFold-5
- Procedure : per-fold StandardScaler fit → model fit → accumulate predictions → CV accuracy/classification report
- Saving : retrain on the full dataset then save `group_classifier.pkl`, `group_scaler.pkl`, `group_features.pkl`
- `run_ablation_study(df)` : compares accuracy across geometry/connector/metadata combinations

**`train_type.py`** : trains the valve type classifier using only VALVE-group samples (same structure)

> v2 does not use a separate training module; the training logic is embedded inside `run_pipeline_v2.py` (see 3.2 below).

---

### 2.7 `src/predict.py` — Inference Engine

**`PipeComponentPredictor` class**

- The constructor loads the group classifier (required) and valve type classifier (optional) from `model_dir`
- `predict_features(features)` : feature dict → group classification → (if VALVE) valve type classification → review decision based on confidence
- `predict_file(json_path)` : single JSON → extract features then predict
- `predict_batch(json_dir, output_csv)` : batch prediction over a folder + CSV save + summary

**Confidence rules** : ≥0.90 auto / 0.70~0.90 review / <0.70 unknown
→ `review_required`, `review_reason` flag items that need human review (Human-in-the-loop)

---

### 2.8 `src/evaluate.py` — Evaluation & Visualization

- `plot_feature_importance` : feature importance bar chart
- Confusion matrix, ablation result visualization, classification report output
- Automatic Korean font setup (Malgun Gothic, etc.), `Agg` backend to support headless environments

---

### 2.9 `api/main.py` — FastAPI Inference Server

- Loads `PipeComponentPredictor` globally at app startup (`startup`) (model folder via the `MODEL_DIR` environment variable)
- `FeatureInput` (Pydantic) : 24-feature input schema + example values
- `PredictionResult` : classification result output schema
- Endpoints : `/health`, `/predict`, `/predict/file`, `/predict/batch`, `/model/info`
- CORS fully open → integrable with web/desktop clients

---

## 3. Entry Points (Pipelines)

### 3.1 `run_pipeline.py` (v1)

```
[1] load_dataset (JSON+xlsx)  →  feature stats / label distribution / matching result CSV
[2] train_group_classifier    →  group classifier
[3] train_type_classifier     →  valve type classifier
[4] generate_evaluation_report
[5] (optional) Ablation Study
[6] (optional) Batch inference
```
CLI options: `--data_dir` (required), `--model_dir`, `--ablation`, `--infer`, `--output_csv`

### 3.2 `run_pipeline_v2.py` (v2, recommended)

```
[1] load_dataset_csv (TrainingModel + expert CSV)  →  NC separation, dataset.csv
[2] Prepare feature matrix (NUMERIC_FEATURE_COLS)
[3] StratifiedKFold(5) training  →  per-fold accuracy
[4] Train final model on full dataset  →  models_v2/*.pkl
[5] Evaluation reports  →  classification_report / confusion_matrix / feature_importance / fold_accuracies CSV
```

**v2 LightGBM parameters (summary)** : num_class=15, num_leaves=31, max_depth=6, learning_rate=0.05, n_estimators=300, class_weight=balanced, early_stopping(30). Compared to v1, the model capacity is increased due to the larger dataset.

---

## 4. Data Processing Pipeline (Overall Flow)

```
[Revit family]
      │  (BIM JSON export)
      ▼
 component.json ──► feature_extractor.extract_features ──► 24 numeric features
                                                              │
 expert CSV / xlsx ─► skey_mapper / label_loader ─► labels (group · valve type)
                                                              │
                                  data_loader(_csv) combines features + labels
                                                              ▼
                                                      pandas DataFrame
                                                              │
                            train (StratifiedKFold / LOO) + StandardScaler
                                                              ▼
                              LightGBM model (.pkl) + Scaler + feature list
                                                              │
                             ┌────────────────┴────────────────┐
                             ▼                                  ▼
                  predict.PipeComponentPredictor        api/main.py (FastAPI)
                  (CLI batch inference)                  (REST inference)
                             │                                  │
                             ▼                                  ▼
              component_group / valve_type / confidence / review_required
```

---

## 5. Key Design Points (Research Perspective)

1. **Three-family feature design** — Rather than 2D drawing OCR, features are extracted directly from BIM 3D geometry, connectors, and metadata to identify components. In particular, connector geometry (count · radius · same-axis/inline) is a key cue for distinguishing valves vs. fittings.
2. **Multi-stage label mapping** — The symbol.xml SKEY system is normalized into standard labels (`skey_mapper`), and label coverage is secured via fallback in the order expert CSV → xlsx → filename rules.
3. **Small → expanded data handling** — v1 uses LeaveOneOut-CV (55 samples), v2 uses Stratified 5-Fold (181 models); the cross-validation strategy switches according to data scale.
4. **Human-in-the-loop** — Confidence thresholds (0.90/0.70) separate auto/review/unknown, and the validation server (`ai_validator`) lets humans verify results.
5. **Integrability** — A FastAPI REST structure that can be decoupled from and integrated with external applications such as C#/DevExpress WinForm.

---

## 6. Module Dependency Summary

```
run_pipeline_v2.py
 ├─ data_loader_csv ─┬─ feature_extractor
 │                   └─ skey_mapper
 └─ skey_mapper

run_pipeline.py (v1)
 ├─ data_loader ─┬─ feature_extractor
 │               ├─ label_loader ─┬─ skey_mapper
 │               │                └─ label_extractor
 │               └─ skey_mapper
 ├─ train_group / train_type ─ data_loader, feature_extractor, skey_mapper
 └─ evaluate

api/main.py
 └─ predict ─┬─ feature_extractor
             └─ skey_mapper
```

---

*Document date: 2026-06-30*
