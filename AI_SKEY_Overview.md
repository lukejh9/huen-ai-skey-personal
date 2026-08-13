# AI_SKEY

> Technical overview of the BIM component classification pipeline, feature extraction, dataset loading, label mapping, prediction workflow, and current model performance.

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Project Structure](#2-project-structure)
3. [Feature Extraction](#3-feature-extraction)
4. [Geometry Features](#4-geometry-features)
5. [Connector Features](#5-connector-features)
6. [Current Numerical Feature List](#6-current-numerical-feature-list)
7. [Feature Groups](#7-feature-groups)
8. [Original BIM Dataset Loader](#8-original-bim-dataset-loader)
9. [Label Priority in `data_loader.py`](#9-label-priority-in-data_loaderpy)
10. [`data_loader.py` Output](#10-data_loaderpy-output)
11. [`get_feature_matrix`](#11-get_feature_matrix)
12. [TrainingModel Dataset](#12-trainingmodel-dataset)
13. [TrainingModel Structure](#13-trainingmodel-structure)
14. [Expert CSV Format](#14-expert-csv-format)
15. [CSV-to-JSON Matching](#15-csv-to-json-matching)
16. [SKEY Exclusions](#16-skey-exclusions)
17. [SKEY Label Mapping](#17-skey-label-mapping)
18. [`data_loader_csv.py` Output](#18-data_loader_csvpy-output)
19. [Dataset Loading Statistics](#19-dataset-loading-statistics)
20. [Component Groups](#20-component-groups)
21. [Machine Learning Classification](#21-machine-learning-classification)
22. [Valve Models](#22-valve-models)
23. [Prediction Pipeline](#23-prediction-pipeline)
24. [Current Model Evaluation](#24-current-model-evaluation)
25. [Overall Classification Performance](#25-overall-classification-performance)
26. [Current Performance Summary](#26-current-performance-summary)
27. [Current Pipeline Summary](#27-current-pipeline-summary)
28. [Main Files](#28-main-files)
29. [Current State](#29-current-state)


## 1. Project Overview

This project uses Revit BIM family JSON data to classify components into component groups and valve types using machine learning.

The overall pipeline consists of:

1.  Extracting numerical features from Revit BIM family JSON files. 
2.  Loading expert-defined labels from different sources. 
3.  Mapping SKEY codes to component groups and valve types. 
4.  Building pandas DataFrames for model training. 
5.  Training a machine-learning classification model. 
6.  Using `predict.py` for component predictions. 
7.  Using `valve_type.py` for valve-type classification. 
8.  Storing trained models and prediction outputs under `models_v2`. 

The project currently uses both the original BIM dataset and a separate `TrainingModel` dataset with expert-defined CSV labels.

---

# 2. Project Structure

The project currently contains the following major components:

```
project/
│
├── src/
│   ├── data_loader.py
│   ├── data_loader_csv.py
│   ├── feature_extractor.py
│   ├── label_loader.py
│   ├── label_extractor.py
│   ├── skey_mapper.py
│   └── ...
│
├── BIM모델_데이터/
│   ├── *.json
│   ├── 학습용레이블_정답.xlsx
│   └── symbol.xml
│
├── TrainingModel/
│   ├── {model_name}/
│   │   ├── component.json
│   │   └── ...
│   │
│   └── ...
│
├── 전문가_정의_csv/
│   ├── {model_name}.csv
│   └── ...
│
└── models_v2/
    ├── valve_models/
    ├── predictions/
    └── ...
```

`models_v2` currently contains both a `valve_models` folder and a `predictions` folder.

---

# 3. Feature Extraction

## `feature_extractor.py`

`feature_extractor.py` converts a Revit BIM family JSON file into a dictionary of numerical and metadata features.

The main function is:

```
extract_features(data)
```

A convenience function is also provided:

```
extract_features_from_file(json_path)
```

The feature extractor handles metadata, geometry, and connector information.

---

## 3.1 Metadata Features

The JSON contains Revit family and type information.

### String metadata

```
meta_family_name
meta_type_name
meta_category
meta_system_class
```

`meta_category` and `meta_system_class` are extracted from `InstanceParameters`.

The following numerical metadata features are also extracted:

```
meta_max_size_mm
meta_min_size_mm
```

These are parsed from values such as:

```
"40 mm"
```

into:

```
40.0
```

---

# 4. Geometry Features

Geometry information is extracted from the `Geometry -> Solid` section of each JSON file.

The following features are currently extracted:

```
geo_surface_area
geo_volume
geo_solid_count
geo_face_count
geo_bbox_x
geo_bbox_y
geo_bbox_z
geo_bbox_diag
geo_aspect_xy
geo_aspect_xz
geo_aspect_yz
geo_compactness
```

### `geo_surface_area`

Total surface area of all solids.

Revit's internal square-foot unit is converted to square meters.

```
ft² → m²
```

### `geo_volume`

Total volume of all solids.

Revit's internal cubic-foot unit is converted to cubic meters.

```
ft³ → m³
```

### `geo_solid_count`

Number of solids in the model.

### `geo_face_count`

Total number of faces across all solids.

### Bounding-box dimensions

The bounding box is calculated by iterating through tessellated points from the edges of the faces.

```
geo_bbox_x
geo_bbox_y
geo_bbox_z
```

All dimensions are converted from feet to meters.

### `geo_bbox_diag`

The bounding-box diagonal:

```
sqrt(x² + y² + z²)
```

### Aspect ratios

```
geo_aspect_xy = bbox_x / bbox_y
geo_aspect_xz = bbox_x / bbox_z
geo_aspect_yz = bbox_y / bbox_z
```

### `geo_compactness`

Currently calculated as:

```
volume / surface_area^1.5
```

This represents a measure of how compact the geometry is.

---

# 5. Connector Features

Connector information is extracted from the JSON `Connectors` array.

The current connector features are:

```
conn_count
conn_radius_mean
conn_radius_std
conn_radius_max
conn_radius_min
conn_radius_range
conn_distance
conn_angle_deg
conn_same_axis
conn_is_inline
```

## Connector radius

Connector radii are converted from feet to meters.

The following statistics are calculated:

```
conn_radius_mean
conn_radius_std
conn_radius_max
conn_radius_min
conn_radius_range
```

`conn_radius_range` is:

```
max_radius - min_radius
```

---

## Connector distance

When at least two connectors exist, the distance between the first two connector origins is calculated.

```
conn_distance
```

The result is converted to meters.

If fewer than two connectors exist, the value is:

```
0.0
```

---

## Connector angle

The angle between the direction vectors of the first two connectors is calculated in degrees.

```
conn_angle_deg
```

---

## Same-axis detection

Two connectors are considered to be on the same axis when their directions are approximately parallel or anti-parallel.

Current threshold:

```
angle < 10°
or
angle > 170°
```

The result is stored as:

```
conn_same_axis
```

where:

```
0 = not same axis
1 = same axis
```

---

## Inline detection

Two connectors are considered inline when they are approximately opposite in direction.

Current threshold:

```
angle > 170°
```

The result is:

```
conn_is_inline
```

where:

```
0 = not inline
1 = inline
```

---

# 6. Current Numerical Feature List

`NUMERIC_FEATURE_COLS` currently contains:

```
meta_max_size_mm
meta_min_size_mm

geo_surface_area
geo_volume
geo_solid_count
geo_face_count
geo_bbox_x
geo_bbox_y
geo_bbox_z
geo_bbox_diag
geo_aspect_xy
geo_aspect_xz
geo_aspect_yz
geo_compactness

conn_count
conn_radius_mean
conn_radius_std
conn_radius_max
conn_radius_min
conn_radius_range
conn_distance
conn_angle_deg
conn_same_axis
conn_is_inline
```

There are currently **24 numerical features**.

---

# 7. Feature Groups

The feature extractor also defines feature groups for ablation studies.

## Geometry

```
geo_surface_area
geo_volume
geo_solid_count
geo_face_count
geo_bbox_x
geo_bbox_y
geo_bbox_z
geo_bbox_diag
geo_aspect_xy
geo_aspect_xz
geo_aspect_yz
geo_compactness
```

## Connector

```
conn_count
conn_radius_mean
conn_radius_std
conn_radius_max
conn_radius_min
conn_radius_range
conn_distance
conn_angle_deg
conn_same_axis
conn_is_inline
```

## Metadata

```
meta_max_size_mm
meta_min_size_mm
```

---

# 8. Original BIM Dataset Loader

## `data_loader.py`

`data_loader.py` loads JSON files from the `BIM모델_데이터` directory and creates a pandas DataFrame containing both features and labels.

Example:

```
from src.data_loader import load_dataset

df = load_dataset(
    data_dir="BIM모델_데이터",
    xlsx_path="BIM모델_데이터/학습용레이블_정답.xlsx",
    xml_path="BIM모델_데이터/symbol.xml",
)
```

---

# 9. Label Priority in `data_loader.py`

The label priority is:

1. `학습용레이블_정답.xlsx` 
2. `symbol.xml` 
3. `component_group / valve_type` 
4.  Filename-based fallback through `label_extractor.py` 

The `LabelRegistry` is initialized using the Excel and XML files.

If the Excel file is unavailable, filename-based labeling is used.

---

# 10. `data_loader.py` Output

Each row contains metadata such as:

```
source_file
meta_family_name
meta_type_name
meta_category
meta_system_class
```

followed by all numerical features.

The label columns are:

```
skey
component_group
valve_type
is_valve
group_label
vtype_label
label_source
```

`group_label` is the integer representation of the component group.

`vtype_label` is the integer representation of the valve type.

---

# 11. `get_feature_matrix`

`data_loader.py` also provides:

```
get_feature_matrix(df, feature_cols=None)
```

This converts the loaded DataFrame into:

```
X
y_group
y_vtype
```

where:

```
X       = numerical feature DataFrame
y_group = group_label
y_vtype = vtype_label
```

If `feature_cols` is not specified, all features in `NUMERIC_FEATURE_COLS` are used.

---

# 12. TrainingModel Dataset

## `data_loader_csv.py`

A second dataset-loading pipeline was added for the `TrainingModel` directory.

This pipeline uses:

```
TrainingModel/
```

together with:

```
전문가_정의_csv/
```

The expert CSV files provide the labels for the models.

---

# 13. TrainingModel Structure

The structure is:

```
TrainingModel/
    {model_name}/
        component.json
        ...
```

Each model has a corresponding expert CSV:

```
전문가_정의_csv/
    {model_name}.csv
```

Only models that have a corresponding CSV file are processed.

Models without an expert CSV are excluded.

---

# 14. Expert CSV Format

The expert CSV contains:

```
FamilyAndTypeName
ItemKey
RevitId
```

### `FamilyAndTypeName`

Contains:

```
FamilyName: TypeName
```

### `ItemKey`

Contains the SKEY code used for labeling.

### `RevitId`

Contains the Revit ID associated with the item.

---

# 15. CSV-to-JSON Matching

`data_loader_csv.py` creates a lookup from:

```
FamilyAndTypeName → ItemKey
```

The lookup attempts matching in several ways.

First, it tries:

```
FamilyName: TypeName
```

Then:

```
FamilyName:TypeName
```

Then the family name by itself.

If exact matching fails, normalized matching is used.

Normalization removes differences involving:

- capitalization 
- whitespace 
- special characters 

The family name is also used for partial matching.

---

# 16. SKEY Exclusions

Two types of SKEY entries are currently excluded from training.

## NC\*\*

SKEYs beginning with:

```
NC
```

represent undetermined items.

These are excluded from training but recorded in `nc_samples`.

## `_P**`

SKEYs beginning with:

```
_P
```

are treated as pipe types and excluded from training.

The helper function is:

```
_is_excluded_skey(skey)
```

---

# 17. SKEY Label Mapping

After an expert CSV SKEY is found, it is passed to the SKEY mapper.

The relevant functions are:

```
parse_symbol_xml()
get_labels_from_skey()
```

The SKEY is converted into:

```
component_group
valve_type
is_valve
```

These are then converted into integer labels using:

```
GROUP_TO_INT
VTYPE_TO_INT
```

---

# 18. `data_loader_csv.py` Output

The resulting DataFrame contains:

```
source_model
source_file
meta_family_name
meta_type_name
meta_category
meta_system_class
```

all numerical features:

```
NUMERIC_FEATURE_COLS
```

and labels:

```
skey
component_group
valve_type
is_valve
group_label
vtype_label
```

The function returns:

```
df, nc_samples
```

where:

```
df = training DataFrame
nc_samples = excluded NC** samples
```

---

# 19. Dataset Loading Statistics

`data_loader_csv.py` tracks:

```
models_with_label
models_no_label
json_total
json_matched
json_nc_skip
json_pipe_skip
json_no_match
json_feat_error
```

The loader prints these statistics when `verbose=True`.

---

# 20. Component Groups

The current component-group classification includes:

```
VALVE
FITTING
FLANGE
GASKET
SUPPORT
INSTRUMENT
FLEXIBLE_JOINT
ACCESSORY
OLET
CAP
COUPLING
UNKNOWN
```

The model's reported evaluation contains these 12 classes.

---

# 21. Machine Learning Classification

The project currently performs component-group classification using the extracted numerical BIM features.

The model predicts the component group represented by:

```
group_label
```

A separate valve-type classification component is also present in the project.

---

# 22. Valve Models

The `models_v2` directory currently contains:

```
models_v2/
    valve_models/
```

The valve models are used for the valve-type portion of the classification pipeline.

A separate:

```
valve_type.py
```

file has been added for valve-type processing.

---

# 23. Prediction Pipeline

A:

```
predict.py
```

file has also been added.

It is used for making predictions from BIM JSON data using the trained models.

Prediction outputs are stored under:

```
models_v2/predictions/
```

---

# 24. Current Model Evaluation

The current component-group classification results are:

| ClassPrecisionRecallF1-scoreSupport |        |        |        |      |
| ----------------------------------- | ------ | ------ | ------ | ---- |
| VALVE                               | 0.9397 | 0.9601 | 0.9498 | 552  |
| FITTING                             | 0.9937 | 0.9949 | 0.9943 | 1576 |
| FLANGE                              | 0.8883 | 0.9091 | 0.8986 | 341  |
| GASKET                              | 0.9328 | 0.9487 | 0.9407 | 117  |
| SUPPORT                             | 0.9939 | 0.9980 | 0.9959 | 488  |
| INSTRUMENT                          | 0.9787 | 0.8519 | 0.9109 | 54   |
| FLEXIBLE\_JOINT                     | 0.8400 | 0.7636 | 0.8000 | 55   |
| ACCESSORY                           | 0.7273 | 0.5000 | 0.5926 | 16   |
| OLET                                | 0.8190 | 0.8407 | 0.8297 | 113  |
| CAP                                 | 0.9450 | 0.9601 | 0.9525 | 376  |
| COUPLING                            | 0.8824 | 0.7407 | 0.8054 | 81   |
| UNKNOWN                             | 0.7805 | 0.6957 | 0.7356 | 46   |

---

# 25. Overall Classification Performance

Current accuracy:

```
0.9567496723460026
```

or approximately:

```
95.67%
```

The current weighted averages are:

```
Precision: 0.9561
Recall:    0.9567
F1-score:  0.9561
```

The macro averages are:

```
Precision: 0.8934
Recall:    0.8470
F1-score:  0.8672
```

The evaluation contains:

```
3815 samples
```

---

# 26. Current Performance Summary

The strongest-performing classes currently include:

```
SUPPORT
FITTING
CAP
VALVE
GASKET
```

The current weaker-performing classes include:

```
ACCESSORY
UNKNOWN
FLEXIBLE_JOINT
COUPLING
OLET
```

The overall accuracy is approximately:

```
95.7%
```

while the macro F1-score is approximately:

```
86.7%
```

This difference reflects the effect of the smaller classes on macro-averaged performance.

---

# 27. Current Pipeline Summary

The current project pipeline can be summarized as:

```
Revit BIM Family
       │
       ▼
     JSON
       │
       ▼
feature_extractor.py
       │
       ├── Metadata
       ├── Geometry
       └── Connectors
       │
       ▼
24 Numerical Features
       │
       ▼
Dataset Loader
       │
       ├── data_loader.py
       │
       └── data_loader_csv.py
       │
       ▼
SKEY / Expert Labels
       │
       ▼
component_group
valve_type
is_valve
       │
       ▼
Machine Learning Model
       │
       ├── Component Classification
       │
       └── Valve Type Classification
       │
       ▼
predict.py
       │
       ▼
models_v2/predictions/
```

---

# 28. Main Files

| File | Purpose |
|---|---|
| `feature_extractor.py` | Extracts numerical and metadata features from BIM JSON |
| `data_loader.py` | Loads BIM JSON files and labels from Excel/XML with filename fallback |
| `data_loader_csv.py` | Loads `TrainingModel` JSON files using expert CSV labels |
| `label_loader.py` | Handles label registry and label extraction |
| `label_extractor.py` | Provides filename-based label fallback |
| `skey_mapper.py` | Maps SKEY codes to component groups and valve types |
| `predict.py` | Performs predictions using trained models |
| `valve_type.py` | Handles valve-type classification |
| `models_v2/valve_models/` | Stores valve-related trained models |
| `models_v2/predictions/` | Stores prediction outputs |

# 29. Current State

The project currently has:

- Revit BIM family JSON feature extraction. 
- 24 numerical features. 
- Geometry-based features. 
- Connector-based features. 
- Metadata size features. 
- Excel/XML-based labeling for the original BIM dataset. 
- Filename-based labeling fallback. 
- Expert CSV-based labeling for the `TrainingModel` dataset. 
- SKEY-based component-group and valve-type mapping. 
- NC\*\* exclusion. 
- `_P**` pipe exclusion. 
- Component-group classification. 
- Separate valve-type classification functionality. 
- `predict.py` prediction functionality. 
- `models_v2/valve_models/` for valve models. 
- `models_v2/predictions/` for prediction outputs. 
- Current component-group classification accuracy of approximately **95.67%** on 3,815 evaluated samples.
