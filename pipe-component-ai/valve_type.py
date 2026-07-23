import sys
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)

_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.skey_mapper import (
    NUM_VTYPE_CLASSES, INT_TO_VTYPE,
    )

# ── 경로 설정 ─────────────────────────────────
_BIM_BASE          = _ROOT.parent / "BIM_Data"
TRAINING_MODEL_DIR = _BIM_BASE / "TrainingModel"
CSV_LABEL_DIR      = _BIM_BASE / "User_Defined_csv"
XML_PATH           = _BIM_BASE / "symbol.xml"
MODEL_DIR          = _ROOT / "models_v2"
VALVE_MODEL_DIR    = MODEL_DIR / "valve_models"

# ── LightGBM 파라미터 ─────────────────────────
LGBM_PARAMS = {
    "objective":         "multiclass",
    "metric":            "multi_logloss",
    "num_class":         NUM_VTYPE_CLASSES,
    "boosting_type":     "gbdt",
    "num_leaves":        31,
    "max_depth":         6,
    "learning_rate":     0.05,
    "n_estimators":      300,
    "min_child_samples": 2,
    "reg_alpha":         0.3,
    "reg_lambda":        1.0,
    "colsample_bytree":  0.8,
    "subsample":         0.8,
    "subsample_freq":    1,
    "class_weight":      "balanced",
    "random_state":      42,
    "verbose":           -1,
    "n_jobs":            -1,
}

ZERO_IMPORTANCE_FEATURES_VALVE = [
    "conn_count",
    "conn_radius_range",
    "conn_radius_std",
    "conn_angle_deg",
    "conn_radius_ratio",
    "conn_same_axis",
    "conn_is_inline",
    "name_kw_valve",
    "name_kw_strainer_inline",
]

N_SPLITS = 5  # Stratified K-Fold 분할 수


# ─────────────────────────────────────────────
def run_valve_pipeline(df, feature_cols, model = lgb.LGBMClassifier(**LGBM_PARAMS)) -> dict:
    print("")
    print("Valve Classifier:")

    feature_cols = [c for c in feature_cols if c not in ZERO_IMPORTANCE_FEATURES_VALVE]

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    
    valve_rows = df[df["is_valve"] == True]
    X_valves = valve_rows[feature_cols].values
    y_valves = valve_rows["vtype_label"].values

    fold_accs = []
    all_y_true = []
    all_y_pred = []
    feature_importance_sum = np.zeros(len(feature_cols))

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_valves, y_valves), 1):
        X_tr, X_val = X_valves[train_idx], X_valves[val_idx]
        y_tr, y_val = y_valves[train_idx], y_valves[val_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)

        model.fit(
            X_tr_s, y_tr,
            eval_set=[(X_val_s, y_val)],
            callbacks=[lgb.early_stopping(30, verbose=False),
                       lgb.log_evaluation(period=-1)],
        )

        y_pred = model.predict(X_val_s)
        acc = accuracy_score(y_val, y_pred)
        fold_accs.append(acc)
        all_y_true.extend(y_val.tolist())
        all_y_pred.extend(y_pred.tolist())
        feature_importance_sum += model.feature_importances_

        print(f"  Fold {fold_idx}/{N_SPLITS}: Val Acc = {acc:.4f}  "
              f"(val_size={len(y_val)})")
        
    mean_acc = np.mean(fold_accs)
    std_acc  = np.std(fold_accs)
    print(f"\n  CV (Valve) 결과: {mean_acc:.4f} ± {std_acc:.4f}")
    print("")
    return {"df": df,
            "feature_cols": feature_cols,
            "fold_accs": fold_accs,
            "all_y_true": all_y_true,
            "all_y_pred": all_y_pred,
            "feature_importance_sum": feature_importance_sum,
            }

def valve_report(df, feature_cols, fold_accs, all_y_true, all_y_pred, feature_importance_sum):
    print("-" * 65)
    X = df[feature_cols].fillna(0.0).values
    y = df["vtype_label"].values
    final_scaler = StandardScaler()
    X_all = final_scaler.fit_transform(X)

    final_model = lgb.LGBMClassifier(**LGBM_PARAMS)
    final_model.fit(X_all, y)

    # 모델 저장
    joblib.dump(final_model,   VALVE_MODEL_DIR / "group_classifier.pkl")
    joblib.dump(final_scaler,  VALVE_MODEL_DIR / "group_scaler.pkl")
    joblib.dump(feature_cols,  VALVE_MODEL_DIR / "feature_cols.pkl")
    joblib.dump(INT_TO_VTYPE,  VALVE_MODEL_DIR / "int_to_group.pkl")

    # 클래스 레이블 정리
    unique_labels = sorted(set(all_y_true))
    label_names   = [INT_TO_VTYPE.get(i, str(i)) for i in unique_labels]

    # Classification report
    cr = classification_report(
        all_y_true, all_y_pred,
        labels=unique_labels,
        target_names=label_names,
        output_dict=True,
    )
    cr_df = pd.DataFrame(cr).T
    cr_df.to_csv(VALVE_MODEL_DIR / "classification_report.csv", encoding="utf-8-sig")
    print("\n  분류 리포트 (CV 전체):")
    print(classification_report(
        all_y_true, all_y_pred,
        labels=unique_labels,
        target_names=label_names,
    ))

    # Confusion matrix
    cm = confusion_matrix(all_y_true, all_y_pred, labels=unique_labels)
    cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
    cm_df.to_csv(VALVE_MODEL_DIR / "confusion_matrix.csv", encoding="utf-8-sig")

    # Feature importance
    fi_df = pd.DataFrame({
        "feature":    feature_cols,
        "importance": feature_importance_sum / N_SPLITS,
    }).sort_values("importance", ascending=False)
    fi_df.to_csv(VALVE_MODEL_DIR / "feature_importance.csv",
                 index=False, encoding="utf-8-sig")
    print(f"\n  피처 중요도 (Valve) Top 10:")
    print(fi_df.head(10).to_string(index=False))

    # Fold별 accuracy 저장
    fold_df = pd.DataFrame({
        "fold":     list(range(1, N_SPLITS + 1)),
        "accuracy": fold_accs,
    })
    fold_df.to_csv(VALVE_MODEL_DIR / "fold_accuracies.csv",
                   index=False, encoding="utf-8-sig")
    print("-" * 65)



if __name__ == "__main__":
    result = run_valve_pipeline()
