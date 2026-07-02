"""
train_type.py
-------------
밸브 타입 분류기 (BALL / GATE / GLOBE / BUTTERFLY / CHECK / CONTROL / SAFETY / AUTO)
LightGBM 학습.

VALVE 그룹 샘플만 사용하여 밸브 세부 타입을 분류한다.

실행:
    python -m src.train_type --data_dir <JSON폴더> --model_dir models/
"""

import argparse
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from src.data_loader import load_dataset, get_feature_matrix
from src.feature_extractor import NUMERIC_FEATURE_COLS, FEATURE_GROUPS
from src.skey_mapper import INT_TO_VTYPE, VALVE_TYPE_CLASSES, NUM_VTYPE_CLASSES


# ─────────────────────────────────────────────
#  LightGBM 파라미터 (소규모 데이터 최적화)
# ─────────────────────────────────────────────

LGBM_PARAMS = {
    "objective":          "multiclass",
    "metric":             "multi_logloss",
    "num_class":          NUM_VTYPE_CLASSES,
    "boosting_type":      "gbdt",
    "num_leaves":         6,
    "max_depth":          3,
    "learning_rate":      0.05,
    "n_estimators":       200,
    "min_child_samples":  1,
    "min_child_weight":   1e-3,
    "reg_alpha":          0.5,
    "reg_lambda":         1.0,
    "colsample_bytree":   0.8,
    "subsample":          0.8,
    "subsample_freq":     1,
    "class_weight":       "balanced",
    "random_state":       42,
    "verbose":            -1,
    "n_jobs":             -1,
}


# ─────────────────────────────────────────────
#  학습 함수
# ─────────────────────────────────────────────

def train_type_classifier(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    cv_strategy: str = "loo",
    model_dir: str | Path = "models",
    save_model: bool = True,
) -> dict:
    """
    밸브 타입 분류기 학습 (VALVE 그룹 샘플만 사용).

    Parameters
    ----------
    df : pd.DataFrame
        load_dataset() 결과 (전체 데이터, 내부에서 VALVE만 필터링)
    feature_cols : list[str] | None
        사용할 피처 컬럼. None → NUMERIC_FEATURE_COLS
    cv_strategy : str
        "loo" 또는 "kfold"
    model_dir : str | Path
        모델 저장 폴더
    save_model : bool
        학습 후 저장 여부

    Returns
    -------
    dict
        {"cv_accuracy", "report", "feature_importance", "model_path"}
    """
    if feature_cols is None:
        feature_cols = NUMERIC_FEATURE_COLS

    # VALVE 그룹만 필터
    valve_df = df[df["is_valve"] == True].copy()

    if len(valve_df) == 0:
        print("[TYPE] VALVE 샘플이 없습니다. 학습 불가.")
        return {"cv_accuracy": 0.0, "report": "", "feature_importance": None, "model_path": None}

    print(f"\n[TYPE] VALVE 샘플: {len(valve_df)}개")

    X, _, y_vtype = get_feature_matrix(valve_df, feature_cols)
    y = y_vtype.values

    # NON_VALVE 제외 확인 (valve 필터 후에는 없어야 함)
    from src.skey_mapper import VTYPE_TO_INT
    non_valve_idx = VTYPE_TO_INT.get("NON_VALVE", len(VALVE_TYPE_CLASSES) - 1)
    valid_mask = y != non_valve_idx
    X, y = X[valid_mask], y[valid_mask]

    unique_classes = np.unique(y)
    print(f"[TYPE] 클래스 분포: { {INT_TO_VTYPE.get(i, str(i)): (y == i).sum() for i in unique_classes} }")

    if len(unique_classes) < 2:
        print("[TYPE] 클래스가 1개뿐입니다. 분류 불가.")
        return {"cv_accuracy": 1.0, "report": "단일 클래스", "feature_importance": None, "model_path": None}

    # ── 교차검증 ───────────────────────────────
    if cv_strategy == "loo" or len(X) < 30:
        cv = LeaveOneOut()
        print(f"[TYPE] LeaveOneOut CV ({len(X)} folds)")
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        print("[TYPE] StratifiedKFold-5 CV")

    y_true_all, y_pred_all = [], []

    for train_idx, test_idx in cv.split(X, y):
        X_tr = X.iloc[train_idx] if hasattr(X, "iloc") else X[train_idx]
        X_te = X.iloc[test_idx]  if hasattr(X, "iloc") else X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        # LOO-CV 에서는 eval_set/early_stopping 사용 불가
        # (test fold 레이블이 학습셋에 없을 수 있음 → LabelEncoder 오류)
        model = lgb.LGBMClassifier(**LGBM_PARAMS)
        model.fit(
            X_tr_s, y_tr,
            callbacks=[lgb.log_evaluation(period=-1)],
        )

        pred = model.predict(X_te_s)
        y_true_all.extend(y_te.tolist())
        y_pred_all.extend(pred.tolist() if hasattr(pred, "tolist") else [pred])

    cv_accuracy = accuracy_score(y_true_all, y_pred_all)
    label_names = [INT_TO_VTYPE.get(i, str(i)) for i in sorted(np.unique(y_true_all))]
    report = classification_report(
        y_true_all, y_pred_all,
        target_names=label_names,
        zero_division=0,
    )

    print(f"\n[TYPE] CV Accuracy: {cv_accuracy:.4f}")
    print(report)

    # ── 전체 VALVE 데이터로 재학습 + 저장 ───────
    model_path = None
    fi_df = None

    if save_model:
        print("[TYPE] 전체 데이터로 최종 모델 학습 중...")
        scaler_final = StandardScaler()
        X_all_s = scaler_final.fit_transform(X)

        final_model = lgb.LGBMClassifier(**LGBM_PARAMS)
        final_model.fit(
            X_all_s, y,
            callbacks=[lgb.log_evaluation(period=-1)],
        )

        fi = final_model.feature_importances_
        fi_df = pd.DataFrame({
            "feature":    list(X.columns),
            "importance": fi,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path_obj = model_dir / "type_classifier.pkl"
        scaler_path    = model_dir / "type_scaler.pkl"
        feat_path      = model_dir / "type_features.pkl"

        joblib.dump(final_model,   model_path_obj)
        joblib.dump(scaler_final,  scaler_path)
        joblib.dump(feature_cols,  feat_path)

        model_path = str(model_path_obj)
        print(f"[TYPE] 모델 저장: {model_path_obj}")
        print(f"[TYPE] 스케일러 저장: {scaler_path}")

        print(f"\n[TYPE] 상위 10 Feature Importance:")
        print(fi_df.head(10).to_string(index=False))

    return {
        "cv_accuracy":        cv_accuracy,
        "report":             report,
        "feature_importance": fi_df,
        "model_path":         model_path,
    }


# ─────────────────────────────────────────────
#  Ablation Study
# ─────────────────────────────────────────────

def run_ablation_study(df: pd.DataFrame) -> pd.DataFrame:
    """밸브 타입 분류기에 대한 피처 그룹별 Ablation Study."""
    experiments = {
        "geometry only":             FEATURE_GROUPS["geometry"],
        "connector only":            FEATURE_GROUPS["connector"],
        "metadata only":             FEATURE_GROUPS["metadata"],
        "geometry + connector":      FEATURE_GROUPS["geometry"] + FEATURE_GROUPS["connector"],
        "geometry + connector + metadata (ALL)": NUMERIC_FEATURE_COLS,
    }

    results = []
    for name, cols in experiments.items():
        available = [c for c in cols if c in df.columns]
        if not available:
            continue
        print(f"\n{'='*50}")
        print(f"[Ablation-Type] {name} ({len(available)} 피처)")
        res = train_type_classifier(df, feature_cols=available, save_model=False)
        results.append({
            "feature_set":   name,
            "features_used": len(available),
            "cv_accuracy":   round(res["cv_accuracy"], 4),
        })

    ablation_df = pd.DataFrame(results)
    print("\n=== Valve Type Ablation Study 결과 ===")
    print(ablation_df.to_string(index=False))
    return ablation_df


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="밸브 타입 분류기 학습")
    p.add_argument("--data_dir",  type=str, required=True,
                   help="BIM JSON 폴더 경로")
    p.add_argument("--model_dir", type=str, default="models",
                   help="모델 저장 폴더 (기본: models/)")
    p.add_argument("--ablation",  action="store_true",
                   help="Ablation Study 수행 여부")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    df = load_dataset(args.data_dir)
    train_type_classifier(df, model_dir=args.model_dir)

    if args.ablation:
        ablation_df = run_ablation_study(df)
        out_path = Path(args.model_dir) / "ablation_type.csv"
        ablation_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"\n[Ablation] 결과 저장: {out_path}")
