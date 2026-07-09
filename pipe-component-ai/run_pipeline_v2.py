"""
run_pipeline_v2.py
------------------
전문가 CSV 레이블 기반 AI 학습 파이프라인 (v2).

변경 사항 (v1 대비):
  - 데이터: TrainingModel/ 다중 폴더 구조 지원
  - 레이블: 전문가_정의_csv/ CSV 파일 기반 (181개 모델)
  - NC**:   학습 제외 (추후 보완 시 재학습 예정)
  - CV:     Stratified 5-Fold (기존 LOO-CV → 데이터 증가로 변경)
  - 출력:   models_v2/ 폴더에 저장

사용:
    cd pipe-component-ai
    python run_pipeline_v2.py
"""

import sys
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd 
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)

_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data_loader_csv import load_dataset_csv
from src.feature_extractor import NUMERIC_FEATURE_COLS
from src.skey_mapper import (
    GROUP_CLASSES, NUM_GROUP_CLASSES,
    INT_TO_GROUP, GROUP_TO_INT,
)

# ── 경로 설정 ─────────────────────────────────
_BIM_BASE          = _ROOT.parent / "BIM_Data"
TRAINING_MODEL_DIR = _BIM_BASE / "TrainingModel"
CSV_LABEL_DIR      = _BIM_BASE / "User_Defined_csv"
XML_PATH           = _BIM_BASE / "symbol.xml"
MODEL_DIR          = _ROOT / "models_v2"

# ── LightGBM 파라미터 ─────────────────────────
LGBM_PARAMS = {
    "objective":         "multiclass",
    "metric":            "multi_logloss",
    "num_class":         NUM_GROUP_CLASSES,
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

N_SPLITS = 5  # Stratified K-Fold 분할 수


# ─────────────────────────────────────────────


def run_pipeline():
    t0 = time.time()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  AI 배관 컴포넌트 인식 – 전문가 CSV 기반 학습 파이프라인 v2")
    print("=" * 65)
    print(f"  TrainingModel : {TRAINING_MODEL_DIR}")
    print(f"  CSV 레이블    : {CSV_LABEL_DIR}")
    print(f"  symbol.xml    : {XML_PATH}")
    print(f"  모델 저장     : {MODEL_DIR}")
    print("=" * 65)

    # ── Step 1: 데이터 로드 ────────────────────
    print("\n[STEP 1] 데이터 로드 (전문가 CSV 레이블 적용)")
    df, nc_samples = load_dataset_csv(
        training_model_dir = TRAINING_MODEL_DIR,
        csv_label_dir      = CSV_LABEL_DIR,
        xml_path           = XML_PATH,
        verbose            = True,
    )

   # meta_category / meta_system_class는 넓은 범주라 그대로 카테고리로 유지
    categorical_cols = [
        "meta_category",
        "meta_system_class",
    ]
    for col in categorical_cols:
        df[col] = df[col].fillna("").astype("category")

    # meta_family_name / meta_type_name은 정확히 일치하는 문자열을 그대로 쓰면
    # 학습에서 본 특정 부품명을 "암기"할 위험이 있음 (새 프로젝트에 일반화 안 됨).
    # 대신 이름에 포함된 키워드를 이진(0/1) 피처로 추출 - 어떤 프로젝트든
    # "GASKET"이라는 단어가 들어가면 항상 같은 신호를 주도록 일반화.
    
    KEYWORD_FEATURES = [
        "VALVE", "GASKET", "ELBOW", "TEE", "FLANGE", "OLET",
        "REDUCER", "COUPLING", "CAP", "STRAINER", "SLEEVE",
        "BELLOWS", "FLEXIBLE", "UNION", "BOLT", "WELD",
        "GAUGE", "METER", "SENSOR", "THERMOMETER", "SIGHT",
        "GLASS", "INSTRUMENT", "INDICATOR",
    ]

    combined_name = (
        df["meta_family_name"].fillna("") + " " + df["meta_type_name"].fillna("")
    ).str.upper()
    keyword_cols = []
    for kw in KEYWORD_FEATURES:
        col_name = f"kw_{kw}"
        df[col_name] = combined_name.str.contains(kw).astype(int)
        keyword_cols.append(col_name)

    # NC** 목록 저장
    if nc_samples:
        nc_df = pd.DataFrame(nc_samples)
        nc_path = MODEL_DIR / "nc_pending_labels.csv"
        nc_df.to_csv(nc_path, index=False, encoding="utf-8-sig")
        print(f"[STEP 1] NC** 미정 항목 {len(nc_samples)}개 저장: {nc_path}")

    # 데이터셋 저장
    df.to_csv(MODEL_DIR / "dataset.csv", index=False, encoding="utf-8-sig")
    print(f"[STEP 1] 데이터셋 저장: {MODEL_DIR / 'dataset.csv'} ({len(df)}개 샘플)")

# ── Step 2: 피처 행렬 준비 ─────────────────
    print("\n[STEP 2] 피처 행렬 준비")

    feature_cols = [
        c for c in NUMERIC_FEATURE_COLS
        if c in df.columns
    ]

    feature_cols.extend(keyword_cols)

    categorical_feature_cols = [
        "meta_category",
        "meta_system_class",
    ]

    feature_cols.extend(categorical_feature_cols)

    print(f"  피처 수  : {len(feature_cols)}")
    print(f"  샘플 수  : {len(df)}")
    print(f"  클래스   : {sorted(df['component_group'].unique())}")

    # 클래스별 샘플 수 출력
    vc = df["component_group"].value_counts()
    print(f"\n  클래스 분포:")
    for grp, cnt in vc.items():
        print(f"    {grp:20s}: {cnt:5d}개")

    # N_SPLITS보다 샘플이 적은 클래스는 CV 불가능 → 제외 (NC**와 동일하게 별도 저장)
    rare_classes = vc[vc < N_SPLITS].index.tolist()
    if rare_classes:
        print(f"\n  [경고] 샘플 부족으로 제외되는 클래스 (< {N_SPLITS}개):")
        for grp in rare_classes:
            print(f"    {grp:20s}: {vc[grp]:5d}개 (제외됨 - 추가 데이터 필요)")
        rare_df = df[df["component_group"].isin(rare_classes)]
        rare_df.to_csv(MODEL_DIR / "rare_classes_excluded.csv",
                        index=False, encoding="utf-8-sig")
        df = df[~df["component_group"].isin(rare_classes)].reset_index(drop=True)
        print(f"  [정보] 제외 후 남은 샘플 수: {len(df)}개")

    # 필터링 후 df 기준으로 X, y 생성 (반드시 필터링 이후에 만들어야 함)
    X = df[feature_cols]
    y = df["group_label"].values

# N_SPLITS보다 샘플이 적은 클래스는 CV 불가능 → 제외 (NC**와 동일하게 별도 저장)
    rare_classes = vc[vc < N_SPLITS].index.tolist()
    if rare_classes:
        print(f"\n  [경고] 샘플 부족으로 제외되는 클래스 (< {N_SPLITS}개):")
        for grp in rare_classes:
            print(f"    {grp:20s}: {vc[grp]:5d}개 (제외됨 - 추가 데이터 필요)")
        rare_df = df[df["component_group"].isin(rare_classes)]
        rare_df.to_csv(MODEL_DIR / "rare_classes_excluded.csv",
                        index=False, encoding="utf-8-sig")
        df = df[~df["component_group"].isin(rare_classes)].reset_index(drop=True)
        print(f"  [정보] 제외 후 남은 샘플 수: {len(df)}개")

    # ── Step 3: Stratified 5-Fold CV ──────────
    print(f"\n[STEP 3] Stratified Group {N_SPLITS}-Fold CV 학습")
    groups = df["source_model"].values
    skf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    fold_accs = []
    all_y_true = []
    all_y_pred = []
    feature_importance_sum = np.zeros(len(feature_cols))

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y, groups), 1):
        
        X_tr = X.iloc[train_idx]
        X_val = X.iloc[val_idx]

        y_tr = y[train_idx]
        y_val = y[val_idx]


        model = lgb.LGBMClassifier(**LGBM_PARAMS)

        model.fit(
            X_tr,
            y_tr,
            categorical_feature=categorical_feature_cols,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(30, verbose=False),
                lgb.log_evaluation(period=-1)
            ],
        )


        y_pred = model.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        fold_accs.append(acc)
        all_y_true.extend(y_val.tolist())
        all_y_pred.extend(y_pred.tolist())
        feature_importance_sum += model.feature_importances_

        print(f"  Fold {fold_idx}/{N_SPLITS}: Val Acc = {acc:.4f}  "
              f"(val_size={len(y_val)})")

    mean_acc = np.mean(fold_accs)
    std_acc  = np.std(fold_accs)
    print(f"\n  CV 결과: {mean_acc:.4f} ± {std_acc:.4f}")

    # ── Step 4: 전체 데이터로 최종 모델 학습 ──
    print("\n[STEP 4] 전체 데이터로 최종 모델 학습")

    final_model = lgb.LGBMClassifier(**LGBM_PARAMS)

    final_model.fit(
        X,
        y,
        categorical_feature=categorical_feature_cols,
    )


    # 모델 저장
    joblib.dump(
        final_model,
        MODEL_DIR / "group_classifier.pkl"
    )

    joblib.dump(
        feature_cols,
        MODEL_DIR / "feature_cols.pkl"
    )

    joblib.dump(
        categorical_feature_cols,
        MODEL_DIR / "categorical_features.pkl"
    )


    category_values = {
        col: df[col].cat.categories.tolist()
        for col in categorical_feature_cols
    }

    joblib.dump(
        category_values,
        MODEL_DIR / "category_values.pkl"
    )

    joblib.dump(
        INT_TO_GROUP,
        MODEL_DIR / "int_to_group.pkl"
    )

    print(f"  모델 저장: {MODEL_DIR}/group_classifier.pkl")

    # ── Step 5: 평가 리포트 ────────────────────
    print("\n[STEP 5] 평가 리포트 생성")

    # 클래스 레이블 정리
    unique_labels = sorted(set(all_y_true))
    label_names   = [INT_TO_GROUP.get(i, str(i)) for i in unique_labels]

    # Classification report
    cr = classification_report(
        all_y_true, all_y_pred,
        labels=unique_labels,
        target_names=label_names,
        output_dict=True,
    )
    cr_df = pd.DataFrame(cr).T
    cr_df.to_csv(MODEL_DIR / "classification_report.csv", encoding="utf-8-sig")
    print("\n  분류 리포트 (CV 전체):")
    print(classification_report(
        all_y_true, all_y_pred,
        labels=unique_labels,
        target_names=label_names,
    ))

    # Confusion matrix
    cm = confusion_matrix(all_y_true, all_y_pred, labels=unique_labels)
    cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
    cm_df.to_csv(MODEL_DIR / "confusion_matrix.csv", encoding="utf-8-sig")
    print(f"  Confusion Matrix 저장: {MODEL_DIR}/confusion_matrix.csv")

    # Feature importance
    fi_df = pd.DataFrame({
        "feature":    feature_cols,
        "importance": feature_importance_sum / N_SPLITS,
    }).sort_values("importance", ascending=False)
    fi_df.to_csv(MODEL_DIR / "feature_importance.csv",
                 index=False, encoding="utf-8-sig")
    print(f"\n  피처 중요도 Top 10:")
    print(fi_df.head(10).to_string(index=False))

    # Fold별 accuracy 저장
    fold_df = pd.DataFrame({
        "fold":     list(range(1, N_SPLITS + 1)),
        "accuracy": fold_accs,
    })
    fold_df.to_csv(MODEL_DIR / "fold_accuracies.csv",
                   index=False, encoding="utf-8-sig")

    # 최종 요약
    elapsed = time.time() - t0
    print("\n" + "=" * 65)
    print("  파이프라인 완료")
    print("=" * 65)
    print(f"  소요 시간                : {elapsed:.1f}초")
    print(f"  학습 모델 수 (CSV)       : {df['source_model'].nunique()}개")
    print(f"  학습 샘플 수             : {len(df)}개")
    print(f"  NC** 제외 샘플           : {len(nc_samples)}개 (추후 보완)")
    print(f"  Stratified {N_SPLITS}-Fold CV Acc : {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"  모델 저장 위치           : {MODEL_DIR}")
    print("=" * 65)

    return {
        "cv_accuracy": mean_acc,
        "cv_std":      std_acc,
        "n_samples":   len(df),
        "n_models":    df["source_model"].nunique(),
        "n_nc_skip":   len(nc_samples),
        "model_dir":   str(MODEL_DIR),
    }


if __name__ == "__main__":
    result = run_pipeline()
