"""
run_pipeline_v2.py
------------------------------
전문가 CSV 레이블 기반 학습 파이프라인 (순수 geometry).

설계 원칙: family/type 이름, 키워드, 카테고리 등 "컴포넌트마다 바뀔 수 있는"
텍스트 정보는 모델에 피처로 넣지 않는다. 오직 형상(geometry)과 커넥터
(connector) 피처만으로 component_group / valve_type을 분류한다 - 이름이나
메타데이터가 없거나 다르게 붙은 새 모델에도 안정적으로 일반화하기 위함.

제외된 피처 (의도적, 버그 아님):
  - kw_*                  : family/type name 기반 키워드 이진 피처 (현재 코드베이스에는
                             생성 로직 자체가 없음 - 이름 참조만 남아 있고 실제로 만들어지지 않음)
  - meta_category         : InstanceParameters 의 Category (텍스트/카테고리)
  - meta_system_class     : System Classification (텍스트/카테고리)
  - meta_max_size_mm      : 사람이 입력한 파라미터 값 (형상에서 측정된 값 아님)
  - meta_min_size_mm      : 〃

사용 피처 (FEATURE_GROUPS["geometry"] + FEATURE_GROUPS["connector"]):
  - solid/face/bbox 기반 형상 피처, 커넥터 반경/거리/각도/분기 피처 전체.

라벨(y)과 피처(X)는 별개다: component_group/valve_type 라벨은 전문가 CSV +
skey_mapper.get_labels_from_skey()로 부여되며, 이 과정은 FamilyAndTypeName
문자열 매칭(data_loader_csv.py)에 의존한다 - 즉 "정답을 어떻게 찾았는지"는
이름 기반이지만, "모델이 예측 시 보는 입력값"에는 이름/키워드가 전혀
들어가지 않는다.

출력: models_pure_geometry/ 폴더에 저장

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
from src.feature_extractor import NUMERIC_FEATURE_COLS, FEATURE_GROUPS
from src.skey_mapper import (
    GROUP_CLASSES, NUM_GROUP_CLASSES,
    INT_TO_GROUP, GROUP_TO_INT,
    VALVE_TYPE_CLASSES, NUM_VTYPE_CLASSES,
    INT_TO_VTYPE, VTYPE_TO_INT,
    skey_prefix,
)

# ── 경로 설정 ─────────────────────────────────
_BIM_BASE          = _ROOT.parent / "BIM_Data"
TRAINING_MODEL_DIR = _BIM_BASE / "TrainingModel"
CSV_LABEL_DIR      = _BIM_BASE / "User_Defined_csv"
XML_PATH           = _BIM_BASE / "symbol.xml"
MODEL_DIR          = _ROOT / "models_pure_geometry"

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

# NC**(미정) 항목에 대해 다국어 키워드 기반 레이블 복구를 시도할지 여부.
# 기본값 False - nc_recovered_labels.csv를 사람이 검토한 뒤에만 켤 것.
# 켜더라도 min_tier="specific"이 기본값이라 "generic"(단일 일반 단어) 매칭은
# 제외됨 - 오매칭 위험이 낮은 항목만 보수적으로 포함.
ENABLE_MULTILINGUAL_RECOVERY = True
MULTILINGUAL_MIN_TIER        = "specific"  # "specific" | "generic"


# ─────────────────────────────────────────────


def _valve_type_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    VALVE로 분류된 컴포넌트를 세부 타입(Knife Gate, 3 Way, Ball, Gate, ...)별로 집계.

    df는 "is_valve" / "valve_type" / "skey" 컬럼을 갖고 있어야 한다
    (data_loader_csv._build_row에서 skey_mapper.get_labels_from_skey()로
    부여됨 - symbol.xml의 SKEY→Description 매핑을 이미 거친 값이므로
    여기서는 그 결과를 그대로 집계만 한다).

    component_group 희소 클래스 필터링(Step 2)과 무관하게 전체 밸브
    통계를 보고 싶으므로, 반드시 그 필터링 이전의 원본 df로 호출할 것.
    """
    required_cols = {"is_valve", "valve_type", "skey"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"밸브 집계에 필요한 컬럼이 없습니다: {sorted(missing)} "
            f"(data_loader_csv.py의 _build_row() 출력 형식을 확인하세요)"
        )

    valves = df[df["is_valve"] == True].copy()  # noqa: E712 (명시적 bool 비교가 더 읽기 쉬움)
    if valves.empty:
        return pd.DataFrame(columns=["valve_type", "count", "pct", "skey_examples"])

    valves["valve_type"] = (
        valves["valve_type"].fillna("UNKNOWN").replace("", "UNKNOWN")
    )

    grouped = (
        valves.groupby("valve_type")
        .agg(
            count=("valve_type", "size"),
            skey_examples=("skey", lambda s: ", ".join(sorted(s.astype(str).unique())[:5])),
        )
        .reset_index()
        .sort_values("count", ascending=False)
    )
    grouped["pct"] = (grouped["count"] / grouped["count"].sum() * 100).round(1)
    return grouped[["valve_type", "count", "pct", "skey_examples"]].reset_index(drop=True)


def _valve_type_by_model(df: pd.DataFrame) -> pd.DataFrame:
    """모델(프로젝트)별 x 밸브 세부 타입별 카운트 피벗 테이블."""
    valves = df[df["is_valve"] == True].copy()  # noqa: E712
    if valves.empty:
        return pd.DataFrame()
    valves["valve_type"] = (
        valves["valve_type"].fillna("UNKNOWN").replace("", "UNKNOWN")
    )
    pivot = pd.pivot_table(
        valves,
        index="source_model",
        columns="valve_type",
        values="skey",
        aggfunc="count",
        fill_value=0,
    )
    pivot["TOTAL"] = pivot.sum(axis=1)
    return pivot.sort_values("TOTAL", ascending=False)


def run_pipeline():
    t0 = time.time()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  AI 배관 컴포넌트 인식 – 전문가 CSV 기반 학습 파이프라인 (순수 geometry)")
    print("=" * 65)
    print(f"  TrainingModel : {TRAINING_MODEL_DIR}")
    print(f"  CSV 레이블    : {CSV_LABEL_DIR}")
    print(f"  symbol.xml    : {XML_PATH}")
    print(f"  모델 저장     : {MODEL_DIR}")
    print("=" * 65)

    # ── Step 1: 데이터 로드 ────────────────────
    print("\n[STEP 1] 데이터 로드 (전문가 CSV 레이블 적용)")
    df, nc_samples, fuzzy_samples, multilingual_samples = load_dataset_csv(
        training_model_dir     = TRAINING_MODEL_DIR,
        csv_label_dir          = CSV_LABEL_DIR,
        xml_path               = XML_PATH,
        verbose                = True,
        enable_multilingual    = ENABLE_MULTILINGUAL_RECOVERY,
        multilingual_min_tier  = MULTILINGUAL_MIN_TIER,
    )

   # 설계상 의도적 제외: meta_category/meta_system_class(카테고리)와
    # kw_*(이름 기반 키워드) 피처는 만들지 않는다 - 이 파이프라인은 오직
    # 형상/커넥터 신호만 모델에 제공하기 위한 것이므로, 이름/메타데이터에서
    # 파생된 피처는 애초에 생성하지 않는다 (kw_*는 애초에 어디서도 생성되지
    # 않는 참조뿐인 이름이기도 함).
    # (df 자체에는 meta_category/meta_family_name 등 원본 컬럼이 여전히
    #  남아있음 - 감사/디버깅용으로만 dataset.csv에 저장되고, feature_cols
    #  에는 포함되지 않는다.)

    # 다국어 복구 목록 저장 (전문가 검증 아님 - 검토 권장)
    if multilingual_samples:
        ml_df = pd.DataFrame(multilingual_samples)
        ml_path = MODEL_DIR / "multilingual_recovered_labels.csv"
        ml_df.to_csv(ml_path, index=False, encoding="utf-8-sig")
        print(f"[STEP 1] 다국어 복구 항목 {len(multilingual_samples)}개 저장: {ml_path} "
              f"(전문가 검증 아님 - 검토 권장)")

    # NC** 목록 저장
    if nc_samples:
        nc_df = pd.DataFrame(nc_samples)
        nc_path = MODEL_DIR / "nc_pending_labels.csv"
        nc_df.to_csv(nc_path, index=False, encoding="utf-8-sig")
        print(f"[STEP 1] NC** 미정 항목 {len(nc_samples)}개 저장: {nc_path}")

    # fuzzy(부분 문자열) 매칭으로 라벨링된 항목 저장 - 학습에는 포함되지만
    # 오매칭 가능성이 있으므로 반드시 검토 권장.
    if fuzzy_samples:
        fuzzy_df = pd.DataFrame(fuzzy_samples)
        fuzzy_path = MODEL_DIR / "fuzzy_matched_labels.csv"
        fuzzy_df.to_csv(fuzzy_path, index=False, encoding="utf-8-sig")
        print(f"[STEP 1] fuzzy 매칭 항목 {len(fuzzy_samples)}개 저장: {fuzzy_path} "
              f"(검토 권장 - 오매칭 가능성 있음)")

    # 데이터셋 저장
    df.to_csv(MODEL_DIR / "dataset.csv", index=False, encoding="utf-8-sig")
    print(f"[STEP 1] 데이터셋 저장: {MODEL_DIR / 'dataset.csv'} ({len(df)}개 샘플)")

    # ── Step 1B: 밸브(Valve) 세부 타입 집계 ────
    # component_group 희소 클래스 필터링(Step 2)이 일어나기 전, 원본 df
    # 기준으로 집계한다 - VALVE 대분류 자체가 필터링될 일은 거의 없지만,
    # 밸브 통계는 학습/CV 대상 여부와 무관하게 항상 전체를 보고 싶기 때문.
    print("\n[STEP 1B] 밸브(Valve) 세부 타입 집계")
    valve_breakdown_df = _valve_type_breakdown(df)
    if valve_breakdown_df.empty:
        print("  [정보] VALVE로 분류된 샘플이 없습니다 (is_valve == True인 행 없음).")
    else:
        valve_breakdown_path = MODEL_DIR / "valve_type_breakdown.csv"
        valve_breakdown_df.to_csv(
            valve_breakdown_path, index=False, encoding="utf-8-sig"
        )
        total_valves = int(valve_breakdown_df["count"].sum())
        print(f"  전체 밸브 샘플 수 : {total_valves}개")
        print(f"  세부 타입 수      : {len(valve_breakdown_df)}개")
        print(valve_breakdown_df.to_string(index=False))
        print(f"  저장: {valve_breakdown_path}")

        # 모델(프로젝트)별 밸브 타입 분포 - 프로젝트 간 비교에 유용
        valve_by_model_df = _valve_type_by_model(df)
        if not valve_by_model_df.empty:
            valve_by_model_path = MODEL_DIR / "valve_type_by_model.csv"
            valve_by_model_df.to_csv(valve_by_model_path, encoding="utf-8-sig")
            print(f"  모델별 밸브 타입 분포 저장: {valve_by_model_path}")

# ── Step 2: 피처 행렬 준비 (PURE GEOMETRY) ──
    print("\n[STEP 2] 피처 행렬 준비 (순수 geometry + connector만 사용)")

    PURE_GEOMETRY_FEATURES = (
        FEATURE_GROUPS["geometry"] + FEATURE_GROUPS["connector"]
    )

    feature_cols = [
        c for c in PURE_GEOMETRY_FEATURES
        if c in df.columns
    ]

    # 카테고리형 피처 없음 (meta_category/meta_system_class 제외) -
    # LightGBM에 categorical_feature 인자 자체를 넘기지 않는다.
    categorical_feature_cols: list = []

    excluded = (
        [c for c in NUMERIC_FEATURE_COLS if c not in feature_cols]
        + ["meta_category", "meta_system_class"]
        + [f"kw_{k}" for k in [
            "VALVE", "GASKET", "ELBOW", "TEE", "FLANGE", "OLET",
            "REDUCER", "COUPLING", "CAP", "STRAINER", "SLEEVE",
            "BELLOWS", "FLEXIBLE", "UNION", "BOLT", "WELD",
            "GAUGE", "METER", "SENSOR", "THERMOMETER", "SIGHT",
            "GLASS", "INSTRUMENT", "INDICATOR",
        ]]
    )
    print(f"  [정보] 제외된 피처 ({len(excluded)}개): {excluded}")
    print(f"  [정보] 사용 피처 ({len(feature_cols)}개): "
          f"geometry {len(FEATURE_GROUPS['geometry'])}개 + "
          f"connector {len(FEATURE_GROUPS['connector'])}개")

    # component_group 레벨 rare-class 필터링 이전 스냅샷.
    # Step 6(Valve 세부 타입 분류기)은 이 필터링과 무관하게 VALVE로 분류된
    # 전체 원본 행을 대상으로 하므로, 여기서 미리 복사해 둔다.
    df_full = df.copy()

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
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(30, verbose=False),
                lgb.log_evaluation(period=-1)
            ],
        )


        y_pred = model.predict(X_val)
        train_acc = accuracy_score(y_tr, model.predict(X_tr))
        acc = accuracy_score(y_val, y_pred)
        fold_accs.append(acc)
        all_y_true.extend(y_val.tolist())
        all_y_pred.extend(y_pred.tolist())
        feature_importance_sum += model.feature_importances_

        print(f"  Fold {fold_idx}/{N_SPLITS}: Val Acc = {acc:.4f}  "
                f"(train_acc={train_acc:.4f}, val_size={len(y_val)})")

    mean_acc = np.mean(fold_accs)
    std_acc  = np.std(fold_accs)
    print(f"\n  CV 결과: {mean_acc:.4f} ± {std_acc:.4f}")

    # ── Step 4: 전체 데이터로 최종 모델 학습 ──
    print("\n[STEP 4] 전체 데이터로 최종 모델 학습")

    final_model = lgb.LGBMClassifier(**LGBM_PARAMS)

    final_model.fit(X, y)


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


    # categorical_feature_cols는 항상 빈 리스트이므로
    # category_values.pkl(카테고리 어휘 스냅샷)은 만들 필요가 없다 -
    # 저장은 하되 빈 dict로 남겨 추론 코드의 파일 구조 기대치만 맞춘다.
    joblib.dump({}, MODEL_DIR / "category_values.pkl")

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

    # ── Step 6: Valve 세부 타입 분류기 ─────────
    # component_group == "VALVE" 로 예측(또는 레이블링)된 행만 따로 모아,
    # 같은 geometry/connector 피처로 valve_type (Ball/Gate/Globe/...) 을
    # 예측하는 2단계(stage-2) 분류기를 별도로 학습한다.
    # 추론 시: group_classifier.pkl 로 먼저 component_group 을 예측하고,
    # 결과가 VALVE 이면 valve_type_classifier.pkl 로 세부 타입을 추가 예측.
    print("\n[STEP 6] Valve 세부 타입(Ball/Gate/Globe/...) 분류기 학습")

    VALVE_LGBM_PARAMS = {**LGBM_PARAMS, "num_class": NUM_VTYPE_CLASSES}

    valve_df = df_full[df_full["component_group"] == "VALVE"].copy()

    # valve_type == "UNKNOWN" 인 행은 skey_mapper가 SKEY 접두어를 구체적인
    # 밸브 타입으로 매핑하지 못했다는 뜻 - 학습에 잘못된 신호를 주지 않도록
    # 제외하고 별도로 저장한다. symbol.xml / _SKEY_TO_VALVE_TYPE 보완이 필요할 수 있음.
    valve_unknown_df = valve_df[valve_df["valve_type"] == "UNKNOWN"]
    if not valve_unknown_df.empty:
        valve_unknown_path = MODEL_DIR / "valve_type_unknown.csv"
        valve_unknown_df.to_csv(valve_unknown_path, index=False, encoding="utf-8-sig")
        print(f"  [경고] valve_type UNKNOWN {len(valve_unknown_df)}개 저장 "
              f"(symbol.xml SKEY→밸브타입 매핑 보완 필요): {valve_unknown_path}")
    valve_df = valve_df[valve_df["valve_type"] != "UNKNOWN"].reset_index(drop=True)

    if valve_df.empty or valve_df["valve_type"].nunique() < 2:
        print("  [정보] 밸브 세부 타입 학습에 필요한 데이터가 부족하여 이 단계를 건너뜁니다 "
              "(VALVE 샘플이 없거나 타입이 1종류뿐).")
        valve_result = None
    else:
        vt_counts = valve_df["valve_type"].value_counts()
        print(f"\n  밸브 세부 타입 분포 (필터링 전, {len(valve_df)}개):")
        for vt, cnt in vt_counts.items():
            print(f"    {vt:20s}: {cnt:5d}개")

        # N_SPLITS보다 샘플이 적은 밸브 타입은 CV 불가능 → 제외 (group 분류기와 동일 원칙)
        rare_vtypes = vt_counts[vt_counts < N_SPLITS].index.tolist()
        if rare_vtypes:
            print(f"\n  [경고] 샘플 부족으로 제외되는 밸브 타입 (< {N_SPLITS}개):")
            for vt in rare_vtypes:
                print(f"    {vt:20s}: {vt_counts[vt]:5d}개 (제외됨 - 추가 데이터 필요)")
            rare_vt_df = valve_df[valve_df["valve_type"].isin(rare_vtypes)]
            rare_vt_df.to_csv(MODEL_DIR / "valve_type_rare_excluded.csv",
                               index=False, encoding="utf-8-sig")
            valve_df = valve_df[~valve_df["valve_type"].isin(rare_vtypes)].reset_index(drop=True)
            print(f"  [정보] 제외 후 남은 밸브 샘플 수: {len(valve_df)}개")

        if valve_df["valve_type"].nunique() < 2:
            print("  [정보] 필터링 후 남은 밸브 타입이 2개 미만이라 이 단계를 건너뜁니다.")
            valve_result = None
        else:
            X_v = valve_df[feature_cols]
            y_v = valve_df["vtype_label"].values
            groups_v = valve_df["source_model"].values

            # group_classifier와 동일한 StratifiedGroupKFold를 쓰되, 밸브 서브셋은
            # 표본이 훨씬 적으므로 (모델 수 / 최소 클래스 크기)에 맞춰 폴드 수를 낮춘다.
            n_valve_groups = valve_df["source_model"].nunique()
            min_vtype_count = int(valve_df["valve_type"].value_counts().min())
            v_splits = max(2, min(N_SPLITS, n_valve_groups, min_vtype_count))
            if v_splits < N_SPLITS:
                print(f"  [정보] 밸브 서브셋 표본이 적어 폴드 수를 {N_SPLITS} → {v_splits}로 축소합니다.")

            skf_v = StratifiedGroupKFold(n_splits=v_splits, shuffle=True, random_state=42)

            v_fold_accs = []
            v_all_true, v_all_pred = [], []
            v_fi_sum = np.zeros(len(feature_cols))

            for fold_idx, (tr_idx, val_idx) in enumerate(skf_v.split(X_v, y_v, groups_v), 1):
                X_tr, X_val = X_v.iloc[tr_idx], X_v.iloc[val_idx]
                y_tr, y_val = y_v[tr_idx], y_v[val_idx]

                vmodel = lgb.LGBMClassifier(**VALVE_LGBM_PARAMS)
                vmodel.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    callbacks=[
                        lgb.early_stopping(30, verbose=False),
                        lgb.log_evaluation(period=-1),
                    ],
                )
                y_pred = vmodel.predict(X_val)
                train_acc = accuracy_score(y_tr, vmodel.predict(X_tr))
                acc = accuracy_score(y_val, y_pred)
                v_fold_accs.append(acc)
                v_all_true.extend(y_val.tolist())
                v_all_pred.extend(y_pred.tolist())
                v_fi_sum += vmodel.feature_importances_

                print(f"  Fold {fold_idx}/{v_splits}: Val Acc = {acc:.4f}  "
                      f"(train_acc={train_acc:.4f}, val_size={len(y_val)})")

            v_mean_acc, v_std_acc = float(np.mean(v_fold_accs)), float(np.std(v_fold_accs))
            print(f"\n  밸브 세부 타입 CV 결과: {v_mean_acc:.4f} ± {v_std_acc:.4f}")

            # 전체 밸브 데이터로 최종 모델 학습
            final_valve_model = lgb.LGBMClassifier(**VALVE_LGBM_PARAMS)
            final_valve_model.fit(X_v, y_v)

            joblib.dump(final_valve_model, MODEL_DIR / "valve_type_classifier.pkl")
            joblib.dump(feature_cols, MODEL_DIR / "valve_feature_cols.pkl")
            joblib.dump(categorical_feature_cols, MODEL_DIR / "valve_categorical_features.pkl")
            joblib.dump(INT_TO_VTYPE, MODEL_DIR / "int_to_vtype.pkl")
            print(f"  모델 저장: {MODEL_DIR}/valve_type_classifier.pkl")

            # 평가 리포트
            v_unique_labels = sorted(set(v_all_true))
            v_label_names = [INT_TO_VTYPE.get(i, str(i)) for i in v_unique_labels]

            v_cr = classification_report(
                v_all_true, v_all_pred,
                labels=v_unique_labels, target_names=v_label_names,
                output_dict=True,
            )
            pd.DataFrame(v_cr).T.to_csv(
                MODEL_DIR / "valve_classification_report.csv", encoding="utf-8-sig")
            print("\n  밸브 세부 타입 분류 리포트 (CV 전체):")
            print(classification_report(
                v_all_true, v_all_pred,
                labels=v_unique_labels, target_names=v_label_names,
            ))

            v_cm = confusion_matrix(v_all_true, v_all_pred, labels=v_unique_labels)
            pd.DataFrame(v_cm, index=v_label_names, columns=v_label_names).to_csv(
                MODEL_DIR / "valve_confusion_matrix.csv", encoding="utf-8-sig")
            print(f"  Confusion Matrix 저장: {MODEL_DIR}/valve_confusion_matrix.csv")

            v_fi_df = pd.DataFrame({
                "feature":    feature_cols,
                "importance": v_fi_sum / v_splits,
            }).sort_values("importance", ascending=False)
            v_fi_df.to_csv(MODEL_DIR / "valve_feature_importance.csv",
                            index=False, encoding="utf-8-sig")
            print(f"\n  밸브 세부 타입 피처 중요도 Top 10:")
            print(v_fi_df.head(10).to_string(index=False))

            pd.DataFrame({
                "fold": list(range(1, v_splits + 1)),
                "accuracy": v_fold_accs,
            }).to_csv(MODEL_DIR / "valve_fold_accuracies.csv",
                      index=False, encoding="utf-8-sig")

            valve_result = {
                "cv_accuracy": v_mean_acc,
                "cv_std": v_std_acc,
                "n_samples": len(valve_df),
                "n_types": valve_df["valve_type"].nunique(),
                "n_splits": v_splits,
            }

    # ── Step 7: 개별 SKEY 단위 분류기 ──────────
    # Step 6의 valve_type(BALL/GATE/GLOBE/...)은 여러 SKEY 접두어를 한
    # 클래스로 묶은 값이다 (예: VB/BVA/BVA2 → 전부 "BALL"). 이 단계는 같은
    # geometry/connector 피처로, 더 세밀한 "개별 SKEY 접두어" 단위(VB vs
    # BVA vs BVA2, VT vs KG, ...)를 예측하는 3단계(stage-3) 분류기를 별도로
    # 학습한다.
    #
    # 주의 - match_tier가 "multilingual_*"인 행은 제외한다: 이런 행은 전문가가
    # 부여한 진짜 SKEY가 없고, family/type 이름을 다국어 키워드 사전과 대조해
    # "이건 아마 BALL 밸브겠다" 정도로 valve_type만 추정한 것이라 skey 컬럼이
    # "MULTILINGUAL:<키워드>" placeholder다 - 어떤 개별 SKEY였는지는 원래
    # 알 수 없다. 따라서 이 단계의 학습 표본은 Step 6(valve_type)보다 훨씬
    # 적어질 수 있다 (특히 BALL/BUTTERFLY처럼 다국어 복구 비중이 큰 타입).
    print("\n[STEP 7] 개별 SKEY 단위(VB/BVA/VT/KG/...) 분류기 학습")

    skey_df = valve_df[~valve_df["match_tier"].str.startswith("multilingual")].copy()
    skey_df["skey_prefix"] = skey_df["skey"].apply(skey_prefix)

    n_multilingual_dropped = len(valve_df) - len(skey_df)
    print(f"  [정보] multilingual 추정 레이블 제외: {n_multilingual_dropped}개 "
          f"(진짜 SKEY 표본 {len(skey_df)}개만 사용)")

    if skey_df.empty or skey_df["skey_prefix"].nunique() < 2:
        print("  [정보] 개별 SKEY 분류 학습에 필요한 데이터가 부족하여 이 단계를 건너뜁니다 "
              "(진짜 SKEY 표본이 없거나 종류가 1개뿐).")
        skey_result = None
    else:
        sk_counts = skey_df["skey_prefix"].value_counts()
        print(f"\n  개별 SKEY 분포 (필터링 전, {len(skey_df)}개):")
        for sk, cnt in sk_counts.items():
            print(f"    {sk:20s}: {cnt:5d}개")

        # N_SPLITS보다 샘플이 적은 SKEY는 CV 불가능 → 제외 (group/valve_type 분류기와 동일 원칙)
        rare_skeys = sk_counts[sk_counts < N_SPLITS].index.tolist()
        if rare_skeys:
            print(f"\n  [경고] 샘플 부족으로 제외되는 SKEY (< {N_SPLITS}개):")
            for sk in rare_skeys:
                print(f"    {sk:20s}: {sk_counts[sk]:5d}개 (제외됨 - 추가 데이터 필요)")
            rare_sk_df = skey_df[skey_df["skey_prefix"].isin(rare_skeys)]
            rare_sk_df.to_csv(MODEL_DIR / "skey_rare_excluded.csv",
                               index=False, encoding="utf-8-sig")
            skey_df = skey_df[~skey_df["skey_prefix"].isin(rare_skeys)].reset_index(drop=True)
            print(f"  [정보] 제외 후 남은 SKEY 샘플 수: {len(skey_df)}개")

        if skey_df["skey_prefix"].nunique() < 2:
            print("  [정보] 필터링 후 남은 SKEY 종류가 2개 미만이라 이 단계를 건너뜁니다.")
            skey_result = None
        else:
            # 이 학습 실행에서 실제로 등장한 SKEY 접두어만으로 클래스 목록을
            # 동적으로 구성한다 (group/valve_type처럼 전체 코드 체계를 미리
            # 하드코딩해두지 않음 - symbol.xml/CSV가 갱신되면 그대로 반영됨).
            SKEY_CLASSES = sorted(skey_df["skey_prefix"].unique())
            SKEY_TO_INT = {s: i for i, s in enumerate(SKEY_CLASSES)}
            INT_TO_SKEY = {i: s for s, i in SKEY_TO_INT.items()}
            NUM_SKEY_CLASSES = len(SKEY_CLASSES)

            SKEY_LGBM_PARAMS = {**LGBM_PARAMS, "num_class": NUM_SKEY_CLASSES}

            skey_df["skey_label"] = skey_df["skey_prefix"].map(SKEY_TO_INT)

            X_sk = skey_df[feature_cols]
            y_sk = skey_df["skey_label"].values
            groups_sk = skey_df["source_model"].values

            n_skey_groups = skey_df["source_model"].nunique()
            min_skey_count = int(skey_df["skey_prefix"].value_counts().min())
            sk_splits = max(2, min(N_SPLITS, n_skey_groups, min_skey_count))
            if sk_splits < N_SPLITS:
                print(f"  [정보] SKEY 서브셋 표본이 적어 폴드 수를 {N_SPLITS} → {sk_splits}로 축소합니다.")

            skf_sk = StratifiedGroupKFold(n_splits=sk_splits, shuffle=True, random_state=42)

            sk_fold_accs = []
            sk_all_true, sk_all_pred = [], []
            sk_fi_sum = np.zeros(len(feature_cols))

            for fold_idx, (tr_idx, val_idx) in enumerate(skf_sk.split(X_sk, y_sk, groups_sk), 1):
                X_tr, X_val = X_sk.iloc[tr_idx], X_sk.iloc[val_idx]
                y_tr, y_val = y_sk[tr_idx], y_sk[val_idx]

                skmodel = lgb.LGBMClassifier(**SKEY_LGBM_PARAMS)
                skmodel.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    callbacks=[
                        lgb.early_stopping(30, verbose=False),
                        lgb.log_evaluation(period=-1),
                    ],
                )
                y_pred = skmodel.predict(X_val)
                train_acc = accuracy_score(y_tr, skmodel.predict(X_tr))
                acc = accuracy_score(y_val, y_pred)
                sk_fold_accs.append(acc)
                sk_all_true.extend(y_val.tolist())
                sk_all_pred.extend(y_pred.tolist())
                sk_fi_sum += skmodel.feature_importances_

                print(f"  Fold {fold_idx}/{sk_splits}: Val Acc = {acc:.4f}  "
                      f"(train_acc={train_acc:.4f}, val_size={len(y_val)})")

            sk_mean_acc, sk_std_acc = float(np.mean(sk_fold_accs)), float(np.std(sk_fold_accs))
            print(f"\n  개별 SKEY CV 결과: {sk_mean_acc:.4f} ± {sk_std_acc:.4f}")

            # 전체 데이터로 최종 모델 학습
            final_skey_model = lgb.LGBMClassifier(**SKEY_LGBM_PARAMS)
            final_skey_model.fit(X_sk, y_sk)

            joblib.dump(final_skey_model, MODEL_DIR / "skey_classifier.pkl")
            joblib.dump(feature_cols, MODEL_DIR / "skey_feature_cols.pkl")
            joblib.dump(categorical_feature_cols, MODEL_DIR / "skey_categorical_features.pkl")
            joblib.dump(INT_TO_SKEY, MODEL_DIR / "int_to_skey.pkl")
            print(f"  모델 저장: {MODEL_DIR}/skey_classifier.pkl")

            # 평가 리포트
            sk_unique_labels = sorted(set(sk_all_true))
            sk_label_names = [INT_TO_SKEY.get(i, str(i)) for i in sk_unique_labels]

            sk_cr = classification_report(
                sk_all_true, sk_all_pred,
                labels=sk_unique_labels, target_names=sk_label_names,
                output_dict=True,
            )
            pd.DataFrame(sk_cr).T.to_csv(
                MODEL_DIR / "skey_classification_report.csv", encoding="utf-8-sig")
            print("\n  개별 SKEY 분류 리포트 (CV 전체):")
            print(classification_report(
                sk_all_true, sk_all_pred,
                labels=sk_unique_labels, target_names=sk_label_names,
            ))

            sk_cm = confusion_matrix(sk_all_true, sk_all_pred, labels=sk_unique_labels)
            pd.DataFrame(sk_cm, index=sk_label_names, columns=sk_label_names).to_csv(
                MODEL_DIR / "skey_confusion_matrix.csv", encoding="utf-8-sig")
            print(f"  Confusion Matrix 저장: {MODEL_DIR}/skey_confusion_matrix.csv")

            sk_fi_df = pd.DataFrame({
                "feature":    feature_cols,
                "importance": sk_fi_sum / sk_splits,
            }).sort_values("importance", ascending=False)
            sk_fi_df.to_csv(MODEL_DIR / "skey_feature_importance.csv",
                             index=False, encoding="utf-8-sig")
            print(f"\n  개별 SKEY 피처 중요도 Top 10:")
            print(sk_fi_df.head(10).to_string(index=False))

            pd.DataFrame({
                "fold": list(range(1, sk_splits + 1)),
                "accuracy": sk_fold_accs,
            }).to_csv(MODEL_DIR / "skey_fold_accuracies.csv",
                      index=False, encoding="utf-8-sig")

            skey_result = {
                "cv_accuracy": sk_mean_acc,
                "cv_std": sk_std_acc,
                "n_samples": len(skey_df),
                "n_skeys": skey_df["skey_prefix"].nunique(),
                "n_splits": sk_splits,
                "n_multilingual_dropped": n_multilingual_dropped,
            }

    # 최종 요약
    elapsed = time.time() - t0
    print("\n" + "=" * 65)
    print("  파이프라인 완료")
    print("=" * 65)
    print(f"  소요 시간                : {elapsed:.1f}초")
    print(f"  학습 모델 수 (CSV)       : {df['source_model'].nunique()}개")
    print(f"  학습 샘플 수             : {len(df)}개")
    print(f"  NC** 제외 샘플           : {len(nc_samples)}개 (추후 보완)")
    print(f"  fuzzy 매칭 샘플 (포함됨) : {len(fuzzy_samples)}개 (검토 권장)")
    print(f"  다국어 복구 샘플 (포함됨): {len(multilingual_samples)}개 "
          f"(min_tier={MULTILINGUAL_MIN_TIER!r}, 검토 권장)")
    print(f"  Stratified {N_SPLITS}-Fold CV Acc : {mean_acc:.4f} ± {std_acc:.4f}")
    if not valve_breakdown_df.empty:
        top_valve = valve_breakdown_df.iloc[0]
        print(f"  밸브(Valve) 총 샘플 수   : {int(valve_breakdown_df['count'].sum())}개 "
              f"({len(valve_breakdown_df)}개 세부 타입, 최다: "
              f"{top_valve['valve_type']} {int(top_valve['count'])}개)")
    if valve_result is not None:
        print(f"  Valve 세부 타입 CV Acc   : {valve_result['cv_accuracy']:.4f} ± "
              f"{valve_result['cv_std']:.4f} ({valve_result['n_samples']}개 샘플, "
              f"{valve_result['n_types']}종)")
    if skey_result is not None:
        print(f"  개별 SKEY CV Acc         : {skey_result['cv_accuracy']:.4f} ± "
              f"{skey_result['cv_std']:.4f} ({skey_result['n_samples']}개 샘플, "
              f"{skey_result['n_skeys']}종, multilingual {skey_result['n_multilingual_dropped']}개 제외)")
    print(f"  모델 저장 위치           : {MODEL_DIR}")
    print("=" * 65)

    return {
        "cv_accuracy": mean_acc,
        "cv_std":      std_acc,
        "n_samples":   len(df),
        "n_models":    df["source_model"].nunique(),
        "n_nc_skip":   len(nc_samples),
        "n_fuzzy_matched": len(fuzzy_samples),
        "n_multilingual_recovered": len(multilingual_samples),
        "valve_type_counts": (
            dict(zip(valve_breakdown_df["valve_type"], valve_breakdown_df["count"]))
            if not valve_breakdown_df.empty else {}
        ),
        "valve_type_classifier": valve_result,
        "skey_classifier": skey_result,
        "model_dir":   str(MODEL_DIR),
    }


if __name__ == "__main__":
    result = run_pipeline()