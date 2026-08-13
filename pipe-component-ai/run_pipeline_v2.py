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
  - meta_max_size_mm      : 사람이 입력한 파라미터 값. 한 차례 재포함해 재학습해
                             봤으나(ABLATION), RealData 실사용 데이터에서 SUPPORT로의
                             쏠림(46→95건)이 발생해 되돌림 - 학습 CSV와 RealData 간
                             "Maximum Size 파라미터 존재 여부" 분포가 달라 생긴 문제로
                             추정. CV 이득은 미미했음 (0.9311→0.9338).
  - meta_min_size_mm      : 〃
  - meta_size_available   : 〃 (위와 함께 도입/제외됨)

사용 피처 (FEATURE_GROUPS["geometry"] + FEATURE_GROUPS["connector"]):
  - solid/face/bbox 기반 형상 피처, 커넥터 반경/거리/각도/분기 피처 전체.

라벨(y)과 피처(X)는 별개다: component_group/valve_type 라벨은 전문가 CSV +
skey_mapper.get_labels_from_skey()로 부여되며, 이 과정은 FamilyAndTypeName
문자열 매칭(data_loader_csv.py)에 의존한다 - 즉 "정답을 어떻게 찾았는지"는
이름 기반이지만, "모델이 예측 시 보는 입력값"에는 이름/키워드가 전혀
들어가지 않는다.

출력: models_v2/ 폴더에 저장

사용:
    cd pipe-component-ai
    python run_pipeline_v2.py

────────────────────────────────────────────────────────────────
콘솔 출력 설계 원칙 (v2 리팩터링에서 추가):
  1. 파일 저장 경로는 헤더/푸터에서 MODEL_DIR을 한 번만 보여주고,
     그 이후 각 저장 로그는 파일명만 출력한다 (풀 경로 반복 금지).
  2. component_group / valve_type / skey, 3개 분류기가 CV → 최종
     학습 → 리포트 저장까지 똑같은 절차를 거치므로, 그 절차는
     _run_cv_stage() 하나로 통합했다. 3배로 반복되던 콘솔 출력과
     코드 블록이 한 곳에서만 정의된다.
  3. 사람들이 실제로 보는 지표는 검증(validation) 정확도이므로 그것을
     주 지표로 크게 보여주고, train accuracy는 "과적합 여부를 보여주는
     보조 수치(gap)"로만 함께 표시한다.
────────────────────────────────────────────────────────────────
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
_BIM_BASE           = _ROOT.parent / "BIM_Data"
TRAINING_MODEL_DIR  = _BIM_BASE / "TrainingModel"
CSV_LABEL_DIR       = _BIM_BASE / "User_Defined_csv"
XML_PATH            = _BIM_BASE / "symbol.xml"
MODEL_DIR           = _ROOT / "models_v2"

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

# ── 범주형 메타데이터 (valve_type / skey 단계 전용) ──────
# component_group(STEP 3) 분류기는 의도적으로 pure-geometry 그대로 둔다 -
# meta_max_size_mm 재포함 실험에서 RealData 쏠림(46→95건, SUPPORT)이
# 발생했던 게 바로 이 단계였기 때문. VALVE 세부 타입(STEP 4/5)은 완전히
# 다른 분류 문제(전부 VALVE인 표본 안에서의 세부 구분)라 같은 실패가
# 그대로 재발한다는 보장은 없지만, 혹시 RealData에서 비슷한 존재율 격차가
# 나타나면 아래 플래그 하나로 즉시 되돌릴 수 있게 해둔다.
ENABLE_VALVE_METADATA   = True
CATEGORICAL_META_FEATURES = FEATURE_GROUPS["metadata_categorical"]  # meta_category, meta_system_class
_MISSING_CATEGORY = "MISSING"  # 결측을 NaN이 아니라 명시적 카테고리로 취급

# True면 valve_type/skey 각 단계를 metadata 없이(baseline) 한 번 더
# 돌려서 CV 정확도를 나란히 출력한다 (모델/리포트 파일은 저장하지 않는
# dry-run - 실제로 저장되는 모델은 ENABLE_VALVE_METADATA 설정을 따름).
# meta_max_size_mm 때처럼 "CV는 오르는데 RealData는 나빠지는" 상황을
# CV 단계에서 완전히 잡아낼 수는 없지만, 최소한 metadata가 CV 상에서
# 실제로 기여하는지는 매 실행마다 바로 확인할 수 있다.
RUN_METADATA_ABLATION = True


# ═══════════════════════════════════════════════
#  출력 / 저장 헬퍼
# ═══════════════════════════════════════════════

def _vwidth(s) -> int:
    """
    문자열의 터미널 표시 폭. 한글/한자 등 CJK 문자는 터미널에서 2칸을
    차지하지만 len()은 1로 센다 - 그래서 한글 헤더와 영문/숫자 데이터를
    같은 표에 섞으면 len() 기준 정렬(ljust/rjust)이 항상 어긋난다.
    이 함수로 실제 표시 폭을 계산해서 맞춘다.
    """
    w = 0
    for ch in str(s):
        code = ord(ch)
        if (0x1100 <= code <= 0x115F or 0x2E80 <= code <= 0xA4CF or
                0xAC00 <= code <= 0xD7A3 or 0xF900 <= code <= 0xFAFF or
                0xFF00 <= code <= 0xFF60 or 0xFFE0 <= code <= 0xFFE6):
            w += 2
        else:
            w += 1
    return w


def _pad(s, width: int, align: str = "right") -> str:
    s = str(s)
    gap = max(0, width - _vwidth(s))
    return (" " * gap + s) if align == "right" else (s + " " * gap)


def _print_table(headers, rows, aligns=None):
    """
    표시 폭 기준으로 정렬된 표를 출력한다 (한글 헤더 + 숫자 데이터가
    섞여도 어긋나지 않음). 첫 컬럼은 기본 왼쪽 정렬, 나머지는 오른쪽 정렬.
    """
    n = len(headers)
    aligns = aligns or (["left"] + ["right"] * (n - 1))
    all_rows = [headers] + [list(r) for r in rows]
    widths = [max(_vwidth(row[i]) for row in all_rows) for i in range(n)]

    def _fmt(cells):
        return "  " + "  ".join(_pad(c, w, a) for c, w, a in zip(cells, widths, aligns))

    print(_fmt(headers))
    print("  " + "-" * (sum(widths) + 2 * (n - 1)))
    for r in rows:
        print(_fmt(r))


def _save(obj, filename: str, index: bool = False):
    """
    MODEL_DIR 아래에 저장하고 파일명만 콘솔에 출력한다.
    DataFrame → to_csv, 그 외(모델/리스트/딕셔너리 등) → joblib.dump.
    전체 경로는 파이프라인 시작/종료 시 헤더에서 이미 한 번 보여줬으므로
    저장할 때마다 반복하지 않는다.
    """
    path = MODEL_DIR / filename
    if isinstance(obj, pd.DataFrame):
        obj.to_csv(path, index=index, encoding="utf-8-sig")
    else:
        joblib.dump(obj, path)
    print(f"    저장: {filename}")
    return path


def _drop_rare_classes(df: pd.DataFrame, label_col: str, min_count: int,
                        excluded_filename: str, item_label: str = "클래스") -> pd.DataFrame:
    """
    min_count(N_SPLITS) 미만 표본을 가진 클래스는 CV 자체가 불가능하므로
    별도 CSV로 빼두고 학습 대상에서 제외한다. group / valve_type / skey
    3개 분류기가 전부 이 로직을 그대로 재사용한다.
    """
    counts = df[label_col].value_counts()
    rare = counts[counts < min_count].index.tolist()
    if not rare:
        return df

    print(f"  [경고] 샘플 부족으로 제외되는 {item_label} (< {min_count}개):")
    for key in rare:
        print(f"    {key:20s}: {counts[key]:5d}개")
    _save(df[df[label_col].isin(rare)], excluded_filename)
    df = df[~df[label_col].isin(rare)].reset_index(drop=True)
    print(f"  [정보] 제외 후 남은 샘플 수: {len(df)}개")
    return df


def _safe_n_splits(df: pd.DataFrame, label_col: str, groups_col: str, n_splits: int) -> int:
    """그룹 수 / 최소 클래스 크기에 맞춰 폴드 수를 안전한 값으로 낮춘다."""
    n_groups = df[groups_col].nunique()
    min_count = int(df[label_col].value_counts().min())
    splits = max(2, min(n_splits, n_groups, min_count))
    if splits < n_splits:
        print(f"  [정보] 표본이 적어 폴드 수를 {n_splits} → {splits}로 축소합니다.")
    return splits


def _fill_missing_categorical(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """범주형 메타데이터 컬럼의 빈 문자열/NaN을 명시적 'MISSING' 카테고리로
    채운다. NaN으로 두면 LightGBM이 결측을 어떻게 다룰지 우리가 통제할 수
    없고, meta_max_size_mm 사례처럼 "값이 없다"는 사실 자체가 소스별로
    분포가 다른 신호일 수 있으므로, 최소한 그 신호를 모델이 명시적으로
    보게(그리고 나중에 감사하기 쉽게) 만든다."""
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str).replace(
                {"": _MISSING_CATEGORY, "nan": _MISSING_CATEGORY, "None": _MISSING_CATEGORY}
            )
    return df


def _apply_categorical_dtype(X: pd.DataFrame, categorical_feature_cols: list,
                              category_values: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """categorical_feature_cols를 pandas Categorical dtype으로 변환한다.
    LightGBM sklearn API는 dtype이 'category'인 컬럼을 자동으로 범주형으로
    취급한다 (categorical_feature='auto' 기본값 - 별도 인자 불필요).

    category_values가 주어지면(추론 시 재현용) 그 카테고리 집합을 그대로
    쓰고, 없으면(학습 시) X에서 관측된 값으로 새로 만들어 반환한다 -
    반환값은 그대로 category_values.pkl로 저장해서 추론 시 동일하게
    재현해야 한다 (그렇지 않으면 학습 때 못 본 카테고리 순서가 갈려
    예측이 어긋날 수 있음)."""
    X = X.copy()
    out_values = dict(category_values) if category_values else {}
    for col in categorical_feature_cols:
        if col not in X.columns:
            continue
        if category_values and col in category_values:
            cats = category_values[col]
        else:
            cats = sorted(X[col].astype(str).unique().tolist())
            out_values[col] = cats
        X[col] = pd.Categorical(X[col].astype(str), categories=cats)
    return X, out_values


def _run_cv_stage(*, stage_name: str, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray,
                   n_splits: int, lgbm_params: dict, int_to_label: dict,
                   classifier_file: str, feature_cols_file: str,
                   categorical_features_file: str, int_to_label_file: str,
                   report_prefix: str, feature_cols: list,
                   categorical_feature_cols: list,
                   category_values_file: str = "category_values.pkl",
                   dry_run: bool = False) -> dict:
    """
    Stratified-Group K-Fold CV → 전체 데이터 최종 학습 → 리포트/모델 저장까지
    한 번에 처리하는 공용 루틴.

    component_group(STEP 3) / valve_type(STEP 4) / skey(STEP 5) 세 분류기가
    이전에는 거의 동일한 코드를 3벌 복사해서 썼는데, 그 결과 콘솔 출력도
    3배로 반복됐다. 이제 이 함수 하나로 통합되어 있으므로, 출력 형식을
    바꾸고 싶으면 여기 한 곳만 고치면 된다.

    화면에는 검증 정확도(Val Acc)를 주 지표로 굵게/먼저 보여주고,
    train accuracy는 과적합 진단용 보조 수치(gap = train - val)로만
    같이 표시한다. 개별 클래스별 리포트 표는 콘솔에 통째로 찍지 않고
    CSV로 저장한 뒤 요약(정확도 + macro/weighted F1)만 보여준다.
    """
    # 범주형 메타데이터 컬럼을 pandas Categorical dtype으로 변환 (전체 X 기준
    # 카테고리 어휘 고정 - 폴드마다 다른 어휘가 생기지 않도록 CV 시작 전에
    # 한 번만 계산한다. 이 category_values는 최종적으로 저장되어 추론 시
    # 동일하게 재현된다).
    X, category_values = _apply_categorical_dtype(X, categorical_feature_cols)

    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_accs, all_true, all_pred = [], [], []
    fi_sum = np.zeros(len(feature_cols))

    print(f"\n  {'Fold':<8}{'Val Acc':>10}{'Train Acc':>12}{'Gap':>8}{'Val N':>8}")
    print("  " + "-" * 46)
    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X, y, groups), 1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        model = lgb.LGBMClassifier(**lgbm_params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(30, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )
        y_pred = model.predict(X_val)
        val_acc = accuracy_score(y_val, y_pred)
        train_acc = accuracy_score(y_tr, model.predict(X_tr))

        fold_accs.append(val_acc)
        all_true.extend(y_val.tolist())
        all_pred.extend(y_pred.tolist())
        fi_sum += model.feature_importances_

        print(f"  {f'{fold_idx}/{n_splits}':<8}{val_acc:>10.4f}{train_acc:>12.4f}"
              f"{train_acc - val_acc:>8.3f}{len(y_val):>8d}")

    mean_acc, std_acc = float(np.mean(fold_accs)), float(np.std(fold_accs))
    print("  " + "-" * 46)
    print(f"  >> {stage_name} Validation Accuracy: {mean_acc:.4f} ± {std_acc:.4f}  "
          f"(← 이 값이 CV 대표 성능입니다. Train Acc는 과적합 확인용 참고치)")

    # 전체 데이터로 최종 모델 학습
    final_model = lgb.LGBMClassifier(**lgbm_params)
    final_model.fit(X, y)

    # 평가 리포트는 dry-run(ablation 비교용)이어도 계산은 한다 - macro_f1
    # 등은 비교에 필요하기 때문. 단, 디스크 저장은 dry_run이 아닐 때만.
    unique_labels = sorted(set(all_true))
    label_names = [int_to_label.get(i, str(i)) for i in unique_labels]

    cr_dict = classification_report(
        all_true, all_pred, labels=unique_labels, target_names=label_names,
        output_dict=True, zero_division=0,
    )
    macro_f1 = cr_dict["macro avg"]["f1-score"]
    weighted_f1 = cr_dict["weighted avg"]["f1-score"]

    fi_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": fi_sum / n_splits,
    }).sort_values("importance", ascending=False)

    if dry_run:
        print(f"  [dry-run - {stage_name}] 모델/리포트 저장 생략 (ablation 비교 전용)")
        print(f"  분류 리포트 요약: macro F1={macro_f1:.4f}, weighted F1={weighted_f1:.4f}")
    else:
        print(f"\n  [저장 - {stage_name}]")
        _save(final_model, classifier_file)
        _save(feature_cols, feature_cols_file)
        _save(categorical_feature_cols, categorical_features_file)
        _save(int_to_label, int_to_label_file)
        _save(category_values, category_values_file)

        cr_file = f"{report_prefix}classification_report.csv"
        _save(pd.DataFrame(cr_dict).T, cr_file, index=True)
        print(f"  분류 리포트 요약: macro F1={macro_f1:.4f}, weighted F1={weighted_f1:.4f} "
              f"(클래스별 전체 표는 {cr_file} 참고)")

        cm = confusion_matrix(all_true, all_pred, labels=unique_labels)
        _save(pd.DataFrame(cm, index=label_names, columns=label_names),
              f"{report_prefix}confusion_matrix.csv", index=True)

        _save(fi_df, f"{report_prefix}feature_importance.csv")
        print(f"  피처 중요도 Top 5: "
              + ", ".join(f"{r.feature}({r.importance:.0f})" for r in fi_df.head(5).itertuples()))

        _save(pd.DataFrame({"fold": list(range(1, n_splits + 1)), "accuracy": fold_accs}),
              f"{report_prefix}fold_accuracies.csv")

    return {
        "cv_accuracy": mean_acc,
        "cv_std": std_acc,
        "n_samples": len(X),
        "n_splits": n_splits,
        "macro_f1": macro_f1,
        "category_values": category_values,
    }


# ═══════════════════════════════════════════════
#  밸브 세부 타입 집계
# ═══════════════════════════════════════════════

def _valve_type_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    VALVE로 분류된 컴포넌트를 세부 타입(Knife Gate, 3 Way, Ball, Gate, ...)별로 집계.

    df는 "is_valve" / "valve_type" / "skey" 컬럼을 갖고 있어야 한다
    (data_loader_csv._build_row에서 skey_mapper.get_labels_from_skey()로
    부여됨 - symbol.xml의 SKEY→Description 매핑을 이미 거친 값이므로
    여기서는 그 결과를 그대로 집계만 한다).

    component_group 희소 클래스 필터링과 무관하게 전체 밸브 통계를 보고
    싶으므로, 반드시 그 필터링 이전의 원본 df로 호출할 것.
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

    valves["valve_type"] = valves["valve_type"].fillna("UNKNOWN").replace("", "UNKNOWN")

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
    valves["valve_type"] = valves["valve_type"].fillna("UNKNOWN").replace("", "UNKNOWN")
    pivot = pd.pivot_table(
        valves, index="source_model", columns="valve_type",
        values="skey", aggfunc="count", fill_value=0,
    )
    pivot["TOTAL"] = pivot.sum(axis=1)
    return pivot.sort_values("TOTAL", ascending=False)


def _skey_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    개별 SKEY 접두어(VB/VY/VV/CK/VT/VG/...)별 집계.
    valve_type_breakdown과 같은 형태로, 어떤 valve_type이 그 SKEY로
    묶이는지도 같이 보여준다 (여러 SKEY가 한 valve_type으로 합쳐지는
    경우를 한눈에 확인하기 위함 - 예: VB/BVA가 둘 다 BALL이면 "BALL"로
    같이 표시됨).

    df는 "skey_prefix" 컬럼을 갖고 있어야 한다 (skey_mapper.skey_prefix()
    로 skey 컬럼에서 파생됨). valve_type 컬럼이 있으면 참고용으로 같이
    보여주고, 없으면 그 컬럼 없이 집계만 한다.
    """
    if df.empty or "skey_prefix" not in df.columns:
        return pd.DataFrame(columns=["skey_prefix", "count", "pct", "valve_type"])

    counts = df["skey_prefix"].value_counts()
    grouped = pd.DataFrame({"skey_prefix": counts.index, "count": counts.values})
    grouped["pct"] = (grouped["count"] / grouped["count"].sum() * 100).round(1)

    if "valve_type" in df.columns:
        vt_map = df.groupby("skey_prefix")["valve_type"].agg(
            lambda s: ", ".join(sorted(s.astype(str).unique()))
        )
        grouped["valve_type"] = grouped["skey_prefix"].map(vt_map)
    else:
        grouped["valve_type"] = ""

    return grouped.sort_values("count", ascending=False).reset_index(drop=True)


# ═══════════════════════════════════════════════
#  메인 파이프라인
# ═══════════════════════════════════════════════

def run_pipeline():
    t0 = time.time()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  AI 배관 컴포넌트 인식 – 전문가 CSV 기반 학습 파이프라인 (순수 geometry)")
    print("=" * 65)
    print(f"  TrainingModel : {TRAINING_MODEL_DIR}")
    print(f"  CSV 레이블    : {CSV_LABEL_DIR}")
    print(f"  symbol.xml    : {XML_PATH}")
    print(f"  모델 저장     : {MODEL_DIR}   (이하 저장 로그는 파일명만 표시)")
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

    if multilingual_samples:
        _save(pd.DataFrame(multilingual_samples), "multilingual_recovered_labels.csv")
        print(f"  다국어 복구 항목 {len(multilingual_samples)}개 (전문가 검증 아님 - 검토 권장)")

    if nc_samples:
        _save(pd.DataFrame(nc_samples), "nc_pending_labels.csv")
        print(f"  NC** 미정 항목 {len(nc_samples)}개")

    if fuzzy_samples:
        _save(pd.DataFrame(fuzzy_samples), "fuzzy_matched_labels.csv")
        print(f"  fuzzy 매칭 항목 {len(fuzzy_samples)}개 (학습엔 포함 - 검토 권장, 오매칭 가능성 있음)")

    _save(df, "dataset.csv")
    print(f"  데이터셋: {len(df)}개 샘플")

    # ── Step 1B: 밸브(Valve) 세부 타입 집계 ────
    # component_group 희소 클래스 필터링(Step 2)이 일어나기 전, 원본 df
    # 기준으로 집계한다 - VALVE 대분류 자체가 필터링될 일은 거의 없지만,
    # 밸브 통계는 학습/CV 대상 여부와 무관하게 항상 전체를 보고 싶기 때문.
    print("\n[STEP 1B] 밸브(Valve) 세부 타입 집계")
    valve_breakdown_df = _valve_type_breakdown(df)
    if valve_breakdown_df.empty:
        print("  [정보] VALVE로 분류된 샘플이 없습니다 (is_valve == True인 행 없음).")
    else:
        total_valves = int(valve_breakdown_df["count"].sum())
        print(f"  전체 밸브 샘플 수: {total_valves}개, 세부 타입 수: {len(valve_breakdown_df)}개")
        _print_table(
            headers=["Valve Type", "Count", "%", "SKEY 예시"],
            rows=[
                [row.valve_type, row.count, f"{row.pct:.1f}", row.skey_examples]
                for row in valve_breakdown_df.itertuples()
            ],
            aligns=["left", "right", "right", "left"],
        )
        _save(valve_breakdown_df, "valve_type_breakdown.csv")

        # 모델(프로젝트)별 밸브 타입 분포 - 프로젝트 간 비교에 유용
        valve_by_model_df = _valve_type_by_model(df)
        if not valve_by_model_df.empty:
            _save(valve_by_model_df, "valve_type_by_model.csv", index=True)

    # ── Step 2: 피처 행렬 준비 (PURE GEOMETRY) ──
    print("\n[STEP 2] 피처 행렬 준비 (순수 geometry + connector만 사용)")

    # [ABLATION - REVERTED] meta_max_size_mm/meta_min_size_mm/meta_size_available를
    # 다시 포함해 재학습한 결과, CV 정확도는 거의 그대로였지만(0.9311→0.9338)
    # 실제 RealData 추론에서 SUPPORT로의 쏠림이 크게 발생했다 (46→95건,
    # CAP/FLANGE/OLET/COUPLING에서 이동). 학습 CSV에서는 "Maximum Size가
    # 없으면 SUPPORT"라는 상관관계가 강했지만, RealData에서는 그 파라미터가
    # 없는 비-SUPPORT 부품이 많아 같은 규칙이 잘못 적용된 것으로 보인다.
    # CV 이득 대비 실사용 데이터 저하가 커서 원래대로 되돌린다.
    PURE_GEOMETRY_FEATURES = FEATURE_GROUPS["geometry"] + FEATURE_GROUPS["connector"]
    feature_cols = [c for c in PURE_GEOMETRY_FEATURES if c in df.columns]

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
    print(f"  사용 피처 {len(feature_cols)}개 "
          f"(geometry {len(FEATURE_GROUPS['geometry'])} + connector {len(FEATURE_GROUPS['connector'])}), "
          f"제외 피처 {len(excluded)}개")

    # component_group 레벨 rare-class 필터링 이전 스냅샷.
    # Step 4(Valve 세부 타입 분류기)는 이 필터링과 무관하게 VALVE로 분류된
    # 전체 원본 행을 대상으로 하므로, 여기서 미리 복사해 둔다.
    df_full = df.copy()

    print(f"  샘플 수: {len(df)}, 클래스: {sorted(df['component_group'].unique())}")
    df = _drop_rare_classes(df, "component_group", N_SPLITS,
                             "rare_classes_excluded.csv", item_label="클래스")

    X = df[feature_cols]
    y = df["group_label"].values
    groups = df["source_model"].values
    n_group_classes_trained = df["component_group"].nunique()

    # ── Step 3: component_group 분류기 ────────
    print(f"\n[STEP 3] component_group 분류기 학습 (Stratified Group {N_SPLITS}-Fold CV)")
    group_result = _run_cv_stage(
        stage_name="component_group",
        X=X, y=y, groups=groups,
        n_splits=N_SPLITS,
        lgbm_params=LGBM_PARAMS,
        int_to_label=INT_TO_GROUP,
        classifier_file="group_classifier.pkl",
        feature_cols_file="feature_cols.pkl",
        categorical_features_file="categorical_features.pkl",
        int_to_label_file="int_to_group.pkl",
        report_prefix="",
        feature_cols=feature_cols,
        categorical_feature_cols=categorical_feature_cols,
    )
    # category_values.pkl은 이제 _run_cv_stage가 직접 저장한다 (group은
    # categorical_feature_cols=[]이므로 빈 dict가 저장됨 - 추론 코드가
    # 기대하는 파일 구조는 그대로 유지됨).

    # ── Step 4: Valve 세부 타입 분류기 ─────────
    # component_group == "VALVE" 로 예측(또는 레이블링)된 행만 따로 모아,
    # 같은 geometry/connector 피처로 valve_type (Ball/Gate/Globe/...) 을
    # 예측하는 2단계(stage-2) 분류기를 별도로 학습한다.
    # 추론 시: group_classifier.pkl 로 먼저 component_group 을 예측하고,
    # 결과가 VALVE 이면 valve_type_classifier.pkl 로 세부 타입을 추가 예측.
    print("\n[STEP 4] valve_type 분류기 학습 (Ball/Gate/Globe/...)")

    valve_df = df_full[df_full["component_group"] == "VALVE"].copy()

    # valve_type == "UNKNOWN" 인 행은 skey_mapper가 SKEY 접두어를 구체적인
    # 밸브 타입으로 매핑하지 못했다는 뜻 - 학습에 잘못된 신호를 주지 않도록
    # 제외하고 별도로 저장한다. symbol.xml / _SKEY_TO_VALVE_TYPE 보완이 필요할 수 있음.
    valve_unknown_df = valve_df[valve_df["valve_type"] == "UNKNOWN"]
    if not valve_unknown_df.empty:
        _save(valve_unknown_df, "valve_type_unknown.csv")
        print(f"  [경고] valve_type UNKNOWN {len(valve_unknown_df)}개 "
              f"(symbol.xml SKEY→밸브타입 매핑 보완 필요)")
    valve_df = valve_df[valve_df["valve_type"] != "UNKNOWN"].reset_index(drop=True)

    valve_result = None
    if valve_df.empty or valve_df["valve_type"].nunique() < 2:
        print("  [정보] 밸브 세부 타입 학습에 필요한 데이터가 부족하여 이 단계를 건너뜁니다 "
              "(VALVE 샘플이 없거나 타입이 1종류뿐).")
    else:
        vt_counts = valve_df["valve_type"].value_counts()
        print(f"  밸브 세부 타입 분포 (필터링 전, {len(valve_df)}개): "
              + ", ".join(f"{vt}={cnt}" for vt, cnt in vt_counts.items()))

        valve_df = _drop_rare_classes(valve_df, "valve_type", N_SPLITS,
                                       "valve_type_rare_excluded.csv", item_label="밸브 타입")

        if valve_df["valve_type"].nunique() < 2:
            print("  [정보] 필터링 후 남은 밸브 타입이 2개 미만이라 이 단계를 건너뜁니다.")
        else:
            v_splits = _safe_n_splits(valve_df, "valve_type", "source_model", N_SPLITS)

            # 범주형 메타데이터 사용 여부에 따라 valve 단계 전용 feature_cols
            # 구성 - component_group(pure geometry) feature_cols에 meta_category/
            # meta_system_class를 덧붙인다. 결측은 "MISSING" 카테고리로 명시.
            valve_meta_cols = CATEGORICAL_META_FEATURES if ENABLE_VALVE_METADATA else []
            valve_feature_cols = feature_cols + valve_meta_cols
            valve_df_filled = _fill_missing_categorical(valve_df, CATEGORICAL_META_FEATURES)

            valve_baseline = None
            if RUN_METADATA_ABLATION and valve_meta_cols:
                print("  [ABLATION - baseline] metadata 없이 valve_type 재학습 (비교용, dry-run)")
                valve_baseline = _run_cv_stage(
                    stage_name="valve_type (baseline, no metadata)",
                    X=valve_df_filled[feature_cols],
                    y=valve_df_filled["vtype_label"].values,
                    groups=valve_df_filled["source_model"].values,
                    n_splits=v_splits,
                    lgbm_params={**LGBM_PARAMS, "num_class": NUM_VTYPE_CLASSES},
                    int_to_label=INT_TO_VTYPE,
                    classifier_file="", feature_cols_file="", categorical_features_file="",
                    int_to_label_file="", report_prefix="valve_baseline_",
                    feature_cols=feature_cols, categorical_feature_cols=[],
                    dry_run=True,
                )

            valve_result = _run_cv_stage(
                stage_name="valve_type" + (" (+ metadata)" if valve_meta_cols else ""),
                X=valve_df_filled[valve_feature_cols],
                y=valve_df_filled["vtype_label"].values,
                groups=valve_df_filled["source_model"].values,
                n_splits=v_splits,
                lgbm_params={**LGBM_PARAMS, "num_class": NUM_VTYPE_CLASSES},
                int_to_label=INT_TO_VTYPE,
                classifier_file="valve_type_classifier.pkl",
                feature_cols_file="valve_feature_cols.pkl",
                categorical_features_file="valve_categorical_features.pkl",
                int_to_label_file="int_to_vtype.pkl",
                category_values_file="valve_category_values.pkl",
                report_prefix="valve_",
                feature_cols=valve_feature_cols,
                categorical_feature_cols=valve_meta_cols,
            )
            valve_result["n_types"] = valve_df["valve_type"].nunique()
            if valve_baseline is not None:
                delta = valve_result["cv_accuracy"] - valve_baseline["cv_accuracy"]
                sign = "+" if delta >= 0 else ""
                print(f"  >> [ABLATION] valve_type metadata 효과: "
                      f"{valve_baseline['cv_accuracy']:.4f} → {valve_result['cv_accuracy']:.4f} "
                      f"({sign}{delta:.4f})")

    # ── Step 5: 개별 SKEY 단위 분류기 ──────────
    # Step 4의 valve_type(BALL/GATE/GLOBE/...)은 여러 SKEY 접두어를 한
    # 클래스로 묶은 값이다 (예: VB/BVA/BVA2 → 전부 "BALL"). 이 단계는 같은
    # geometry/connector 피처로, 더 세밀한 "개별 SKEY 접두어" 단위(VB vs
    # BVA vs BVA2, VT vs KG, ...)를 예측하는 3단계(stage-3) 분류기를 별도로
    # 학습한다.
    #
    # 주의 - match_tier가 "multilingual_*"인 행은 제외한다: 이런 행은 전문가가
    # 부여한 진짜 SKEY가 없고, family/type 이름을 다국어 키워드 사전과 대조해
    # "이건 아마 BALL 밸브겠다" 정도로 valve_type만 추정한 것이라 skey 컬럼이
    # "MULTILINGUAL:<키워드>" placeholder다 - 어떤 개별 SKEY였는지는 원래
    # 알 수 없다. 따라서 이 단계의 학습 표본은 Step 4(valve_type)보다 훨씬
    # 적어질 수 있다 (특히 BALL/BUTTERFLY처럼 다국어 복구 비중이 큰 타입).
    #
    # 추가 - match_tier가 "fuzzy"인 행도 이 단계에서는 제외한다. fuzzy는
    # family/type 이름의 부분/유사 문자열 매칭이라 오매칭 위험이 실제로
    # 있고(위 "fuzzy 매칭 항목 ... 학습엔 포함 - 검토 권장, 오매칭 가능성
    # 있음" 로그가 이미 그 위험을 인지하고 있었음), Step 4(valve_type,
    # ~7클래스)보다 이 단계(개별 SKEY, ~6클래스 세분류)가 라벨 노이즈에
    # 훨씬 민감하다. group/valve_type 단계는 클래스가 더 넓어 오매칭 한
    # 건의 영향이 상대적으로 작으므로 그대로 둔다 - fuzzy 제외는 skey
    # 단계 전용이다.
    print("\n[STEP 5] 개별 SKEY 단위(VB/BVA/VT/KG/...) 분류기 학습")

    _skey_excluded_tiers = ("multilingual", "fuzzy")
    skey_df = valve_df[~valve_df["match_tier"].str.startswith(_skey_excluded_tiers)].copy() \
        if valve_result is not None else pd.DataFrame()
    skey_result = None

    if skey_df.empty:
        print("  [정보] 개별 SKEY 분류 학습에 필요한 데이터가 없어 이 단계를 건너뜁니다.")
    else:
        skey_df["skey_prefix"] = skey_df["skey"].apply(skey_prefix)
        n_excluded = len(valve_df) - len(skey_df)
        print(f"  multilingual/fuzzy 추정 레이블 제외: {n_excluded}개 "
              f"(exact/normalized SKEY 표본 {len(skey_df)}개만 사용)")

        if skey_df["skey_prefix"].nunique() < 2:
            print("  [정보] 진짜 SKEY 종류가 1개뿐이라 이 단계를 건너뜁니다.")
        else:
            skey_breakdown_df = _skey_breakdown(skey_df)
            print(f"  개별 SKEY 분포 (필터링 전, {len(skey_df)}개, {len(skey_breakdown_df)}종):")
            _print_table(
                headers=["SKEY", "Count", "%", "Valve Type"],
                rows=[
                    [row.skey_prefix, row.count, f"{row.pct:.1f}", row.valve_type]
                    for row in skey_breakdown_df.itertuples()
                ],
                aligns=["left", "right", "right", "left"],
            )
            _save(skey_breakdown_df, "skey_breakdown.csv")

            skey_df = _drop_rare_classes(skey_df, "skey_prefix", N_SPLITS,
                                          "skey_rare_excluded.csv", item_label="SKEY")

            if skey_df["skey_prefix"].nunique() < 2:
                print("  [정보] 필터링 후 남은 SKEY 종류가 2개 미만이라 이 단계를 건너뜁니다.")
            else:
                # 이 학습 실행에서 실제로 등장한 SKEY 접두어만으로 클래스 목록을
                # 동적으로 구성한다 (group/valve_type처럼 전체 코드 체계를 미리
                # 하드코딩해두지 않음 - symbol.xml/CSV가 갱신되면 그대로 반영됨).
                SKEY_CLASSES = sorted(skey_df["skey_prefix"].unique())
                SKEY_TO_INT = {s: i for i, s in enumerate(SKEY_CLASSES)}
                INT_TO_SKEY = {i: s for s, i in SKEY_TO_INT.items()}
                skey_df["skey_label"] = skey_df["skey_prefix"].map(SKEY_TO_INT)

                sk_splits = _safe_n_splits(skey_df, "skey_prefix", "source_model", N_SPLITS)

                skey_meta_cols = CATEGORICAL_META_FEATURES if ENABLE_VALVE_METADATA else []
                skey_feature_cols = feature_cols + skey_meta_cols
                skey_df_filled = _fill_missing_categorical(skey_df, CATEGORICAL_META_FEATURES)

                skey_baseline = None
                if RUN_METADATA_ABLATION and skey_meta_cols:
                    print("  [ABLATION - baseline] metadata 없이 skey 재학습 (비교용, dry-run)")
                    skey_baseline = _run_cv_stage(
                        stage_name="skey (baseline, no metadata)",
                        X=skey_df_filled[feature_cols],
                        y=skey_df_filled["skey_label"].values,
                        groups=skey_df_filled["source_model"].values,
                        n_splits=sk_splits,
                        lgbm_params={**LGBM_PARAMS, "num_class": len(SKEY_CLASSES)},
                        int_to_label=INT_TO_SKEY,
                        classifier_file="", feature_cols_file="", categorical_features_file="",
                        int_to_label_file="", report_prefix="skey_baseline_",
                        feature_cols=feature_cols, categorical_feature_cols=[],
                        dry_run=True,
                    )

                skey_result = _run_cv_stage(
                    stage_name="skey" + (" (+ metadata)" if skey_meta_cols else ""),
                    X=skey_df_filled[skey_feature_cols],
                    y=skey_df_filled["skey_label"].values,
                    groups=skey_df_filled["source_model"].values,
                    n_splits=sk_splits,
                    lgbm_params={**LGBM_PARAMS, "num_class": len(SKEY_CLASSES)},
                    int_to_label=INT_TO_SKEY,
                    classifier_file="skey_classifier.pkl",
                    feature_cols_file="skey_feature_cols.pkl",
                    categorical_features_file="skey_categorical_features.pkl",
                    int_to_label_file="int_to_skey.pkl",
                    category_values_file="skey_category_values.pkl",
                    report_prefix="skey_",
                    feature_cols=skey_feature_cols,
                    categorical_feature_cols=skey_meta_cols,
                )
                skey_result["n_skeys"] = skey_df["skey_prefix"].nunique()
                skey_result["n_multilingual_or_fuzzy_dropped"] = n_excluded
                if skey_baseline is not None:
                    delta = skey_result["cv_accuracy"] - skey_baseline["cv_accuracy"]
                    sign = "+" if delta >= 0 else ""
                    print(f"  >> [ABLATION] skey metadata 효과: "
                          f"{skey_baseline['cv_accuracy']:.4f} → {skey_result['cv_accuracy']:.4f} "
                          f"({sign}{delta:.4f})")

    # ── 최종 요약 ──────────────────────────────
    elapsed = time.time() - t0
    print("\n" + "=" * 65)
    print("  파이프라인 완료  (검증 정확도 = Validation Accuracy 기준)")
    print("=" * 65)
    print(f"  소요 시간           : {elapsed:.1f}초")
    print(f"  학습 모델(프로젝트) : {df['source_model'].nunique()}개  |  학습 샘플: {len(df)}개")
    print(f"  제외 - NC** {len(nc_samples)}개, fuzzy 매칭(포함됨) {len(fuzzy_samples)}개, "
          f"다국어 복구(포함됨) {len(multilingual_samples)}개")
    summary_rows = [
        ["component_group", f"{group_result['cv_accuracy']:.4f} ± {group_result['cv_std']:.3f}",
         group_result["n_samples"], n_group_classes_trained],
    ]
    if valve_result is not None:
        summary_rows.append(
            ["valve_type", f"{valve_result['cv_accuracy']:.4f} ± {valve_result['cv_std']:.3f}",
             valve_result["n_samples"], valve_result["n_types"]]
        )
    if skey_result is not None:
        summary_rows.append(
            ["skey (개별)", f"{skey_result['cv_accuracy']:.4f} ± {skey_result['cv_std']:.3f}",
             skey_result["n_samples"], skey_result["n_skeys"]]
        )
    _print_table(
        headers=["분류기", "Val Acc", "표본 수", "클래스 수"],
        rows=summary_rows,
        aligns=["left", "right", "right", "right"],
    )
    print(f"  모델 저장 위치: {MODEL_DIR}")
    print("=" * 65)

    return {
        "cv_accuracy": group_result["cv_accuracy"],
        "cv_std":      group_result["cv_std"],
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