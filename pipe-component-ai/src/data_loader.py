"""
data_loader.py
--------------
BIM모델_데이터 폴더의 JSON 파일들을 일괄 로드하여
피처 + 레이블이 포함된 pandas DataFrame 을 생성한다.

레이블 우선순위:
  1. 학습용레이블_정답.xlsx  (SKY → symbol.xml → component_group / valve_type)
  2. 파일명 기반 규칙 (label_extractor.py fallback)

사용 예:
    from src.data_loader import load_dataset

    df = load_dataset(
        data_dir  = "BIM모델_데이터",
        xlsx_path = "BIM모델_데이터/학습용레이블_정답.xlsx",
        xml_path  = "BIM모델_데이터/symbol.xml",
    )
"""

import json
from pathlib import Path

import pandas as pd

from src.feature_extractor import extract_features, NUMERIC_FEATURE_COLS
from src.label_loader import (
    LabelRegistry,
    extract_labels_with_fallback,
)
from src.skey_mapper import GROUP_TO_INT, VTYPE_TO_INT


# ─────────────────────────────────────────────
#  JSON 로더
# ─────────────────────────────────────────────

def _load_json(path: Path) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] JSON 로드 실패: {path.name} → {e}")
        return None


# ─────────────────────────────────────────────
#  데이터셋 빌더
# ─────────────────────────────────────────────

def load_dataset(
    data_dir:   str | Path,
    xlsx_path:  str | Path | None = None,
    xml_path:   str | Path | None = None,
    verbose:    bool = True,
) -> pd.DataFrame:
    """
    data_dir 내 모든 *.json 파일을 읽어 피처+레이블 DataFrame 반환.

    Parameters
    ----------
    data_dir  : JSON 파일이 있는 폴더 경로
    xlsx_path : 학습용레이블_정답.xlsx 경로
                (None 이면 data_dir/학습용레이블_정답.xlsx 자동 탐색)
    xml_path  : symbol.xml 경로
                (None 이면 data_dir/symbol.xml 자동 탐색)
    verbose   : 진행 상황 출력 여부

    Returns
    -------
    pd.DataFrame
        컬럼:
          source_file, meta_family_name, meta_type_name,
          meta_category, meta_system_class,
          [NUMERIC_FEATURE_COLS],
          skey, component_group, valve_type, is_valve,
          group_label, vtype_label, label_source
    """
    data_dir = Path(data_dir)
    json_files = sorted(data_dir.glob("*.json"))

    if not json_files:
        raise FileNotFoundError(f"JSON 파일을 찾을 수 없습니다: {data_dir}")

    # ── xlsx / xml 경로 자동 탐색 ──────────────
    if xlsx_path is None:
        cand = data_dir / "학습용레이블_정답.xlsx"
        xlsx_path = cand if cand.exists() else None

    if xml_path is None:
        cand = data_dir / "symbol.xml"
        xml_path = cand if cand.exists() else None

    # ── LabelRegistry 초기화 ────────────────────
    registry: LabelRegistry | None = None
    if xlsx_path and Path(xlsx_path).exists():
        try:
            registry = LabelRegistry(xlsx_path=xlsx_path, xml_path=xml_path)
        except Exception as e:
            print(f"[WARN] LabelRegistry 초기화 실패: {e} → 파일명 기반 fallback 사용")
    else:
        if verbose:
            print(f"[INFO] xlsx 파일 없음 → 파일명 기반 레이블 사용")

    if verbose:
        print(f"[INFO] JSON 파일 {len(json_files)}개 발견 → 처리 시작")

    # ── JSON 순회 ────────────────────────────────
    rows    = []
    skipped = []
    label_sources = {"xlsx": 0, "filename": 0}

    for jf in json_files:
        if verbose:
            print(f"  로드: {jf.name}")

        data = _load_json(jf)
        if data is None:
            skipped.append(jf.name)
            continue

        # 피처 추출
        try:
            feat = extract_features(data)
        except Exception as e:
            print(f"[WARN] 피처 추출 실패: {jf.name} → {e}")
            skipped.append(jf.name)
            continue

        # 레이블 추출
        family_name = feat.get("meta_family_name", "")
        type_name   = feat.get("meta_type_name",   "")

        labels = extract_labels_with_fallback(
            family_name   = family_name,
            type_name     = type_name,
            filename_stem = jf.stem,
            registry      = registry,
        )

        src = "xlsx" if labels.get("skey", "") else "filename"
        label_sources[src] += 1

        row = {
            "source_file":       jf.stem,
            "meta_family_name":  family_name,
            "meta_type_name":    type_name,
            "meta_category":     feat.get("meta_category",     ""),
            "meta_system_class": feat.get("meta_system_class", ""),
        }

        # 수치 피처
        for col in NUMERIC_FEATURE_COLS:
            row[col] = feat.get(col, 0.0)

        # 레이블
        row["skey"]            = labels.get("skey", "")
        row["component_group"] = labels["component_group"]
        row["valve_type"]      = labels["valve_type"]
        row["is_valve"]        = labels["is_valve"]
        row["group_label"]     = GROUP_TO_INT.get(labels["component_group"],
                                                   len(GROUP_TO_INT) - 1)
        row["vtype_label"]     = VTYPE_TO_INT.get(labels["valve_type"],
                                                   len(VTYPE_TO_INT) - 1)
        row["label_source"]    = src

        rows.append(row)

    if not rows:
        raise ValueError("유효한 데이터가 없습니다.")

    df = pd.DataFrame(rows)

    if verbose:
        print(f"\n[INFO] 데이터셋 로드 완료: {len(df)}개 샘플")
        print(f"[INFO] 레이블 출처: xlsx={label_sources['xlsx']}개 / filename={label_sources['filename']}개")
        print(f"[INFO] 스킵된 파일: {len(skipped)}개 → {skipped}")
        print(f"\n컴포넌트 그룹 분포:\n{df['component_group'].value_counts().to_string()}")
        print(f"\n밸브 타입 분포:\n{df['valve_type'].value_counts().to_string()}")

    return df


def get_feature_matrix(df: pd.DataFrame, feature_cols: list[str] | None = None):
    """
    DataFrame → (X, y_group, y_vtype) 반환.

    Parameters
    ----------
    df           : load_dataset() 결과
    feature_cols : 사용할 피처 컬럼 목록. None 이면 NUMERIC_FEATURE_COLS 전부

    Returns
    -------
    X        : pd.DataFrame  (수치 피처)
    y_group  : pd.Series     (group_label)
    y_vtype  : pd.Series     (vtype_label)
    """
    if feature_cols is None:
        feature_cols = NUMERIC_FEATURE_COLS

    available = [c for c in feature_cols if c in df.columns]
    X       = df[available].copy()
    y_group = df["group_label"]
    y_vtype = df["vtype_label"]

    return X, y_group, y_vtype


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    data_path = sys.argv[1] if len(sys.argv) > 1 else "../../BIM모델_데이터"
    df = load_dataset(data_path, verbose=True)

    print("\n=== 상위 5행 ===")
    print(df[["source_file", "skey", "component_group", "valve_type",
              "label_source", "geo_volume", "conn_count"]].head().to_string())

    print("\n=== 피처 통계 ===")
    print(df[NUMERIC_FEATURE_COLS].describe().T.to_string())
