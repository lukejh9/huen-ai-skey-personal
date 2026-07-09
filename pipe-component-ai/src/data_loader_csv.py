"""
data_loader_csv.py
------------------
TrainingModel 폴더 구조 + 전문가_정의_csv/ CSV 레이블을 읽어
학습용 DataFrame을 생성한다.

구조:
  TrainingModel/
    {model_name}/          ← 모델별 폴더
      component.json
      ...
  전문가_정의_csv/
    {model_name}.csv       ← 모델별 전문가 레이블

CSV 컬럼: FamilyAndTypeName, ItemKey, RevitId
  - FamilyAndTypeName: "FamilyName: TypeName"
  - ItemKey: SKEY 코드 (NC** = 미정, _P** = 파이프 제외)

NC** 처리: 학습 대상에서 제외
_P** 처리: 파이프 타입으로 제외

병렬화 (변경 사항):
  - 모델 폴더 단위로 ProcessPoolExecutor 사용 (216개 폴더는 서로 독립적)
  - extract_features()가 CPU 연산 위주라 스레드 대신 프로세스 사용
    (파이썬 GIL 때문에 스레드로는 CPU 작업이 실제 병렬화되지 않음)
"""

import csv as csv_module
import json
import re
from pathlib import Path
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

import pandas as pd

from src.feature_extractor import extract_features, NUMERIC_FEATURE_COLS
from src.skey_mapper import (
    parse_symbol_xml,
    get_labels_from_skey,
    GROUP_TO_INT, VTYPE_TO_INT,
)


# ── 제외할 SKEY 패턴 ──────────────────────────
def _is_excluded_skey(skey: str) -> bool:
    """NC** (미정) 또는 _P** (파이프) → 학습 제외."""
    s = skey.strip().upper()
    if s.startswith("NC"):
        return True
    if s.startswith("_P"):
        return True
    return False


# ── CSV 파싱 ──────────────────────────────────
def _load_model_csv(csv_path: Path) -> dict[str, str]:
    """
    CSV 파일을 읽어 {normalize(FamilyAndTypeName): ItemKey} 반환.
    FamilyAndTypeName = "FamilyName: TypeName"
    """
    lookup = {}
    try:
        # UTF-8 BOM 대응
        with open(csv_path, encoding="utf-8-sig", newline="") as f:
            reader = csv_module.DictReader(f)
            for row in reader:
                fat = row.get("FamilyAndTypeName", "").strip()
                key = row.get("ItemKey", "").strip()
                if fat and key:
                    lookup[fat] = key
    except Exception as e:
        print(f"[WARN] CSV 읽기 실패: {csv_path.name} → {e}")
    return lookup


def _normalize(text: str) -> str:
    """비교용 정규화: 대소문자/공백/특수문자 무시."""
    return re.sub(r"[\s_\-\.\(\)]+", "", text).upper()


def _find_skey(
    family_name: str,
    type_name: str,
    csv_lookup: dict[str, str],
) -> Optional[str]:
    """
    CSV lookup에서 (family_name, type_name)에 해당하는 SKEY 반환.
    None이면 매칭 실패.
    """
    # 1) "FamilyName: TypeName" 형식으로 조합
    candidates = []
    if type_name:
        candidates.append(f"{family_name}: {type_name}")
        candidates.append(f"{family_name}:{type_name}")
    candidates.append(family_name)

    # 정확 매칭
    for cand in candidates:
        if cand in csv_lookup:
            return csv_lookup[cand]

    # 정규화 매칭
    norm_cands = [_normalize(c) for c in candidates]
    norm_map = {_normalize(k): v for k, v in csv_lookup.items()}
    for nc in norm_cands:
        if nc in norm_map:
            return norm_map[nc]

    # 패밀리명만으로 부분 매칭
    norm_fam = _normalize(family_name)
    for k, v in norm_map.items():
        if norm_fam in k or k in norm_fam:
            return v

    return None


# ── JSON 로더 ─────────────────────────────────
def _load_json(path: Path) -> Optional[dict]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# ── 모델 1개 처리 (워커 함수 - 병렬 실행됨) ──────
def _process_one_model(args):
    """
    모델 폴더 1개를 처리 - 별도 프로세스에서 실행됨.
    Windows에서 pickling 되려면 반드시 모듈 최상위 함수여야 함.

    Returns
    -------
    dict with keys: rows, nc_samples, stats (partial counts for this model)
    """
    model_dir_str, csv_path_str, skey_map = args
    model_dir = Path(model_dir_str)
    csv_path = Path(csv_path_str)
    model_name = model_dir.name

    csv_lookup = _load_model_csv(csv_path)

    rows = []
    nc_samples = []
    stats = {
        "json_total":      0,
        "json_matched":    0,
        "json_nc_skip":    0,
        "json_pipe_skip":  0,
        "json_no_match":   0,
        "json_feat_error": 0,
    }

    json_files = sorted(model_dir.glob("*.json"))
    stats["json_total"] = len(json_files)

    for jf in json_files:
        data = _load_json(jf)
        if data is None:
            continue

        try:
            feat = extract_features(data)
        except Exception:
            stats["json_feat_error"] += 1
            continue

        family_name = feat.get("meta_family_name", "")
        type_name   = feat.get("meta_type_name",   "")

        skey = _find_skey(family_name, type_name, csv_lookup)

        if skey is None:
            stats["json_no_match"] += 1
            continue

        if skey.strip().upper().startswith("NC"):
            stats["json_nc_skip"] += 1
            nc_samples.append({
                "model": model_name,
                "file": jf.name,
                "family": family_name,
                "type": type_name,
                "skey": skey,
            })
            continue

        if _is_excluded_skey(skey):
            stats["json_pipe_skip"] += 1
            continue

        labels = get_labels_from_skey(skey, skey_map)
        stats["json_matched"] += 1

        row = {
            "source_model":      model_name,
            "source_file":       jf.stem,
            "meta_family_name":  family_name,
            "meta_type_name":    type_name,
            "meta_category":     feat.get("meta_category",     ""),
            "meta_system_class": feat.get("meta_system_class", ""),
        }
        for col in NUMERIC_FEATURE_COLS:
            row[col] = feat.get(col, 0.0)

        row["skey"]            = skey
        row["component_group"] = labels["component_group"]
        row["valve_type"]      = labels["valve_type"]
        row["is_valve"]        = labels["is_valve"]
        row["group_label"]     = GROUP_TO_INT.get(labels["component_group"],
                                                   len(GROUP_TO_INT) - 1)
        row["vtype_label"]     = VTYPE_TO_INT.get(labels["valve_type"],
                                                   len(VTYPE_TO_INT) - 1)
        rows.append(row)

    return {"rows": rows, "nc_samples": nc_samples, "stats": stats}


# ── 메인 데이터 로드 함수 (병렬화) ──────────────
def load_dataset_csv(
    training_model_dir: str | Path,
    csv_label_dir: str | Path,
    xml_path: str | Path | None = None,
    verbose: bool = True,
    n_workers: int | None = None,
) -> pd.DataFrame:
    """
    TrainingModel 폴더 구조 + CSV 레이블로 학습 데이터셋 생성.
    모델 폴더 단위로 병렬 처리 (ProcessPoolExecutor).

    Parameters
    ----------
    training_model_dir : TrainingModel 폴더 경로
    csv_label_dir      : 전문가_정의_csv 폴더 경로
    xml_path           : symbol.xml 경로 (SKEY 매핑용)
    verbose            : 진행 상황 출력
    n_workers          : 병렬 워커 수 (기본값: CPU 코어 수)

    Returns
    -------
    pd.DataFrame with columns:
        source_model, source_file, meta_family_name, meta_type_name,
        [NUMERIC_FEATURE_COLS],
        skey, component_group, valve_type, is_valve,
        group_label, vtype_label
    """
    training_model_dir = Path(training_model_dir)
    csv_label_dir      = Path(csv_label_dir)

    if n_workers is None:
        n_workers = os.cpu_count() or 4

    # symbol.xml 파싱
    skey_map = {}
    if xml_path and Path(xml_path).exists():
        skey_map = parse_symbol_xml(Path(xml_path))
        if verbose:
            print(f"[INFO] symbol.xml 로드: {len(skey_map)}개 SKEY")

    # CSV 파일 목록 → 모델명 집합
    csv_files = {
        f.stem: f
        for f in csv_label_dir.glob("*.csv")
        if f.is_file()
    }
    if verbose:
        print(f"[INFO] 전문가 CSV 파일: {len(csv_files)}개 모델")

    # TrainingModel 폴더 목록
    model_dirs = sorted([
        d for d in training_model_dir.iterdir()
        if d.is_dir()
    ])
    if verbose:
        print(f"[INFO] TrainingModel 폴더: {len(model_dirs)}개")

    # 레이블 있는 모델만 작업 목록에 추가
    tasks = []
    models_no_label = 0
    for model_dir in model_dirs:
        model_name = model_dir.name
        if model_name not in csv_files:
            models_no_label += 1
            continue
        tasks.append((str(model_dir), str(csv_files[model_name]), skey_map))

    if verbose:
        print(f"[INFO] 병렬 처리 워커 수: {n_workers}개 "
              f"(모델 {len(tasks)}개 처리 예정)")

    rows = []
    nc_samples = []
    stats = {
        "models_with_label": len(tasks),
        "models_no_label":   models_no_label,
        "json_total":        0,
        "json_matched":      0,
        "json_nc_skip":      0,
        "json_pipe_skip":    0,
        "json_no_match":     0,
        "json_feat_error":   0,
    }

    completed = 0
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_process_one_model, task): task for task in tasks}
        for future in as_completed(futures):
            result = future.result()
            rows.extend(result["rows"])
            nc_samples.extend(result["nc_samples"])
            for k, v in result["stats"].items():
                stats[k] += v

            completed += 1
            if verbose and completed % 20 == 0:
                print(f"  [진행] {completed}/{len(tasks)} 모델 처리 완료")

    if verbose:
        print(f"\n=== 데이터 로드 결과 ===")
        print(f"  레이블 있는 모델  : {stats['models_with_label']}개")
        print(f"  레이블 없는 모델  : {stats['models_no_label']}개 (제외)")
        print(f"  JSON 총계         : {stats['json_total']}개")
        print(f"  학습 사용         : {stats['json_matched']}개")
        print(f"  NC** 제외         : {stats['json_nc_skip']}개 (추후 보완 예정)")
        print(f"  Pipe 제외         : {stats['json_pipe_skip']}개")
        print(f"  CSV 매칭 실패     : {stats['json_no_match']}개")
        print(f"  피처 추출 오류    : {stats['json_feat_error']}개")

    if not rows:
        raise ValueError("유효한 학습 데이터가 없습니다.")

    df = pd.DataFrame(rows)

    if verbose:
        print(f"\n컴포넌트 그룹 분포:\n{df['component_group'].value_counts().to_string()}")
        print(f"\n총 학습 샘플: {len(df)}개")

    return df, nc_samples