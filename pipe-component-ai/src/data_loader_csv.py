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
from src.multilingual_recovery import recover_row as _recover_multilingual


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


# 부분(fuzzy) 매칭을 허용할 최소 정규화 문자열 길이.
# 이보다 짧으면 우연히 겹칠 확률이 높아 매칭을 포기한다 (오매칭 방지).
_MIN_FUZZY_LEN = 4

# 매칭 신뢰도 등급 (감사/필터링용)
MATCH_TIER_EXACT      = "exact"       # "FamilyName: TypeName" 원문 그대로 일치
MATCH_TIER_NORMALIZED = "normalized"  # 공백/특수문자/대소문자만 다름
MATCH_TIER_FUZZY       = "fuzzy"      # 부분 문자열 포함 관계로 추정한 매칭 (검증 필요)
MATCH_TIER_NONE        = "no_match"


def _find_skey(
    family_name: str,
    type_name: str,
    csv_lookup: dict[str, str],
) -> tuple[Optional[str], str]:
    """
    CSV lookup에서 (family_name, type_name)에 해당하는 SKEY 반환.

    Returns
    -------
    (skey, match_tier) : (Optional[str], str)
        skey가 None이면 매칭 실패 (match_tier == MATCH_TIER_NONE).
        match_tier는 매칭이 얼마나 신뢰할 수 있는지를 나타낸다:
          - "exact"      : 원문 그대로 일치 → 가장 신뢰도 높음
          - "normalized" : 공백/기호/대소문자 차이만 있음 → 신뢰도 높음
          - "fuzzy"      : 부분 문자열 포함으로 추정 → 반드시 감사(audit) 필요
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
            return csv_lookup[cand], MATCH_TIER_EXACT

    # 정규화 매칭
    norm_cands = [_normalize(c) for c in candidates]
    norm_map = {_normalize(k): v for k, v in csv_lookup.items()}
    for nc in norm_cands:
        if nc in norm_map:
            return norm_map[nc], MATCH_TIER_NORMALIZED

    # 패밀리명만으로 부분(fuzzy) 매칭 - 최후의 수단.
    # - 너무 짧은 문자열은 우연히 포함 관계가 성립하기 쉬우므로 제외
    # - 여러 후보가 겹치면 "가장 긴 공통 구간"을 가진, 즉 가장 구체적인
    #   매칭을 선택한다 (기존 코드는 dict 순회 순서상 첫 매칭을 그냥 반환해서
    #   재현성이 없고 품질이 더 나쁜 매칭이 이길 수 있었음).
    norm_fam = _normalize(family_name)
    if len(norm_fam) >= _MIN_FUZZY_LEN:
        best_match = None
        best_score = -1
        for k, v in norm_map.items():
            if len(k) < _MIN_FUZZY_LEN:
                continue
            if norm_fam in k or k in norm_fam:
                score = min(len(norm_fam), len(k))
                if score > best_score:
                    best_score = score
                    best_match = v
        if best_match is not None:
            return best_match, MATCH_TIER_FUZZY

    return None, MATCH_TIER_NONE


# ── JSON 로더 ─────────────────────────────────
def _load_json(path: Path) -> Optional[dict]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _build_row(model_name, jf, family_name, type_name, feat, skey, match_tier, labels) -> dict:
    """Assemble one training row dict. Shared by the real-SKEY path and the
    multilingual-recovery path so both produce identically-shaped rows."""
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
    row["match_tier"]      = match_tier
    row["component_group"] = labels["component_group"]
    row["valve_type"]      = labels["valve_type"]
    row["is_valve"]        = labels["is_valve"]
    row["group_label"]     = GROUP_TO_INT.get(labels["component_group"],
                                               len(GROUP_TO_INT) - 1)
    row["vtype_label"]     = VTYPE_TO_INT.get(labels["valve_type"],
                                               len(VTYPE_TO_INT) - 1)
    return row


# ── 모델 1개 처리 (워커 함수 - 병렬 실행됨) ──────
def _process_one_model(args):
    """
    모델 폴더 1개를 처리 - 별도 프로세스에서 실행됨.
    Windows에서 pickling 되려면 반드시 모듈 최상위 함수여야 함.

    Returns
    -------
    dict with keys: rows, nc_samples, multilingual_samples, stats
        (partial counts for this model)
    """
    (model_dir_str, csv_path_str, skey_map,
     enable_multilingual, multilingual_min_tier) = args
    model_dir = Path(model_dir_str)
    csv_path = Path(csv_path_str)
    model_name = model_dir.name

    csv_lookup = _load_model_csv(csv_path)

    rows = []
    nc_samples = []
    fuzzy_samples = []
    multilingual_samples = []
    stats = {
        "json_total":      0,
        "json_matched":    0,
        "json_nc_skip":    0,
        "json_pipe_skip":  0,
        "json_no_match":   0,
        "json_feat_error": 0,
        "json_multilingual_recovered": 0,
        "match_exact":      0,
        "match_normalized": 0,
        "match_fuzzy":      0,
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

        skey, match_tier = _find_skey(family_name, type_name, csv_lookup)

        if skey is None:
            stats["json_no_match"] += 1
            continue

        stats[f"match_{match_tier}"] = stats.get(f"match_{match_tier}", 0) + 1

        if match_tier == MATCH_TIER_FUZZY:
            # 부분 문자열 매칭은 오매칭 가능성이 있으므로 별도 감사 목록에 기록.
            # 학습 데이터에서 자동으로 빼지는 않되(데이터 손실 방지),
            # match_tier 컬럼으로 표시해 두어 나중에 필터링/검토할 수 있게 한다.
            fuzzy_samples.append({
                "model": model_name,
                "file": jf.name,
                "family": family_name,
                "type": type_name,
                "matched_skey": skey,
            })

        if skey.strip().upper().startswith("NC"):
            recovered_group = None
            if enable_multilingual:
                recovered_group, recovered_vtype, matched_kw, tier = _recover_multilingual(
                    family_name, type_name, min_tier=multilingual_min_tier,
                )

            if recovered_group is not None:
                # Recovered via multilingual keyword match - NOT an expert
                # label. Included in training but tagged so it can be
                # filtered out later (mirrors the existing "fuzzy" pattern).
                stats["json_multilingual_recovered"] += 1
                labels = {
                    "component_group": recovered_group,
                    "valve_type":      recovered_vtype or "UNKNOWN",
                    "is_valve":        recovered_group == "VALVE",
                }
                multilingual_samples.append({
                    "model": model_name,
                    "file": jf.name,
                    "family": family_name,
                    "type": type_name,
                    "recovered_group": recovered_group,
                    "matched_keyword": matched_kw,
                    "confidence_tier": tier,
                })
                rows.append(_build_row(
                    model_name, jf, family_name, type_name, feat,
                    skey=f"MULTILINGUAL:{matched_kw}",
                    match_tier=f"multilingual_{tier}",
                    labels=labels,
                ))
                continue

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
        rows.append(_build_row(model_name, jf, family_name, type_name, feat,
                                skey, match_tier, labels))

    return {
        "rows": rows,
        "nc_samples": nc_samples,
        "fuzzy_samples": fuzzy_samples,
        "multilingual_samples": multilingual_samples,
        "stats": stats,
    }


# ── 메인 데이터 로드 함수 (병렬화) ──────────────
def load_dataset_csv(
    training_model_dir: str | Path,
    csv_label_dir: str | Path,
    xml_path: str | Path | None = None,
    verbose: bool = True,
    n_workers: int | None = None,
    enable_multilingual: bool = False,
    multilingual_min_tier: str = "specific",
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
    enable_multilingual : bool, default False
        CSV 레이블이 "NC**" (미정)인 항목에 대해, family/type 이름을
        다국어 키워드 사전(src/multilingual_recovery.py)과 대조해서
        component_group을 복구 시도한다. 기본값 False - 이 기능을 켜기
        전에 nc_recovered_labels.csv를 사람이 검토했는지 확인할 것.
        (전문가가 부여한 진짜 SKEY가 아니라 추정 레이블이므로 반드시
        match_tier="multilingual_*"로 표시되어 이후 필터링 가능함.)
    multilingual_min_tier : "specific" (기본값, 보수적) | "generic"
        복구 시 허용할 최소 신뢰도 등급. "specific"는 다중 문자/복합어
        (예: "球阀"=볼밸브)만 허용하고, 단일 일반 단어(예: 그냥 "VALVE"/
        "阀")로만 매칭된 항목은 무시한다. "generic"으로 낮추면 그런
        항목도 포함하지만 오매칭 위험이 커지므로 사전에 반드시 검토 필요.

    Returns
    -------
    (df, nc_samples, fuzzy_samples, multilingual_samples) : tuple
        df : pd.DataFrame with columns:
            source_model, source_file, meta_family_name, meta_type_name,
            [NUMERIC_FEATURE_COLS],
            skey, match_tier, component_group, valve_type,
            group_label, vtype_label
            (match_tier is one of "exact" / "normalized" / "fuzzy" /
             "multilingual_specific" / "multilingual_generic" -
             see _find_skey; filter to match_tier not in
             {"fuzzy", "multilingual_specific", "multilingual_generic"}
             if you want to train only on genuine expert labels)
        nc_samples : list[dict]
            JSON files whose expert-labeled SKEY was "NC**" (미정) and
            could NOT be recovered (or enable_multilingual=False) -
            excluded from training, kept for manual review.
        fuzzy_samples : list[dict]
            JSON files whose CSV label was resolved via substring/fuzzy
            matching rather than an exact or normalized match. These ARE
            included in df (tagged match_tier="fuzzy") but should be
            spot-checked, since fuzzy matching can silently mislabel data.
        multilingual_samples : list[dict]
            JSON files whose "NC**" label was recovered via the
            multilingual keyword pass. These ARE included in df (tagged
            match_tier="multilingual_specific"/"multilingual_generic") but
            are NOT expert-verified - spot-check before trusting fully.
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
        tasks.append((
            str(model_dir), str(csv_files[model_name]), skey_map,
            enable_multilingual, multilingual_min_tier,
        ))

    if verbose:
        print(f"[INFO] 병렬 처리 워커 수: {n_workers}개 "
              f"(모델 {len(tasks)}개 처리 예정)")

    rows = []
    nc_samples = []
    fuzzy_samples = []
    multilingual_samples = []
    stats = {
        "models_with_label": len(tasks),
        "models_no_label":   models_no_label,
        "json_total":        0,
        "json_matched":      0,
        "json_nc_skip":      0,
        "json_pipe_skip":    0,
        "json_no_match":     0,
        "json_feat_error":   0,
        "json_multilingual_recovered": 0,
        "match_exact":       0,
        "match_normalized":  0,
        "match_fuzzy":       0,
    }

    completed = 0
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_process_one_model, task): task for task in tasks}
        for future in as_completed(futures):
            result = future.result()
            rows.extend(result["rows"])
            nc_samples.extend(result["nc_samples"])
            fuzzy_samples.extend(result["fuzzy_samples"])
            multilingual_samples.extend(result["multilingual_samples"])
            for k, v in result["stats"].items():
                stats[k] = stats.get(k, 0) + v

            completed += 1
            if verbose and completed % 20 == 0:
                print(f"  [진행] {completed}/{len(tasks)} 모델 처리 완료")

    if verbose:
        print(f"\n=== 데이터 로드 결과 ===")
        print(f"  레이블 있는 모델  : {stats['models_with_label']}개")
        print(f"  레이블 없는 모델  : {stats['models_no_label']}개 (제외)")
        print(f"  JSON 총계         : {stats['json_total']}개")
        print(f"  학습 사용         : {stats['json_matched']}개")
        if enable_multilingual:
            print(f"  다국어 복구 사용  : {stats['json_multilingual_recovered']}개 "
                  f"(min_tier={multilingual_min_tier!r}, 전문가 검증 아님 - 검토 권장)")
        print(f"  NC** 제외         : {stats['json_nc_skip']}개 (추후 보완 예정)")
        print(f"  Pipe 제외         : {stats['json_pipe_skip']}개")
        print(f"  CSV 매칭 실패     : {stats['json_no_match']}개")
        print(f"  피처 추출 오류    : {stats['json_feat_error']}개")
        print(f"\n  --- CSV 매칭 신뢰도 (감사용) ---")
        print(f"  exact (원문 일치)      : {stats['match_exact']}개")
        print(f"  normalized (정규화 일치): {stats['match_normalized']}개")
        print(f"  fuzzy (부분 매칭 추정)  : {stats['match_fuzzy']}개  "
              f"← 검토 권장, fuzzy_matched_labels.csv 확인")

    if not rows:
        raise ValueError("유효한 학습 데이터가 없습니다.")

    df = pd.DataFrame(rows)

    if verbose:
        print(f"\n컴포넌트 그룹 분포:\n{df['component_group'].value_counts().to_string()}")
        print(f"\n총 학습 샘플: {len(df)}개")

    return df, nc_samples, fuzzy_samples, multilingual_samples