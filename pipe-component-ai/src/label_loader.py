"""
label_loader.py
---------------
학습용레이블_정답.xlsx 를 읽어 ProductCode → 레이블 딕셔너리를 반환한다.

xlsx 컬럼 구조 (예상):
    ProductCode  : "패밀리명: 타입명"  (예: "T_BALL VALVE_FL_STS_LEVER_10K_A2A: 15A-200A")
    SKY          : symbol.xml 의 SKEY 값  (예: "VB**")
    (추가 컬럼 있어도 무시)

사용 예:
    from src.label_loader import LabelRegistry

    registry = LabelRegistry(
        xlsx_path  = "BIM모델_데이터/학습용레이블_정답.xlsx",
        xml_path   = "BIM모델_데이터/symbol.xml",
    )
    labels = registry.get_labels("T_BALL VALVE_FL_STS_LEVER_10K_A2A", "15A-200A")
    # → {"component_group": "VALVE", "valve_type": "BALL", "is_valve": True, "skey": "VB**"}
"""

import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.skey_mapper import (
    parse_symbol_xml,
    get_labels_from_skey,
    SkeyInfo,
    GROUP_CLASSES, VALVE_TYPE_CLASSES,
    GROUP_TO_INT, INT_TO_GROUP,
    VTYPE_TO_INT, INT_TO_VTYPE,
    NUM_GROUP_CLASSES, NUM_VTYPE_CLASSES,
)
from src.label_extractor import extract_labels as fallback_labels


# ─────────────────────────────────────────────
#  xlsx 컬럼명 후보 (대소문자 무관)
# ─────────────────────────────────────────────

_PRODUCT_CODE_CANDIDATES = [
    "productcode", "product code", "product_code",
    "제품코드", "패밀리", "familyname", "family",
]

_SKY_CANDIDATES = [
    "sky", "skey", "symbol", "심볼", "iso_symbol",
    "iso symbol", "symbolkey", "symbol_key",
]


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """DataFrame 컬럼 중 후보 목록에 매칭되는 컬럼명 반환."""
    col_lower = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand in col_lower:
            return col_lower[cand]
    return None


# ─────────────────────────────────────────────
#  ProductCode 파싱
# ─────────────────────────────────────────────

def _parse_product_code(product_code: str) -> tuple[str, str]:
    """
    "패밀리명: 타입명" 형식을 (family_name, type_name) 으로 분리.

    구분자는 ": " 또는 ":" 를 사용.
    예)
        "T_BALL VALVE_FL_STS_LEVER_10K_A2A: 15A-200A" → ("T_BALL VALVE_FL_STS_LEVER_10K_A2A", "15A-200A")
        "T_AUTOVALVE_40A_40A:40A" → ("T_AUTOVALVE_40A_40A", "40A")
    """
    pc = str(product_code).strip()
    # 첫 번째 ":" 기준으로 분리
    idx = pc.find(":")
    if idx == -1:
        return pc, ""
    family = pc[:idx].strip()
    tname  = pc[idx + 1:].strip()
    return family, tname


def _normalize_family_name(name: str) -> str:
    """비교용 정규화: 대소문자, 공백, 특수문자 무시."""
    return re.sub(r"[\s_\-]+", "", name).upper()


# ─────────────────────────────────────────────
#  LabelRegistry 클래스
# ─────────────────────────────────────────────

class LabelRegistry:
    """
    xlsx 정답 레이블 + symbol.xml SKEY 매핑을 통합 관리.

    Attributes
    ----------
    xlsx_path  : xlsx 파일 경로
    xml_path   : symbol.xml 파일 경로
    df_labels  : 원본 DataFrame
    skey_map   : SKEY → SkeyInfo 딕셔너리
    lookup     : {normalize(family_name): {type_name: labels}} 딕셔너리
    """

    def __init__(
        self,
        xlsx_path: str | Path,
        xml_path:  str | Path | None = None,
    ):
        self.xlsx_path = Path(xlsx_path)
        self.xml_path  = Path(xml_path) if xml_path else None

        # symbol.xml 파싱
        if self.xml_path and self.xml_path.exists():
            self.skey_map: dict[str, SkeyInfo] = parse_symbol_xml(self.xml_path)
            print(f"[LABEL] symbol.xml 로드: {len(self.skey_map)}개 SKEY")
        else:
            self.skey_map = {}
            print(f"[LABEL] symbol.xml 없음 → 규칙 기반 SKEY 매핑만 사용")

        # xlsx 읽기
        self.df_labels: pd.DataFrame = self._load_xlsx()
        self.lookup: dict[str, dict[str, dict]] = self._build_lookup()
        print(f"[LABEL] xlsx 레이블 로드: {len(self.df_labels)}개 항목")

    # ── xlsx 로드 ────────────────────────────────

    def _load_xlsx(self) -> pd.DataFrame:
        if not self.xlsx_path.exists():
            raise FileNotFoundError(f"레이블 파일을 찾을 수 없습니다: {self.xlsx_path}")

        # 모든 시트 읽기 후 합치기
        all_sheets = pd.read_excel(self.xlsx_path, sheet_name=None, dtype=str)
        frames = []
        for sheet_name, df in all_sheets.items():
            df = df.dropna(how="all")
            if len(df) > 0:
                df["__sheet__"] = sheet_name
                frames.append(df)

        if not frames:
            raise ValueError("xlsx 파일에 유효한 데이터가 없습니다.")

        combined = pd.concat(frames, ignore_index=True)

        # 컬럼 탐지
        pc_col  = _find_column(combined, _PRODUCT_CODE_CANDIDATES)
        sky_col = _find_column(combined, _SKY_CANDIDATES)

        if pc_col is None:
            # 컬럼명 목록 출력 후 첫 번째 컬럼 사용
            print(f"[WARN] ProductCode 컬럼을 찾을 수 없음. 컬럼 목록: {combined.columns.tolist()}")
            print(f"[WARN] 첫 번째 컬럼을 ProductCode로 사용합니다.")
            pc_col = combined.columns[0]

        if sky_col is None:
            print(f"[WARN] SKY 컬럼을 찾을 수 없음. 컬럼 목록: {combined.columns.tolist()}")
            # 두 번째 컬럼 시도
            if len(combined.columns) > 1:
                sky_col = combined.columns[1]
                print(f"[WARN] 두 번째 컬럼 '{sky_col}'을 SKY로 사용합니다.")

        print(f"[LABEL] ProductCode 컬럼: '{pc_col}', SKY 컬럼: '{sky_col}'")

        # 필요 컬럼만 추출
        result = combined[[pc_col, sky_col]].copy()
        result.columns = ["ProductCode", "SKY"]
        result = result.dropna(subset=["ProductCode", "SKY"])
        result["ProductCode"] = result["ProductCode"].str.strip()
        result["SKY"]         = result["SKY"].str.strip()

        return result

    # ── lookup 테이블 빌드 ───────────────────────

    def _build_lookup(self) -> dict[str, dict[str, dict]]:
        """
        {normalize(family_name): {type_name: labels_dict}} 구조 생성.
        type_name 이 비어 있을 때는 "" 키 사용.
        """
        lookup: dict[str, dict[str, dict]] = {}

        for _, row in self.df_labels.iterrows():
            pc   = str(row["ProductCode"]).strip()
            sky  = str(row["SKY"]).strip()

            family, tname = _parse_product_code(pc)
            key = _normalize_family_name(family)

            labels = get_labels_from_skey(sky, self.skey_map)
            labels["product_code"] = pc

            if key not in lookup:
                lookup[key] = {}
            lookup[key][tname] = labels

        return lookup

    # ── 레이블 조회 ──────────────────────────────

    def get_labels(
        self,
        family_name: str,
        type_name:   str = "",
    ) -> dict | None:
        """
        (family_name, type_name) → 레이블 딕셔너리 반환.
        매칭 실패 시 None 반환.

        Parameters
        ----------
        family_name : str
            Revit FamilyName  (예: "T_BALL VALVE_FL_STS_LEVER_10K_A2A")
        type_name   : str
            Revit TypeName    (예: "15A-200A")

        Returns
        -------
        dict | None
            {"component_group", "valve_type", "is_valve", "skey"} 또는 None
        """
        key = _normalize_family_name(family_name)

        if key not in self.lookup:
            return None

        type_lookup = self.lookup[key]

        # 1) 정확한 type_name 매칭
        if type_name in type_lookup:
            return type_lookup[type_name]

        # 2) type_name 이 비어 있으면 첫 번째 값 사용
        if not type_name and type_lookup:
            return next(iter(type_lookup.values()))

        # 3) 부분 매칭 (type_name 이 포함된 키 탐색)
        for key_t, labels in type_lookup.items():
            if type_name in key_t or key_t in type_name:
                return labels

        # 4) family_name 매치가 있으면 첫 번째 값 사용
        if type_lookup:
            return next(iter(type_lookup.values()))

        return None

    def get_labels_from_filename(self, filename_stem: str) -> dict | None:
        """
        파일명 stem (예: "T_BALL VALVE_FL_STS_LEVER_10K_A2A_15A-200A") 으로 조회.
        파일명 형식에서 FamilyName 과 TypeName 을 추출하여 get_labels 호출.
        """
        # 파일명 = FamilyName_TypeName (언더스코어 구분)
        # JSON 파일명은 대부분 "FamilyName_TypeName" 형식
        # 마지막 언더스코어 기준으로 분리 시도
        stem = filename_stem.strip()

        # 방법 1: FamilyName이 정확히 lookup 에 있는 경우
        key = _normalize_family_name(stem)
        if key in self.lookup:
            return next(iter(self.lookup[key].values()))

        # 방법 2: 뒤에서부터 언더스코어 제거하며 탐색
        parts = stem.split("_")
        for i in range(len(parts), 0, -1):
            candidate_family = "_".join(parts[:i])
            candidate_type   = "_".join(parts[i:])
            key = _normalize_family_name(candidate_family)
            if key in self.lookup:
                return self.get_labels(candidate_family, candidate_type)

        return None

    # ── 전체 레이블 DataFrame ────────────────────

    def to_dataframe(self) -> pd.DataFrame:
        """전체 레이블 정보를 DataFrame 으로 반환."""
        rows = []
        for norm_family, type_dict in self.lookup.items():
            for tname, labels in type_dict.items():
                row = {
                    "norm_family":     norm_family,
                    "type_name":       tname,
                    **labels,
                    "group_label":     GROUP_TO_INT.get(labels["component_group"], 14),
                    "vtype_label":     VTYPE_TO_INT.get(labels["valve_type"], 22),
                }
                rows.append(row)
        return pd.DataFrame(rows)

    def summary(self) -> None:
        """레이블 분포 요약 출력."""
        df = self.to_dataframe()
        print(f"\n=== 레이블 분포 ===")
        print(f"총 항목: {len(df)}")
        print(f"\n[component_group]\n{df['component_group'].value_counts().to_string()}")
        print(f"\n[valve_type]\n{df['valve_type'].value_counts().to_string()}")


# ─────────────────────────────────────────────
#  편의 함수 (registry 없이 단일 파일용)
# ─────────────────────────────────────────────

def extract_labels_with_fallback(
    family_name:  str,
    type_name:    str,
    filename_stem: str,
    registry: LabelRegistry | None,
) -> dict:
    """
    registry → xlsx 레이블 우선, 없으면 파일명 기반 fallback.

    Returns
    -------
    dict
        {"component_group", "valve_type", "is_valve", "skey"(옵션)}
    """
    if registry is not None:
        # 1) FamilyName + TypeName 으로 조회
        labels = registry.get_labels(family_name, type_name)
        if labels:
            return labels

        # 2) 파일명 stem 으로 재시도
        labels = registry.get_labels_from_filename(filename_stem)
        if labels:
            return labels

    # 3) 파일명 기반 fallback
    fb = fallback_labels(filename_stem)
    fb["skey"] = ""
    return fb


# ─────────────────────────────────────────────
#  CLI 테스트
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    xlsx_p = sys.argv[1] if len(sys.argv) > 1 else "../../BIM모델_데이터/학습용레이블_정답.xlsx"
    xml_p  = sys.argv[2] if len(sys.argv) > 2 else "../../BIM모델_데이터/symbol.xml"

    registry = LabelRegistry(xlsx_path=xlsx_p, xml_path=xml_p)
    registry.summary()

    # xlsx 전체 내용 출력
    print("\n=== xlsx 원본 데이터 (상위 20행) ===")
    print(registry.df_labels.head(20).to_string(index=False))
