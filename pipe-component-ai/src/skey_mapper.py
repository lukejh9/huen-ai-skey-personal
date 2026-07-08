"""
skey_mapper.py
--------------
symbol.xml 의 SKEY → component_group / valve_type 변환 모듈.

symbol.xml 구조:
    <Symbol SKEY="VB**" Category="Valves" Description="Valve - Ball"
            PCF_Identification="VALVE" ...>

SKEY 체계:
  - 앞 2~4자리 + 와일드카드(**)로 구성
  - 대부분 대문자 접두어로 타입 구분 가능
  - ** 는 연결 타입(BW/FL/SC 등)을 나타내는 와일드카드

학습 레이블 체계 (component_group):
    VALVE, FITTING, FLANGE, GASKET, SLEEVE, SUPPORT,
    INSTRUMENT, FLEXIBLE_JOINT, ACCESSORY, COUPLING,
    OLET, CAP, TRAP, PIPE, UNKNOWN

밸브 타입 (valve_type, VALVE 그룹만):
    BALL, GATE, GLOBE, BUTTERFLY, CHECK, CONTROL,
    SAFETY, PRESSURE_REDUCING, DIAPHRAGM, COCK,
    NEEDLE, PLUG, SLIDE, THREE_WAY, FOUR_WAY,
    KNIFE_GATE, BASIC, AUTO, UNKNOWN
"""

import re
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple


# ─────────────────────────────────────────────
#  데이터 구조
# ─────────────────────────────────────────────

class SkeyInfo(NamedTuple):
    skey:           str
    category:       str
    description:    str
    pcf_id:         str
    component_group: str
    valve_type:      str


# ─────────────────────────────────────────────
#  SKEY prefix → component_group 매핑 규칙
#  (우선순위 높은 순으로 배치)
# ─────────────────────────────────────────────

# PCF_Identification 기반 그룹 매핑 (우선)
_PCF_TO_GROUP: dict[str, str] = {
    "VALVE":                    "VALVE",
    "VALVE-ANGLE":              "VALVE",
    "VALVE-3WAY":               "VALVE",
    "VALVE-4WAY":               "VALVE",
    "INSTRUMENT":               "INSTRUMENT",
    "INSTRUMENT-DIAL":          "INSTRUMENT",
    "INSTRUMENT-OFFSET":        "INSTRUMENT",
    "INSTRUMENT-RETURN":        "INSTRUMENT",
    "INSTRUMENT-ANGLE":         "INSTRUMENT",
    "INSTRUMENT-3WAY":          "INSTRUMENT",
    "INSTRUMENT-4WAY":          "INSTRUMENT",
    "FLANGE":                   "FLANGE",
    "FLANGE-BLIND":             "FLANGE",
    "FLANGE-REDUCING-CONCENTRIC": "FLANGE",
    "LAPJOINT-STUBEND":         "FLANGE",
    "GASKET":                   "GASKET",
    "ELBOW":                    "FITTING",
    "ELBOW-REDUCING":           "FITTING",
    "TEE":                      "FITTING",
    "TEE-SET-ON":               "FITTING",
    "CROSS":                    "FITTING",
    "REDUCER-CONCENTRIC":       "FITTING",
    "REDUCER-ECCENTRIC":        "FITTING",
    "BEND":                     "FITTING",
    "OLET":                     "OLET",
    "CAP":                      "CAP",
    "COUPLING":                 "COUPLING",
    "CONNECTOR":                "COUPLING",
    "UNION":                    "COUPLING",
    "CLAMP":                    "COUPLING",
    "SUPPORT":                  "SUPPORT",
    "NOZZLE":                   "SUPPORT",
    "FILTER":                   "ACCESSORY",
    "FILTER-OFFSET":            "ACCESSORY",
    "FILTER-RETURN":            "ACCESSORY",
    "TRAP":                     "TRAP",
    "TRAP-ANGLE":               "TRAP",
    "TRAP-RETURN":              "TRAP",
    "PIPE":                     "PIPE",
    "PIPE-FIXED":               "PIPE",
    "WELD":                     "JOINT",
    "BOLT":                     "JOINT",
    "FLOW-ARROW":               "INSTRUMENT",
    "MISC-COMPONENT":           "ACCESSORY",   # 기본값, 아래에서 세분화
    "MISC-COMPONENT-OFFSET":    "ACCESSORY",
    "MISC-COMPONENT-RETURN":    "ACCESSORY",
}

# SKEY 패턴 → 세분화 (PCF_ID 가 MISC-COMPONENT 등 애매한 경우)
_SKEY_OVERRIDE: dict[str, str] = {
    "EX":   "FLEXIBLE_JOINT",   # Expansion Bellows
    "FX":   "FLEXIBLE_JOINT",   # Flexible Hose
    "SG":   "INSTRUMENT",       # Sight Glass
    "FM":   "INSTRUMENT",       # Flow Meter
    "IDPL": "INSTRUMENT",       # Pressure Gauge (Dial)
    "IDFL": "INSTRUMENT",       # Pressure Gauge Flanged
    "IRFL": "INSTRUMENT",       # Level Gauge
    "FLOR": "INSTRUMENT",       # Floor Symbol
    "WALL": "INSTRUMENT",       # Wall Symbol
    "SLVE": "SLEEVE",           # Penetration Sleeve
    "GASK": "GASKET",           # Gasket (misc)
    "OB":   "ACCESSORY",        # Spectacle Blind (open)
    "SB":   "ACCESSORY",        # Spectacle Blind (closed)
    "SS":   "SUPPORT",          # Puddle Pipe
    "_B":   "JOINT",            # Bolt
    "_G":   "GASKET",           # Gasket
    "WW":   "JOINT",            # Weld
    "SC":   "JOINT",            # Thread
    "PTRP": "TRAP",             # P-Trap
}

# SKEY → valve_type (VALVE 그룹만)
_SKEY_TO_VALVE_TYPE: dict[str, str] = {
    "VB":    "BALL",
    "BVA":   "BALL",
    "BVA2":  "BALL",
    "VT":    "GATE",
    "KG":    "GATE",            # Knife Gate
    "VG":    "GLOBE",
    "VY":    "BUTTERFLY",
    "VC":    "CHECK",
    "CK":    "CHECK",
    "CV":    "CONTROL",
    "CA":    "CONTROL",         # Instrument Angle control valve
    "PCV":   "PRESSURE_REDUCING",
    "VX":    "PRESSURE_REDUCING",
    "AX":    "PRESSURE_REDUCING",
    "VR":    "SAFETY",
    "AR":    "SAFETY",          # Relief/Vent angle
    "BRV":   "BREATHER",
    "VD":    "DIAPHRAGM",
    "DV":    "DIAPHRAGM",
    "VK":    "COCK",
    "VN":    "NEEDLE",
    "VP":    "PLUG",
    "VS":    "SLIDE",
    "V3":    "THREE_WAY",
    "V4":    "FOUR_WAY",
    "VV":    "BASIC",
    "AV":    "ANGLE",
}


# ─────────────────────────────────────────────
#  레이블 클래스 정의 (학습용)
# ─────────────────────────────────────────────

GROUP_CLASSES = [
    "VALVE", "FITTING", "FLANGE", "GASKET",
    "SLEEVE", "SUPPORT", "INSTRUMENT",
    "FLEXIBLE_JOINT", "ACCESSORY", "OLET",
    "CAP", "COUPLING", "TRAP", "PIPE", "UNKNOWN",
]

VALVE_TYPE_CLASSES = [
    "BALL", "GATE", "GLOBE", "BUTTERFLY", "CHECK",
    "CONTROL", "SAFETY", "PRESSURE_REDUCING", "DIAPHRAGM",
    "COCK", "NEEDLE", "PLUG", "SLIDE", "THREE_WAY", "FOUR_WAY",
    "KNIFE_GATE", "BASIC", "BREATHER", "ANGLE",
    "AUTO", "UNKNOWN", "NON_VALVE",
]

GROUP_TO_INT  = {g: i for i, g in enumerate(GROUP_CLASSES)}
INT_TO_GROUP  = {i: g for g, i in GROUP_TO_INT.items()}
VTYPE_TO_INT  = {v: i for i, v in enumerate(VALVE_TYPE_CLASSES)}
INT_TO_VTYPE  = {i: v for v, i in VTYPE_TO_INT.items()}
NUM_GROUP_CLASSES = len(GROUP_CLASSES)
NUM_VTYPE_CLASSES = len(VALVE_TYPE_CLASSES)


# ─────────────────────────────────────────────
#  symbol.xml 파서
# ─────────────────────────────────────────────

def _skey_prefix(skey: str) -> str:
    """SKEY에서 와일드카드(**) 및 EndType 부분을 제거한 순수 접두어 반환."""
    # "VB**" → "VB", "01HG" → "01HG", "_G" → "_G"
    prefix = re.sub(r"\*+$", "", skey).strip()
    return prefix


def _derive_labels(skey: str, category: str, description: str, pcf_id: str) -> tuple[str, str]:
    """
    SKEY / Category / PCF_Identification → (component_group, valve_type) 반환.
    """
    prefix = _skey_prefix(skey)

    # ── SKEY 오버라이드 우선 ─────────────────
    # 정확 매칭
    if prefix in _SKEY_OVERRIDE:
        group = _SKEY_OVERRIDE[prefix]
        vtype = _derive_valve_type(prefix, group)
        return group, vtype

    # prefix 앞부분으로 오버라이드 검사 (ex: "EX" matches "EX**")
    for key, grp in _SKEY_OVERRIDE.items():
        if prefix.startswith(key):
            vtype = _derive_valve_type(prefix, grp)
            return grp, vtype

    # ── PCF_Identification 기반 매핑 ─────────
    pcf_id_clean = pcf_id.strip()
    if pcf_id_clean in _PCF_TO_GROUP:
        group = _PCF_TO_GROUP[pcf_id_clean]
        # MISC-COMPONENT → symbol Category로 세분화
        if group == "ACCESSORY":
            group = _refine_misc(category, description, prefix)
        vtype = _derive_valve_type(prefix, group)
        return group, vtype

    # ── Category 텍스트 기반 fallback ─────────
    cat_upper = category.upper()
    if "VALVE" in cat_upper:
        group = "VALVE"
    elif "FLANGE" in cat_upper:
        group = "FLANGE"
    elif "GASKET" in cat_upper:
        group = "GASKET"
    elif "FILTER" in cat_upper or "STRAINER" in cat_upper:
        group = "ACCESSORY"
    elif "SUPPORT" in cat_upper:
        group = "SUPPORT"
    elif "INSTRUMENT" in cat_upper:
        group = "INSTRUMENT"
    elif "ELBOW" in cat_upper or "BEND" in cat_upper:
        group = "FITTING"
    elif "TEE" in cat_upper or "CROSS" in cat_upper:
        group = "FITTING"
    elif "REDUCER" in cat_upper:
        group = "FITTING"
    elif "OLET" in cat_upper:
        group = "OLET"
    elif "CAP" in cat_upper or "PLUG" in cat_upper:
        group = "CAP"
    elif "COUPLING" in cat_upper or "UNION" in cat_upper:
        group = "COUPLING"
    elif "TRAP" in cat_upper:
        group = "TRAP"
    elif "PIPE" in cat_upper:
        group = "PIPE"
    else:
        group = "UNKNOWN"

    vtype = _derive_valve_type(prefix, group)
    return group, vtype


def _refine_misc(category: str, description: str, prefix: str) -> str:
    """MISC-COMPONENT 항목을 Description/Prefix 기반으로 세분화."""
    desc_up = description.upper()
    cat_up  = category.upper()

    if "BELLOWS" in desc_up or "EXPANSION" in desc_up:
        return "FLEXIBLE_JOINT"
    if "FLEXIBLE" in desc_up or "HOSE" in desc_up:
        return "FLEXIBLE_JOINT"
    if "SIGHT" in desc_up or "GLASS" in desc_up:
        return "INSTRUMENT"
    if "SIGHT" in cat_up:
        return "INSTRUMENT"
    if "TRAP" in desc_up:
        return "TRAP"
    if "SLEEVE" in desc_up or "PENETRATION" in desc_up:
        return "SLEEVE"
    if "STRAINER" in desc_up or "FILTER" in desc_up:
        return "ACCESSORY"
    if "SPECTACLE" in desc_up:
        return "ACCESSORY"
    return "ACCESSORY"


def _derive_valve_type(prefix: str, group: str) -> str:
    """VALVE 그룹일 때 valve_type 반환, 아니면 NON_VALVE."""
    if group != "VALVE":
        return "NON_VALVE"

    # 정확 매칭
    if prefix in _SKEY_TO_VALVE_TYPE:
        return _SKEY_TO_VALVE_TYPE[prefix]

    # prefix 시작 매칭
    for key, vtype in _SKEY_TO_VALVE_TYPE.items():
        if prefix.startswith(key):
            return vtype

    return "UNKNOWN"


def parse_symbol_xml(xml_path: str | Path) -> dict[str, SkeyInfo]:
    """
    symbol.xml 파싱 → {skey: SkeyInfo} 딕셔너리 반환.

    Parameters
    ----------
    xml_path : str | Path
        symbol.xml 파일 경로

    Returns
    -------
    dict[str, SkeyInfo]
        SKEY 문자열 → SkeyInfo 매핑
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    result: dict[str, SkeyInfo] = {}

    for sym in root.findall("Symbol"):
        skey    = sym.get("SKEY", "").strip()
        cat     = sym.get("Category", "").strip()
        desc    = sym.get("Description", "").strip()
        pcf_id  = sym.get("PCF_Identification", "").strip()

        if not skey:
            continue

        group, vtype = _derive_labels(skey, cat, desc, pcf_id)

        result[skey] = SkeyInfo(
            skey=skey,
            category=cat,
            description=desc,
            pcf_id=pcf_id,
            component_group=group,
            valve_type=vtype,
        )

    return result


def get_labels_from_skey(
    skey_value: str,
    skey_map: dict[str, SkeyInfo] | None = None,
) -> dict:
    """
    SKY 값(xlsx 컬럼) → {"component_group", "valve_type", "is_valve"} 반환.

    Parameters
    ----------
    skey_value : str
        xlsx의 SKY 열 값 (예: "VB**", "EL**", "FLBL")
    skey_map : dict | None
        parse_symbol_xml() 결과. None 이면 규칙 기반으로만 처리.

    Returns
    -------
    dict
    """
    skey_clean = str(skey_value).strip()

    # 1) 정확 매칭
    if skey_map and skey_clean in skey_map:
        info = skey_map[skey_clean]
        return {
            "component_group": info.component_group,
            "valve_type":      info.valve_type,
            "is_valve":        info.component_group == "VALVE",
            "skey":            skey_clean,
        }

    # 2) prefix 매칭 (와일드카드 제거 후)
    prefix = _skey_prefix(skey_clean)
    if skey_map:
        # 가장 긴 prefix match 우선
        candidates = {k: v for k, v in skey_map.items()
                      if _skey_prefix(k) == prefix}
        if candidates:
            info = next(iter(candidates.values()))
            return {
                "component_group": info.component_group,
                "valve_type":      info.valve_type,
                "is_valve":        info.component_group == "VALVE",
                "skey":            skey_clean,
            }

    # 3) 규칙 기반 fallback
    group, vtype = _derive_labels(
        skey_clean,
        category="",
        description="",
        pcf_id="",
    )
    return {
        "component_group": group,
        "valve_type":      vtype,
        "is_valve":        group == "VALVE",
        "skey":            skey_clean,
    }


# ─────────────────────────────────────────────
#  테스트용
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    xml_path = sys.argv[1] if len(sys.argv) > 1 else "../../BIM모델_데이터/symbol.xml"
    skey_map = parse_symbol_xml(xml_path)

    print(f"총 SKEY 수: {len(skey_map)}\n")
    print(f"{'SKEY':<10} {'그룹':<18} {'밸브타입':<20} {'설명'}")
    print("-" * 80)
    for skey, info in sorted(skey_map.items()):
        if info.category in ("Spindles", "Hidden", "End Types"):
            continue
        print(f"{skey:<10} {info.component_group:<18} {info.valve_type:<20} {info.description[:40]}")

    # 그룹별 분포
    from collections import Counter
    groups = Counter(v.component_group for v in skey_map.values()
                     if v.category not in ("Spindles", "Hidden", "End Types"))
    print(f"\n그룹 분포: {dict(groups)}")