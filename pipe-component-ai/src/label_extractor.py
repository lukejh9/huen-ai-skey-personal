"""
label_extractor.py
------------------
BIM 패밀리 파일명 또는 FamilyName 에서 컴포넌트 그룹 / 밸브 타입 레이블을 자동 추출.

컴포넌트 그룹 (component_group):
    VALVE      : 밸브류 (ball, gate, globe, butterfly, check, control, safety, auto)
    FITTING    : 배관 피팅 (elbow, tee, reducer, weldlet 등)
    FLANGE     : 플랜지류 (flange, blind flange)
    GASKET     : 개스킷류 (gasket, blind gasket)
    SLEEVE     : 슬리브류 (pipe floor sleeve, pipe wall sleeve)
    SUPPORT    : 지지대류 (pipe shoe, u-bolt, channel, hanger)
    INSTRUMENT : 계측기류 (pressure gauge, thermometer, sight glass, F.D.)
    FLEXIBLE_JOINT : 신축이음 (bellows, expansion joint)
    ACCESSORY  : 기타 부속 (strainer 등)
    UNKNOWN    : 미분류

밸브 타입 (valve_type):
    BALL, GATE, GLOBE, BUTTERFLY, CHECK, CONTROL, SAFETY, RELIEF, AUTO, UNKNOWN
    VALVE 그룹이 아닌 경우 → NON_VALVE
"""

import re
from pathlib import Path


# ─────────────────────────────────────────────
#  규칙 기반 매핑 테이블
#  순서 중요: 더 구체적인 패턴을 앞에 배치
# ─────────────────────────────────────────────

_GROUP_RULES: list[tuple[str, str]] = [
    # ── VALVE ──────────────────────────────────
    (r"BALL\s*VALVE",           "VALVE"),
    (r"GATE\s*VALVE",           "VALVE"),
    (r"GLOBE\s*VALVE",          "VALVE"),
    (r"BUTTERFLY\s*VALVE",      "VALVE"),
    (r"CHECK\s*VALVE",          "VALVE"),
    (r"CONTROL\s*VALVE",        "VALVE"),
    (r"SAFETY\s*VALVE",         "VALVE"),
    (r"RELIEF\s*VALVE",         "VALVE"),
    (r"AUTOVALVE",              "VALVE"),
    (r"\bVALVE\b",              "VALVE"),

    # ── GASKET (FLANGE 앞에 배치) ───────────────
    (r"BLIND\s*GASKET",         "GASKET"),
    (r"\bGASKET\b",             "GASKET"),

    # ── SLEEVE (SUPPORT/PIPE 앞에) ──────────────
    (r"PIPE\s*FLOOR\s*SLEEVE",  "SLEEVE"),
    (r"PIPE\s*WALL\s*SLEEVE",   "SLEEVE"),
    (r"\bSLEEVE\b",             "SLEEVE"),

    # ── SUPPORT ────────────────────────────────
    (r"PIPE\s*SHOE",            "SUPPORT"),
    (r"\bU\s*[\-]?\s*BOLT\b",   "SUPPORT"),   # U-BOLT
    (r"SHAPE\s*STEEL",          "SUPPORT"),   # SHAPE STEEL CHANNEL
    (r"\bCHANNEL\b",            "SUPPORT"),   # UPPER_CHANNEL, SIDE_CHANNEL 등
    (r"\bSHOE\b",               "SUPPORT"),
    (r"\bHANGER\b",             "SUPPORT"),
    (r"\bSUPPORT\b",            "SUPPORT"),

    # ── INSTRUMENT ─────────────────────────────
    (r"PRESSURE\s*GAUGE",       "INSTRUMENT"),
    (r"SIGHT\s*GLASS",          "INSTRUMENT"),
    (r"온도계",                  "INSTRUMENT"),   # 한글 온도계(thermometer)
    (r"THERMOMETER",            "INSTRUMENT"),
    (r"FLOW\s*METER",           "INSTRUMENT"),
    (r"\bGAUGE\b",              "INSTRUMENT"),
    (r"\bF\.D\b",               "INSTRUMENT"),   # Flow Direction Indicator

    # ── FLEXIBLE_JOINT ─────────────────────────
    (r"BELLOWS",                "FLEXIBLE_JOINT"),
    (r"FLEXIBLE\s*JOINT",       "FLEXIBLE_JOINT"),
    (r"EXPANSION\s*JOINT",      "FLEXIBLE_JOINT"),

    # ── FLANGE ─────────────────────────────────
    (r"BLIND\s*FLANGE",         "FLANGE"),
    (r"\bFLANGE\b",             "FLANGE"),

    # ── ACCESSORY (STRAINER 등) ─────────────────
    (r"\bSTRAINER\b",           "ACCESSORY"),
    (r"\bFILTER\b",             "ACCESSORY"),

    # ── FITTING (배관 피팅) ─────────────────────
    (r"\bELBOW\b",              "FITTING"),
    (r"\bTEE\b",                "FITTING"),
    (r"\bREDUCER\b",            "FITTING"),
    (r"\bWELDLET\b",            "FITTING"),
    (r"\bCOUPLING\b",           "FITTING"),
    (r"\bCAP\b",                "FITTING"),
    (r"\bSOCKET\b",             "FITTING"),
]

_VALVE_TYPE_RULES: list[tuple[str, str]] = [
    (r"BALL\s*VALVE",           "BALL"),
    (r"GATE\s*VALVE",           "GATE"),
    (r"GLOBE\s*VALVE",          "GLOBE"),
    (r"BUTTERFLY\s*VALVE",      "BUTTERFLY"),
    (r"CHECK\s*VALVE",          "CHECK"),
    (r"CONTROL\s*VALVE",        "CONTROL"),
    (r"SAFETY\s*VALVE",         "SAFETY"),
    (r"RELIEF\s*VALVE",         "RELIEF"),
    (r"AUTOVALVE",              "AUTO"),
]


def _normalize(text: str) -> str:
    """대문자화 + 특수문자 일부 정규화 (언더스코어·하이픈 → 공백)"""
    return text.upper().replace("_", " ").replace("-", " ")


def extract_component_group(name: str) -> str:
    """
    FamilyName 또는 파일명 → component_group 문자열 반환.

    Parameters
    ----------
    name : str
        FamilyName, TypeName, 또는 파일 경로 문자열

    Returns
    -------
    str
        컴포넌트 그룹 레이블
    """
    # 파일 경로인 경우 stem만 사용
    stem = Path(name).stem if ("/" in name or "\\" in name or name.endswith(".json")) else name
    # T_ 접두어 제거
    stem = re.sub(r"^T_", "", stem, flags=re.IGNORECASE)
    text = _normalize(stem)

    for pattern, group in _GROUP_RULES:
        if re.search(pattern, text):
            return group
    return "UNKNOWN"


def extract_valve_type(name: str) -> str:
    """
    FamilyName 또는 파일명 → valve_type 문자열 반환.
    VALVE 그룹이 아닌 경우 'NON_VALVE' 반환.

    Parameters
    ----------
    name : str
        FamilyName, TypeName, 또는 파일 경로 문자열

    Returns
    -------
    str
        밸브 타입 레이블 또는 'NON_VALVE'
    """
    group = extract_component_group(name)
    if group != "VALVE":
        return "NON_VALVE"

    stem = Path(name).stem if ("/" in name or "\\" in name or name.endswith(".json")) else name
    stem = re.sub(r"^T_", "", stem, flags=re.IGNORECASE)
    text = _normalize(stem)

    for pattern, vtype in _VALVE_TYPE_RULES:
        if re.search(pattern, text):
            return vtype
    return "UNKNOWN"


def extract_labels(name: str) -> dict:
    """
    파일명/FamilyName 으로부터 전체 레이블 딕셔너리 반환.

    Returns
    -------
    dict
        {"component_group": str, "valve_type": str, "is_valve": bool}
    """
    group = extract_component_group(name)
    vtype = extract_valve_type(name)
    return {
        "component_group": group,
        "valve_type":      vtype,
        "is_valve":        group == "VALVE",
    }


# ─────────────────────────────────────────────
#  그룹 / 타입 → 정수 인코딩
# ─────────────────────────────────────────────

GROUP_CLASSES = [
    "VALVE", "FITTING", "FLANGE", "GASKET",
    "SLEEVE", "SUPPORT", "INSTRUMENT",
    "FLEXIBLE_JOINT", "ACCESSORY", "UNKNOWN"
]

VALVE_TYPE_CLASSES = [
    "BALL", "GATE", "GLOBE", "BUTTERFLY",
    "CHECK", "CONTROL", "SAFETY", "RELIEF", "AUTO", "UNKNOWN", "NON_VALVE"
]

GROUP_TO_INT   = {g: i for i, g in enumerate(GROUP_CLASSES)}
INT_TO_GROUP   = {i: g for g, i in GROUP_TO_INT.items()}
VTYPE_TO_INT   = {v: i for i, v in enumerate(VALVE_TYPE_CLASSES)}
INT_TO_VTYPE   = {i: v for v, i in VTYPE_TO_INT.items()}


# ─────────────────────────────────────────────
#  LightGBM num_class 업데이트용 상수
# ─────────────────────────────────────────────

NUM_GROUP_CLASSES = len(GROUP_CLASSES)
NUM_VTYPE_CLASSES = len(VALVE_TYPE_CLASSES)


if __name__ == "__main__":
    # 간단 테스트 (전체 55개 파일명 검증)
    test_names = [
        # VALVE
        "T_BALL VALVE_FL_STS_LEVER_10K_A2A_15A-200A.json",
        "T_GLOBE VALVE_FL_CSS_10K_A2A_STS 10K(15A-40A).json",
        "T_SAFETY VALVE_PT_25A_T_SAFETY VALVE_PT_25A.json",
        "T_CONTROL VALVE_40A_40A.json",
        "T_AUTOVALVE_40A_40A.json",
        # GASKET
        "T_GASKET_FF_10K_A2A_3.0T.json",
        "T_BLIND GASKET_FF_10K_A2A_3.0T.json",
        # SLEEVE
        "T_PIPE FLOOR SLEEVE_A2A_150L.json",
        "T_PIPE WALL SLEEVE_A2A_100L.json",
        # SUPPORT
        "T_PIPE SHOE_01_STS관용.json",
        "T_U-BOLT_02_STS관용.json",
        "T_SHAPE STEEL_CHANNEL_100x50x5x7.5.json",
        "T_UPPER_CHANNEL_1F_1S_ㄷ100x50.json",
        "T_SIDE_CHANNEL_1S_ㄷ150x75.json",
        # INSTRUMENT
        "T_PRESSURE GAUGE_GENERAL_60_A_60Φ.json",
        "T_F.D_80A.json",
        "T_Sight Glass_50A_A2A_50A.json",
        "T_온도계+웰_100_R_Hor_A2A_온도계-100Ø.json",
        # FLEXIBLE_JOINT
        "T_Bellows Flexible Joint_A2A_10K.json",
        # FLANGE
        "T_STEEL BLIND FLANGE_A2A_STS BLIND FLANGE-10K.json",
        "T_STEEL FLANGE_SOP_A2A_STS FLANGE-10K.json",
        # ACCESSORY
        "T_STRAINER_FL_CI_10K_강관 10K(20A-500A).json",
        # FITTING
        "T_STS PIPE ELBOW_WELD_A2A_LONG.json",
        "T_STS관 REDUCER_WELD_A2A_STANDARD.json",
        "T_STS관 TEE_WELD_A2A_STANDARD.json",
        "T_STS관 WELDLET_A2A_STS관 티뽑기(용접).json",
    ]
    print(f"{'파일명':<55} {'그룹':<15} {'밸브타입'}")
    print("-" * 90)
    for n in test_names:
        lbl = extract_labels(n)
        print(f"{n:<55} {lbl['component_group']:<15} {lbl['valve_type']}")
