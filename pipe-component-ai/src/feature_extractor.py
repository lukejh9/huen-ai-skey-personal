"""
feature_extractor.py
--------------------
Revit BIM 패밀리 JSON → 수치 피처 벡터 추출.

추출 피처 목록:
  [메타데이터]
    meta_category        : InstanceParameters 의 Category 값 (str)
    meta_system_class    : System Classification 값 (str)
    meta_max_size_mm     : Maximum Size (mm, float)
    meta_min_size_mm     : Minimum Size (mm, float)

  [형상 (Geometry)]
    geo_surface_area     : 전체 솔리드 SurfaceArea 합 (Revit 단위 → m²)
    geo_volume           : 전체 솔리드 Volume 합 (Revit 단위 → m³)
    geo_solid_count      : 솔리드 수
    geo_face_count       : 총 Face 수
    geo_bbox_x           : BBox 가로 (m)
    geo_bbox_y           : BBox 세로 (m)
    geo_bbox_z           : BBox 높이 (m)
    geo_bbox_diag        : BBox 대각선 길이 (m)
    geo_aspect_xy        : bbox_x / bbox_y
    geo_aspect_xz        : bbox_x / bbox_z
    geo_aspect_yz        : bbox_y / bbox_z
    geo_compactness      : volume / (surface_area^1.5)  (형상 밀집도)
    geo_density          : geo face count / geo volume

  [커넥터 (Connector)]
    conn_count           : 커넥터 수
    conn_radius_mean     : Radius 평균 (m)
    conn_radius_std      : Radius 표준편차 (m)
    conn_radius_max      : Radius 최대값 (m)
    conn_radius_min      : Radius 최솟값 (m)
    conn_distance        : 첫 두 커넥터 간 거리 (2개 이상일 때, m)
    conn_angle_deg       : 두 커넥터 방향 벡터 사이 각도 (degree)

Revit 내부 단위 변환:
  길이 : feet → meter  (* 0.3048)
  면적 : feet² → m²    (* 0.3048²)
  부피 : feet³ → m³    (* 0.3048³)
"""

import json
import math
from pathlib import Path
from typing import Any


# Revit 내부 단위 변환 상수
_FT_TO_M  = 0.3048
_FT2_TO_M2 = _FT_TO_M ** 2
_FT3_TO_M3 = _FT_TO_M ** 3

_EPS = 1e-9  # 0 나눗셈 방지


# ─────────────────────────────────────────────
#  유틸리티 함수
# ─────────────────────────────────────────────

def _vec_norm(v: dict) -> float:
    return math.sqrt(v["X"] ** 2 + v["Y"] ** 2 + v["Z"] ** 2)


def _vec_dot(a: dict, b: dict) -> float:
    return a["X"] * b["X"] + a["Y"] * b["Y"] + a["Z"] * b["Z"]


def _vec_dist(a: dict, b: dict) -> float:
    return math.sqrt(
        (a["X"] - b["X"]) ** 2 +
        (a["Y"] - b["Y"]) ** 2 +
        (a["Z"] - b["Z"]) ** 2
    )


def _angle_between_deg(a: dict, b: dict) -> float:
    """두 방향 벡터 사이 각도 (degree)"""
    na, nb = _vec_norm(a), _vec_norm(b)
    if na < _EPS or nb < _EPS:
        return 0.0
    cos_val = max(-1.0, min(1.0, _vec_dot(a, b) / (na * nb)))
    return math.degrees(math.acos(cos_val))


def _parse_mm_value(val_str: str) -> float:
    """'40 mm' → 40.0, 실패 시 0.0"""
    try:
        return float(str(val_str).split()[0])
    except Exception:
        return 0.0


# ─────────────────────────────────────────────
#  메인 추출 함수
# ─────────────────────────────────────────────

def extract_features(data: dict) -> dict:
    """
    JSON dict (json.load 결과) → 피처 딕셔너리 반환.

    Parameters
    ----------
    data : dict
        BIM JSON 전체 내용

    Returns
    -------
    dict
        피처 딕셔너리 (모두 수치형 or str)
    """
    feat: dict[str, Any] = {}

    # ── 메타데이터 ─────────────────────────────
    feat["meta_family_name"] = data.get("FamilyName", "")
    feat["meta_type_name"]   = data.get("TypeName", "")

    params = data.get("InstanceParameters", [])
    param_map: dict[str, str] = {}
    for p in params:
        name = p.get("Name", "")
        val  = str(p.get("Value", ""))
        if name not in param_map:   # 첫 번째 값 우선
            param_map[name] = val

    feat["meta_category"]     = param_map.get("Category", "")
    feat["meta_system_class"] = param_map.get("System Classification", "")
    feat["meta_max_size_mm"]  = _parse_mm_value(param_map.get("Maximum Size", "0"))
    feat["meta_min_size_mm"]  = _parse_mm_value(param_map.get("Minimum Size", "0"))

    # ── 형상 피처 ──────────────────────────────
    solids = data.get("Geometry", {}).get("Solid", [])
    total_surface_area = sum(s.get("SurfaceArea", 0.0) for s in solids)
    total_volume       = sum(s.get("Volume", 0.0) for s in solids)
    solid_count        = len(solids)

    feat["geo_surface_area"] = total_surface_area * _FT2_TO_M2
    feat["geo_volume"]       = total_volume * _FT3_TO_M3
    feat["geo_solid_count"]  = solid_count

    # BBox: 모든 솔리드의 Face → EdgeLoop → Tessellate 점 순회
    face_count = 0
    xs, ys, zs = [], [], []

    for solid in solids:
        for face in solid.get("Faces", []):
            face_count += 1
            for loop in face.get("EdgeLoops", []):
                for edge in loop:
                    for pt in edge.get("Tessellate", []):
                        xs.append(pt["X"])
                        ys.append(pt["Y"])
                        zs.append(pt["Z"])

    feat["geo_face_count"] = face_count

    if xs:
        bbox_x = (max(xs) - min(xs)) * _FT_TO_M
        bbox_y = (max(ys) - min(ys)) * _FT_TO_M
        bbox_z = (max(zs) - min(zs)) * _FT_TO_M
    else:
        bbox_x = bbox_y = bbox_z = 0.0

    feat["geo_bbox_x"]    = bbox_x
    feat["geo_bbox_y"]    = bbox_y
    feat["geo_bbox_z"]    = bbox_z
    feat["geo_bbox_diag"] = math.sqrt(bbox_x ** 2 + bbox_y ** 2 + bbox_z ** 2)
    feat["geo_aspect_xy"] = bbox_x / (bbox_y + _EPS)
    feat["geo_aspect_xz"] = bbox_x / (bbox_z + _EPS)
    feat["geo_aspect_yz"] = bbox_y / (bbox_z + _EPS)

    sa = feat["geo_surface_area"]
    vol = feat["geo_volume"]
    feat["geo_compactness"] = vol / (sa ** 1.5 + _EPS)
    feat["geo_density"] = feat["geo_face_count"] / feat["geo_volume"] if feat["geo_volume"] > 0 else 0

    # ── 커넥터 피처 ────────────────────────────
    connectors = data.get("Connectors", [])
    conn_count = len(connectors)
    feat["conn_count"] = conn_count

    radii = [c.get("Radius", 0.0) * _FT_TO_M for c in connectors]

    if radii:
        feat["conn_radius_mean"]  = sum(radii) / len(radii)
        feat["conn_radius_std"]   = _std(radii)
        feat["conn_radius_max"]   = max(radii)
        feat["conn_radius_min"]   = min(radii)
    else:
        feat["conn_radius_mean"]  = 0.0
        feat["conn_radius_std"]   = 0.0
        feat["conn_radius_max"]   = 0.0
        feat["conn_radius_min"]   = 0.0
        feat["conn_radius_range"] = 0.0

    # 두 커넥터가 있을 때 거리 / 각도 / 동축 판단
    if conn_count >= 2:
        c0, c1 = connectors[0], connectors[1]
        dist = _vec_dist(c0["Origin"], c1["Origin"]) * _FT_TO_M
        angle_deg = _angle_between_deg(c0["Direction"], c1["Direction"])

        # 동축(same_axis): 두 방향 벡터가 평행 (0° 또는 180°)
        # 인라인: 동축 + 반대 방향 (각도 ≈ 180°)
    else:
        dist      = 0.0
        angle_deg = 0.0

    feat["conn_distance"]  = dist
    feat["conn_angle_deg"] = angle_deg

    return feat


def extract_features_from_file(json_path: str | Path) -> dict:
    """
    JSON 파일 경로 → 피처 딕셔너리 반환.
    파일명 정보를 'source_file' 키로 함께 반환.
    """
    json_path = Path(json_path)
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    feat = extract_features(data)
    feat["source_file"] = json_path.stem
    return feat


# ─────────────────────────────────────────────
#  수치 유틸
# ─────────────────────────────────────────────

def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(variance)


# ─────────────────────────────────────────────
#  피처 컬럼 정의 (학습용 수치 피처만)
# ─────────────────────────────────────────────

NUMERIC_FEATURE_COLS = [
    # 메타데이터 수치
    "meta_max_size_mm",
    "meta_min_size_mm",
    # 형상
    "geo_surface_area",
    "geo_volume",
    "geo_solid_count",
    "geo_face_count",
    "geo_bbox_x",
    "geo_bbox_y",
    "geo_bbox_z",
    "geo_bbox_diag",
    "geo_aspect_xy",
    "geo_aspect_xz",
    "geo_aspect_yz",
    "geo_compactness",
    "geo_density",
    # 커넥터
    "conn_count",
    "conn_radius_mean",
    "conn_radius_std",
    "conn_radius_max",
    "conn_radius_min",
    "conn_distance",
    "conn_angle_deg",
]

# 피처 그룹 (Ablation study 용)
FEATURE_GROUPS = {
    "geometry": [
        "geo_surface_area", "geo_volume", "geo_solid_count", "geo_face_count",
        "geo_bbox_x", "geo_bbox_y", "geo_bbox_z", "geo_bbox_diag",
        "geo_aspect_xy", "geo_aspect_xz", "geo_aspect_yz", "geo_compactness",
        "geo_density",
    ],
    "connector": [
        "conn_count", "conn_radius_mean", "conn_radius_std",
        "conn_radius_max", "conn_radius_min",
        "conn_distance", "conn_angle_deg",
    ],
    "metadata": [
        "meta_max_size_mm", "meta_min_size_mm",
    ],
}


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python feature_extractor.py <json_file>")
        sys.exit(1)

    feat = extract_features_from_file(sys.argv[1])
    print("\n=== 추출된 피처 ===")
    for k, v in feat.items():
        print(f"  {k:<25} : {v}")
