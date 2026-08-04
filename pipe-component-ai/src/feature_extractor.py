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

  [커넥터 (Connector)]
    conn_count           : 커넥터 수
    conn_radius_mean     : Radius 평균 (m)
    conn_radius_std      : Radius 표준편차 (m)
    conn_radius_max      : Radius 최대값 (m)
    conn_radius_min      : Radius 최솟값 (m)
    conn_radius_range    : max - min
    conn_distance        : 첫 두 커넥터 간 거리 (2개 이상일 때, m)
    conn_angle_deg       : 두 커넥터 방향 벡터 사이 각도 (degree)
    conn_same_axis       : 두 커넥터가 동일 축 여부 (0/1)
    conn_is_inline       : 두 커넥터가 인라인 (같은 방향 반대, 동축) 여부 (0/1)

    (주: 위 목록은 최초 작성 당시 문서고, conn_density/conn_is_two_way/
     geo_volume_per_face/conn_branch_* 등 이후 추가된 피처는 코드
     NUMERIC_FEATURE_COLS / FEATURE_GROUPS 쪽이 최신 소스임.)

  [순수 geometry 분류기 보강 - ACCESSORY / INSTRUMENT / WELD 판별용]
    이 세 클래스는 kw_*(이름) 피처를 뺀 순수 geometry 모델에서 정확도가
    크게 떨어졌던 클래스들이다. 원인이 서로 다르므로 각각 다른 피처로 대응:

    - ACCESSORY (strainer/spectacle blind/filter 등 잡화 묶음)
      → 단일 vs 다중 솔리드 구성비로 구분 시도.
        geo_bbox_volume, geo_fill_ratio, geo_max_solid_frac,
        geo_solid_volume_cv, geo_convexity
    - INSTRUMENT (gauge/meter/sight glass 등, 커넥터 1개인 경우가 많음)
      → 기존 conn_distance류는 커넥터가 2개 이상이어야 값이 생기므로,
        커넥터 1개짜리 부품(주로 계기)에서는 전부 0이 되어 판별력을
        잃는다. conn_is_one_way + conn_body_size_ratio로 보완.
        conn_is_one_way, conn_is_four_plus_way, conn_body_size_ratio
    - WELD (거의 마커/주석에 가까운 얇은 형상)
      → 실제 솔리드 부피가 bbox 부피에 비해 거의 0에 가까운 "얇은 형상"
        여부를 연속값으로 제공 (geo_fill_ratio가 여기에도 기여).

    geo_bbox_volume     : bbox_x * bbox_y * bbox_z (m³) - solid 채움 비율 계산용
    geo_fill_ratio      : geo_volume / geo_bbox_volume - bbox 대비 실제
                          솔리드가 얼마나 채워져 있는지 (thin/marker 형상일수록 0에 가까움)
    geo_max_solid_frac  : 가장 큰 솔리드의 부피 / 전체 솔리드 부피 합
                          (단일 바디는 1에 가까움, spectacle blind처럼 크기가
                           비슷한 2개 솔리드로 구성되면 ~0.5)
    geo_solid_volume_cv : 솔리드별 부피의 변동계수(std/mean) - 여러 솔리드의
                          크기가 균일한지(낮음) 아니면 들쭉날쭉한지(높음)
    geo_convexity       : 실제 솔리드 부피 / convex hull 부피 (0~1) - 메쉬형
                          스트레이너처럼 오목/천공 형상일수록 낮아짐
                          (scipy 미설치 시 1.0으로 대체)
    conn_is_one_way     : 커넥터가 정확히 1개인지 (0/1) - dead-end 부착형
                          계기(가우지 등)에서 흔함
    conn_is_four_plus_way : 커넥터가 4개 이상인지 (0/1)
    conn_body_size_ratio : geo_bbox_diag / conn_radius_mean - 커넥터 지름 대비
                          바디 크기 비율. 계기는 접속구가 작아도 다이얼/바디가
                          커서 이 값이 크게 나오는 경향, 밸브/피팅은 배관
                          사이즈에 body가 비례해 이 값이 상대적으로 안정적임.
                          (커넥터가 1개뿐이라 conn_distance류가 0이 되는
                           경우에도 항상 값이 존재하는 게 핵심)

Revit 내부 단위 변환:
  길이 : feet → meter  (* 0.3048)
  면적 : feet² → m²    (* 0.3048²)
  부피 : feet³ → m³    (* 0.3048³)
"""

import json
import math
from pathlib import Path
from typing import Any

try:
    from scipy.spatial import ConvexHull  # noqa: F401
    _HAS_SCIPY = True
except ImportError:  # scipy가 없는 환경에서도 나머지 피처는 그대로 동작해야 함
    ConvexHull = None
    _HAS_SCIPY = False

# convex hull 계산 비용을 제한하기 위한 점군 상한 (초과 시 균등 서브샘플링)
_MAX_HULL_POINTS = 2000


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
    feat["geo_volume_per_face"] = (
        vol / (feat["geo_face_count"] + _EPS)
    )

    # ── 순수 geometry 보강 (ACCESSORY 판별용) ──
    # bbox 대비 실제 솔리드 채움 비율 - 마커/주석성 얇은 형상(WELD 등)일수록
    # 0에 가깝고, 속이 꽉 찬 단일 바디(CAP/FLANGE 등)일수록 1에 가까움.
    feat["geo_bbox_volume"] = bbox_x * bbox_y * bbox_z
    feat["geo_fill_ratio"] = vol / (feat["geo_bbox_volume"] + _EPS)

    # 솔리드별 부피 분포 - 단일 바디 vs 다중 솔리드 조립체(예: spectacle
    # blind = 크기가 비슷한 원판 2개) 구분에 사용.
    solid_volumes = [s.get("Volume", 0.0) for s in solids]
    if solid_volumes:
        _sv_sum = sum(solid_volumes) + _EPS
        feat["geo_max_solid_frac"] = max(solid_volumes) / _sv_sum
        _sv_mean = sum(solid_volumes) / len(solid_volumes)
        feat["geo_solid_volume_cv"] = (
            _std(solid_volumes) / (_sv_mean + _EPS)
            if len(solid_volumes) > 1 else 0.0
        )
    else:
        feat["geo_max_solid_frac"]  = 0.0
        feat["geo_solid_volume_cv"] = 0.0

    # Convexity: 실제 솔리드 부피 / convex hull 부피. 천공/메쉬형 스트레이너
    # 바스켓처럼 오목한 형상일수록 hull 부피가 실제 부피보다 크게 나와
    # 이 비율이 낮아진다. (같은 xs/ys/zs 점군에서 계산하므로 raw feet^3
    # 단위끼리 나눠 비율만 쓰는 것이므로 단위 변환은 필요 없음)
    feat["geo_convexity"] = 1.0
    if _HAS_SCIPY and len(xs) >= 4 and total_volume > _EPS:
        pts = list(zip(xs, ys, zs))
        if len(pts) > _MAX_HULL_POINTS:
            step = max(1, len(pts) // _MAX_HULL_POINTS)
            pts = pts[::step]
        try:
            hull_volume = ConvexHull(pts).volume
            if hull_volume > _EPS:
                feat["geo_convexity"] = min(1.0, total_volume / hull_volume)
        except Exception:
            # 동일 평면상의 점들만 있는 등 hull이 정의되지 않는 퇴화 케이스
            feat["geo_convexity"] = 1.0

    # ── 커넥터 피처 ────────────────────────────
    connectors = data.get("Connectors", [])
    conn_count = len(connectors)

    feat["conn_count"] = conn_count
    feat["conn_density"] = (
        conn_count / (feat["geo_volume"] + _EPS)
    )

    # Connector topology features
    # 1-way: dead-end 부착형 (gauge/meter 등 계기류에서 흔함)
    # 2-way: elbow, coupling, valve, pipe connection
    # 3-way: tee, olet, branch components
    # 4-way 이상: cross, 4-way valve 등
    feat["conn_is_zero_way"] = int(conn_count == 0)
    feat["conn_is_one_way"] = int(conn_count == 1)
    feat["conn_is_two_way"] = int(conn_count == 2)
    feat["conn_is_three_way"] = int(conn_count == 3)
    feat["conn_is_four_plus_way"] = int(conn_count >= 4)

    radii = [c.get("Radius", 0.0) * _FT_TO_M for c in connectors]

    if radii:
        feat["conn_radius_mean"]  = sum(radii) / len(radii)
        feat["conn_radius_std"]   = _std(radii)
        feat["conn_radius_max"]   = max(radii)
        feat["conn_radius_min"]   = min(radii)
        feat["conn_radius_range"] = max(radii) - min(radii)
        feat["conn_radius_ratio"] = (
            max(radii) / (min(radii) + _EPS)
            if radii else 0.0
        )

    else:
        feat["conn_radius_mean"]  = 0.0
        feat["conn_radius_std"]   = 0.0
        feat["conn_radius_max"]   = 0.0
        feat["conn_radius_min"]   = 0.0
        feat["conn_radius_range"] = 0.0
        feat["conn_radius_ratio"] = 0.0

    # 바디 크기 / 커넥터 반경 비율. conn_distance류는 커넥터가 2개
    # 이상이어야 값이 생기지만, 이 피처는 커넥터가 1개뿐인 부품
    # (계기류에서 흔함)에서도 항상 값이 존재한다는 게 핵심 - 접속구는
    # 작아도 다이얼/바디가 큰 계기류를 밸브/피팅과 구분하는 데 쓰인다.
    #
    # conn_count == 0 (SUPPORT/BOLT/GASKET 등 커넥터가 애초에 없는 부품)일
    # 때는 conn_radius_mean이 0.0이 되어 이 비율이 geo_bbox_diag / _EPS로
    # 발산한다 (정상 범위가 대략 1~200인데 반해 1e7~1e10대로 튐) - 실제
    # 형상 신호가 아니라 0-나눗셈 방지용 _EPS의 부작용이므로, 커넥터가
    # 없는 경우엔 비율 자체를 정의하지 않고 0.0으로 둔다. 이 경우 구분
    # 신호는 위에서 추가한 conn_is_zero_way 플래그가 대신 제공한다.
    feat["conn_body_size_ratio"] = (
        feat["geo_bbox_diag"] / (feat["conn_radius_mean"] + _EPS)
        if conn_count > 0 else 0.0
    )

    # 두 커넥터가 있을 때 거리 / 각도 / 동축 판단
    if conn_count >= 2:
        c0, c1 = connectors[0], connectors[1]
        dist = _vec_dist(c0["Origin"], c1["Origin"]) * _FT_TO_M
        angle_deg = _angle_between_deg(c0["Direction"], c1["Direction"])

        # 동축(same_axis): 두 방향 벡터가 평행 (0° 또는 180°)
        same_axis = int(angle_deg < 10.0 or angle_deg > 170.0)
        # 인라인: 동축 + 반대 방향 (각도 ≈ 180°)
        is_inline = int(angle_deg > 170.0)
    else:
        dist      = 0.0
        angle_deg = 0.0
        same_axis = 0
        is_inline = 0

    feat["conn_distance"]  = dist

    feat["conn_distance_ratio"] = (
        dist / (feat["geo_bbox_diag"] + _EPS)
    )

    feat["conn_length_ratio"] = (
        dist / (feat["conn_radius_mean"] + _EPS)
    )

    feat["conn_angle_deg"] = angle_deg
    feat["conn_same_axis"] = same_axis
    feat["conn_is_inline"] = is_inline

    # 세 번째 커넥터(분기) - OLET/TEE 등 분기형 피팅과
    # COUPLING/스트레이트 피팅을 구분하는 데 유용한 피처

    branch_angles = []

    for i in range(conn_count):
        for j in range(i + 1, conn_count):
            angle = _angle_between_deg(
                connectors[i]["Direction"],
                connectors[j]["Direction"]
            )

            if 10.0 <= angle <= 170.0:
                branch_angles.append(angle)


    if branch_angles:
        branch_angle_deg = min(branch_angles)
        has_branch = 1
    else:
        branch_angle_deg = 0.0
        has_branch = 0

    branch_dist = 0.0

    if has_branch:
        branch_pairs = []

        for i in range(conn_count):
            for j in range(i + 1, conn_count):
                angle = _angle_between_deg(
                    connectors[i]["Direction"],
                    connectors[j]["Direction"]
                )

                if 10.0 <= angle <= 170.0:
                    d = _vec_dist(
                        connectors[i]["Origin"],
                        connectors[j]["Origin"]
                    ) * _FT_TO_M

                    branch_pairs.append(d)

        if branch_pairs:
            branch_dist = min(branch_pairs)

    feat["conn_branch_angle_deg"] = branch_angle_deg
    feat["conn_branch_distance"]  = branch_dist
    feat["conn_has_branch"]       = has_branch

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
    # metadata
    "meta_max_size_mm",
    "meta_min_size_mm",

    # geometry
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
    "geo_volume_per_face",
    "geo_bbox_volume",
    "geo_fill_ratio",
    "geo_max_solid_frac",
    "geo_solid_volume_cv",
    "geo_convexity",

    # connector
    "conn_count",
    "conn_density",
    "conn_is_zero_way",
    "conn_is_one_way",
    "conn_is_two_way",
    "conn_is_three_way",
    "conn_is_four_plus_way",
    "conn_radius_mean",
    "conn_radius_std",
    "conn_radius_max",
    "conn_radius_min",
    "conn_radius_range",
    "conn_radius_ratio",
    "conn_body_size_ratio",
    "conn_distance",
    "conn_distance_ratio",
    "conn_length_ratio",
    "conn_angle_deg",
    "conn_same_axis",
    "conn_is_inline",
    "conn_branch_angle_deg",
    "conn_branch_distance",
    "conn_has_branch",
]

# 피처 그룹 (Ablation study 용)
FEATURE_GROUPS = {
    "geometry": [
        "geo_surface_area", "geo_volume", "geo_solid_count", "geo_face_count",
        "geo_bbox_x", "geo_bbox_y", "geo_bbox_z", "geo_bbox_diag",
        "geo_aspect_xy", "geo_aspect_xz", "geo_aspect_yz", "geo_compactness", "geo_volume_per_face",
        "geo_bbox_volume", "geo_fill_ratio", "geo_max_solid_frac", "geo_solid_volume_cv", "geo_convexity",
    ],
    "connector": [
        "conn_count", "conn_density",
        "conn_is_zero_way", "conn_is_one_way", "conn_is_two_way", "conn_is_three_way", "conn_is_four_plus_way",
        "conn_radius_mean", "conn_radius_std", "conn_radius_max", "conn_radius_min",
        "conn_radius_range", "conn_radius_ratio", "conn_body_size_ratio",
        "conn_distance", "conn_distance_ratio", "conn_length_ratio", "conn_angle_deg", "conn_same_axis", "conn_is_inline",
        "conn_branch_angle_deg", "conn_branch_distance", "conn_has_branch",
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