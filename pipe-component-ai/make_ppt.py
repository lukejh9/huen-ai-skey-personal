"""
make_ppt.py
-----------
AI 기반 배관 컴포넌트 인식 모듈 설명 자료 생성 스크립트.
실행: pip install python-pptx  →  python make_ppt.py
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
import copy
from pathlib import Path

# ── 색상 팔레트 ─────────────────────────────────────
def rgb(h): return RGBColor(int(h[0:2],16), int(h[2:4],16), int(h[4:6],16))

NAVY    = rgb("1E2761")
TEAL    = rgb("028090")
TEAL_LT = rgb("00A896")
MINT    = rgb("02C39A")
WHITE   = rgb("FFFFFF")
OFF_W   = rgb("F4F7FA")
GRAY    = rgb("64748B")
GRAY_LT = rgb("E2E8F0")
CODE_BG = rgb("1E293B")
CODE_TX = rgb("E2E8F0")
CODE_KW = rgb("7DD3FC")
CODE_ST = rgb("86EFAC")
CODE_NM = rgb("FCA5A5")
CODE_CM = rgb("64748B")
CODE_FN = rgb("FCD34D")
GREEN   = rgb("059669")
AMBER   = rgb("D97706")
RED     = rgb("DC2626")
PURPLE  = rgb("7C3AED")
ICE     = rgb("CADCFC")

W  = Inches(10)
H  = Inches(5.625)
IN = Inches

def prs_new():
    prs = Presentation()
    prs.slide_width  = W
    prs.slide_height = H
    return prs

def blank(prs):
    layout = prs.slide_layouts[6]  # blank
    return prs.slides.add_slide(layout)

def rect(slide, x, y, w, h, fill_c=None, line_c=None, line_w=Pt(0)):
    from pptx.util import Emu
    shape = slide.shapes.add_shape(1, IN(x), IN(y), IN(w), IN(h))  # MSO_SHAPE.RECTANGLE=1
    shape.line.width = line_w
    if fill_c:
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill_c
    else:
        shape.fill.background()
    if line_c:
        shape.line.color.rgb = line_c
    else:
        shape.line.fill.background()
    return shape

def txt(slide, text, x, y, w, h, size=12, bold=False, color=WHITE, align=PP_ALIGN.LEFT, italic=False, font="Calibri", wrap=True):
    tb = slide.shapes.add_textbox(IN(x), IN(y), IN(w), IN(h))
    tf = tb.text_frame
    tf.word_wrap = wrap
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    run.font.name = font
    return tb

def code_box(slide, x, y, w, h):
    """어두운 코드 배경 박스"""
    return rect(slide, x, y, w, h, fill_c=CODE_BG)

def code_line(slide, lines_data, x, y, w, line_h=0.2, font_size=9.5):
    """
    lines_data: list of list of (text, color) tuples
    각 항목이 한 줄
    """
    for i, segments in enumerate(lines_data):
        tb = slide.shapes.add_textbox(IN(x), IN(y + i*line_h), IN(w), IN(line_h + 0.02))
        tf = tb.text_frame
        tf.word_wrap = False
        p = tf.paragraphs[0]
        for seg_text, seg_color in segments:
            run = p.add_run()
            run.text = seg_text
            run.font.size = Pt(font_size)
            run.font.color.rgb = seg_color
            run.font.name = "Consolas"

def add_header(slide, title, color=NAVY, size=20):
    rect(slide, 0, 0, 10, 0.72, fill_c=color)
    txt(slide, title, 0.4, 0.08, 9.2, 0.58, size=size, bold=True, color=WHITE, align=PP_ALIGN.LEFT)

def add_card(slide, x, y, w, h, header_text, header_color, items, item_size=10.5):
    rect(slide, x, y, w, h, fill_c=WHITE, line_c=GRAY_LT, line_w=Pt(0.5))
    rect(slide, x, y, w, 0.36, fill_c=header_color)
    txt(slide, header_text, x+0.1, y+0.04, w-0.15, 0.3, size=12, bold=True, color=WHITE)
    for j, item in enumerate(items):
        txt(slide, ("• " if item else "") + item, x+0.12, y+0.42+j*0.42, w-0.2, 0.38,
            size=item_size, color=GRAY, font="Consolas" if item and not item.startswith("※") else "Calibri")


# ═══════════════════════════════════════════════════════════════
prs = prs_new()

# ──────────────────────────────────────────────────────────────
# SLIDE 1 – 표지
# ──────────────────────────────────────────────────────────────
s = blank(prs)
rect(s, 0, 0, 10, 5.625, fill_c=NAVY)
rect(s, 0, 0, 0.18, 5.625, fill_c=TEAL)

txt(s, "BIM 기반 AI 배관 컴포넌트 인식 모듈",
    0.55, 1.0, 7.5, 1.0, size=34, bold=True, color=WHITE, font="Calibri")
rect(s, 0.55, 2.18, 3.5, 0.05, fill_c=TEAL)
txt(s, "전처리 · 라벨링 · 학습 · 분석 결과",
    0.55, 2.35, 7, 0.45, size=16, color=ICE, font="Calibri")
txt(s, "LightGBM  ·  Python  ·  Revit BIM JSON  ·  FastAPI",
    0.55, 2.9, 7, 0.35, size=13, color=TEAL_LT, font="Calibri")
txt(s, "Namespace: RailSweng",
    0.55, 4.85, 4, 0.3, size=11, color=rgb("7BA7BC"), font="Calibri")

# ──────────────────────────────────────────────────────────────
# SLIDE 2 – 전체 파이프라인
# ──────────────────────────────────────────────────────────────
s = blank(prs)
rect(s, 0, 0, 10, 5.625, fill_c=OFF_W)
add_header(s, "전체 시스템 파이프라인", color=NAVY, size=22)

steps = [
    ("BIM JSON\n데이터",     "Revit 패밀리\n형상·커넥터·속성", NAVY),
    ("피처 추출\n(전처리)",  "24개 수치 피처\n형상·커넥터·메타", TEAL),
    ("라벨링",               "xlsx 정답\n+ 파일명 fallback",  PURPLE),
    ("LightGBM\n학습",       "그룹 분류기\n타입 분류기",       AMBER),
    ("추론 결과\n(FastAPI)",  "그룹·타입·신뢰도\n검토 여부",    GREEN),
]

bw, bh, sx, sy, gap = 1.55, 1.3, 0.22, 1.2, 0.38
for i, (lbl, sub, col) in enumerate(steps):
    x = sx + i*(bw+gap)
    rect(s, x, sy, bw, bh, fill_c=col)
    txt(s, lbl, x+0.06, sy+0.08, bw-0.1, 0.55, size=12, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    txt(s, sub, x+0.06, sy+0.68, bw-0.1, 0.55, size=9.5, color=GRAY_LT, align=PP_ALIGN.CENTER)
    if i < len(steps)-1:
        rect(s, x+bw+0.04, sy+bh/2-0.02, gap-0.08, 0.04, fill_c=TEAL)

col_info = [
    ("feature_extractor.py", ["SurfaceArea, Volume", "BBox X/Y/Z", "Connector Radius/Angle"]),
    ("label_loader / skey_mapper", ["xlsx → SKY 매핑", "symbol.xml SKEY 파싱", "파일명 fallback"]),
    ("train_group / train_type", ["LOO-CV 교차검증", "LightGBM + 정규화", "Feature Importance"]),
    ("predict.py / api/main.py", ["배치·단건 추론", "신뢰도 임계값", "POST /predict"]),
]
for i, (title, items) in enumerate(col_info):
    x = 0.22 + i*2.45
    rect(s, x, 2.72, 2.3, 2.6, fill_c=WHITE, line_c=GRAY_LT, line_w=Pt(0.5))
    rect(s, x, 2.72, 2.3, 0.3, fill_c=GRAY_LT)
    txt(s, title, x+0.06, 2.73, 2.18, 0.27, size=8.5, bold=True, color=NAVY)
    for j, item in enumerate(items):
        txt(s, "• "+item, x+0.1, 3.08+j*0.55, 2.1, 0.5, size=10, color=GRAY)

# ──────────────────────────────────────────────────────────────
# SLIDE 3 – 데이터 구조
# ──────────────────────────────────────────────────────────────
s = blank(prs)
rect(s, 0, 0, 10, 5.625, fill_c=OFF_W)
add_header(s, "1단계 · 데이터 구조 — Revit BIM JSON", color=NAVY)
rect(s, 0, 0.72, 10, 0.05, fill_c=TEAL)

fields = [
    ("Id",                  "Revit Element ID"),
    ("FamilyName",          "패밀리명 (분류 기준)"),
    ("TypeName",            "타입명 (구경 등)"),
    ("Geometry.Solid",      "SurfaceArea / Volume / Faces"),
    ("Connectors[]",        "Origin / Direction / Radius"),
    ("InstanceParameters",  "Category, Size, System 등"),
]
txt(s, "JSON 최상위 구조", 0.35, 0.95, 4, 0.32, size=12, bold=True, color=NAVY)
for i, (key, desc) in enumerate(fields):
    y = 1.35 + i*0.52
    rect(s, 0.35, y, 4.2, 0.44, fill_c=WHITE, line_c=GRAY_LT, line_w=Pt(0.5))
    rect(s, 0.35, y, 0.06, 0.44, fill_c=TEAL)
    txt(s, key,  0.5,  y+0.04, 1.55, 0.36, size=10.5, bold=True, color=TEAL, font="Consolas")
    txt(s, desc, 2.1,  y+0.04, 2.35, 0.36, size=10.5, color=GRAY)

# 우: 코드 박스
code_box(s, 5.0, 0.88, 4.75, 4.5)
txt(s, "T_BALL VALVE_FL_STS_LEVER_10K_A2A_15A-200A.json",
    5.1, 0.91, 4.55, 0.26, size=8.5, color=rgb("94A3B8"), font="Consolas")
rect(s, 5.0, 1.2, 4.75, 0.03, fill_c=rgb("334155"))

code_line(s, [
    [("{", CODE_TX)],
    [('  "Id": ', CODE_KW), ("1294553,", CODE_NM)],
    [('  "FamilyName": ', CODE_KW), ('"T_BALL VALVE...",', CODE_ST)],
    [('  "TypeName": ', CODE_KW), ('"15A-200A",', CODE_ST)],
    [('  "Geometry": {', CODE_TX)],
    [('    "Solid": [{', CODE_TX)],
    [('      "SurfaceArea": ', CODE_KW), ("0.170,", CODE_NM)],
    [('      "Volume": ', CODE_KW), ("0.0024,", CODE_NM)],
    [('      "Faces": [...]', CODE_TX)],
    [('  ]}', CODE_TX)],
    [('  "Connectors": [', CODE_TX)],
    [('    { "Radius": ', CODE_KW), ("0.0656 },", CODE_NM)],
    [('    { "Radius": ', CODE_KW), ("0.0656 }", CODE_NM)],
    [('  ]', CODE_TX)],
    [("}", CODE_TX)],
], x=5.1, y=1.26, w=4.55, line_h=0.205)

# 배지
rect(s, 5.0, 4.88, 2.1, 0.42, fill_c=TEAL)
txt(s, "총 55개 JSON 파일", 5.0, 4.88, 2.1, 0.42, size=11, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
rect(s, 7.2, 4.88, 2.3, 0.42, fill_c=PURPLE)
txt(s, "파일당 500KB~2.2MB", 7.2, 4.88, 2.3, 0.42, size=11, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

# ──────────────────────────────────────────────────────────────
# SLIDE 4 – 전처리 / 피처 추출
# ──────────────────────────────────────────────────────────────
s = blank(prs)
rect(s, 0, 0, 10, 5.625, fill_c=OFF_W)
add_header(s, "2단계 · 전처리 — 피처 추출 (feature_extractor.py)", color=TEAL)

fgroups = [
    ("형상 피처 (12개)",     NAVY,   ["geo_surface_area  (m²)", "geo_volume  (m³)", "geo_bbox_x/y/z  (m)", "geo_aspect_xy/xz/yz", "geo_compactness"]),
    ("커넥터 피처 (10개)",   PURPLE, ["conn_count", "conn_radius_mean/std", "conn_radius_max/min", "conn_distance  (m)", "conn_angle_deg / same_axis"]),
    ("메타데이터 피처 (2개)",AMBER,  ["meta_max_size_mm", "meta_min_size_mm", "", "※ Revit 단위 변환", "   feet → meter (×0.3048)"]),
]
for i, (title, col, items) in enumerate(fgroups):
    x = 0.25 + i*3.25
    add_card(s, x, 0.88, 3.0, 2.78, title, col, items)

# 코드 스니펫
code_box(s, 0.25, 3.83, 9.5, 1.6)
txt(s, "feature_extractor.py  —  핵심 추출 로직",
    0.38, 3.86, 6, 0.25, size=9, color=rgb("94A3B8"), font="Consolas")
rect(s, 0.25, 4.13, 9.5, 0.03, fill_c=rgb("334155"))

code_line(s, [
    [("def ", CODE_KW), ("extract_features", CODE_FN), ("(data: dict) -> dict:", CODE_TX)],
    [("    solids = data[", CODE_TX), ('"Geometry"', CODE_ST), ("][", CODE_TX), ('"Solid"', CODE_ST), ("]", CODE_TX)],
    [("    feat[", CODE_TX), ('"geo_surface_area"', CODE_ST), ("] = sum(s[", CODE_TX), ('"SurfaceArea"', CODE_ST), ("] for s in solids) * ", CODE_TX), ("0.0929", CODE_NM)],
    [("    # BBox: Tessellate 포인트 전체 min/max X,Y,Z → bbox_x = (max_x-min_x)*0.3048", CODE_CM)],
    [("    feat[", CODE_TX), ('"conn_distance"', CODE_ST), ("] = _vec_dist(conns[0]['Origin'], conns[1]['Origin']) * ", CODE_TX), ("0.3048", CODE_NM)],
], x=0.38, y=4.18, w=9.3, line_h=0.21)

# ──────────────────────────────────────────────────────────────
# SLIDE 5 – 라벨링
# ──────────────────────────────────────────────────────────────
s = blank(prs)
rect(s, 0, 0, 10, 5.625, fill_c=OFF_W)
add_header(s, "3단계 · 라벨링 — 자동 레이블 & 정답 매핑", color=PURPLE)

methods = [
    ("① xlsx 정답 레이블 (우선)", PURPLE,
     "학습용레이블_정답.xlsx\nProductCode(패밀리명:타입명) → SKY\nsymbol.xml SKEY → 컴포넌트 그룹/밸브타입"),
    ("② 파일명 기반 Fallback", TEAL,
     "xlsx 매칭 실패 시 자동 적용\n파일명 패턴 → 정규식 → 레이블\nT_BALL VALVE → VALVE/BALL 등"),
]
for i, (title, col, body) in enumerate(methods):
    y = 0.88 + i*1.85
    rect(s, 0.25, y, 4.3, 1.65, fill_c=WHITE, line_c=GRAY_LT, line_w=Pt(0.5))
    rect(s, 0.25, y, 0.07, 1.65, fill_c=col)
    txt(s, title, 0.42, y+0.08, 4.0, 0.35, size=12, bold=True, color=col)
    txt(s, body,  0.42, y+0.48, 4.0, 1.1, size=11, color=GRAY)

# 클래스 목록
txt(s, "컴포넌트 그룹 클래스 (9종)", 0.25, 4.6, 4.3, 0.3, size=11, bold=True, color=NAVY)
classes = [("VALVE","D97706"),("FITTING","059669"),("FLANGE","7C3AED"),("GASKET","028090"),
           ("SLEEVE","1E2761"),("SUPPORT","DC2626"),("INSTRUMENT","0891B2"),("FLEXIBLE_JOINT","DB2777"),("ACCESSORY","64748B")]
for i, (cl, ch) in enumerate(classes):
    col_idx = 0 if i < 5 else 1
    row_idx = i if i < 5 else i-5
    x = 0.25 + col_idx*2.2
    y = 4.96 + row_idx*0.29
    rect(s, x, y, 2.0, 0.25, fill_c=rgb(ch+"22"), line_c=rgb(ch), line_w=Pt(0.5))
    txt(s, cl, x+0.06, y+0.01, 1.9, 0.23, size=9.5, bold=True, color=rgb(ch), font="Consolas")

# 코드 박스
code_box(s, 4.8, 0.88, 4.95, 4.5)
txt(s, "label_loader.py  —  LabelRegistry.get_labels()",
    4.92, 0.91, 4.7, 0.25, size=9, color=rgb("94A3B8"), font="Consolas")
rect(s, 4.8, 1.18, 4.95, 0.03, fill_c=rgb("334155"))

code_line(s, [
    [("class ", CODE_KW), ("LabelRegistry", CODE_FN), (":", CODE_TX)],
    [("  def ", CODE_KW), ("get_labels", CODE_FN), ("(self, family, type_name):", CODE_TX)],
    [("    key = _normalize(family)", CODE_TX)],
    [("    # ① xlsx 정확 매칭", CODE_CM)],
    [("    if", CODE_KW), (" key in self.lookup:", CODE_TX)],
    [("      return ", CODE_KW), ("self.lookup[key][type_name]", CODE_TX)],
    [("    # ② Fallback: 파일명 기반", CODE_CM)],
    [("    return ", CODE_KW), ("fallback_labels(family)", CODE_TX)],
    [("", CODE_TX)],
    [("# skey_mapper.py — SKEY → 레이블", CODE_CM)],
    [("def ", CODE_KW), ("get_labels_from_skey", CODE_FN), ("(skey, skey_map):", CODE_TX)],
    [('    # "VB**" → VALVE / BALL', CODE_CM)],
    [('    # "EL**" → FITTING / NON_VALVE', CODE_CM)],
    [("    info = skey_map[skey]", CODE_TX)],
    [("    return ", CODE_KW), ("{ group: info.component_group,", CODE_TX)],
    [("             valve_type: info.valve_type }", CODE_TX)],
], x=4.92, y=1.23, w=4.7, line_h=0.205)

# ──────────────────────────────────────────────────────────────
# SLIDE 6 – 학습
# ──────────────────────────────────────────────────────────────
s = blank(prs)
rect(s, 0, 0, 10, 5.625, fill_c=OFF_W)
add_header(s, "4단계 · 학습 — LightGBM 분류기 (train_group / train_type)", color=AMBER)

models_info = [
    ("컴포넌트 그룹 분류기", "9개 클래스 (multiclass)", NAVY,
     ["objective: multiclass", "num_leaves: 8", "max_depth: 4", "learning_rate: 0.05", "reg_alpha: 0.5  (L1)", "reg_lambda: 1.0  (L2)", "class_weight: balanced"]),
    ("밸브 타입 분류기", "VALVE 샘플만 학습", PURPLE,
     ["objective: multiclass", "num_leaves: 6", "max_depth: 3", "min_child_samples: 1", "subsample: 0.8", "colsample_bytree: 0.8", "n_estimators: 200"]),
]
for i, (title, sub, col, params) in enumerate(models_info):
    x = 0.25 + i*4.85
    rect(s, x, 0.88, 4.55, 3.15, fill_c=WHITE, line_c=GRAY_LT, line_w=Pt(0.5))
    rect(s, x, 0.88, 4.55, 0.44, fill_c=col)
    txt(s, title, x+0.1, 0.88, 4.3, 0.28, size=13, bold=True, color=WHITE)
    txt(s, sub,   x+0.1, 1.1,  4.3, 0.22, size=10, color=GRAY_LT)
    for j, p in enumerate(params):
        txt(s, "• "+p, x+0.12, 1.42+j*0.38, 4.3, 0.34, size=10.5, color=GRAY, font="Consolas")

rect(s, 0.25, 4.18, 4.55, 0.55, fill_c=rgb("1E276118"), line_c=NAVY, line_w=Pt(0.5))
txt(s, "LeaveOneOut CV — 소규모 데이터(55개) 최적화", 0.35, 4.18, 4.35, 0.55, size=11, color=NAVY)
rect(s, 5.1,  4.18, 4.55, 0.55, fill_c=rgb("7C3AED18"), line_c=PURPLE, line_w=Pt(0.5))
txt(s, "VALVE 그룹 샘플만 필터 후 학습 (8개 현재)", 5.2, 4.18, 4.35, 0.55, size=11, color=PURPLE)

code_box(s, 0.25, 4.88, 9.5, 0.65)
code_line(s, [
    [("for ", CODE_KW), ("train_idx, test_idx ", CODE_TX), ("in ", CODE_KW), ("LeaveOneOut().split(X, y):", CODE_TX)],
    [("    model = LGBMClassifier(**LGBM_PARAMS).fit(scaler.fit_transform(X_tr), y_tr)  ", CODE_TX), ("# eval_set 미사용 (단일 샘플 fold)", CODE_CM)],
], x=0.38, y=4.93, w=9.3, line_h=0.21)

# ──────────────────────────────────────────────────────────────
# SLIDE 7 – 분석결과 그룹
# ──────────────────────────────────────────────────────────────
s = blank(prs)
rect(s, 0, 0, 10, 5.625, fill_c=OFF_W)
add_header(s, "5단계 · 분석 결과 — 컴포넌트 그룹 분류 성능", color=GREEN)

# 큰 숫자 카드
rect(s, 0.25, 0.88, 2.3, 1.75, fill_c=WHITE, line_c=GRAY_LT, line_w=Pt(0.5))
txt(s, "81.8%", 0.25, 0.95, 2.3, 0.85, size=40, bold=True, color=GREEN, align=PP_ALIGN.CENTER)
txt(s, "LOO-CV Accuracy", 0.25, 1.75, 2.3, 0.4, size=11, color=GRAY, align=PP_ALIGN.CENTER)
txt(s, "55샘플 / 9클래스", 0.25, 2.12, 2.3, 0.3, size=10, color=GRAY, align=PP_ALIGN.CENTER)

# 클래스별 결과 막대 (수동)
class_results = [
    ("GASKET",   1.00, GREEN),
    ("FITTING",  0.95, GREEN),
    ("SUPPORT",  0.91, GREEN),
    ("VALVE",    0.89, TEAL),
    ("SLEEVE",   0.67, AMBER),
    ("FLANGE",   0.57, AMBER),
    ("INSTRMNT", 0.44, RED),
    ("FLEX_JT",  0.00, RED),
    ("ACCSSRY",  0.00, RED),
]
bar_max_w = 5.5
txt(s, "클래스별 F1 Score", 2.85, 0.88, 6.8, 0.32, size=12, bold=True, color=NAVY)
for i, (cls, f1, col) in enumerate(class_results):
    y = 1.28 + i*0.46
    rect(s, 2.85, y, 6.9, 0.38, fill_c=WHITE, line_c=GRAY_LT, line_w=Pt(0.3))
    bw = max(0.05, f1 * bar_max_w)
    rect(s, 2.85, y+0.04, bw, 0.3, fill_c=col)
    txt(s, cls, 2.88, y+0.04, 1.1, 0.3, size=10, bold=True, color=WHITE if f1 > 0.2 else col, font="Consolas")
    txt(s, f"F1={f1:.2f}", 2.85+bw+0.05, y+0.04, 1.0, 0.3, size=10, color=GRAY)

# 해석
notes = [
    ("GASKET 100%, FITTING 95% — 형상/커넥터 특성 명확", GREEN),
    ("FLEXIBLE_JOINT·ACCESSORY 0% — 각 1개 샘플 구조적 한계", RED),
    ("INSTRUMENT 44% — 계측기 간 형상 유사성 높음", AMBER),
    ("Weighted Avg F1=0.80, 가중평균 기준 양호한 수준", TEAL),
]
for i, (note, col) in enumerate(notes):
    x = 0.25 + (i%2)*4.85
    y = 4.38 + (i//2)*0.5
    rect(s, x, y, 0.06, 0.38, fill_c=col)
    txt(s, note, x+0.12, y, 4.55, 0.38, size=10.5, color=GRAY)

# ──────────────────────────────────────────────────────────────
# SLIDE 8 – Ablation Study
# ──────────────────────────────────────────────────────────────
s = blank(prs)
rect(s, 0, 0, 10, 5.625, fill_c=OFF_W)
add_header(s, "6단계 · 분석 결과 — Ablation Study (피처 기여도)", color=NAVY)

ablation_data = [
    ("Geo+Conn+Meta (ALL)", 81.8, TEAL),
    ("Geo+Connector",       81.8, TEAL),
    ("Geometry+Meta",       78.2, NAVY),
    ("Geometry only",       78.2, NAVY),
    ("Connector only",      45.5, AMBER),
    ("Connector+Meta",      43.6, AMBER),
    ("Metadata only",       20.0, RED),
]
bar_unit = 5.5 / 100
for i, (lbl, acc, col) in enumerate(ablation_data):
    y = 0.93 + i*0.6
    rect(s, 0.25, y, 6.0, 0.5, fill_c=WHITE, line_c=GRAY_LT, line_w=Pt(0.3))
    bw = max(0.05, acc * bar_unit * 5.7)
    rect(s, 0.25, y+0.05, bw, 0.4, fill_c=col)
    txt(s, lbl, 0.3, y+0.07, 2.5, 0.36, size=10, bold=True,
        color=WHITE if acc > 30 else col, font="Calibri")
    txt(s, f"{acc:.1f}%", 0.25+bw+0.05, y+0.07, 0.8, 0.36, size=11, bold=True, color=col)

# 우: 인사이트 카드
insights = [
    ("핵심 피처: 형상", NAVY,
     "형상 피처만으로 78.2% 달성\nGeometry는 필수 피처 그룹"),
    ("커넥터의 보완 역할", TEAL,
     "형상+커넥터 = 81.8%\n커넥터 단독 45.5% — 보조 역할"),
    ("메타데이터 기여 미미", AMBER,
     "구경(mm) 정보만으로는 20%\n그룹 간 구경 중첩 → 단독 판별 불가"),
]
for i, (title, col, body) in enumerate(insights):
    y = 0.93 + i*1.45
    rect(s, 6.55, y, 3.2, 1.3, fill_c=WHITE, line_c=GRAY_LT, line_w=Pt(0.5))
    rect(s, 6.55, y, 0.07, 1.3, fill_c=col)
    txt(s, title, 6.72, y+0.06, 2.9, 0.35, size=12, bold=True, color=col)
    txt(s, body,  6.72, y+0.45, 2.9, 0.78, size=11, color=GRAY)

rect(s, 0.25, 5.08, 9.5, 0.42, fill_c=NAVY)
txt(s, "▶  결론: 형상 피처가 핵심 — 커넥터 보완 / 메타데이터 구경 보조 역할",
    0.4, 5.08, 9.2, 0.42, size=13, bold=True, color=WHITE)

# ──────────────────────────────────────────────────────────────
# SLIDE 9 – 밸브 타입 & FastAPI
# ──────────────────────────────────────────────────────────────
s = blank(prs)
rect(s, 0, 0, 10, 5.625, fill_c=OFF_W)
add_header(s, "7단계 · 밸브 타입 분류 & FastAPI 추론 서버", color=PURPLE)

valve_types = [
    ("AUTO",    0.80, 2, GREEN),
    ("BALL",    0.33, 3, AMBER),
    ("GLOBE",   0.00, 1, RED),
    ("CONTROL", 0.00, 1, RED),
    ("SAFETY",  0.00, 1, RED),
]
txt(s, "밸브 타입 분류 현황", 0.25, 0.88, 4.5, 0.3, size=12, bold=True, color=NAVY)
for i, (vt, f1, n, col) in enumerate(valve_types):
    y = 1.25 + i*0.57
    rect(s, 0.25, y, 4.5, 0.48, fill_c=WHITE, line_c=GRAY_LT, line_w=Pt(0.5))
    bw = max(0.05, f1*2.4)
    rect(s, 1.6, y+0.1, bw, 0.28, fill_c=col)
    txt(s, vt, 0.4, y+0.06, 1.1, 0.36, size=11, bold=True, color=col, font="Consolas")
    txt(s, f"F1={f1:.2f}  (n={n})", 1.65+bw, y+0.06, 1.5, 0.36, size=10, color=GRAY)

rect(s, 0.25, 4.18, 4.5, 0.55, fill_c=rgb("DC262618"), line_c=RED, line_w=Pt(0.5))
txt(s, "LOO-CV 37.5%  —  8개 샘플/5타입 → 클래스당 최소 15개 필요",
    0.35, 4.18, 4.3, 0.55, size=10.5, color=RED)

# FastAPI 코드
code_box(s, 5.0, 0.88, 4.75, 3.55)
txt(s, "api/main.py  —  FastAPI POST /predict",
    5.12, 0.91, 4.5, 0.25, size=9, color=rgb("94A3B8"), font="Consolas")
rect(s, 5.0, 1.18, 4.75, 0.03, fill_c=rgb("334155"))

code_line(s, [
    [("@app.post", CODE_FN), ("('/predict')", CODE_ST)],
    [("async def ", CODE_KW), ("predict", CODE_FN), ("(body: FeatureInput):", CODE_TX)],
    [("    features = body.model_dump()", CODE_TX)],
    [("    result  = predictor.predict_features(features)", CODE_TX)],
    [("    return ", CODE_KW), ("PredictionResult(**result)", CODE_TX)],
    [("", CODE_TX)],
    [("class ", CODE_KW), ("PredictionResult", CODE_FN), ("(BaseModel):", CODE_TX)],
    [("  component_group: ", CODE_TX), ("str   ", CODE_KW), ("# VALVE, FITTING...", CODE_CM)],
    [("  group_confidence: ", CODE_TX), ("float ", CODE_KW), ("# 0.0 ~ 1.0", CODE_CM)],
    [("  valve_type:       ", CODE_TX), ("str   ", CODE_KW), ("# BALL, GATE...", CODE_CM)],
    [("  review_required:  ", CODE_TX), ("bool  ", CODE_KW), ("# conf < 0.9", CODE_CM)],
    [("  inference_ms:     ", CODE_TX), ("float", CODE_KW)],
], x=5.1, y=1.23, w=4.6, line_h=0.195)

# 신뢰도 임계값
thresholds = [
    ("≥ 0.90", "자동 처리", GREEN),
    ("0.70 ~ 0.90", "검토 권장", AMBER),
    ("< 0.70", "미지 (unknown)", RED),
]
for i, (rng, label, col) in enumerate(thresholds):
    y = 4.58 + i*0.29
    rect(s, 5.0, y, 4.75, 0.26, fill_c=rgb("1E293B"), line_c=col, line_w=Pt(0.5))
    txt(s, rng, 5.08, y, 1.4, 0.26, size=10, bold=True, color=col, font="Consolas")
    txt(s, "→  " + label, 6.5, y, 3.1, 0.26, size=10, color=col)

# ──────────────────────────────────────────────────────────────
# SLIDE 10 – 결론
# ──────────────────────────────────────────────────────────────
s = blank(prs)
rect(s, 0, 0, 10, 5.625, fill_c=NAVY)
rect(s, 0, 0, 0.18, 5.625, fill_c=TEAL)

txt(s, "결론 및 향후 과제", 0.45, 0.45, 7, 0.55, size=26, bold=True, color=WHITE)
rect(s, 0.45, 1.1, 3.0, 0.05, fill_c=TEAL)

# 숫자 카드
for i, (label, val, desc, col) in enumerate([
    ("그룹 분류", "81.8%", "55개 샘플 LOO-CV", MINT),
    ("타입 분류", "37.5%", "8개 샘플 → 데이터 부족", AMBER),
]):
    x = 0.45 + i*3.2
    rect(s, x, 1.3, 2.8, 1.1, fill_c=rgb("FFFFFF12"), line_c=rgb("FFFFFF30"), line_w=Pt(0.5))
    txt(s, val,   x, 1.35, 2.8, 0.6, size=32, bold=True, color=col, align=PP_ALIGN.CENTER)
    txt(s, label+" — "+desc, x, 1.92, 2.8, 0.38, size=10.5, color=ICE, align=PP_ALIGN.CENTER)

txt(s, "향후 과제", 0.45, 2.65, 5, 0.38, size=15, bold=True, color=TEAL_LT)

todos = [
    "밸브 타입 분류: 클래스당 15개 이상 확보 → 78%+ 목표",
    "xlsx 정답 레이블 ProductCode 컬럼명 확인 후 매칭 적용",
    "C# Revit Add-in 연동 → ExternalCommand → POST /predict",
    "ISO 심볼 매핑: VALVE/BALL → ISO_VLV_BALL 자동 연결",
]
for i, td in enumerate(todos):
    y = 3.1 + i*0.5
    rect(s, 0.45, y+0.04, 0.22, 0.22, fill_c=TEAL)
    txt(s, td, 0.78, y, 6.8, 0.38, size=12, color=ICE)

txt(s, "Namespace: RailSweng  ·  Python + LightGBM + FastAPI  ·  C# Revit Add-in 연동 예정",
    0.45, 5.2, 9, 0.27, size=10, color=rgb("7BA7BC"))


# ── 저장 ────────────────────────────────────────────────────
out_path = Path(__file__).parent / "AI모델_설명자료.pptx"
prs.save(out_path)
print(f"✅  PPT 저장 완료: {out_path}")
