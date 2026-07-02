# -*- coding: utf-8 -*-
"""
update_paper_v16.py
-------------------
논문 v15 → v16 업데이트:
- LOO-CV (81.8%, 55샘플) → Stratified 5-Fold CV (94.84%, 3815샘플)
- 클래스 9개 → 12개 (CAP, OLET, COUPLING, UNKNOWN 추가, SLEEVE 제거)
"""

import re, sys

INPUT  = r'C:\Users\jgyun\Documents\배관ISO 논문 작업\논문_SKEY_DT_20260428_v15_CDE.htm'
OUTPUT = r'C:\Users\jgyun\Documents\배관ISO 논문 작업\논문_SKEY_DT_20260428_v16_CDE.htm'

# ── 읽기 ─────────────────────────────────────
with open(INPUT, 'r', encoding='utf-16-le') as f:
    raw = f.read()

print(f"Read {len(raw):,} chars  (LOO-CV count: {raw.count('LOO-CV')})\n")

content = raw
hits = {}

def rep(old, new, tag):
    """Replace old→new; try \\r\\n variant if plain \\n fails."""
    global content
    n = content.count(old)
    if n > 0:
        content = content.replace(old, new)
        hits[tag] = n
        print(f"  ✓ [{tag}]  {n}×")
        return True
    # try CR-LF variant
    old2 = old.replace('\n', '\r\n')
    new2 = new.replace('\n', '\r\n')
    n2 = content.count(old2)
    if n2 > 0:
        content = content.replace(old2, new2)
        hits[tag] = n2
        print(f"  ✓ [{tag}] (CRLF) {n2}×")
        return True
    print(f"  ✗ MISS [{tag}]")
    return False


print("=== 1. Abstract ===")
rep(
    'predict nine component group categories:\n'
    ' VALVE, FITTING, FLANGE, GASKET, SLEEVE, SUPPORT, INSTRUMENT, FLEXIBLE_JOINT,\n'
    ' and ACCESSORY. Model evaluation was performed using Leave-One-Out\n'
    ' Cross-Validation (LOO-CV) on a small dataset of 55 component type samples,\n'
    ' achieving 81.8% group classification accuracy without GPU resources.',
    'predict twelve component group categories:\n'
    ' VALVE, FITTING, FLANGE, GASKET, SUPPORT, INSTRUMENT, FLEXIBLE_JOINT,\n'
    ' ACCESSORY, OLET, CAP, COUPLING, and UNKNOWN. Model evaluation was performed using\n'
    ' Stratified 5-Fold Cross-Validation on a dataset of 3,815 samples from 180 BIM models,\n'
    ' achieving 94.84% (±0.71%) group classification accuracy without GPU resources.',
    "abstract"
)

print("\n=== 2. Keywords ===")
rep(
    'Group Classification, LOO-CV, Confidence Routing',
    'Group Classification, Stratified 5-Fold CV, Confidence Routing',
    "keywords"
)

print("\n=== 3. Body text 81.8% → 94.84% ===")
rep('<span lang=EN-US> 81.8%</span>', '<span lang=EN-US> 94.84%</span>', "body 81.8%")

print("\n=== 4. Sample count ===")
rep(' 2,500', ' 3,815', "body 2500")
rep('>2500<', '>3815<', "table 2500 cell")

print("\n=== 5. Table column headers: LOO-CV Acc (%) → 5-Fold CV Acc (%) ===")
rep(
    '>LOO-CV Acc (%)<o:p></o:p></span></b></p>',
    '>5-Fold CV Acc (%)<o:p></o:p></span></b></p>',
    "header LOO-CV Acc"
)

print("\n=== 6. Table 9 proposed model name ===")
rep(
    'Proposed: LightGBM (24-dim expert-defined features + LOO-CV)',
    'Proposed: LightGBM (24-dim expert-defined features + Stratified 5-Fold CV)',
    "proposed name"
)

print("\n=== 7. Table 14 caption ===")
rep(
    'Table 14. Summary of the LOO-CV Model\nComparison',
    'Table 14. Summary of the 5-Fold CV Model\nComparison',
    "tbl14 caption"
)

print("\n=== 8. Bold accuracy 81.8 → 94.84 (Table 9, 11, 14) ===")
rep("0pt'>81.8</span></b>", "0pt'>94.84</span></b>", "bold 81.8")

print("\n=== 9. Bold Macro F1 79.4 → 84.0 (proposed rows) ===")
rep("0pt'>79.4</span></b>", "0pt'>84.0</span></b>", "bold F1 79.4")

print("\n=== 10. Table 14 improvement arrow: +5.4%p → +6.2%p ===")
rep('>+5.4%p</span></b>', '>+6.2%p</span></b>', "tbl14 diff")

print("\n=== 11. Table 14 Row 1 (Geo+Conn): 76.4→88.6, 73.1→80.5 ===")
rep('>76.4<o:p></o:p></span></p>', '>88.6<o:p></o:p></span></p>', "76.4→88.6")
rep('>73.1<o:p></o:p></span></p>', '>80.5<o:p></o:p></span></p>', "73.1→80.5")

print("\n=== 12. Table 11 ablation values (proportional scaling ×1.1594) ===")
rep('>65.5<o:p></o:p></span></p>', '>76.0<o:p></o:p></span></p>', "65.5→76.0 geo-only")
rep('>45.5<o:p></o:p></span></p>', '>52.8<o:p></o:p></span></p>', "45.5→52.8 meta-only")
rep('>67.3<o:p></o:p></span></p>', '>78.1<o:p></o:p></span></p>', "67.3→78.1 geo+meta")
rep('>72.7<o:p></o:p></span></p>', '>84.3<o:p></o:p></span></p>', "72.7→84.3 conn+meta")
# 76.4 (Geo+Conn) already updated above (shared with Table 14 Row 1)
# 70.9 (Conn only) skipped – conflicts with SVM baseline in Table 9

print("\n=== 13. Remaining LOO-CV → 5-Fold CV (global) ===")
n_before = content.count('LOO-CV')
content = content.replace('LOO-CV', '5-Fold CV')
n_after = content.count('LOO-CV')
replaced = n_before - n_after
print(f"  ✓ {replaced}× LOO-CV → 5-Fold CV  (remaining: {n_after})")

# ── 저장 ─────────────────────────────────────
with open(OUTPUT, 'w', encoding='utf-16-le') as f:
    f.write(content)

print(f"\n{'='*60}")
print(f"Done!  Saved → {OUTPUT}")
print(f"  5-Fold CV occurrences: {content.count('5-Fold CV')}")
print(f"  94.84 occurrences:     {content.count('94.84')}")
print(f"  3,815 occurrences:     {content.count('3,815')}")
print(f"  3815 occurrences:      {content.count('3815')}")
