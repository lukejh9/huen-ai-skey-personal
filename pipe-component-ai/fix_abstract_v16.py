# -*- coding: utf-8 -*-
"""
fix_abstract_v16.py
-------------------
v16 파일의 abstract 영역 수정 (step 1 MISS 보완)
- "nine" → "twelve"
- SLEEVE 제거, OLET/CAP/COUPLING/UNKNOWN 추가
- "Leave-One-Out\n Cross-Validation (5-Fold CV)" → "Stratified 5-Fold\n Cross-Validation"
- "55 component type samples" → "3,815 samples from 180 BIM models"
"""

TARGET = r'C:\Users\jgyun\Documents\배관ISO 논문 작업\논문_SKEY_DT_20260428_v16_CDE.htm'

with open(TARGET, 'r', encoding='utf-16-le') as f:
    content = f.read()

print(f"Read {len(content):,} chars\n")

# ── 1. 현재 abstract 근방 내용 확인 ──────────────
idx = content.find('Leave-One-Out')
if idx >= 0:
    ctx = content[idx-200:idx+400]
    print("=== Found 'Leave-One-Out' context ===")
    print(repr(ctx[:300]))
    print()
else:
    print("'Leave-One-Out' NOT found in v16 – may already be replaced\n")

idx2 = content.find('predict nine')
if idx2 >= 0:
    print("=== Found 'predict nine' context ===")
    print(repr(content[idx2:idx2+300]))
    print()
else:
    print("'predict nine' NOT found\n")

# ── 2. 수정 ──────────────────────────────────────
changes = 0

def rep(old, new, tag):
    global content, changes
    n = content.count(old)
    if n > 0:
        content = content.replace(old, new)
        changes += n
        print(f"  ✓ [{tag}]  {n}×")
    else:
        # Try CRLF variant
        old2 = old.replace('\n', '\r\n')
        n2 = content.count(old2)
        if n2 > 0:
            content = content.replace(old2, new.replace('\n', '\r\n'))
            changes += n2
            print(f"  ✓ [{tag}] (CRLF)  {n2}×")
        else:
            print(f"  ✗ MISS [{tag}]")

# "predict nine" → "predict twelve"
rep('predict nine component group', 'predict twelve component group', "nine→twelve")

# Class list: remove SLEEVE, add new classes
rep(
    'VALVE, FITTING, FLANGE, GASKET, SLEEVE, SUPPORT, INSTRUMENT, FLEXIBLE_JOINT,\n'
    ' and ACCESSORY.',
    'VALVE, FITTING, FLANGE, GASKET, SUPPORT, INSTRUMENT, FLEXIBLE_JOINT,\n'
    ' ACCESSORY, OLET, CAP, COUPLING, and UNKNOWN.',
    "class list"
)

# "Leave-One-Out\n Cross-Validation (5-Fold CV)" → "Stratified 5-Fold\n Cross-Validation"
rep(
    'Leave-One-Out\n Cross-Validation (5-Fold CV)',
    'Stratified 5-Fold\n Cross-Validation',
    "Leave-One-Out → Stratified 5-Fold"
)

# "a small dataset of 55 component type samples" → "a dataset of 3,815 samples from 180 BIM models"
rep(
    'a small dataset of 55 component type samples',
    'a dataset of 3,815 samples from 180 BIM models',
    "55 samples → 3815"
)

# Also handle any remaining "Leave-One-Out" references
n_loo = content.count('Leave-One-Out')
if n_loo > 0:
    content = content.replace('Leave-One-Out', 'Stratified 5-Fold')
    print(f"  ✓ [Leave-One-Out global]  {n_loo}×")
    changes += n_loo

# ── 3. 저장 ──────────────────────────────────────
with open(TARGET, 'w', encoding='utf-16-le') as f:
    f.write(content)

print(f"\n{'='*50}")
print(f"Done! {changes} changes applied → {TARGET}")
print(f"  'twelve' count:  {content.count('twelve')}")
print(f"  '94.84' count:   {content.count('94.84')}")
print(f"  '3,815' count:   {content.count('3,815')}")
print(f"  'Stratified' count: {content.count('Stratified')}")
print(f"  'LOO-CV' remaining: {content.count('LOO-CV')}")
print(f"  'Leave-One-Out' remaining: {content.count('Leave-One-Out')}")
