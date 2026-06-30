# -*- coding: utf-8 -*-
"""
fix_classlist_v16.py
--------------------
v16 파일의 클래스 목록 수정 (정확한 line break 패턴 사용)
"""

TARGET = r'C:\Users\jgyun\Documents\배관ISO 논문 작업\논문_SKEY_DT_20260428_v16_CDE.htm'

with open(TARGET, 'r', encoding='utf-16-le') as f:
    content = f.read()

print(f"Read {len(content):,} chars\n")

changes = 0

def rep(old, new, tag):
    global content, changes
    n = content.count(old)
    if n > 0:
        content = content.replace(old, new)
        changes += n
        print(f"  OK [{tag}]  {n}x")
        return True
    # CRLF variant
    old2 = old.replace('\n', '\r\n')
    n2 = content.count(old2)
    if n2 > 0:
        content = content.replace(old2, new.replace('\n', '\r\n'))
        changes += n2
        print(f"  OK [{tag}] (CRLF)  {n2}x")
        return True
    print(f"  MISS [{tag}]")
    # Show context for debugging
    # Find partial match
    partial = old[:30]
    idx = content.find(partial)
    if idx >= 0:
        print(f"    Partial '{partial[:20]}...' found at {idx}:")
        print(f"    context: {repr(content[idx:idx+200])}")
    return False

print("=== Fix class list (correct line break pattern) ===")
rep(
    'VALVE, FITTING, FLANGE, GASKET, SLEEVE, SUPPORT\n'
    ', INSTRUMENT, FLEXIBLE_JOINT,\n'
    'and ACCESSORY.',
    'VALVE, FITTING, FLANGE, GASKET, SUPPORT, INSTRUMENT, FLEXIBLE_JOINT,\n'
    'ACCESSORY, OLET, CAP, COUPLING, and UNKNOWN.',
    "class list (\\n after SUPPORT)"
)

print("\n=== Fix redundant (5-Fold CV) in Cross-Validation ===")
rep(
    'Cross-Validation (5-Fold CV)',
    'Cross-Validation',
    "remove redundant (5-Fold CV)"
)

print("\n=== Final summary ===")
with open(TARGET, 'w', encoding='utf-16-le') as f:
    f.write(content)

print(f"Done! {changes} changes applied")
print(f"  'twelve' count:    {content.count('twelve')}")
print(f"  'UNKNOWN' count:   {content.count('UNKNOWN')}")
print(f"  'OLET' count:      {content.count('OLET')}")
print(f"  'SLEEVE' count:    {content.count('SLEEVE')}")
print(f"  'Stratified' count:{content.count('Stratified')}")
print(f"  '94.84' count:     {content.count('94.84')}")
print(f"  '3,815' count:     {content.count('3,815')}")
print(f"  'LOO-CV' remain:   {content.count('LOO-CV')}")
