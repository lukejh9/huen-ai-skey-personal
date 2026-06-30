# -*- coding: utf-8 -*-
"""
fix_table7_v16.py
-----------------
reduce_tables_v16.py의 cascade 오류 수정:
  Table 12 -> 7 -> 3 으로 두 번 바뀐 문제 해결
  두 번째 'Table 3.' 캡션을 'Table 7.'로 되돌림
  (첫 번째 Table 3. = 원래 Table 7, 두 번째 Table 3. = 원래 Table 12)
"""

import re

TARGET = r'C:\Users\jgyun\Documents\배관ISO 논문 작업\논문_SKEY_DT_20260428_v16_CDE.htm'

with open(TARGET, 'r', encoding='utf-16-le') as f:
    content = f.read()

print(f"Read {len(content):,} chars")

# Count Table 3. occurrences before fix
n_t3 = content.count('Table 3.')
print(f"  'Table 3.' count: {n_t3} (expected: 2)")

# Replace the SECOND occurrence of 'Table 3.' with 'Table 7.'
first_pos = content.find('Table 3.')
if first_pos == -1:
    print("ERROR: 'Table 3.' not found!")
else:
    second_pos = content.find('Table 3.', first_pos + 1)
    if second_pos == -1:
        print("Only one 'Table 3.' found – nothing to fix!")
    else:
        content = content[:second_pos] + 'Table 7.' + content[second_pos + len('Table 3.'):]
        print(f"  Fixed: second 'Table 3.' at pos {second_pos} -> 'Table 7.'")

# Verify
caps = re.findall(r'Table \d+\.', content)
unique_caps = sorted(set(caps))
print(f"\nTable captions remaining ({len(caps)} total, {len(unique_caps)} unique):")
for c in unique_caps:
    cnt = caps.count(c)
    print(f"  {c}  (x{cnt})")

# Check for Table 7 in content
t7_caps = len(re.findall(r'Table 7\.', content))
print(f"\n'Table 7.' in file: {t7_caps}")

# Save
with open(TARGET, 'w', encoding='utf-16-le') as f:
    f.write(content)

print(f"\nSaved -> {TARGET}")
print("Done.")
