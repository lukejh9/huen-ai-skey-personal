# -*- coding: utf-8 -*-
"""
reduce_tables_v16.py
--------------------
논문 v16: 16개 테이블 → 8개 테이블로 축소
제거: Table 3, 4, 5, 6, 8, 14, 15, 16  (본문 텍스트로 교체)
유지: Table 1, 2, 7, 9, 10, 11, 12, 13
재번호: 7→3, 9→4, 10→5, 11→6, 12→7, 13→8
"""

import re

TARGET = r'C:\Users\jgyun\Documents\배관ISO 논문 작업\논문_SKEY_DT_20260428_v16_CDE.htm'

with open(TARGET, 'r', encoding='utf-16-le') as f:
    content = f.read()

orig_len = len(content)
print(f"Read {orig_len:,} chars")
table_refs_before = len(re.findall(r'Table \d+', content))
print(f"'Table N' refs before: {table_refs_before}\n")


# ════════════════════════════════════════════════════════
#  Helper: find and remove/replace a table section
# ════════════════════════════════════════════════════════
def remove_table(content, table_num, replacement=''):
    """
    Find <p class=tblcaption>...Table N....</p> and the
    following <table>...</table> block, replace entire chunk.
    """
    # Find caption line containing 'Table N.'
    # The caption paragraph may span multiple lines in the HTML
    cap_idx = content.find(f'Table {table_num}.')
    if cap_idx == -1:
        print(f"  ✗ Table {table_num}: 'Table {table_num}.' not found")
        return content

    # Walk back to find start of the <p class=tblcaption> that contains this
    para_start = content.rfind('<p ', 0, cap_idx)
    if para_start == -1:
        para_start = cap_idx

    # Walk forward to find the <table that follows (within next 2000 chars)
    search_window = content[cap_idx: cap_idx + 3000]
    table_rel = search_window.find('<table ')
    if table_rel == -1:
        print(f"  ✗ Table {table_num}: <table> not found after caption")
        return content
    table_abs = cap_idx + table_rel

    # Find matching </table> tracking nested depth
    depth = 0
    pos = table_abs
    table_end = -1
    while pos < len(content):
        nxt_open  = content.find('<table', pos)
        nxt_close = content.find('</table>', pos)

        if nxt_open != -1 and (nxt_close == -1 or nxt_open < nxt_close):
            depth += 1
            pos = nxt_open + 6
        elif nxt_close != -1:
            depth -= 1
            pos = nxt_close + 8
            if depth == 0:
                table_end = pos
                break
        else:
            break

    if table_end == -1:
        print(f"  ✗ Table {table_num}: </table> not found")
        return content

    removed_len = table_end - para_start
    content = content[:para_start] + replacement + content[table_end:]
    print(f"  OK Table {table_num}: -{removed_len:,} chars, +{len(replacement)} chars replacement")
    return content


# ════════════════════════════════════════════════════════
#  Replacement text helper
# ════════════════════════════════════════════════════════
def para(en_text):
    """Simple English paragraph in Word Web Page style."""
    return (
        "\n<p class=MsoNormal align=left style='margin-top:3.0pt;"
        "text-align:left;\ntext-indent:0cm'>"
        "<span lang=EN-US style='font-size:9.0pt'>"
        + en_text +
        "<o:p></o:p></span></p>\n\n"
    )


# ════════════════════════════════════════════════════════
#  Replacement texts (English inline description)
# ════════════════════════════════════════════════════════
R = {
    3: para(
        "The SKEY code is structured as seven hierarchical fields: "
        "Type (major component category, e.g., V = Valve), "
        "Class (functional subclass, e.g., G = Gate), "
        "Material (e.g., A = Carbon Steel), "
        "Rating (pressure rating, e.g., 150 = ANSI #150), "
        "End Connection type (e.g., F = Flanged), "
        "Size (nominal pipe size, e.g., 0200 = 2 inches), "
        "and Special attribute (auxiliary type parameters). "
        "All fields are automatically extracted from Revit family parameters "
        "and connector geometry via the BIM Add-in."
    ),
    4: para(
        "The confidence score c from the LightGBM classifier determines the routing stage: "
        "(1) Auto (c &ge; 0.8, ~65% of cases): AI prediction is applied directly to Ez-ISO; "
        "(2) Review (0.5 &le; c &lt; 0.8, ~25%): the prediction is presented to an engineer "
        "for verification before confirmation; "
        "(3) Unknown (c &lt; 0.5, ~10%): the component is flagged for manual classification "
        "by domain experts."
    ),
    5: para(
        "The Construction Digital Twin employs a two-layer architecture. "
        "Layer 1 (BIM Layer) captures Revit-sourced attributes "
        "(SKEY, nominal size, pressure rating, prediction confidence) "
        "through the Add-in and Ez-ISO pipeline. "
        "Layer 2 consists of a Welding Record Layer "
        "(Ez-Spool scan data: spool_no, weld_point_id, install_date, weld_status) "
        "and an IWP Progress Layer "
        "(construction management system: iwp_id, plan_date, actual_date, progress_pct), "
        "enabling integrated monitoring of design intent and construction execution."
    ),
    6: para(
        "The LightGBM classifier was configured with: "
        "objective=multiclass (12 component groups), num_leaves=8, max_depth=4, "
        "learning_rate=0.05, n_estimators=200, "
        "reg_alpha=0.5, reg_lambda=1.0, "
        "colsample_bytree=0.8, subsample=0.8, "
        "class_weight=&#x2018;balanced&#x2019;, and min_child_samples=1. "
        "These settings balance regularization and boosting stability "
        "for the 3,815-sample, 12-class dataset."
    ),
    8: para(
        "All experiments were conducted on a standard Windows 11 Pro desktop "
        "(Intel Core i7, Python 3.11, LightGBM 4.x, scikit-learn) "
        "without GPU acceleration. "
        "The Revit Add-in was developed in C# .NET Framework 4.8 "
        "with the Autodesk Revit 2023 API "
        "(DevExpress XtraEditors 24.2, Newtonsoft.Json 13.0.4). "
        "This configuration confirms practical deployability "
        "in standard construction office environments."
    ),
    14: para(
        "Among the feature configurations evaluated, "
        "the 22-dimensional Geometry+Connector subset achieved 88.6% accuracy (Macro F1: 80.5%), "
        "while the full 24-dimensional feature set (Geometry + Connector + Metadata) "
        "reached 94.84% accuracy (Macro F1: 84.0%), a +6.2%p improvement. "
        "This confirms that the two metadata features "
        "(max_size_mm, min_size_mm) provide meaningful discriminative information "
        "beyond geometric and topological cues."
    ),
    15: para(
        "Statistical significance of the metadata contribution was confirmed "
        "by a paired t-test comparing 24-dim vs. 22-dim configurations "
        "(t = 3.21, df = 54, p = 0.0022, 95% CI: [+2.0%p, +8.8%p], "
        "Cohen&#x2019;s d = 0.43 [medium effect size]). "
        "The result confirms that the observed improvement is both statistically "
        "and practically significant at &alpha; = 0.05."
    ),
    16: para(
        "Misclassification analysis (see confusion matrix) reveals three primary error types: "
        "(1) Similar-geometry conflicts (50%, e.g., VALVE vs. INSTRUMENT "
        "for small gauge body components) "
        "&#x2014; addressable through finer connector direction and angle features; "
        "(2) Sparse-class overfitting (30%, e.g., FLEXIBLE_JOINT with few training samples) "
        "&#x2014; reducible by additional sample collection or SMOTE augmentation; "
        "(3) Feature-range outliers (20%, e.g., distorted bounding-box values "
        "in non-standard Revit families) "
        "&#x2014; mitigable through outlier-detection preprocessing."
    ),
}


# ════════════════════════════════════════════════════════
#  Step 1: Remove tables (highest → lowest to preserve offsets)
# ════════════════════════════════════════════════════════
print("=== Step 1: Remove/replace tables ===")
for tnum in sorted(R.keys(), reverse=True):
    content = remove_table(content, tnum, R[tnum])


# ════════════════════════════════════════════════════════
#  Step 2: Renumber remaining tables (captions + cross-refs)
#  Old: 1,2,7,9,10,11,12,13  →  New: 1,2,3,4,5,6,7,8
# ════════════════════════════════════════════════════════
print("\n=== Step 2: Renumber remaining tables ===")
REMAP = {7:3, 9:4, 10:5, 11:6, 12:7, 13:8}

# Process highest-old-number first to avoid cascade errors
for old, new in sorted(REMAP.items(), reverse=True):
    # Caption: 'Table OLD.' → 'Table NEW.'
    n1 = content.count(f'Table {old}.')
    content = content.replace(f'Table {old}.', f'Table {new}.')
    # Cross-refs: 'Table OLD' not followed by digit (avoid Table 10 → Table 40 issues)
    n2 = len(re.findall(rf'\bTable {old}\b(?!\d)', content))
    content = re.sub(rf'\bTable {old}\b(?!\d)', f'Table {new}', content)
    print(f"  Table {old} → Table {new}  (captions: {n1}, refs: {n2})")


# ════════════════════════════════════════════════════════
#  Step 3: Fix stale hyperparameter mention (num_class 9→12)
# ════════════════════════════════════════════════════════
print("\n=== Step 3: Misc text fixes ===")
# Hyperparameter objective description (from old Table 6 remnants in text)
n = content.count('nine component groups')
content = content.replace('nine component groups', 'twelve component groups')
print(f"  'nine component groups' → 'twelve component groups': {n}x")

n = content.count('Nine component groups')
content = content.replace('Nine component groups', 'Twelve component groups')
print(f"  'Nine component groups' → 'Twelve component groups': {n}x")


# ════════════════════════════════════════════════════════
#  Save
# ════════════════════════════════════════════════════════
with open(TARGET, 'w', encoding='utf-16-le') as f:
    f.write(content)

# ════════════════════════════════════════════════════════
#  Verification
# ════════════════════════════════════════════════════════
print(f"\n{'='*55}")
print(f"Saved → {TARGET}")
print(f"  File size: {orig_len:,} → {len(content):,} chars")

caps = re.findall(r'Table \d+\.', content)
unique_caps = sorted(set(caps))
print(f"\n  Table captions remaining ({len(caps)} total, {len(unique_caps)} unique):")
for c in unique_caps:
    cnt = caps.count(c)
    print(f"    {c}  (x{cnt})")

# Check for stale references to removed tables
stale = []
for n in [3,4,5,6,8,14,15,16]:
    found = re.findall(rf'\bTable {n}\b', content)
    if found:
        stale.append(f"Table {n} ({len(found)}x)")
if stale:
    print(f"\n  WARNING - stale refs to removed tables: {', '.join(stale)}")
    print("  (These may be in Korean body text — please review manually)")
else:
    print(f"\n  No stale refs to removed table numbers found.")
