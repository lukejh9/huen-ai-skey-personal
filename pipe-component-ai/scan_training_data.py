"""
scan_training_data.py
----------------------
DRY RUN ONLY — makes no changes to your files.

Recursively scans TrainingModel/ for all .json files (no matter how deeply
nested), and compares folder/file names against your CSV label names in
User_Defined_csv/ to see which ones would match.

Run this first from your pipe-component-ai folder:
    python scan_training_data.py

It will print a report. Nothing gets moved or copied.
"""

from pathlib import Path
import difflib

# ── EDIT THESE PATHS IF NEEDED ──────────────────────────────
ROOT = Path(__file__).parent
BIM_BASE = ROOT.parent / "BIM_Data"
TRAINING_MODEL_DIR = BIM_BASE / "TrainingModel"
CSV_LABEL_DIR = BIM_BASE / "User_Defined_csv"
# ─────────────────────────────────────────────────────────────


def main():
    print("=" * 70)
    print("  TrainingModel JSON Scanner (DRY RUN — no files will be changed)")
    print("=" * 70)
    print(f"  Scanning: {TRAINING_MODEL_DIR}")
    print(f"  Labels:   {CSV_LABEL_DIR}")
    print("=" * 70)

    if not TRAINING_MODEL_DIR.exists():
        print(f"[ERROR] TrainingModel folder not found: {TRAINING_MODEL_DIR}")
        return
    if not CSV_LABEL_DIR.exists():
        print(f"[ERROR] CSV label folder not found: {CSV_LABEL_DIR}")
        return

    # 1. Find ALL json files anywhere under TrainingModel, at any depth
    all_json_files = list(TRAINING_MODEL_DIR.rglob("*.json"))
    print(f"\n[INFO] Total JSON files found (any depth): {len(all_json_files)}")

    # 2. Group json files by their immediate parent folder name
    #    (this is likely the "model name" the loader expects)
    folder_to_jsons = {}
    for jf in all_json_files:
        parent_name = jf.parent.name
        folder_to_jsons.setdefault(parent_name, []).append(jf)

    print(f"[INFO] Distinct parent-folder names containing JSON: {len(folder_to_jsons)}")

    # 3. Get CSV label names (without .csv extension)
    csv_files = list(CSV_LABEL_DIR.glob("*.csv"))
    csv_names = [c.stem for c in csv_files]
    print(f"[INFO] Total CSV label files: {len(csv_names)}")

    # 4. Try exact matches first
    exact_matches = []
    unmatched_folders = []
    for folder_name in folder_to_jsons:
        if folder_name in csv_names:
            exact_matches.append(folder_name)
        else:
            unmatched_folders.append(folder_name)

    print(f"\n[RESULT] Exact name matches (folder == csv name): {len(exact_matches)}")
    print(f"[RESULT] Folders with JSON but NO exact CSV match:  {len(unmatched_folders)}")

    # 5. For unmatched folders, try fuzzy matching to suggest likely pairs
    print("\n" + "=" * 70)
    print("  FUZZY MATCH SUGGESTIONS for unmatched folders")
    print("  (these are guesses — verify before using)")
    print("=" * 70)

    fuzzy_suggestions = []
    for folder_name in unmatched_folders:
        close = difflib.get_close_matches(folder_name, csv_names, n=1, cutoff=0.6)
        if close:
            fuzzy_suggestions.append((folder_name, close[0]))
            print(f"  '{folder_name}'  ~=  '{close[0]}.csv'  ({len(folder_to_jsons[folder_name])} json files)")

    still_unmatched = [f for f in unmatched_folders if f not in [x[0] for x in fuzzy_suggestions]]

    print("\n" + "=" * 70)
    print("  FOLDERS WITH NO MATCH AT ALL (need manual review)")
    print("=" * 70)
    for folder_name in still_unmatched:
        print(f"  '{folder_name}'  ({len(folder_to_jsons[folder_name])} json files) "
              f"-- path: {folder_to_jsons[folder_name][0].parent}")

    # 6. CSVs that have NO matching folder anywhere (truly missing data)
    matched_csv_names = set(exact_matches) | set(x[1] for x in fuzzy_suggestions)
    csvs_with_no_data = [c for c in csv_names if c not in matched_csv_names]

    print("\n" + "=" * 70)
    print(f"  CSV LABELS WITH NO JSON DATA FOUND ANYWHERE: {len(csvs_with_no_data)}")
    print("=" * 70)
    for c in csvs_with_no_data[:30]:
        print(f"  {c}.csv")
    if len(csvs_with_no_data) > 30:
        print(f"  ... and {len(csvs_with_no_data) - 30} more")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Total JSON files found         : {len(all_json_files)}")
    print(f"  Distinct model folders (JSON)  : {len(folder_to_jsons)}")
    print(f"  Total CSV labels               : {len(csv_names)}")
    print(f"  Exact matches                  : {len(exact_matches)}")
    print(f"  Fuzzy match suggestions        : {len(fuzzy_suggestions)}")
    print(f"  Unmatched folders (no guess)   : {len(still_unmatched)}")
    print(f"  CSVs with zero JSON data       : {len(csvs_with_no_data)}")
    print("=" * 70)
    print("\n  No files were changed. This was a scan only.")
    print("  Review the results above before running any copy/move script.")


if __name__ == "__main__":
    main()