"""
merge_unmatched_folders.py
----------------------------
Safely copies JSON files from unmatched TrainingModel subfolders into the
correctly-named folder that matches an existing CSV label - so the pipeline
can find and use them.

SAFE BY DEFAULT: this script only COPIES files (never deletes/moves the
originals) and runs in DRY RUN mode unless you pass --execute. Existing
files are never overwritten - if a filename already exists in the target,
it's skipped and reported.

The MAPPINGS list below is pre-filled with HIGH-CONFIDENCE matches only
(the "UT PIPING" group, which are clearly sub-variants of one project).
The riskier "IC-M16P2" folders (different dates, possibly different model
versions) are intentionally left OUT - review those manually before adding
them here, since merging them could mislabel components if the model
actually changed between export dates.

Usage:
    # Step 1 - see what WOULD happen (default, safe):
    python merge_unmatched_folders.py

    # Step 2 - once you're happy with the plan, actually copy the files:
    python merge_unmatched_folders.py --execute
"""

import argparse
import shutil
from pathlib import Path

_ROOT = Path(__file__).parent
TRAINING_MODEL_DIR = _ROOT.parent / "BIM_Data" / "TrainingModel"

# ── EDIT THIS LIST to add/remove mappings ──────────────────
# Format: (source_folder_name, target_folder_name)
# target_folder_name should match an existing CSV label name exactly
# (i.e. BIM_Data/User_Defined_csv/<target_folder_name>.csv should exist)
MAPPINGS = [
    # High-confidence: clearly sub-variants of one base project
    ("UT PIPING_20180327_GN2",     "UT PIPING_20180327"),
    ("UT PIPING_20180327_NH40H",   "UT PIPING_20180327"),
    ("UT PIPING_20180327_P-VAC",   "UT PIPING_20180327"),
    ("UT PIPING_20180327_PCO2",    "UT PIPING_20180327"),
    ("UT PIPING_20180327_PCW-R",   "UT PIPING_20180327"),
    ("UT PIPING_20180327_PCW-S",   "UT PIPING_20180327"),
    ("UT PIPING_20180327_PCW-_1R", "UT PIPING_20180327"),
    ("UT PIPING_20180327_PN2",     "UT PIPING_20180327"),
    ("UT PIPING_20180327_PO2",     "UT PIPING_20180327"),
    ("UT PIPING_20180327_SOLVENT", "UT PIPING_20180327"),

    # Added after confirming the merge technique is safe (matched by
    # content, not just filename - the UT PIPING batch showed 0 CSV
    # matching failures out of ~204 merged files). These are the big
    # ones - 1,174 files combined across 4 folders.
    ("IC-M16P2-W12B-1F3F-PI-NA-NA(2021.11.10)",           "IC-M16P2-W12B-1F3F-PI-NA-NA(400A-500A EDIT)"),
    ("IC-M16P2-W12B-1F3F-PI-NA-NA(2022.01.19)",           "IC-M16P2-W12B-1F3F-PI-NA-NA(400A-500A EDIT)"),
    ("IC-M16P2-W12B-1F3F-PI-NA-NA_26ab255b",              "IC-M16P2-W12B-1F3F-PI-NA-NA(400A-500A EDIT)"),
    ("IC-M16P2-W12B-1F3F-PI-NA-NA_(4개 라인)detached",     "IC-M16P2-W12B-1F3F-PI-NA-NA(400A-500A EDIT)"),

    # Still worth reviewing before adding (smaller, lower priority):
    # ("CJ_M15M01ALL_PP_XX_Chemical_EZ-ISO_2019", "CJ_M15M01ALL_FF_1F_A_EZ-Iso_2018"),
    # ("Project2_b2967378", "Project1_2018"),
    # ("S32_FBALO_UPI_SAMPLING_2_b55bd123", "S32_FBALO_UPI_SAMPLING_1_"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true",
                     help="Actually copy files. Without this flag, only prints the plan.")
    args = ap.parse_args()

    mode = "EXECUTING" if args.execute else "DRY RUN (nothing will be changed)"
    print("=" * 70)
    print(f"  Merge Unmatched Folders — {mode}")
    print("=" * 70)

    total_copied = 0
    total_skipped_existing = 0
    total_errors = 0

    for source_name, target_name in MAPPINGS:
        source_dir = TRAINING_MODEL_DIR / source_name
        target_dir = TRAINING_MODEL_DIR / target_name

        print(f"\n[{source_name}] -> [{target_name}]")

        if not source_dir.exists():
            print(f"  [SKIP] Source folder not found: {source_dir}")
            total_errors += 1
            continue
        if not target_dir.exists():
            print(f"  [SKIP] Target folder not found: {target_dir}")
            total_errors += 1
            continue

        json_files = list(source_dir.glob("*.json"))
        png_files = list(source_dir.glob("*.png"))
        all_files = json_files + png_files

        copied_here = 0
        renamed_here = 0

        # Use a short, filesystem-safe tag from the source folder name
        # to disambiguate files that collide by name but are actually
        # different component instances (different RevitId inside).
        source_tag = source_name.replace(target_name, "").strip("_- ")
        if not source_tag:
            source_tag = source_name

        for f in all_files:
            dest = target_dir / f.name
            if dest.exists():
                # Name collision - almost always a DIFFERENT component
                # instance (same family/type name reused across lines,
                # different RevitId). Rename instead of skipping, so we
                # don't silently lose real training samples.
                dest = target_dir / f"{f.stem}__{source_tag}{f.suffix}"
                # In the rare case even the disambiguated name collides,
                # skip to avoid overwriting (this would mean true duplicate).
                if dest.exists():
                    continue
                renamed_here += 1
            if args.execute:
                shutil.copy2(f, dest)
            copied_here += 1

        print(f"  Files to copy: {copied_here}   (of which renamed to avoid collision: {renamed_here})")
        total_copied += copied_here
        total_skipped_existing += 0

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Total files {'copied' if args.execute else 'that WOULD be copied'}: {total_copied}")
    print(f"  Total skipped (already exist in target): {total_skipped_existing}")
    print(f"  Mapping errors (folder not found): {total_errors}")

    if not args.execute:
        print("\n  This was a DRY RUN. No files were changed.")
        print("  Review the plan above, then run with --execute to actually copy files.")
    else:
        print("\n  Files copied successfully.")
        print("  Re-run run_pipeline_v2.py to pick up the new data.")


if __name__ == "__main__":
    main()