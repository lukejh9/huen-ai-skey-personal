@echo off
chcp 65001 > nul
echo ============================================================
echo  Updating paper: v15 to v16 (5-Fold CV results)
echo ============================================================
echo  Working dir: %CD%
python update_paper_v16.py
echo.
echo ============================================================
echo  Done. Press any key to close.
echo ============================================================
pause
