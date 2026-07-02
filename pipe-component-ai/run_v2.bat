@echo off
chcp 65001 > nul
echo ============================================================
echo  AI Pipeline v2 - Training with Expert CSV Labels
echo ============================================================
echo  Working dir: %CD%
echo ============================================================
python run_pipeline_v2.py
echo.
echo ============================================================
echo  Done. Press any key to close.
echo ============================================================
pause
