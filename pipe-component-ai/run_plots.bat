@echo off
chcp 65001 > nul
echo ============================================================
echo  Generating visualization plots for models_v2/
echo ============================================================
echo  Working dir: %CD%
python generate_plots.py
echo.
echo ============================================================
echo  Done. Press any key to close.
echo ============================================================
pause
