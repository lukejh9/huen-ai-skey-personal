@echo off
chcp 65001 > nul
echo ============================================================
echo  Fix abstract in v16 paper
echo ============================================================
echo  Working dir: %CD%
python fix_abstract_v16.py
echo.
echo ============================================================
echo  Done. Press any key to close.
echo ============================================================
pause
