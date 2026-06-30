@echo off
chcp 65001 > nul
echo ============================================================
echo  Fix Table 7 cascade rename issue in v16
echo ============================================================
python fix_table7_v16.py
echo.
echo ============================================================
echo  Done. Press any key to close.
echo ============================================================
pause
