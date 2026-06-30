@echo off
chcp 65001 > nul
echo ============================================================
echo  Reduce tables v16: 16 tables to 8 tables
echo ============================================================
python reduce_tables_v16.py
echo.
echo ============================================================
echo  Done. Press any key to close.
echo ============================================================
pause
