@echo off
echo ============================================================
echo  AI Pipe Component Validation Server
echo  Open browser: http://localhost:8765/ai_validator.html
echo ============================================================
cd /d "%~dp0"
python ai_validator_server.py
echo.
echo ============================================================
echo  Server stopped. Press any key to close.
echo ============================================================
pause
