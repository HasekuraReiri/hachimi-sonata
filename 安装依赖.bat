@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"
echo Installing / updating dependencies...
where py >nul 2>nul
if not errorlevel 1 (
  py -3 -m pip install --upgrade pip
  py -3 -m pip install -r requirements.txt
) else (
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
)
echo.
echo Done. Double-click start.bat or run: python main.py
pause
