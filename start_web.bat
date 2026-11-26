@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"
echo Hachimi Sonata · starting web UI (auto port + open browser)...
where py >nul 2>nul
if not errorlevel 1 (
  py -3 main.py web
) else (
  python main.py web
)
if errorlevel 1 pause
