@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"

rem 优先使用系统 Python 启动器（已签名，不会触发 exe 的安全验证）
if exist "%WINDIR%\pyw.exe" (
  start "" "%WINDIR%\pyw.exe" -3 "%~dp0launcher.py"
  exit /b 0
)
where pyw >nul 2>nul
if not errorlevel 1 (
  start "" pyw -3 "%~dp0launcher.py"
  exit /b 0
)
where pythonw >nul 2>nul
if not errorlevel 1 (
  start "" pythonw "%~dp0launcher.py"
  exit /b 0
)
if exist "HachimiSonata.exe" (
  start "" "HachimiSonata.exe"
  exit /b 0
)
echo [ERROR] 未找到 Python / HachimiSonata.exe
pause
