@echo off
REM Wrapper for Task Scheduler - keeps stdout silent, but errors go to log
chcp 65001 >nul
set PYTHONIOENCODING=utf-8

REM Use full path to script (Task Scheduler may run from C:\Windows\System32)
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

REM Try to find python (prefer the one used by the user)
where python >nul 2>nul
if %ERRORLEVEL%==0 (
    python "%SCRIPT_DIR%vps_watchdog.py"
) else if exist "C:\Users\hlin2\AppData\Local\Programs\Python\Python310\python.exe" (
    "C:\Users\hlin2\AppData\Local\Programs\Python\Python310\python.exe" "%SCRIPT_DIR%vps_watchdog.py"
) else if exist "C:\Program Files\Python310\python.exe" (
    "C:\Program Files\Python310\python.exe" "%SCRIPT_DIR%vps_watchdog.py"
) else (
    echo Python not found! >> "%SCRIPT_DIR%vps_watchdog.log"
    exit /b 1
)
