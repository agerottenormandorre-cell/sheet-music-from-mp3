@echo off
chcp 65001 >nul
cd /d "%~dp0"
rem Установка всего, что нужно программе. Подробности - в setup.ps1.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" %*
echo.
pause
