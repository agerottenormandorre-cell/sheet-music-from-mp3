@echo off
chcp 65001 >nul
cd /d "%~dp0"
rem Без аргументов: все аудиофайлы из этой папки.
rem Можно перетащить на этот файл одну или несколько записей.
".venv\Scripts\python.exe" -W ignore transcribe.py %*
echo.
echo Результаты в папке out\
pause
