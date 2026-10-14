@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "LOG=%~dp0setup_log.txt"
set "PY=%~dp0.venv\Scripts\python.exe"
set "BASEPY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"

echo ==== setup %date% %time% ==== > "%LOG%"

if not exist "%PY%" (
    echo [1/3] Создаю окружение .venv ...
    "%BASEPY%" -m venv --system-site-packages .venv >> "%LOG%" 2>&1
)

rem не даём pip заменить уже установленный PyTorch с CUDA
(
echo torch==2.12.0+cu130
echo torchaudio==2.11.0+cu130
echo torchvision==0.27.0+cu130
) > .venv\constraints.txt

set TRY=0
:install
set /a TRY+=1
echo [2/3] Устанавливаю библиотеки, попытка %TRY% из 3 ...
rem transkun ставится без зависимостей: его пакет ncls нужен только для обучения
rem и без компилятора C++ под Windows не собирается
"%PY%" -m pip install --timeout 120 --retries 10 -c .venv\constraints.txt --no-deps transkun >> "%LOG%" 2>&1
if not errorlevel 1 "%PY%" -m pip install --timeout 120 --retries 10 -c .venv\constraints.txt pretty_midi music21 soundfile soxr moduleconf beat-this verovio >> "%LOG%" 2>&1
if errorlevel 1 (
    echo pip: попытка %TRY% не удалась >> "%LOG%"
    if %TRY% lss 3 goto install
    echo.
    echo ОШИБКА: не удалось установить библиотеки, подробности в setup_log.txt
    pause
    exit /b 1
)
echo pip install OK >> "%LOG%"

echo [3/3] Скачиваю модель для определения долей ...
"%PY%" -c "from beat_this.inference import Audio2Frames; Audio2Frames(checkpoint_path='final0', device='cpu'); print('beat_this checkpoint OK')" >> "%LOG%" 2>&1
if errorlevel 1 (
    echo ОШИБКА при скачивании модели, подробности в setup_log.txt
) else (
    echo    готово
)

echo Собираю программу "Ноты из записи" ...
"%PY%" build_app.py --shortcut >> "%LOG%" 2>&1
if errorlevel 1 echo ОШИБКА при сборке программы, подробности в setup_log.txt

echo.
echo Установка завершена. Запускайте "Ноты из записи" (ярлык на рабочем столе)
pause
