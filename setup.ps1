# Installs everything "Ноты из записи" needs on a Windows 10/11 computer:
#   1. Python 3.12 (downloaded from python.org if there is none; installed for this user only)
#   2. a virtual environment .venv with PyTorch: the CUDA build for NVIDIA cards, the CPU
#      build otherwise; a PyTorch already installed in that Python is reused
#   3. the libraries from requirements.txt, transkun, the beat model
#   4. the program "Ноты из записи.exe" and a desktop shortcut
# Run it through setup.bat. Can be run again at any time to repair the installation.
#
#   -Cpu         the CPU build of PyTorch even with an NVIDIA card
#   -Fresh       ignore a PyTorch installed in Python itself, put everything into .venv
#   -NoShortcut  no desktop shortcut

param([switch]$Cpu, [switch]$Fresh, [switch]$NoShortcut)

# Continue: in Windows PowerShell 5.1 "Stop" turns any stderr line of python or pip into a fatal error
$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"   # Invoke-WebRequest is very slow with the progress bar
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Log = Join-Path $Root "setup_log.txt"
$Venv = Join-Path $Root ".venv"
$Py = Join-Path $Venv "Scripts\python.exe"
$PythonVersion = "3.12.10"
$TorchVersion = "2.12.0"
$TorchIndex = "https://download.pytorch.org/whl"

"==== setup $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ====" | Out-File $Log -Encoding utf8

function Say([string]$text) {
    Write-Host $text
    $text | Out-File $Log -Append -Encoding utf8
}

function Fail([string]$text) {
    Say ""
    Say "ОШИБКА: $text"
    Say "Подробности в файле setup_log.txt. Запустите setup.bat ещё раз - установка продолжится с того же места."
    exit 1
}

# runs a program, its output goes to the log; returns the exit code
function Run([string]$exe, [string[]]$arguments) {
    & $exe @arguments 2>&1 | ForEach-Object { "$_" } | Out-File $Log -Append -Encoding utf8
    return $LASTEXITCODE
}

# the same, up to three times: downloads fail on a poor connection
function Retry([string]$what, [string]$exe, [string[]]$arguments) {
    for ($try = 1; $try -le 3; $try++) {
        if ((Run $exe $arguments) -eq 0) { return }
        Say "   $what`: попытка $try из 3 не удалась"
        Start-Sleep -Seconds 5
    }
    Fail "$what - не удалось (проверьте интернет)"
}

function Test-Python([string]$exe) {
    if (-not $exe -or -not (Test-Path $exe)) { return $false }
    $v = & $exe -c "import sys, tkinter; print('%d.%d' % sys.version_info[:2])" 2>$null
    return $v -eq "3.12"
}

# ------------------------------------------------------------------ 1. Python 3.12

Say "[1/5] Python 3.12"
$Base = $null
$candidates = @(
    (Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"),
    "C:\Program Files\Python312\python.exe"
)
foreach ($c in $candidates) { if (Test-Python $c) { $Base = $c; break } }
if (-not $Base -and (Get-Command py -ErrorAction SilentlyContinue)) {
    $c = & py -3.12 -c "import sys; print(sys.executable)" 2>$null
    if (Test-Python $c) { $Base = $c }
}
if (-not $Base) {
    Say "   не найден, скачиваю Python $PythonVersion с python.org ..."
    $installer = Join-Path $env:TEMP "python-$PythonVersion-amd64.exe"
    $url = "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-amd64.exe"
    $ok = $false
    for ($try = 1; $try -le 3 -and -not $ok; $try++) {
        try { Invoke-WebRequest $url -OutFile $installer -UseBasicParsing -ErrorAction Stop; $ok = $true }
        catch { Say "   загрузка Python: попытка $try из 3 не удалась"; Start-Sleep -Seconds 5 }
    }
    if (-not $ok) { Fail "не удалось скачать Python" }
    Say "   устанавливаю (только для этого пользователя) ..."
    $p = Start-Process $installer -Wait -PassThru -ArgumentList @(
        "/quiet", "InstallAllUsers=0", "PrependPath=0", "Include_test=0", "Include_tcltk=1",
        "Include_launcher=0", "Shortcuts=0", "AssociateFiles=0")
    $Base = Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"
    if (-not (Test-Python $Base)) { Fail "Python не установился (код $($p.ExitCode))" }
}
Say "   $Base"

# ------------------------------------------------------------------ 2. environment and PyTorch

Say "[2/5] Окружение и PyTorch"
$baseTorch = $null
if (-not $Fresh) {
    $baseTorch = & $Base -c "import torch, torchaudio; print(torch.__version__ + ' ' + torchaudio.__version__)" 2>$null
}
if (-not (Test-Path $Py)) {
    $venvArgs = @("-m", "venv")
    if ($baseTorch) { $venvArgs += "--system-site-packages" }   # reuse the installed PyTorch
    $venvArgs += $Venv
    if ((Run $Base $venvArgs) -ne 0) { Fail "не удалось создать окружение .venv" }
}
$constraints = Join-Path $Venv "constraints.txt"
$haveTorch = & $Py -c "import torch, torchaudio; print(torch.__version__ + ' ' + torchaudio.__version__)" 2>$null
if ($haveTorch) {
    Say "   PyTorch уже есть: $haveTorch"
} else {
    $nvidia = (Get-Command nvidia-smi -ErrorAction SilentlyContinue) -or (Test-Path "$env:SystemRoot\System32\nvidia-smi.exe")
    if ($nvidia -and -not $Cpu) {
        $build = "cu126"
        Say "   найдена видеокарта NVIDIA: качаю PyTorch для видеокарты (около 3 ГБ, это надолго) ..."
    } else {
        $build = "cpu"
        if ($Cpu) { Say "   качаю PyTorch для процессора (около 250 МБ) ..." }
        else { Say "   видеокарты NVIDIA нет: качаю PyTorch для процессора (около 250 МБ) ..." }
    }
    Retry "PyTorch" $Py @("-m", "pip", "install", "--timeout", "120", "--retries", "10",
                          "torch==$TorchVersion", "torchaudio", "--index-url", "$TorchIndex/$build")
    $haveTorch = & $Py -c "import torch, torchaudio; print(torch.__version__ + ' ' + torchaudio.__version__)" 2>$null
    if (-not $haveTorch) { Fail "PyTorch установился, но не запускается" }
    Say "   $haveTorch"
}
# pip must never replace the PyTorch build chosen above
$tv = $haveTorch.Split(" ")
"torch==$($tv[0])`ntorchaudio==$($tv[1])" | Out-File $constraints -Encoding ascii

# ------------------------------------------------------------------ 3. libraries

Say "[3/5] Библиотеки"
Retry "библиотеки" $Py @("-m", "pip", "install", "--timeout", "120", "--retries", "10",
                         "-c", $constraints, "-r", (Join-Path $Root "requirements.txt"))
# transkun without its dependencies: ncls is only needed for training and does not build on Windows
Retry "transkun" $Py @("-m", "pip", "install", "--timeout", "120", "--retries", "10",
                       "--no-deps", "transkun==2.0.1")

# ------------------------------------------------------------------ 4. models

Say "[4/5] Модель для определения долей"
Retry "модель долей" $Py @("-c", "from beat_this.inference import Audio2Frames; Audio2Frames(checkpoint_path='final0', device='cpu'); print('beat model OK')")
# load both networks for real: a missing module only shows up then
$probe = "import sys; sys.path.insert(0, r'$Root'); import torch, transcribe, trim_editor, beat_editor; " +
         "transcribe.load_transkun('cpu'); from beat_this.inference import Audio2Frames; " +
         "Audio2Frames(checkpoint_path='final0', device='cpu'); print('cuda' if torch.cuda.is_available() else 'cpu')"
$check = & $Py -c $probe 2>$null | Select-Object -Last 1
if ($check -ne "cuda" -and $check -ne "cpu") {
    Run $Py @("-c", $probe) | Out-Null   # the error goes to the log
    Fail "программа не запускается в новом окружении"
}
if ($check -eq "cuda") { Say "   распознавание будет идти на видеокарте" }
else { Say "   распознавание будет идти на процессоре (медленнее: трёхминутная запись - несколько минут)" }

# ------------------------------------------------------------------ 5. the program

Say "[5/5] Программа и ярлык"
$buildArgs = @((Join-Path $Root "build_app.py"))
if (-not $NoShortcut) { $buildArgs += "--shortcut" }
if ((Run $Py $buildArgs) -ne 0) { Fail "не удалось собрать программу" }

Say ""
Say "Готово. Запускайте `"Ноты из записи`" (ярлык на рабочем столе)."
