@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion
cd /d "%~dp0"

echo ============================================================
echo   SegFormer + YOLO Video Demo  -  One-Click Launcher
echo   URL: http://127.0.0.1:8000
echo ============================================================
echo.

REM ---------- 1. virtual env ----------
if not exist ".venv\Scripts\python.exe" (
    echo [1/5] Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create venv. Install Python 3.11+ first.
        pause
        exit /b 1
    )
) else (
    echo [1/5] Virtual environment ready.
)

set PY=.venv\Scripts\python.exe

REM ---------- 2. dependencies ----------
"%PY%" -c "import fastapi, uvicorn, ultralytics, transformers, cv2, torch" >nul 2>nul
if errorlevel 1 (
    echo [2/5] Installing dependencies ^(first run, this may take a while^)...
    where nvidia-smi >nul 2>nul
    if !errorlevel!==0 (
        echo       NVIDIA GPU detected - installing CUDA build of PyTorch...
        "%PY%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
    ) else (
        echo       No NVIDIA GPU - installing CPU build of PyTorch...
        "%PY%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    )
    "%PY%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Dependency installation failed.
        pause
        exit /b 1
    )
) else (
    echo [2/5] Dependencies ready.
)

REM ---------- 3. YOLO weights ----------
if not exist "weights\yolo\best.pt" (
    echo [3/5] Downloading YOLO weights...
    if not exist "weights\yolo" mkdir "weights\yolo"
    curl.exe -L -o "weights\yolo\best.pt" "https://raw.githubusercontent.com/yyh061129vic-debug/driver-vision-risk-warning-system/main/team%20submissions%20second%20week/YOLOv11s-P2/weights/best.pt"
    if not exist "weights\yolo\best.pt" (
        echo [WARN] YOLO weight download failed. Fusion mode will fall back to SegFormer only.
    )
) else (
    echo [3/5] YOLO weights ready.
)

REM ---------- 4. frontend build ----------
if not exist "web\dist\index.html" (
    echo [4/5] Building frontend...
    pushd web
    if not exist node_modules (
        call npm install
    )
    call npm run build
    popd
    if errorlevel 1 (
        echo [WARN] Frontend build failed. The API will still work, UI is served only if built.
    )
) else (
    echo [4/5] Frontend already built.
)

REM ---------- 5. start server ----------
echo [5/5] Starting backend at http://127.0.0.1:8000 ...
echo       First start downloads the SegFormer pretrained weights (~100MB).
echo       Keep this window open. Close it to stop the server.
echo.
set HF_ENDPOINT=https://hf-mirror.com
start "" cmd /c "timeout /t 6 >nul & start "" http://127.0.0.1:8000"
"%PY%" -m uvicorn server.main:app --host 127.0.0.1 --port 8000

pause
