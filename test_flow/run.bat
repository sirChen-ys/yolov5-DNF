@echo off
rem ==========================================================================
rem  DNF test / run entry point
rem
rem    test_flow\run.bat              run everything (labels, val, sim, live)
rem    test_flow\run.bat val          val metrics only
rem    test_flow\run.bat sim 100      main-loop simulation, 100 frames
rem    test_flow\run.bat live         live screen capture + detection check
rem    test_flow\run.bat labels       dataset label statistics
rem
rem  Override the interpreter with:  set DNF_PYTHON=C:\path\to\python.exe
rem ==========================================================================
setlocal

chcp 65001 >nul 2>nul
set "PYTHONIOENCODING=utf-8"

set "ROOT=%~dp0.."
pushd "%ROOT%" || (echo [ERROR] cannot enter %ROOT% & exit /b 2)

rem ---------- locate a python that has torch + ultralytics ----------
set "PY="
if defined DNF_PYTHON if exist "%DNF_PYTHON%" set "PY=%DNF_PYTHON%"
if not defined PY if exist "%ROOT%\.venv\Scripts\python.exe" set "PY=%ROOT%\.venv\Scripts\python.exe"
if not defined PY if exist "%ROOT%venv\Scripts\python.exe" set "PY=%ROOT%venv\Scripts\python.exe"
if not defined PY if exist "E:\Model\yolo_cropDisease_detection_web\yolo_cropDisease_detection_flask\.venv\Scripts\python.exe" set "PY=E:\Model\yolo_cropDisease_detection_web\yolo_cropDisease_detection_flask\.venv\Scripts\python.exe"
if not defined PY set "PY=python"

"%PY%" -c "import torch, ultralytics, cv2" >nul 2>nul
if errorlevel 1 (
  echo [ERROR] No Python with torch + ultralytics + opencv found.
  echo         Tried: %PY%
  echo.
  echo   Create a venv for this project ^(one time^):
  echo     cd /d "%ROOT%"
  echo     python -m venv .venv
  echo     .venv\Scripts\python.exe -m pip install -U pip
  echo     .venv\Scripts\python.exe -m pip install -r requirements.txt
  echo.
  echo   For CUDA torch, install separately:
  echo     .venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
  echo.
  echo   Then re-run this script, or point DNF_PYTHON at another interpreter.
  popd
  exit /b 2
)

echo Interpreter : %PY%
echo Working dir : %ROOT%
echo.

set "TASK=%~1"
if "%TASK%"=="" set "TASK=all"
set "FRAMES=%~2"
if "%FRAMES%"=="" set "FRAMES=50"

set "FAIL=0"

if /i "%TASK%"=="all" (
  call :run labels ""
  call :run val ""
  call :run sim "%FRAMES%"
  call :run skill ""
  call :run settle ""
  call :run live ""
  goto :done
)
call :run "%TASK%" "%FRAMES%"

:done
popd
if not "%FAIL%"=="0" (
  echo.
  echo %FAIL% task^(s^) failed.
  exit /b 1
)
echo.
echo All checks passed.
exit /b 0

rem --------------------------------------------------------------------------
:run
set "T=%~1"
set "N=%~2"
echo ==========================================================================
echo ^>^>^> %T%
echo ==========================================================================
if /i "%T%"=="labels" ( "%PY%" test_flow\label_stats.py & goto :check )
if /i "%T%"=="val"    ( "%PY%" test_flow\val_metrics.py & goto :check )
if /i "%T%"=="sim"    ( "%PY%" test_flow\simulate_main2.py --frames %N% & goto :check )
if /i "%T%"=="skill"  ( "%PY%" test_flow\test_skill_bar.py & goto :check )
if /i "%T%"=="settle" ( "%PY%" test_flow\test_settlement.py & goto :check )
if /i "%T%"=="live"   ( "%PY%" test_flow\live_capture_check.py --frames 3 & goto :check )
echo [ERROR] unknown task: %T%
echo         valid: all ^| labels ^| val ^| sim ^| skill ^| settle ^| live
set "FAIL=1"
goto :eof

:check
if errorlevel 1 (
  echo [FAILED] %T%
  set "FAIL=1"
) else (
  echo [OK] %T%
)
echo.
goto :eof
