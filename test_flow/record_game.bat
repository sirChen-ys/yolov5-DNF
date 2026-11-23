@echo off
rem ==========================================================================
rem  Record the game client area for annotation (doors / exits).
rem
rem  Double-click this file, then IMMEDIATELY click the game window:
rem  the recorder starts after a 5 second countdown.
rem
rem  Do NOT switch to other windows while recording, otherwise the
rem  covering window gets recorded instead of the game.
rem
rem  Default: 10 minutes -> E:\GameData\raw\doors_<timestamp>.mp4
rem
rem  Custom duration (e.g. 5 minutes):
rem     test_flow\record_game.bat 300
rem ==========================================================================
setlocal

chcp 65001 >nul 2>nul
set "PYTHONIOENCODING=utf-8"

set "ROOT=%~dp0.."
pushd "%ROOT%" || (echo [ERROR] cannot enter %ROOT% & exit /b 2)

set "PY="
if defined DNF_PYTHON if exist "%DNF_PYTHON%" set "PY=%DNF_PYTHON%"
if not defined PY if exist "%ROOT%\.venv\Scripts\python.exe" set "PY=%ROOT%\.venv\Scripts\python.exe"
if not defined PY if exist "E:\Model\yolo_cropDisease_detection_web\yolo_cropDisease_detection_flask\.venv\Scripts\python.exe" set "PY=E:\Model\yolo_cropDisease_detection_web\yolo_cropDisease_detection_flask\.venv\Scripts\python.exe"
if not defined PY set "PY=python"

set "SECONDS=%~1"
if "%SECONDS%"=="" set "SECONDS=600"

"%PY%" -c "import cv2, win32gui" >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python with cv2 + pywin32 not found: %PY%
  echo         Set DNF_PYTHON to a suitable python.exe first.
  popd
  pause
  exit /b 2
)

"%PY%" test_flow\record_game.py --seconds %SECONDS%

popd
echo.
pause
