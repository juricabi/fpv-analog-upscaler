@echo off
setlocal
cd /d "%~dp0"
if "%~1"=="" goto :help

if not exist ".venv\Scripts\python.exe" goto :nosetup
if not exist "tools\ffmpeg\bin\ffmpeg.exe" goto :nosetup
if not exist "models\2xLiveActionV1_SPAN.onnx" goto :nosetup

:run
".venv\Scripts\python.exe" enhance.py %*
if errorlevel 1 (
    echo.
    echo  Something went wrong - see the messages above.
    pause
    exit /b 1
)
echo.
echo  Finished. Output is in the "output" folder.
pause
exit /b 0

:nosetup
echo.
echo  This folder has not been set up yet: the Python environment, FFmpeg or the
echo  AI models are missing. A one-time setup downloads them (about 2 GB).
echo.
choice /c YN /m "  Run the setup now"
if errorlevel 2 exit /b 1
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; & '.\setup.ps1'"
if errorlevel 1 (
    echo.
    echo  Setup failed - see the messages above. Fix the problem and drop the file again.
    pause
    exit /b 1
)
echo.
echo  Setup done, starting the enhancement ...
echo.
goto :run

:help
echo.
echo  Drag one or more DVR videos onto this file, or run:
echo    enhance.bat ORQA0084.MP4 [options]
echo.
echo  Useful options:
echo    --model vhs2hd        VHS-trained model, cleaner but ~15x slower
echo    --sharpen 0           no extra sharpening, default is 0.4
echo    --pad 1920x1080       add black side bars for a 16:9 canvas
echo    --prescale auto       faster: downscale to true analog res first
echo    --denoise off         no temporal denoise, default is light
echo    --encoder hevc_amf    GPU H.265 encode instead of CPU H.264
echo    --throttle 0.5        use only half the GPU, keeps the PC usable meanwhile
echo    --low-priority        run in the background at low CPU priority
echo    --passes 2            run the AI twice, more texture, slower
echo    --model denoise1x+liveaction   cleaner edges and calmer grass, about 4x slower
echo    --mc-denoise 300      calmer but slightly softer picture, default is 150, 0 = off
echo    --list-models
echo.
pause
exit /b 1
