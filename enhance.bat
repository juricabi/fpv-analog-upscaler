@echo off
setlocal
cd /d "%~dp0"
if "%~1"=="" goto :help
".venv\Scripts\python.exe" enhance.py %*
echo.
echo  Finished. Output is in the "output" folder.
pause
exit /b 0

:help
echo.
echo  Drag one or more DVR videos onto this file, or run:
echo    enhance.bat ORQA0084.MP4 [options]
echo.
echo  Useful options:
echo    --model vhs2hd        VHS-trained model, cleaner but ~15x slower
echo    --sharpen 0           no extra sharpening, default is 0.25
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
