# One-time setup on a fresh Windows PC (needs Python 3.10+ and an AMD/Intel/NVIDIA GPU with DirectX 12).
# Run:  powershell -ExecutionPolicy Bypass -File setup.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv")) { py -3 -m venv .venv }
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

if (-not (Test-Path "tools\ffmpeg\bin\ffmpeg.exe")) {
    New-Item -ItemType Directory -Force tools | Out-Null
    Invoke-WebRequest "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip" -OutFile tools\ffmpeg.zip
    Expand-Archive tools\ffmpeg.zip -DestinationPath tools -Force
    Remove-Item tools\ffmpeg.zip
    Get-ChildItem tools -Directory -Filter "ffmpeg-*-essentials_build" | Rename-Item -NewName "ffmpeg"
}

New-Item -ItemType Directory -Force models | Out-Null
$models = @{
    "2xLiveActionV1_SPAN.onnx"           = "https://github.com/jcj83429/upscaling/raw/f73a3a02874360ec6ced18f8bdd8e43b5d7bba57/2xLiveActionV1_SPAN/2xLiveActionV1_SPAN_490000.onnx"
    "2xNomosUni_compact_otf_medium.onnx" = "https://github.com/Phhofm/models/releases/download/2xNomosUni_compact_otf_medium/2xNomosUni_compact_otf_medium_fp32_opset17.onnx"
    "2xNomosUni_span_multijpg.onnx"      = "https://github.com/Phhofm/models/releases/download/2xNomosUni_span_multijpg/2xNomosUni_span_multijpg_fp32_opset17.onnx"
    "4xLSDIRCompactR3.onnx"              = "https://github.com/Phhofm/models/releases/download/4xLSDIRCompact3/4xLSDIRCompactR3_fp32_opset17.onnx"
    "2xVHS2HD-RealPLKSR.pth"             = "https://github.com/starinspace/StarinspaceUpscale/releases/download/Models/2xVHS2HD-RealPLKSR.pth"
    "2xNomosUni_esrgan_multijpg.onnx"    = "https://github.com/Phhofm/models/releases/download/2xNomosUni_esrgan_multijpg/2xNomosUni_esrgan_multijpg_fp16_opset17.onnx"
    "4xNomos2_otf_esrgan.onnx"           = "https://github.com/Phhofm/models/releases/download/4xNomos2_otf_esrgan/4xNomos2_otf_esrgan_fp16_opset17.onnx"
    "1xDeNoise_realplksr_otf.onnx"       = "https://github.com/Phhofm/models/releases/download/1xDeNoise_realplksr_otf/1xDeNoise_realplksr_otf_fp32.onnx"
    "1xSuperScale_SPAN.safetensors"      = "https://github.com/Kim2091/Kim2091-Models/releases/download/1x-SuperScale/1x-SuperScale_SPAN.safetensors"
}
foreach ($name in $models.Keys) {
    if (-not (Test-Path "models\$name")) { Invoke-WebRequest $models[$name] -OutFile "models\$name" }
}
$convert = @{ "models\2xVHS2HD-RealPLKSR.pth" = "models\2xVHS2HD.onnx"; "models\1xSuperScale_SPAN.safetensors" = "models\1xSuperScale_SPAN.onnx" }
$needConvert = $convert.Keys | Where-Object { -not (Test-Path $convert[$_]) }
if ($needConvert) {
    Write-Host "Converting PyTorch models to ONNX - installs CPU PyTorch (~250 MB) ..."
    .\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    .\.venv\Scripts\python.exe -m pip install spandrel
    foreach ($src in $needConvert) { .\.venv\Scripts\python.exe tools\convert_pth_to_onnx.py $src $convert[$src] }
}
if (-not (Test-Path "tools\vapoursynth-portable\vspipe.bat")) {
    Write-Host "Installing portable VapourSynth (temporal denoise pre-filter) ..."
    Push-Location tools
    Invoke-WebRequest "https://github.com/vapoursynth/vapoursynth/releases/download/R80/Install-Portable-VapourSynth-R80.zip" -OutFile vs-install.zip
    Expand-Archive vs-install.zip -DestinationPath . -Force
    Remove-Item vs-install.zip
    & powershell -NoProfile -ExecutionPolicy Bypass -File Install-Portable-VapourSynth-R80.ps1 -PythonVersionMinor 12 -Unattended
    Pop-Location
    $vs = "tools\vapoursynth-portable"
    & "$vs\python.exe" -m pip install --no-warn-script-location vsrepo
    & "$vs\Scripts\vsrepo.exe" update
    & "$vs\Scripts\vsrepo.exe" install zsmooth
    Invoke-WebRequest "https://github.com/vapoursynth/bestsource/releases/download/R22/BestSource-R22-win64-msvc.zip" -OutFile "$vs\bs.zip"
    Expand-Archive "$vs\bs.zip" -DestinationPath "$vs\Lib\site-packages\vapoursynth\plugins" -Force
    Remove-Item "$vs\bs.zip"
}
Write-Host "`nDone. Try:  .\enhance.bat yourclip.mp4"
