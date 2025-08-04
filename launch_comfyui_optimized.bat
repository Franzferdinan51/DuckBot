@echo off
echo Starting ComfyUI with BEAST MODE Optimization...
echo.
echo HARDWARE DETECTED: 128GB DDR5 + Ryzen 9 7950X3D
echo HIGH VRAM MODE: Maximum quality settings enabled
echo CPU OFFLOAD: 32GB buffer (MASSIVE overflow capacity with 128GB RAM!)
echo ULTRA FAST ATTENTION: Flash Attention + Advanced modes
echo.

cd /d "C:\Users\Ryan\Desktop\DuckBotV1\DiscordBot\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable"

python_embeded\python.exe ComfyUI\main.py ^
    --highvram ^
    --cpu-offload-gb 32 ^
    --attention-pytorch ^
    --use-quad-cross-attention ^
    --fp16-unet ^
    --force-fp16 ^
    --listen 127.0.0.1 ^
    --port 8188

pause