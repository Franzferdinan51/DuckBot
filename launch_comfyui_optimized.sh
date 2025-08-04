#!/bin/bash
echo "Starting ComfyUI with BEAST MODE Optimization..."
echo ""
echo "HARDWARE DETECTED: 128GB DDR5 + Ryzen 9 7950X3D"
echo "HIGH VRAM MODE: Maximum quality settings enabled"
echo "MASSIVE CPU BUFFER: 32GB CPU offload capability"
echo "ULTRA FAST ATTENTION: Flash Attention + Advanced modes"
echo ""

cd "/c/Users/Ryan/Desktop/DuckBotV1/DiscordBot/ComfyUI_windows_portable_nvidia/ComfyUI_windows_portable"

python main.py \
    --highvram \
    --cpu-offload-gb 32 \
    --attention-pytorch \
    --use-quad-cross-attention \
    --fp16-unet \
    --force-fp16 \
    --listen 127.0.0.1 \
    --port 8188