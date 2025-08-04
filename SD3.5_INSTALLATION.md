# 🎨 Stable Diffusion 3.5 Large FP8 Installation

## **Model Information**
- **Model:** Stable Diffusion 3.5 Large FP8 Scaled
- **URL:** https://huggingface.co/Comfy-Org/stable-diffusion-3.5-fp8/resolve/main/sd3.5_large_fp8_scaled.safetensors
- **File Size:** ~8.9 GB 
- **Format:** FP8 (Memory Optimized)
- **Quality:** State-of-the-art image generation

## **Installation Steps**

### **1. Download the Model**
```bash
# Option 1: Direct download
wget https://huggingface.co/Comfy-Org/stable-diffusion-3.5-fp8/resolve/main/sd3.5_large_fp8_scaled.safetensors

# Option 2: Using curl
curl -L -o sd3.5_large_fp8_scaled.safetensors https://huggingface.co/Comfy-Org/stable-diffusion-3.5-fp8/resolve/main/sd3.5_large_fp8_scaled.safetensors?download=true
```

### **2. Place in Your ComfyUI Models Folder**
```
C:\Users\Ryan\Desktop\DuckBotV1\DiscordBot\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\models\checkpoints\sd3.5_large_fp8_scaled.safetensors
```

### **3. Verify Installation**
1. Start ComfyUI
2. Check that `sd3.5_large_fp8_scaled.safetensors` appears in the checkpoint dropdown
3. Test generation with DuckBot

## **Optimized Settings**

### **Default Workflow Settings**
- **Sampler:** DPM++ 2M Karras (optimized for SD3.5)
- **Steps:** 25 (increased from 20 for better quality)
- **CFG:** 7.0 (optimal for SD3.5)
- **Scheduler:** Karras (better noise scheduling)

### **Memory Benefits with FP8**
- **50% Less VRAM** compared to FP16
- **Faster Generation** due to reduced memory bandwidth
- **Same Quality** as full precision
- **Perfect for your 128GB RAM** system with overflow handling

## **Performance Comparison**

| Model | VRAM Usage | Quality | Speed | Compatibility |
|-------|------------|---------|-------|---------------|
| SD 1.5 | ~3GB | Good | Fast | High |
| SDXL | ~6GB | Better | Medium | Medium |
| **SD3.5 FP8** | **~4GB** | **Excellent** | **Fast** | **High** |

## **Expected Improvements**
- **Better Text Rendering** - Superior text integration in images
- **Higher Detail** - More intricate and coherent details
- **Better Composition** - Improved spatial understanding
- **Style Consistency** - More consistent artistic styles
- **Prompt Adherence** - Better following of complex prompts

## **Troubleshooting**

### **Model Not Loading**
- Verify file integrity (should be ~8.9GB)
- Check file permissions
- Ensure ComfyUI has access to models folder

### **Out of Memory**
- Use `--lowvram` flag with ComfyUI
- Your 32GB CPU offload buffer should handle any overflow
- Reduce resolution if needed (4K → 1440p → 1080p)

### **Slow Generation**
- Ensure using optimized sampler settings
- Check that FP8 precision is being used
- Verify Flash Attention is enabled

---

**🎯 You're now ready to generate stunning images with Stable Diffusion 3.5!**