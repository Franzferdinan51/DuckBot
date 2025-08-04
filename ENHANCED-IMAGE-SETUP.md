# 🎨 Enhanced Image Generation Setup Guide

Your DuckBot now supports **3 powerful image generation models**:

- **🌟 FLUX.1 Schnell** - Ultra-fast, highest quality (2-4 seconds)
- **🎨 Stable Diffusion XL** - Versatile, great for art (10-15 seconds) 
- **🚀 Stable Diffusion 1.5** - Fast and reliable fallback (5-8 seconds)

## 🚀 Quick Setup

### **1. Add Enhanced Image Generation Code**

Copy the code from `enhanced_image_integration.py` and add it to your main bot file (`DuckBot-v2.1-MULTI-SERVER.py`).

### **2. Download Model Files**

You need to download the model files and place them in your ComfyUI models folder:

#### **📁 ComfyUI Models Folder Structure:**
```
ComfyUI/
├── models/
│   ├── checkpoints/          # Main model files go here
│   │   ├── flux1-schnell.safetensors          # FLUX.1 model
│   │   ├── sd_xl_base_1.0.safetensors         # SDXL base
│   │   ├── sd_xl_refiner_1.0.safetensors      # SDXL refiner
│   │   └── v1-5-pruned-emaonly.ckpt           # SD 1.5
│   ├── clip/                 # Text encoder files
│   │   └── t5xxl_fp16.safetensors             # FLUX.1 text encoder
│   └── ...
```

## 📥 Model Download Links

### **🌟 FLUX.1 Schnell (Recommended - Best Quality)**

**Main Model:**
- **File:** `flux1-schnell.safetensors` (23.8 GB)
- **Link:** https://huggingface.co/black-forest-labs/FLUX.1-schnell/blob/main/flux1-schnell.safetensors
- **Place in:** `ComfyUI/models/checkpoints/`

**Text Encoder:**
- **File:** `t5xxl_fp16.safetensors` (4.89 GB)
- **Link:** https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/t5xxl_fp16.safetensors
- **Place in:** `ComfyUI/models/clip/`

### **🎨 Stable Diffusion XL (Great for Art)**

**Base Model:**
- **File:** `sd_xl_base_1.0.safetensors` (6.94 GB)
- **Link:** https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0.safetensors
- **Place in:** `ComfyUI/models/checkpoints/`

**Refiner Model:**
- **File:** `sd_xl_refiner_1.0.safetensors` (6.08 GB) 
- **Link:** https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/blob/main/sd_xl_refiner_1.0.safetensors
- **Place in:** `ComfyUI/models/checkpoints/`

### **🚀 Stable Diffusion 1.5 (Fast Fallback)**

**Model:**
- **File:** `v1-5-pruned-emaonly.ckpt` (4.27 GB)
- **Link:** https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.ckpt
- **Place in:** `ComfyUI/models/checkpoints/`

## 🛠️ Installation Steps

### **Step 1: Download Models**

Pick the models you want. Start with **SD 1.5** for testing, then add **FLUX.1** for best quality:

```bash
# Option 1: Use git lfs (if you have it)
cd ComfyUI/models/checkpoints/
git lfs install
wget https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt

# Option 2: Direct download (Windows)
# Use your browser to download the files from the links above
```

### **Step 2: Verify Workflow Files**

Make sure these workflow files exist in your bot directory:
- ✅ `workflow_api.json` (SD 1.5)
- ✅ `workflow_sdxl_api.json` (SDXL) 
- ✅ `workflow_flux_api.json` (FLUX.1)

### **Step 3: Test the Models**

Start ComfyUI and your bot, then test:

```
/model_info          # See which models are available
/generate_advanced   # Try advanced generation
/generate_style      # Try style presets
```

## 🎯 New Commands

### **`/generate_advanced`**
Choose specific models for generation:
- 🌟 **FLUX.1** - Best quality, photorealistic
- 🎨 **SDXL** - Great for art and creative styles  
- 🚀 **SD 1.5** - Fast and reliable
- 🎯 **Auto** - Automatically picks the best available model

### **`/model_info`**
View all available models, their speeds, and capabilities.

### **`/generate_style`**
Generate with predefined artistic styles:
- 📸 Photorealistic
- 🎨 Digital Art
- 🖼️ Oil Painting
- ✏️ Pencil Sketch
- 🌸 Anime Style
- 🏛️ Classical Art
- 🌟 Fantasy
- 🤖 Cyberpunk

## 💾 Storage Requirements

| Model | Size | Quality | Speed | Use Case |
|-------|------|---------|-------|----------|
| **FLUX.1** | ~29 GB | ⭐⭐⭐⭐⭐ | Very Fast | Photorealism, portraits |
| **SDXL** | ~13 GB | ⭐⭐⭐⭐ | Medium | Art, creative styles |  
| **SD 1.5** | ~4 GB | ⭐⭐⭐ | Fast | Quick generation, fallback |

**Recommended:** Start with **SD 1.5** + **FLUX.1** for best balance of quality and speed.

## 🔧 ComfyUI Setup

### **Required ComfyUI Nodes**
All models use standard ComfyUI nodes - no custom nodes needed!

### **Memory Requirements**
- **FLUX.1:** 12+ GB VRAM (or system RAM with CPU mode)
- **SDXL:** 8+ GB VRAM 
- **SD 1.5:** 4+ GB VRAM

### **Performance Tips**

**For Lower VRAM:**
```python
# In ComfyUI settings, enable:
- CPU mode for text encoder
- Model offloading 
- Low VRAM mode
```

**For Faster Generation:**
```python
# FLUX.1 settings:
- Steps: 4 (already optimized)
- CFG: 1.0 (distilled model)
- Sampler: euler (fastest)

# SDXL settings:  
- Steps: 20-25 base + 10-15 refiner
- CFG: 7.0
- Sampler: dpmpp_2m

# SD 1.5 settings:
- Steps: 15-25
- CFG: 7-9
- Sampler: dpmpp_2m or euler_a
```

## 🚀 Usage Examples

### **Basic Usage:**
```
/generate_advanced prompt:a cat driving a car model:Auto
```

### **Style-Specific:**
```
/generate_style prompt:a magical forest style:Fantasy
```

### **Model-Specific:**
```
/generate_advanced prompt:professional headshot model:FLUX.1 (Best Quality)
```

## 🎯 Model Comparison

| Feature | FLUX.1 | SDXL | SD 1.5 |
|---------|--------|------|--------|
| **Quality** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Speed** | ⚡⚡⚡⚡⚡ | ⚡⚡⚡ | ⚡⚡⚡⚡ |
| **Photorealism** | Excellent | Good | Fair |
| **Artistic Styles** | Good | Excellent | Very Good |
| **Text Rendering** | Excellent | Good | Poor |
| **Hands/Faces** | Excellent | Good | Fair |

## 🛟 Troubleshooting

### **"Model Not Available" Error**
- Check that model files are in the correct ComfyUI folders
- Verify workflow JSON files exist
- Use `/model_info` to see what's detected

### **Out of Memory Errors**
- Enable CPU mode in ComfyUI settings
- Try smaller batch sizes
- Use SD 1.5 for lower memory usage

### **Slow Generation**
- Check GPU utilization in Task Manager  
- Enable GPU optimization in ComfyUI
- Consider upgrading to FLUX.1 (actually faster than SDXL!)

### **Poor Quality Images**
- Try FLUX.1 for photorealism
- Use style presets with `/generate_style`
- Add quality terms: "masterpiece, best quality, highly detailed"

## 🎉 You're Ready!

Your DuckBot now has **state-of-the-art image generation** with multiple models! 

Start with `/model_info` to see what's available, then try `/generate_advanced` with different models to see the quality differences.

**Pro Tip:** FLUX.1 Schnell is often faster AND higher quality than other models - it's worth the download! 🌟