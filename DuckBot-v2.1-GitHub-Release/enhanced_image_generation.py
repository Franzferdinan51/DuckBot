# Enhanced Image Generation with Multiple Models
# Add this to your DuckBot for better image generation

import json
import os
from typing import Dict, List, Optional

# Available image generation models
IMAGE_MODELS = {
    "flux": {
        "name": "FLUX.1 Schnell",
        "description": "⚡ Ultra-fast, highest quality photorealistic images",
        "workflow": "workflow_flux_api.json",
        "steps": 4,
        "cfg": 1.0,
        "resolution": "1024x1024",
        "speed": "Very Fast (2-4s)",
        "best_for": "Photorealism, portraits, detailed scenes"
    },
    "sdxl": {
        "name": "Stable Diffusion XL",
        "description": "🎨 High-quality versatile generation with refiner",
        "workflow": "workflow_sdxl_api.json", 
        "steps": 25,
        "cfg": 7.0,
        "resolution": "1024x1024",
        "speed": "Medium (10-15s)",
        "best_for": "Art, creative styles, general purpose"
    },
    "sd15": {
        "name": "Stable Diffusion 1.5",
        "description": "🚀 Fast, reliable, wide style compatibility",
        "workflow": "workflow_api.json",
        "steps": 20,
        "cfg": 8.0,
        "resolution": "512x512",
        "speed": "Fast (5-8s)",
        "best_for": "Quick generation, artistic styles"
    }
}

# Default model preference order
MODEL_PRIORITY = ["flux", "sdxl", "sd15"]

def get_available_models() -> List[str]:
    """Check which model workflows are available."""
    available = []
    for model_id, model_info in IMAGE_MODELS.items():
        workflow_path = model_info["workflow"]
        if os.path.exists(workflow_path):
            available.append(model_id)
    return available

def get_best_available_model() -> str:
    """Get the best available model based on priority."""
    available = get_available_models()
    for model_id in MODEL_PRIORITY:
        if model_id in available:
            return model_id
    return "sd15"  # Fallback

async def load_workflow_for_model(model_id: str) -> Optional[Dict]:
    """Load and return the workflow for a specific model."""
    if model_id not in IMAGE_MODELS:
        return None
        
    workflow_path = IMAGE_MODELS[model_id]["workflow"]
    
    try:
        with open(workflow_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading workflow for {model_id}: {e}")
        return None

def update_workflow_prompt(workflow: Dict, prompt: str, model_id: str) -> Dict:
    """Update workflow with new prompt, handling different model structures."""
    
    if model_id == "flux":
        # FLUX uses node 3 for positive prompt, node 7 for negative
        workflow["3"]["inputs"]["text"] = prompt
        # Enhanced negative prompt for FLUX
        workflow["7"]["inputs"]["text"] = "blurry, low quality, watermark, text, bad anatomy, deformed, noise, artifacts, oversaturated"
        
    elif model_id == "sdxl":
        # SDXL uses node 3 for positive prompt, node 7 for negative
        enhanced_prompt = f"{prompt}, masterpiece, best quality, highly detailed, sharp focus"
        workflow["3"]["inputs"]["text"] = enhanced_prompt
        workflow["7"]["inputs"]["text"] = "blurry, low quality, watermark, text, bad anatomy, deformed, pixelated, noise, artifacts, oversaturated, cartoon, anime"
        
    elif model_id == "sd15":
        # SD 1.5 uses node 3 for positive prompt, node 7 for negative
        workflow["3"]["inputs"]["text"] = prompt
        workflow["7"]["inputs"]["text"] = "text, watermark, blurry, low quality"
    
    return workflow

async def enhanced_process_image_generation(queue_item, model_id: str = None):
    """Enhanced image generation with model selection."""
    start_time = time.time()
    
    # Determine which model to use
    if not model_id:
        model_id = get_best_available_model()
    
    model_info = IMAGE_MODELS.get(model_id, IMAGE_MODELS["sd15"])
    
    try:
        # Load workflow for the selected model
        prompt_workflow = await load_workflow_for_model(model_id)
        
        if not prompt_workflow:
            await queue_item.status_message.edit(
                content=f"❌ Error loading {model_info['name']} workflow"
            )
            return
        
        # Update workflow with prompt and model-specific enhancements
        prompt_workflow = update_workflow_prompt(prompt_workflow, queue_item.prompt, model_id)
        
        # Update seed
        seed_node = "6"  # Most models use node 6 for KSampler
        prompt_workflow[seed_node]["inputs"]["seed"] = torch.randint(1, 1125899906842624, (1,)).item()
        
        # Update status with model info
        await queue_item.status_message.edit(
            content=f"🎨 **Generating with {model_info['name']}**\n"
                   f"Prompt: `{queue_item.prompt}`\n"
                   f"⚡ {model_info['speed']} | 📐 {model_info['resolution']}"
        )
        
        # Generate image
        images_data = await run_comfyui_workflow(prompt_workflow, is_video=False)
        
        if not images_data:
            await queue_item.status_message.edit(
                content=f"❌ **{model_info['name']} generation failed**\nPrompt: `{queue_item.prompt}`"
            )
            return
        
        # Create Discord files
        image_files = []
        for i, data in enumerate(images_data):
            filename = f"{model_id}_image_{uuid.uuid4()}.png"
            image_files.append(discord.File(fp=BytesIO(data), filename=filename))
        
        # Success message with model info
        embed = discord.Embed(
            title="✅ Image Generation Complete!",
            description=f"**Model:** {model_info['name']}\n**Prompt:** {queue_item.prompt}",
            color=0x00ff88
        )
        embed.add_field(name="⚡ Generation Time", value=f"{time.time() - start_time:.1f}s", inline=True)
        embed.add_field(name="📐 Resolution", value=model_info['resolution'], inline=True)
        embed.add_field(name="🎯 Best For", value=model_info['best_for'], inline=False)
        
        await queue_item.status_message.edit(content="", embed=embed)
        await queue_item.interaction.followup.send(files=image_files)
        
        # Update average time
        actual_time = time.time() - start_time
        update_average_time("image", actual_time)
        
        # Store artwork with model info if Neo4j enabled
        if NEO4J_ENABLED:
            await store_artwork(
                user_id=queue_item.interaction.user.id,
                prompt=queue_item.prompt,
                style=f"AI Generated - {model_info['name']}",
            )
        
    except Exception as e:
        await queue_item.status_message.edit(
            content=f"❌ **{model_info['name']} generation error**\n"
                   f"Prompt: `{queue_item.prompt}`\n"
                   f"Error: {str(e)[:100]}..."
        )

# Enhanced generate command with model selection
@bot.tree.command(name="generate_advanced", description="Generate images with model selection")
@app_commands.describe(
    prompt="The prompt for the image",
    model="Image generation model to use"
)
@app_commands.choices(model=[
    app_commands.Choice(name="🌟 FLUX.1 (Best Quality)", value="flux"),
    app_commands.Choice(name="🎨 SDXL (Versatile)", value="sdxl"), 
    app_commands.Choice(name="🚀 SD 1.5 (Fast)", value="sd15"),
    app_commands.Choice(name="🎯 Auto (Best Available)", value="auto")
])
async def generate_advanced_command(interaction: discord.Interaction, prompt: str, model: str = "auto"):
    await interaction.response.defer(ephemeral=False)
    
    # Determine model to use
    if model == "auto":
        selected_model = get_best_available_model()
    else:
        selected_model = model
    
    # Check if model is available
    available_models = get_available_models()
    if selected_model not in available_models:
        embed = discord.Embed(
            title="❌ Model Not Available",
            description=f"The {IMAGE_MODELS[selected_model]['name']} model is not available.",
            color=0xe74c3c
        )
        embed.add_field(
            name="📋 Available Models",
            value="\n".join([f"• {IMAGE_MODELS[m]['name']}" for m in available_models]),
            inline=False
        )
        embed.add_field(
            name="💡 Solution",
            value="Download the model files and place them in your ComfyUI models folder",
            inline=False
        )
        await interaction.followup.send(embed=embed)
        return
    
    # Create queue item with model specification
    queue_item = QueueItem(interaction, prompt, "image")
    queue_item.model_id = selected_model
    
    # Add to queue with model-specific processing
    async def process_with_model(queue_item):
        await enhanced_process_image_generation(queue_item, selected_model)
    
    # Add to server queue
    server_queue = get_server_queue(interaction.guild.id)
    async with server_queue['lock']:
        server_queue['queue'].append(queue_item)
        position = len(server_queue['queue'])
        
        model_info = IMAGE_MODELS[selected_model]
        
        if position == 1 and not server_queue['currently_processing']:
            queue_item.status_message = await interaction.followup.send(
                f"🎨 **Starting {model_info['name']} generation**\n"
                f"Prompt: `{prompt}`\n"
                f"🚀 Processing now... ({model_info['speed']})"
            )
        else:
            wait_time = calculate_estimated_wait(position, "image")
            wait_str = f"\n⏱️ Estimated wait: {format_time(wait_time)}" if wait_time > 0 else ""
            queue_item.status_message = await interaction.followup.send(
                f"🎨 **{model_info['name']} generation queued**\n"
                f"Prompt: `{prompt}`\n"
                f"📍 Position {position} in queue{wait_str}"
            )
    
    # Start processing if not already running
    if not server_queue['currently_processing']:
        asyncio.create_task(process_queue_with_models(interaction.guild.id))

async def process_queue_with_models(server_id: int):
    """Enhanced queue processing with model support."""
    server_queue = get_server_queue(server_id)
    
    async with server_queue['lock']:
        if server_queue['currently_processing'] or not server_queue['queue']:
            return
        server_queue['currently_processing'] = True
    
    while server_queue['queue']:
        async with server_queue['lock']:
            if not server_queue['queue']:
                break
            current_item = server_queue['queue'].popleft()
        
        try:
            # Process with specified model or auto-select
            model_id = getattr(current_item, 'model_id', get_best_available_model())
            await enhanced_process_image_generation(current_item, model_id)
        except Exception as e:
            model_name = IMAGE_MODELS.get(getattr(current_item, 'model_id', 'sd15'), {}).get('name', 'Unknown')
            await current_item.status_message.edit(
                content=f"❌ **{model_name} generation failed**\n"
                       f"Prompt: `{current_item.prompt}`\n"
                       f"Error: {str(e)[:100]}..."
            )
    
    server_queue['currently_processing'] = False

@bot.tree.command(name="model_info", description="View available image generation models")
async def model_info_command(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=False)
    
    available_models = get_available_models()
    
    embed = discord.Embed(
        title="🎨 Available Image Generation Models",
        description="Choose the best model for your needs!",
        color=0x3498db
    )
    
    for model_id in MODEL_PRIORITY:
        if model_id in IMAGE_MODELS:
            model_info = IMAGE_MODELS[model_id]
            status = "✅ Available" if model_id in available_models else "❌ Not Installed"
            
            embed.add_field(
                name=f"{model_info['name']} {status}",
                value=f"{model_info['description']}\n"
                      f"**Speed:** {model_info['speed']}\n"
                      f"**Resolution:** {model_info['resolution']}\n"
                      f"**Best For:** {model_info['best_for']}",
                inline=False
            )
    
    embed.add_field(
        name="🚀 Usage",
        value="Use `/generate_advanced` to select a model, or `/generate` for auto-selection",
        inline=False
    )
    
    await interaction.followup.send(embed=embed)