# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DuckBot is a multi-server Discord bot with AI chat, image/video generation, social analytics, and interactive adventures. The bot integrates with ComfyUI for content generation, LM Studio for AI chat, and Neo4j for analytics and memory storage.

## Core Architecture

### Main Components

- **DuckBot-v2.2.3-MULTI-SERVER-Enhanced-Adventures.py**: Latest production bot with ultra-high quality image generation, improved error handling, and enhanced VAE support
- **enhanced_ask_command.py**: LM Studio AI chat with progressive fallback system
- **enhanced_image_generation.py**: Multi-model image generation (FLUX.1, SDXL, SD1.5)
- **enhanced_image_integration.py**: Advanced image generation with style presets
- **workflow_*.json**: ComfyUI workflow configurations for different models

### Key Dependencies

```python
# Core dependencies from requirements.txt
discord.py>=2.3.0     # Discord bot framework
python-dotenv>=1.0.0  # Environment variable management
requests>=2.31.0      # HTTP requests
aiohttp>=3.8.0        # Async HTTP client
websockets>=11.0.0    # WebSocket communication with ComfyUI
torch>=2.0.0          # PyTorch for random seed generation
neo4j>=5.12.0         # Neo4j database driver
opencv-python>=4.8.0  # Video processing for MP4 creation
Pillow>=10.0.0        # Image processing
```

### Bot Architecture

The bot uses:
- **Discord.py** with slash commands (`@bot.tree.command`)
- **Server isolation** - All data (memories, knowledge, analytics) is per-server
- **Progressive fallback** - AI systems gracefully degrade when services unavailable
- **Queue systems** - Both image and video generation use intelligent queuing
- **Multi-model support** - FLUX.1 Schnell, SDXL, SD1.5 with automatic fallback
- **WebSocket monitoring** - Real-time progress tracking for ComfyUI workflows
- **Enhanced memory** - Personal memories and server knowledge bases via Neo4j

## Development Commands

### Environment Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Configure `.env` file:
   ```env
   # Discord (REQUIRED)
   DISCORD_TOKEN=your_discord_bot_token
   
   # Neo4j (OPTIONAL - for analytics/memory)
   NEO4J_ENABLED=true
   NEO4J_URI=neo4j://127.0.0.1:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=your_password
   
   # ComfyUI (OPTIONAL - for image/video generation)
   COMFYUI_URL=http://127.0.0.1:8188
   
   # LM Studio (OPTIONAL - for AI chat)
   LM_STUDIO_URL=http://localhost:1234/v1/chat/completions
   ```

### Running the Bot
```bash
python DuckBot-v2.2.3-MULTI-SERVER-Enhanced-Adventures.py
```

### Service Dependencies

**ComfyUI** (Image/Video Generation):
- Runs on `127.0.0.1:8188`
- Models required: FLUX.1 Schnell, SDXL 1.0, SD 1.5, WAN 2.2
- Workflow files: `workflow_flux_api.json`, `workflow_sdxl_api.json`, `workflow_api.json`, `workflow_video_api.json`

**LM Studio** (AI Chat):
- Runs on `localhost:1234`
- Supports plugins: duckduckgo, web-search, wikipedia, dice, js-code-sandbox
- Progressive fallback when plugins fail

**Neo4j** (Analytics/Memory):
- Runs on `localhost:7687`
- Stores user interactions, memories, knowledge base
- Automatic cleanup at 10GB limit

## Multi-Model Image Generation

### Model Hierarchy
1. **FLUX.1 Schnell** - Primary (ultra-fast, photorealistic, 2-4s)
2. **SDXL 1.0** - Secondary (versatile, high-quality, 10-15s) 
3. **SD 1.5** - Fallback (reliable, fast, 5-8s)

### Workflow Node Structure
- **workflow_api.json** (SD 1.5): Prompt in node "3", seed in node "6"
- **workflow_sdxl_api.json** (SDXL): Prompt in node "3", seed in node "6" 
- **workflow_flux_api.json** (FLUX.1): Prompt in node "3", seed in node "6"
- **workflow_video_api.json** (WAN 2.2): Prompt in node "6", seed in node "9"

## Command Categories (28 total)

### AI Chat (4 commands)
- `/ask` - Enhanced AI with plugins and fallback
- `/ask_simple` - Basic AI without plugins
- `/ask_enhanced` - AI with personal memory
- `/lm_health` - Connection diagnostics

### Image Generation (4 commands)
- `/generate` - Basic generation (SD 1.5)
- `/generate_advanced` - Model selection (FLUX.1/SDXL/SD1.5/Auto)
- `/generate_style` - Style presets (photorealistic, anime, cyberpunk, etc.)
- `/model_info` - Available models and capabilities

### Video Generation (1 command)
- `/animate` - 10-second videos with WAN 2.2

### Analytics (3 commands)
- `/server_stats` - Server activity statistics
- `/my_connections` - User similarity analysis
- `/channel_insights` - Channel activity patterns

### Knowledge & Memory (4 commands)
- `/learn` - Teach bot information (server-specific)
- `/ask_knowledge` - Query knowledge base
- `/remember` - Store personal memories
- `/my_context` - View stored memories

### Interactive Adventures (4 commands)
- `/start_adventure` - Begin text adventures (fantasy, sci-fi, D&D, etc.)
- `/join_adventure` - Join multiplayer adventures
- `/continue_adventure` - Progress story with dice mechanics
- `/adventure_status` - View character and party status

### Creative Tools (2 commands)
- `/save_idea` - Store creative concepts
- `/random_idea_combo` - Generate inspiration

### Management (5 commands - Admin only)
- `/server_config` - Configure server settings
- `/storage_status` - Database usage monitoring
- `/force_cleanup` - Manual cleanup
- `/global_stats` - Cross-server statistics
- `/server_info` - Server configuration info

## Error Handling & Fallbacks

### AI Chat System
- Primary: LM Studio with all plugins
- Fallback 1: LM Studio with safe plugins only (duckduckgo, wikipedia, dice)
- Fallback 2: LM Studio without plugins
- Fallback 3: Static responses with guidance

### Image Generation
- Model selection follows priority: FLUX.1 → SDXL → SD 1.5
- Automatic workflow switching based on availability
- Queue system prevents overload

### Database Operations
- Neo4j optional - features gracefully disable if unavailable
- Automatic connection retry with exponential backoff
- Storage cleanup prevents database bloat

## Important Development Notes

- **Server Isolation**: All user data is completely isolated per Discord server
- **Queue Management**: Both image and video generation use intelligent queuing with position tracking
- **WebSocket Handling**: ComfyUI integration uses proper WebSocket lifecycle management
- **Memory Management**: Neo4j auto-cleanup prevents storage issues
- **Client ID Management**: Each ComfyUI request uses unique CLIENT_ID for tracking
- **Discord Interaction Handling**: Proper deferral for long-running operations (image/video generation)
- **Progressive Enhancement**: All features work independently - missing services don't break core functionality

## File Structure

```
DiscordBot/
├── DuckBot-v2.2.2-MULTI-SERVER-Enhanced-Adventures.py  # Main production bot
├── enhanced_ask_command.py                              # AI chat with fallback system  
├── enhanced_image_generation.py                         # Multi-model image generation
├── enhanced_image_integration.py                        # Style presets and advanced features
├── workflow_api.json                                    # SD 1.5 workflow
├── workflow_sdxl_api.json                              # SDXL workflow
├── workflow_flux_api.json                              # FLUX.1 workflow
├── workflow_video_api.json                             # WAN 2.2 video workflow
├── requirements.txt                                     # Dependencies
├── .env                                                # Environment configuration
└── README.md                                           # Complete setup guide
```