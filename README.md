# 🦆 DuckBot v2.2 Multi-Server Edition

**Advanced Discord Bot with AI Chat, Image/Video Generation, Analytics & Multiplayer Adventures**

![Discord](https://img.shields.io/badge/Discord-Bot-7289da?style=for-the-badge&logo=discord)
![Python](https://img.shields.io/badge/Python-3.9+-3776ab?style=for-the-badge&logo=python)
![Neo4j](https://img.shields.io/badge/Neo4j-Database-4581c3?style=for-the-badge&logo=neo4j)

---

## 🌟 Features

### 🤖 **AI Chat System**
- **LM Studio Integration** - Local AI chat with plugin support
- **Progressive Fallback** - Automatic retry logic when plugins fail
- **Enhanced Memory** - Server-specific memory storage
- **Web Search & Tools** - DuckDuckGo, Wikipedia, code sandbox plugins

### 🎨 **Advanced Image Generation**
- **Multiple AI Models:**
  - 🌟 **FLUX.1 Schnell** - Ultra-fast, photorealistic (2-4s)
  - 🎨 **SDXL 1.0** - High-quality, versatile (10-15s)
  - 🚀 **SD 1.5** - Fast, reliable fallback (5-8s)
- **Style Presets** - Photorealistic, Digital Art, Oil Painting, Anime, etc.
- **Smart Queue System** - Per-server processing with position tracking

### 🎬 **Video Generation**
- **WAN 2.2 Integration** - Text-to-video generation
- **240-frame Videos** - 10-second clips at 24 FPS
- **Queue Management** - Intelligent processing across servers

### 📊 **Analytics & Social Features**
- **Neo4j Database** - Graph-based user relationships
- **Server Analytics** - Message counts, active users, insights
- **Social Connections** - Find users with similar interests
- **Knowledge Base** - Server-specific learning system

### 🧠 **Memory & Learning**
- **Personal Memory** - Bot remembers user preferences
- **Knowledge Storage** - Teach the bot facts and concepts
- **Adventure Mode** - Interactive text adventures
- **Creative Tools** - Idea storage and inspiration

---

## 🚀 Quick Start

### 1. **Bot Invitation** 🤖

**Your bot's Client ID: `1397321689830002768`**

**🔗 Quick Invite URL (copy & paste in browser):**
```
https://discord.com/api/oauth2/authorize?client_id=1397321689830002768&permissions=8&scope=bot%20applications.commands
```

**⚠️ CRITICAL - Before inviting, enable these in Discord Developer Portal:**
1. Go to https://discord.com/developers/applications
2. Select your DuckBot app (ID: 1397321689830002768)
3. Go to **Bot** section
4. Enable **MESSAGE CONTENT INTENT** ✅
5. Enable **SERVER MEMBERS INTENT** ✅

**Common invitation failure causes:**
- Missing MESSAGE CONTENT INTENT (most common)
- You don't have "Manage Server" permission on target server
- Bot is set to private instead of public

### 2. **First Test**
After invitation:
- Use `/ping` to verify bot is responding
- Try `/lm_health` to check AI connections
- Test image generation with `/generate hello world`

---

## 🛠️ Installation & Setup

### Prerequisites
- **Python 3.9+**
- **Discord Developer Account**
- **Neo4j Database** (optional, for analytics)
- **LM Studio** (optional, for AI chat)
- **ComfyUI** (optional, for image/video generation)

### 1. **Clone & Install Dependencies**
```bash
git clone <your-repo>
cd DiscordBot
pip install -r requirements.txt
```

### 2. **Environment Configuration**
Create `.env` file:
```env
# Discord Configuration (REQUIRED)
DISCORD_TOKEN="your_discord_bot_token_here"

# Neo4j Database (OPTIONAL - for analytics/memory)
NEO4J_ENABLED=true
NEO4J_URI=neo4j://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# ComfyUI (OPTIONAL - for image/video generation)
COMFYUI_URL=http://127.0.0.1:8188

# LM Studio (OPTIONAL - for AI chat)
LM_STUDIO_URL=http://localhost:1234/v1/chat/completions
```

---

## 🔧 Service Connections

### **Discord Bot Setup**
1. **Create Application:**
   - Go to https://discord.com/developers/applications
   - Click "New Application" → Name it "DuckBot"
   - Copy Application ID: `1397321689830002768`

2. **Bot Configuration:**
   - Go to "Bot" section
   - Click "Reset Token" → Copy to `DISCORD_TOKEN` in .env
   - Enable **MESSAGE CONTENT INTENT** ⚠️ **CRITICAL**
   - Enable **SERVER MEMBERS INTENT** ⚠️ **CRITICAL**
   - Set bot to "Public" if others should invite it

3. **OAuth2 URL Generator:**
   - Select scopes: `bot` + `applications.commands`
   - Select permissions: `Administrator` (or specific permissions)

### **Neo4j Database Setup** (Optional)
```bash
# Install Neo4j Desktop or use Docker
docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest

# Or install Neo4j Desktop:
# https://neo4j.com/download/
```

**Features enabled with Neo4j:**
- `/server_stats` - Server analytics
- `/my_connections` - Social connections
- `/learn` / `/ask_knowledge` - Knowledge base
- `/remember` / `/my_context` - Personal memory
- Message/user analytics and storage

### **LM Studio Setup** (Optional)
1. **Download:** https://lmstudio.ai/
2. **Load Model:** Download and load any chat model (recommend 7B-13B models)
3. **Start Server:** Local Server → Start Server
4. **Plugin Configuration:** 
   - Basic plugins: `dice`
   - Advanced: `duckduckgo`, `wikipedia`, `web-search`

**Features enabled with LM Studio:**
- `/ask` - Enhanced AI chat with plugins
- `/ask_simple` - Basic AI chat
- `/ask_enhanced` - AI with personal memory
- `/lm_health` - Connection diagnostics

### **ComfyUI Setup** (Optional)
1. **Install ComfyUI:** https://github.com/comfyanonymous/ComfyUI
2. **Download Models:**
   - **FLUX.1 Schnell:** `flux1-schnell.safetensors` + `t5xxl_fp16.safetensors`
   - **SDXL 1.0:** `sd_xl_base_1.0.safetensors` + `sd_xl_refiner_1.0.safetensors`
   - **SD 1.5:** `v1-5-pruned-emaonly.ckpt`
3. **Workflow Files:** Ensure these exist:
   - `workflow_api.json` (SD 1.5)
   - `workflow_sdxl_api.json` (SDXL)
   - `workflow_flux_api.json` (FLUX.1)
   - `workflow_video_api.json` (WAN 2.2)

**Features enabled with ComfyUI:**
- `/generate` - Basic image generation
- `/generate_advanced` - Model selection
- `/generate_style` - Style presets
- `/animate` - Video generation
- `/model_info` - Available models

---

## 📋 Command Reference

### **🔧 Basic Commands**
- `/ping` - Test bot responsiveness

### **🤖 AI Chat** (Requires LM Studio)
- `/ask` - Enhanced AI with plugins & fallback
- `/ask_simple` - Reliable AI without plugins
- `/ask_enhanced` - AI with personal memory
- `/lm_health` - Check LM Studio connection

### **🎨 Image Generation** (Requires ComfyUI)
- `/generate` - Basic image generation (SD 1.5)
- `/generate_advanced` - Choose model (FLUX.1/SDXL/SD1.5)
- `/generate_style` - Apply artistic styles
- `/model_info` - View available models

### **🎬 Video Generation** (Requires ComfyUI + WAN 2.2)
- `/animate` - Generate 10-second videos

### **📊 Analytics** (Requires Neo4j)
- `/server_stats` - Server activity statistics
- `/my_connections` - Find similar users
- `/channel_insights` - Channel activity analysis

### **🧠 Knowledge & Memory** (Requires Neo4j)
- `/learn` - Teach bot new information
- `/ask_knowledge` - Query knowledge base
- `/remember` - Store personal memories
- `/my_context` - View stored memories

### **🎮 Interactive Features**
- `/start_adventure` - Begin text adventure
- `/continue_adventure` - Progress adventure
- `/save_idea` - Store creative ideas
- `/random_idea_combo` - Get inspiration

### **🔧 Management** (Admin Only)
- `/server_config` - Configure server settings
- `/storage_status` - Check database usage
- `/force_cleanup` - Clean old data
- `/global_stats` - Cross-server statistics

---

## 🏃‍♂️ Running the Bot

### **Start the Bot**
```bash
cd DiscordBot
python DuckBot-v2.1-MULTI-SERVER.py
```

### **Expected Startup Output**
```
Starting DuckBot v2.1 Multi-Server Edition...
✅ Neo4j connected successfully to neo4j://127.0.0.1:7687
📊 Analytics, memory, and social features enabled!
✅ Enhanced ask system loaded!
✅ Enhanced image generation system loaded!
📋 New commands available:
   • /generate_advanced - Choose specific models
   • /model_info - View available models
   • /generate_style - Apply artistic styles
🔄 Your existing /generate command will continue to work as before!

--- DuckBot v2.1 Multi-Server Command Summary ---
✅ Total Commands Available: 28
   • Basic: ping (1)
   • AI Chat: ask, ask_simple, ask_enhanced, lm_health (4)
   • Image Generation: generate, generate_advanced, generate_style, model_info (4)
   • Video Generation: animate (1)
   • Analytics: server_stats, my_connections, channel_insights (3)
   • Knowledge: learn, ask_knowledge (2)
   • Memory: remember, my_context (2)
   • Interactive: start_adventure, continue_adventure, save_idea, random_idea_combo (4)
   • Management: server_info, server_config, storage_status, force_cleanup, global_stats (5)
   • Enhanced Features: ask_enhanced, generate_advanced (2)

🦆 DuckBot v2.1 Multi-Server is ready! All enhanced features loaded.
2024-12-XX XX:XX:XX INFO     discord.client Logging in using static token
2024-12-XX XX:XX:XX INFO     discord.gateway Shard ID None has connected to Gateway
🦆 DuckBot is online and ready!
🔄 Synced 28 slash commands globally
```

---

## 🔍 Troubleshooting

### **Bot Won't Join Server**
1. **Check Intents:** MESSAGE CONTENT INTENT must be enabled
2. **Verify Permissions:** You need "Manage Server" on target server
3. **Try Incognito:** Use private browser window for invite URL
4. **Check Bot Settings:** Ensure bot is set to "Public"

### **Commands Not Appearing**
1. **Wait 1 hour:** Slash commands can take time to sync globally
2. **Try Refresh:** Restart Discord client
3. **Check Permissions:** Bot needs "Use Slash Commands" permission

### **LM Studio Issues**
- Use `/lm_health` to diagnose connection issues
- Try `/ask_simple` for reliable chat without plugins
- Check LM Studio server is running on localhost:1234

### **ComfyUI Issues**
- Verify ComfyUI is running on localhost:8188
- Check workflow JSON files exist
- Use `/model_info` to see available models

### **Neo4j Issues**
- Check database is running on localhost:7687
- Verify credentials in .env file
- Some features will be disabled if Neo4j unavailable

---

## 📁 File Structure

```
DiscordBot/
├── DuckBot-v2.1-MULTI-SERVER.py    # Main bot file
├── .env                             # Environment configuration
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── COMMAND-LIST.md                  # Detailed command reference
├── INVITE-URL.md                    # Bot invitation instructions
├── workflow_api.json                # SD 1.5 workflow
├── workflow_sdxl_api.json           # SDXL workflow  
├── workflow_flux_api.json           # FLUX.1 workflow
├── workflow_video_api.json          # WAN 2.2 video workflow
└── enhanced_*.py                    # Additional feature modules
```

---

## 🤝 Support & Development

### **Feature Requests**
DuckBot v2.1 includes 28 slash commands across multiple categories. All major features are implemented with robust error handling and multi-server support.

### **Server Isolation**
All data (memories, knowledge, analytics) is completely isolated per server. No cross-contamination between different Discord servers.

### **Performance**
- **Image Generation:** 2-15 seconds depending on model
- **Video Generation:** 2-5 minutes for 10-second clips
- **AI Chat:** 1-5 seconds with fallback support
- **Database:** Efficient Neo4j graph queries with 10GB limit

### **Security**
- Environment variables for all sensitive data
- No hardcoded tokens or credentials
- Server-isolated data storage
- Automatic cleanup of old data

---

## 📊 System Requirements

### **Minimum Setup (Basic Chat Bot)**
- Discord Bot Token
- Python 3.9+
- 1GB RAM

### **Recommended Setup (All Features)**
- Discord Bot Token ✅
- Neo4j Database (analytics/memory) ✅
- LM Studio + AI Model (chat) ✅
- ComfyUI + Models (image/video) ✅
- 8GB RAM, GPU recommended for AI generation

### **Storage Requirements**
- **Neo4j Database:** Up to 10GB (automatic cleanup)
- **AI Models:** 2-15GB depending on models chosen
- **ComfyUI Output:** Temporary files, auto-cleaned

---

**🦆 DuckBot v2.1 Multi-Server - Ready to quack in your Discord server!**

*Built with ❤️ for the Discord community*