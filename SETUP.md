# DuckBot Portable Setup Guide

## 🚀 Quick Setup Instructions

### Prerequisites
1. **Python 3.8+** installed on your system
2. **Discord Bot Token** from Discord Developer Portal
3. **ComfyUI** running on local machine (for image/video generation)
4. **LM Studio** (optional, for AI chat features)

### Setup Steps

#### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 2. Configure Environment
1. Copy `.env.example` to `.env`
2. Edit `.env` and add your Discord bot token:
   ```
   DISCORD_TOKEN=your_discord_bot_token_here
   NEO4J_ENABLED=false
   ```

#### 3. Start Required Services

**ComfyUI Server:**
- Download ComfyUI from: https://github.com/comfyanonymous/ComfyUI
- Install WAN 2.2 model for video generation
- Start ComfyUI server on `127.0.0.1:8188`

**LM Studio (Optional):**
- Download from: https://lmstudio.ai/
- Load a model and start server on `localhost:1234`
- Enable plugins: duckduckgo, web-search, visit-website, wikipedia, rag-v1, js-code-sandbox, query-neo4j, dice

#### 4. Run DuckBot
```bash
python DuckBot-v1.2VID.py
```

## 🎯 Bot Commands

### Core Features
- `/ping` - Test bot connectivity
- `/generate <prompt>` - Generate AI images (requires ComfyUI)
- `/animate <prompt>` - Generate AI videos (requires ComfyUI + WAN 2.2)
- `/ask <prompt>` - Chat with AI (requires LM Studio)

### Queue System
- Automatic queuing for generation requests
- Real-time position tracking
- Wait time estimates

## 🛠️ Optional: Neo4j Social Analytics

To enable social analytics features:

1. **Install Neo4j Desktop:** https://neo4j.com/download/
2. **Create database** with credentials: `neo4j` / `password`
3. **Update .env:**
   ```
   NEO4J_ENABLED=true
   ```
4. **Restart bot** - analytics commands will be available:
   - `/server_stats` - Server social analytics
   - `/my_connections` - Find similar users
   - `/channel_insights` - Channel activity patterns
   - `/storage_status` - Database health
   - `/force_cleanup` - Manual cleanup (admin only)

## 📁 File Structure

- `DuckBot-v1.2VID.py` - Main bot file
- `workflow_api.json` - Image generation workflow
- `workflow_video_api.json` - Video generation workflow  
- `requirements.txt` - Python dependencies
- `.env.example` - Environment template
- `CLAUDE.md` - Developer documentation

## ⚠️ Troubleshooting

**Bot won't start:**
- Check Discord token in `.env`
- Ensure Python dependencies installed

**Generation fails:**
- Verify ComfyUI is running on port 8188
- Check that required models are installed

**Neo4j errors:**
- Set `NEO4J_ENABLED=false` in `.env` to disable
- Or install Neo4j Desktop and create database

## 🎊 You're Ready!

Your DuckBot should now be running with image generation, video generation, and AI chat capabilities!