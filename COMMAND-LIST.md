# 🦆 DuckBot v2.1 Multi-Server - Complete Command List

## 🔧 **Basic Commands**
- **`/ping`** - Test if the bot is responsive
  - Simple "Pong!" response to verify bot is working

## 🤖 **AI Chat Commands**
- **`/ask`** - Enhanced AI chat with plugins and retry logic
  - Uses LM Studio with web search, Wikipedia, code sandbox, etc.
  - Automatic fallback if plugins fail
  - Server-aware responses
- **`/ask_simple`** - Simple AI chat without plugins (more reliable)
  - Basic LM Studio connection without any plugins
  - Use if `/ask` is having issues
- **`/lm_health`** - Check LM Studio connection health
  - Tests basic connection and plugin functionality
  - Shows connection status and troubleshooting tips

## 🎨 **Image Generation Commands**
- **`/generate`** - Generate images using ComfyUI (Stable Diffusion 1.5)
  - Basic image generation with queue system
  - Uses default SD 1.5 workflow
- **`/generate_advanced`** - Choose specific image models
  - 🌟 **FLUX.1 (Best Quality)** - Ultra-fast, photorealistic (2-4s)
  - 🎨 **SDXL (Versatile)** - High-quality art and creative styles (10-15s)
  - 🚀 **SD 1.5 (Fast)** - Fast and reliable fallback (5-8s)
  - 🎯 **Auto (Best Available)** - Automatically picks best model
- **`/generate_style`** - Apply artistic style presets
  - 📸 Photorealistic - Professional photography style
  - 🎨 Digital Art - Concept art and digital painting
  - 🖼️ Oil Painting - Classical painting style
  - ✏️ Pencil Sketch - Hand-drawn sketches
  - 🌸 Anime Style - Japanese animation style
  - 🏛️ Classical Art - Museum-quality classical painting
  - 🌟 Fantasy - Magical and mystical themes
  - 🤖 Cyberpunk - Futuristic neon aesthetic
- **`/model_info`** - View available image generation models
  - Shows which models are installed and their capabilities
  - Provides download links and requirements

## 🎬 **Video Generation Commands**
- **`/animate`** - Generate videos using ComfyUI and WAN 2.2
  - 10-second videos at 24 FPS (240 frames)
  - Uses WAN 2.2 text-to-video model
  - Queue system with position tracking

## 📊 **Analytics & Social Commands**
- **`/server_stats`** - Get social analytics about this Discord server
  - Total messages, users, knowledge entries
  - Most active users in the server
  - Server-specific data only
- **`/my_connections`** - Find users with similar interests (private)
  - Shows users with shared interests in this server
  - Based on Neo4j relationship analysis
- **`/channel_insights`** - Get insights about current channel activity
  - Message counts, unique users, activity patterns
  - Channel-specific analytics

## 🧠 **Knowledge Management Commands**
- **`/learn`** - Teach DuckBot something new (server-specific)
  - Stores information in Neo4j database
  - Categories: fact, how-to, definition, concept, etc.
  - Server-isolated knowledge base
- **`/ask_knowledge`** - Query this server's knowledge base
  - Search through server-specific learned information
  - Shows related concepts and contributors

## 💾 **Personal Memory Commands**
- **`/remember`** - Store something for DuckBot to remember about you (private)
  - Personal memories stored per server
  - Private to you and server-specific
- **`/my_context`** - See what DuckBot remembers about you (private)
  - View your stored memories in this server
  - Shows up to 10 most recent memories

## 🎮 **Interactive Adventure Commands**
- **`/start_adventure`** - Begin an interactive text adventure
  - Choose themes: fantasy, sci-fi, mystery, etc.
  - Server-specific adventure state
- **`/continue_adventure`** - Continue your adventure
  - Make choices to progress your story
  - Dynamic story generation based on your actions

## 💡 **Creative Tools Commands**
- **`/save_idea`** - Store a creative idea
  - Categories: art, story, game, invention, etc.
  - Automatic tag extraction for organization
  - Server-specific idea storage
- **`/random_idea_combo`** - Get random creative inspiration
  - Combines themes, objects, actions, and moods
  - Great for brainstorming and creative blocks

## 🔧 **Server Management Commands**
- **`/server_info`** - Get information about this server's DuckBot setup
  - Shows server configuration and enabled features
- **`/server_config`** - Configure server settings (Admin only)
  - Enable/disable features per server
  - Requires "Manage Server" permissions
- **`/global_stats`** - View DuckBot statistics across all servers (Global Admin only)
  - Cross-server statistics for bot administrators
  - Requires Global Admin permissions

## 💾 **Database Management Commands**
- **`/storage_status`** - Check Neo4j database storage usage (private)
  - Shows node/relationship counts
  - Estimated storage usage vs 10GB limit
  - Health status indicators
- **`/force_cleanup`** - Manually trigger database cleanup (Admin only)
  - Removes messages older than 90 days
  - Requires "Manage Server" permissions
  - Helps manage storage usage

## 🏷️ **Command Categories Summary**

### **🟢 Public Commands (everyone can see responses):**
- `/ping`, `/ask`, `/generate`, `/generate_advanced`, `/generate_style`, `/animate`
- `/model_info`, `/server_stats`, `/channel_insights`, `/ask_knowledge`
- `/start_adventure`, `/continue_adventure`, `/random_idea_combo`

### **🟡 Private Commands (only you see responses):**
- `/ask_simple`, `/lm_health`, `/my_connections`, `/remember`, `/my_context`
- `/save_idea`, `/storage_status`

### **🔴 Admin-Only Commands:**
- `/server_config` - Requires "Manage Server" permissions
- `/force_cleanup` - Requires "Manage Server" permissions  
- `/global_stats` - Requires Global Admin ID in environment

## 🌟 **Key Features**

### **🔄 Queue System**
- All generation commands use intelligent queuing
- Shows position in queue and estimated wait times
- Per-server queue isolation

### **🗄️ Server Isolation**
- All data is server-specific (no cross-contamination)
- Knowledge, memories, adventures are per-server
- Analytics only show data from current server

### **🛡️ Error Handling**
- Robust retry logic for LM Studio connections
- Graceful fallbacks when services are unavailable
- Helpful error messages with suggestions

### **💾 Storage Management**
- 10GB database storage limit with automatic monitoring
- Cleanup commands for database maintenance
- Smart data retention policies

### **🎯 Smart Features**
- Auto model selection for best available quality
- Progressive plugin fallbacks for AI chat
- Context-aware responses with server and user information

## 🚀 **Getting Started**

1. **Basic Test**: `/ping` - Verify bot is working
2. **AI Chat**: `/ask` - Talk to the AI assistant  
3. **Image Generation**: `/generate_advanced` - Create images with model choice
4. **Knowledge**: `/learn` - Teach the bot something new
5. **Analytics**: `/server_stats` - See server activity data

## 📋 **Notes**

- Most commands require Neo4j database for data storage
- Image/video generation requires ComfyUI server running
- AI chat requires LM Studio with loaded model
- Some features are server-specific and isolated
- Admin commands require appropriate Discord permissions

---

**Total Commands: 24** | **Multi-Server Support** ✅ | **Queue System** ✅ | **Database Storage** ✅