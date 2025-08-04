# 🦆 DuckBot v2.2 Release Notes

## 🎉 Major New Features

### 🎲 **Multiplayer Adventure System**
- **Party Mode**: Up to 6 players can join the same adventure
- **Collaborative Storytelling**: Shared narrative progression for all party members
- **Auto-Matching**: Easy `/join_adventure` command to find and join parties
- **Individual Characters**: Each player gets their own stats and character progression

### 🎯 **D&D Adventure Theme**
- **Full D&D Mechanics**: Complete 6-stat system (STR/DEX/CON/INT/WIS/CHA)
- **Real Dice Rolls**: d20 system with proper modifiers and difficulty classes
- **Smart Skill Detection**: Actions automatically trigger appropriate ability checks
- **Success Levels**: Critical Success, Great Success, Success, Failure, Critical Failure
- **Character Progression**: Level, HP, gold, equipment, and class system

### 🎮 **Basic Dice Mechanics for All Themes**
- **Universal Dice System**: All adventure themes now have dice mechanics
- **Simple Stats**: Luck and Skill stats (1d6) for non-D&D themes
- **Action Resolution**: Dice rolls determine success/failure of player actions
- **Balanced Gameplay**: Appropriate difficulty curves for each theme

### 🛠️ **Enhanced ComfyUI Integration**
- **Fixed WebSocket Issues**: Resolved timeout compatibility problems
- **Better Error Handling**: Improved connection stability and error reporting
- **Health Monitoring**: `/comfyui_health` command for system diagnostics
- **Image Upload Fix**: Resolved Discord image delivery issues

## 🔧 **Technical Improvements**

### 📊 **Database Architecture**
- **Player-Story Separation**: Individual Player nodes linked to shared Story nodes
- **Party Management**: Proper tracking of party membership and limits
- **Character Persistance**: Individual character data storage per player
- **Server Isolation**: All multiplayer features maintain server-specific data

### 🎭 **Adventure System Overhaul**
- **AI-Powered Stories**: Dynamic story generation using LM Studio
- **Rich Fallback System**: 45+ themed story elements across multiple phases
- **Progressive Difficulty**: Story complexity increases as adventures develop
- **Context Awareness**: Stories remember and build on previous player actions

### 🎲 **Dice System Implementation**
- **Smart Action Parsing**: Automatic detection of action types for appropriate rolls
- **Stat-Based Modifiers**: Character stats affect dice roll outcomes
- **Difficulty Scaling**: Different challenges require different skill thresholds
- **Success Narratives**: Outcomes vary based on roll results and success levels

## 📋 **New Commands**

- **`/join_adventure`** - Join an existing party adventure
- **`/adventure_status`** - Enhanced status with character sheets and party info
- **`/comfyui_health`** - Check ComfyUI connection and model availability

## 🔄 **Enhanced Commands**

- **`/start_adventure`** - Now supports party mode with `party_mode: true`
- **`/continue_adventure`** - Now includes dice mechanics for all themes
- **`/adventure_status`** - Shows character stats, party members, and dice results
- **`/end_adventure`** - Works for both solo and party adventures

## 🎮 **Adventure Themes**

### 🎲 **D&D (New!)**
- Full tabletop RPG experience with dice mechanics
- Character creation with random ability scores
- Classic D&D setting with quests, taverns, and dungeons

### 🎯 **Enhanced Basic Themes**
- **Fantasy**: Dice-driven medieval adventures
- **Sci-Fi**: Space exploration with skill checks
- **Mystery**: Investigation with luck and skill rolls
- **Cyberpunk**: High-tech adventures with dice mechanics
- **Horror**: Survival horror with tension-building dice
- **Steampunk**: Victorian-era adventures with skill tests

## 🚀 **Performance & Stability**

- **WebSocket Stability**: Fixed timeout compatibility issues
- **Image Generation**: Resolved Discord upload failures
- **Command Sync**: Enhanced global command synchronization
- **Database Efficiency**: Optimized queries for multiplayer scenarios
- **Error Handling**: Improved fallback systems for AI failures

## 📊 **Statistics**

- **Total Commands**: 32 (up from 28)
- **Adventure Themes**: 7 (including new D&D theme)
- **Max Party Size**: 6 players per adventure
- **Database Nodes**: Story + Player architecture
- **Dice Systems**: d20 (D&D) + d6 (Basic themes)

## 🎯 **Coming Soon**

- Character class selection for D&D adventures
- Equipment and item management systems
- Adventure save/load functionality
- Cross-server party adventures
- Custom adventure creation tools

---

## 🎮 **How to Use New Features**

### Start a Party Adventure:
```
/start_adventure dnd true
```

### Join Someone's Adventure:
```
/join_adventure
```

### Take Actions with Dice:
```
/continue_adventure attack the goblin with my sword
🎲 Attack Check: 15 + 3 = 18
✨ Great Success! Your sword strikes true...
```

### Check Your Character:
```
/adventure_status
```

---

**🦆 DuckBot v2.2 - Now with multiplayer adventures and D&D mechanics!**

*Compatible with Python 3.9+, Discord.py 2.3+, Neo4j 5.12+*