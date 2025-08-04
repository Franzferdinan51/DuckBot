# 📋 DuckBot v2.2 Changelog

## 🎉 **New Features**

### 🎲 **Multiplayer Adventure System**
- **Added** `/join_adventure` command for joining party adventures
- **Added** Party mode support in `/start_adventure` with `party_mode: true`
- **Added** Up to 6 players per adventure support
- **Added** Shared story progression for all party members
- **Added** Individual character creation for each party member

### 🎯 **D&D Adventure Theme**
- **Added** Full D&D theme with 6-stat system (STR/DEX/CON/INT/WIS/CHA)
- **Added** d20 dice system with proper modifiers and difficulty classes
- **Added** Smart skill detection for automatic ability checks
- **Added** Character progression with Level, HP, Gold, Equipment
- **Added** Critical success/failure mechanics

### 🎮 **Universal Dice Mechanics**
- **Added** Basic dice system for all non-D&D themes
- **Added** Luck and Skill stats (1d6) for simple themes
- **Added** Action-based dice roll triggers
- **Added** Success level narratives (Critical Success → Critical Failure)

### 🛠️**ComfyUI Enhancements**
- **Added** `/comfyui_health` command for connection diagnostics
- **Fixed** WebSocket timeout compatibility issues
- **Fixed** Image upload failures to Discord
- **Improved** Error handling and connection stability

### 🐛 **Critical Bug Fixes**
- **Fixed** Systematic `/ask` command failures (every other command failing)
- **Fixed** Race conditions in queue processing causing stuck operations
- **Fixed** LM Studio health check timeout too aggressive (3s → 10s)
- **Fixed** Queue processing flags not being reset after exceptions
- **Added** Thread-safe queue processing with proper exception handling

## 🔧 **Technical Changes**

### 📊 **Database Architecture**
- **Restructured** Player-Story relationship with separate Player nodes
- **Added** Party management and membership tracking
- **Added** Individual character data storage per player
- **Maintained** Server isolation for all multiplayer features

### 🎭 **Adventure System**
- **Enhanced** Story generation with AI-powered responses
- **Expanded** Fallback system with 45+ themed story elements
- **Added** Progressive story phases (Opening → Development → Climax → Resolution)
- **Improved** Context awareness and choice consequences

### 🎲 **Dice Implementation**
- **Added** Smart action parsing for appropriate skill checks
- **Added** Stat-based modifiers affecting roll outcomes
- **Added** Difficulty scaling based on action complexity
- **Added** Dynamic success narratives based on roll results

## 📋 **Command Changes**

### **New Commands (4)**
- `/join_adventure` - Join existing party adventures
- `/adventure_status` - Enhanced character and party status
- `/end_adventure` - End current adventure (solo or party)
- `/comfyui_health` - ComfyUI connection diagnostics

### **Enhanced Commands**
- `/start_adventure` - Added `party_mode` parameter
- `/continue_adventure` - Added universal dice mechanics
- All adventure commands now support multiplayer

## 🎮 **Adventure Themes**

### **Enhanced Themes (7 total)**
- **D&D** *(New!)* - Full tabletop RPG experience
- **Fantasy** - Medieval adventures with dice mechanics
- **Sci-Fi** - Space exploration with skill checks
- **Mystery** - Investigation with luck-based rolls
- **Cyberpunk** - High-tech adventures with dice
- **Horror** - Survival horror with tension mechanics
- **Steampunk** - Victorian adventures with skill tests

## 🚀 **Performance & Stability**

### **Fixes**
- **Fixed** WebSocket `timeout` parameter compatibility
- **Fixed** Discord image upload delivery issues
- **Fixed** F-string backslash syntax errors
- **Improved** Command synchronization stability

### **Optimizations**
- **Optimized** Database queries for multiplayer scenarios
- **Enhanced** Error handling with graceful fallbacks
- **Improved** Memory management for party adventures
- **Streamlined** Dice calculation algorithms

## 📊 **Statistics**

| Metric | v2.1 | v2.2 | Change |
|--------|------|------|--------|
| Total Commands | 28 | 32 | +4 |
| Adventure Themes | 6 | 7 | +1 |
| Max Party Size | 1 | 6 | +5 |
| Dice Systems | 1 (D&D only) | 2 (D&D + Basic) | +1 |
| Database Nodes | Story only | Story + Player | Enhanced |

## 🎯 **Migration Notes**

### **Existing Adventures**
- All existing solo adventures continue to work normally
- New dice mechanics apply to new adventures only
- Database migration is automatic and backwards compatible

### **Command Compatibility**
- All existing commands maintain full functionality
- New parameters are optional with sensible defaults
- No breaking changes to existing workflows

## 🔄 **Upgrade Instructions**

1. **Backup** your current bot files
2. **Replace** `DuckBot-v2.1-MULTI-SERVER.py` with `DuckBot-v2.2-MULTI-SERVER.py`
3. **Restart** the bot to load new commands
4. **Test** multiplayer adventures with `/start_adventure dnd true`

---

**📅 Release Date:** August 2, 2025  
**🔢 Version:** 2.2.1  
**🦆 Codename:** "Multiplayer Adventures" (Hotfix)

*Full backwards compatibility maintained*