# Enhanced Ask Command with Robust Error Handling
# Add this to your DuckBot to replace the existing ask command

import asyncio
import json
import requests
from typing import Dict, Any, Optional

# LM Studio Configuration
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
LM_STUDIO_TIMEOUT = 60  # seconds
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

# Plugin configurations (safe defaults)
SAFE_PLUGIN_CONFIG = {
    "plugins": {
        "duckduckgo": {"enabled": True, "priority": 1},
        "wikipedia": {"enabled": True, "priority": 2},
        "dice": {"enabled": True, "priority": 3}
    },
    "max_plugin_calls": 2,
    "plugin_timeout": 10
}

FULL_PLUGIN_CONFIG = {
    "plugins": {
        "duckduckgo": {"enabled": True, "priority": 1},
        "web-search": {"enabled": True, "priority": 2},
        "visit-website": {"enabled": True, "priority": 3},
        "wikipedia": {"enabled": True, "priority": 1},
        "rag-v1": {"enabled": True, "priority": 1, "sources": ["web", "news", "wikipedia"]},
        "js-code-sandbox": {"enabled": True, "priority": 2},
        "dice": {"enabled": True, "priority": 3}
    },
    "max_plugin_calls": 3,
    "plugin_timeout": 15
}

async def check_lm_studio_health() -> bool:
    """Check if LM Studio is responding."""
    try:
        health_payload = {
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 10,
            "temperature": 0
        }
        
        response = requests.post(
            LM_STUDIO_URL, 
            headers={"Content-Type": "application/json"}, 
            data=json.dumps(health_payload),
            timeout=10
        )
        return response.status_code == 200
    except:
        return False

async def call_lm_studio_with_retry(payload: Dict[str, Any], max_retries: int = MAX_RETRIES) -> Optional[Dict[str, Any]]:
    """Call LM Studio with retry logic and error handling."""
    
    for attempt in range(max_retries):
        try:
            # Add timeout to prevent hanging
            response = requests.post(
                LM_STUDIO_URL,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=LM_STUDIO_TIMEOUT
            )
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 500:
                print(f"LM Studio 500 error (attempt {attempt + 1}/{max_retries})")
                
                # Try with simpler configuration on retry
                if attempt == 0 and "plugins" in payload:
                    print("Retrying with safe plugin configuration...")
                    payload.update(SAFE_PLUGIN_CONFIG)
                elif attempt == 1 and "plugins" in payload:
                    print("Retrying without plugins...")
                    payload.pop("plugins", None)
                    payload.pop("max_plugin_calls", None)
                    payload.pop("plugin_timeout", None)
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                    continue
            else:
                print(f"LM Studio error {response.status_code}: {response.text}")
                break
                
        except requests.exceptions.Timeout:
            print(f"LM Studio timeout (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                await asyncio.sleep(RETRY_DELAY)
                continue
        except requests.exceptions.RequestException as e:
            print(f"LM Studio connection error: {e}")
            break
        except Exception as e:
            print(f"Unexpected error calling LM Studio: {e}")
            break
    
    return None

async def get_fallback_response(prompt: str, user_name: str) -> str:
    """Generate a fallback response when LM Studio is unavailable."""
    fallback_responses = [
        f"I'm having trouble connecting to my AI brain right now, {user_name}! 🤖 Could you try asking again in a moment?",
        f"Oops! My AI circuits are a bit scrambled at the moment, {user_name}. Give me a second to reboot! ⚡",
        f"Sorry {user_name}, I'm experiencing some technical difficulties. Try your question again shortly! 🛠️",
        f"My AI assistant seems to be taking a coffee break, {user_name}! ☕ Please retry in a moment.",
        f"Houston, we have a problem! 🚀 My AI systems are temporarily offline, {user_name}. Try again soon!"
    ]
    
    import random
    base_response = random.choice(fallback_responses)
    
    # Add helpful suggestions based on the prompt
    if any(word in prompt.lower() for word in ["generate", "image", "picture", "art"]):
        base_response += "\n\n💡 *In the meantime, you can try `/generate` for AI image creation!*"
    elif any(word in prompt.lower() for word in ["animate", "video", "movie"]):
        base_response += "\n\n💡 *You might want to try `/animate` for AI video generation!*"
    elif any(word in prompt.lower() for word in ["learn", "teach", "knowledge"]):
        base_response += "\n\n💡 *You can use `/learn` to teach me new information while I'm recovering!*"
    
    return base_response

@bot.tree.command(name="ask", description="Ask a question to the local LLM via LM Studio (Enhanced)")
@app_commands.describe(prompt="Your question or request")
async def enhanced_ask_command(interaction: discord.Interaction, prompt: str):
    """Enhanced ask command with robust error handling and fallbacks."""
    await interaction.response.defer(ephemeral=False)
    
    user_name = interaction.user.display_name
    
    # Check LM Studio health first
    if not await check_lm_studio_health():
        fallback_msg = await get_fallback_response(prompt, user_name)
        await interaction.followup.send(fallback_msg)
        return
    
    # Build the payload with progressive fallback strategy
    base_payload = {
        "messages": [
            {
                "role": "system",
                "content": f"""You are DuckBot, a helpful AI assistant for Discord. The user's name is {user_name}. 
                
Be helpful, concise, and engaging. If you need to use tools or search for information, do so when appropriate. 
Keep responses under 1500 characters when possible to fit Discord's message limits well."""
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.7,
        "max_tokens": 800,
        "stream": False
    }
    
    # Try with full plugin configuration first
    full_payload = {**base_payload, **FULL_PLUGIN_CONFIG}
    
    try:
        # Attempt to call LM Studio
        data = await call_lm_studio_with_retry(full_payload)
        
        if data and "choices" in data and len(data["choices"]) > 0:
            ai_response = data["choices"][0]["message"]["content"]
            
            # Handle long responses
            if len(ai_response) > 2000:
                # Split into chunks
                chunks = [ai_response[i:i+1900] for i in range(0, len(ai_response), 1900)]
                await interaction.followup.send(chunks[0])
                for chunk in chunks[1:]:
                    await interaction.followup.send(chunk)
            else:
                await interaction.followup.send(ai_response)
                
        else:
            # Fallback response
            fallback_msg = await get_fallback_response(prompt, user_name)
            await interaction.followup.send(fallback_msg)
            
    except Exception as e:
        print(f"Enhanced ask command error: {e}")
        fallback_msg = await get_fallback_response(prompt, user_name)
        await interaction.followup.send(fallback_msg)

# Alternative: Simple ask command without plugins
@bot.tree.command(name="ask_simple", description="Ask a question without plugins (more reliable)")
@app_commands.describe(prompt="Your question or request")
async def simple_ask_command(interaction: discord.Interaction, prompt: str):
    """Simple ask command without plugins for maximum reliability."""
    await interaction.response.defer(ephemeral=False)
    
    payload = {
        "messages": [
            {
                "role": "system",
                "content": f"You are DuckBot, a helpful AI assistant. The user is {interaction.user.display_name}. Be concise and helpful."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.7,
        "max_tokens": 500,
        "stream": False
    }
    
    try:
        response = requests.post(
            LM_STUDIO_URL,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            if "choices" in data and len(data["choices"]) > 0:
                ai_response = data["choices"][0]["message"]["content"]
                await interaction.followup.send(ai_response)
            else:
                await interaction.followup.send("🤖 I received an unexpected response format from my AI brain!")
        else:
            await interaction.followup.send(f"🔧 AI system returned error {response.status_code}. Try `/ask_simple` for a more basic version!")
            
    except requests.exceptions.Timeout:
        await interaction.followup.send("⏰ My AI is thinking too hard! The request timed out. Try a simpler question!")
    except Exception as e:
        await interaction.followup.send(f"🚨 Something went wrong: {str(e)[:100]}... Try again in a moment!")

# Health check command for debugging
@bot.tree.command(name="lm_health", description="Check LM Studio connection health")
async def lm_health_command(interaction: discord.Interaction):
    """Check the health of LM Studio connection."""
    await interaction.response.defer(ephemeral=True)
    
    # Test basic connection
    basic_health = await check_lm_studio_health()
    
    embed = discord.Embed(
        title="🏥 LM Studio Health Check",
        color=0x00ff00 if basic_health else 0xff0000
    )
    
    embed.add_field(
        name="Basic Connection",
        value="✅ Connected" if basic_health else "❌ Failed",
        inline=True
    )
    
    # Test with plugins
    if basic_health:
        try:
            plugin_payload = {
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 10,
                **SAFE_PLUGIN_CONFIG
            }
            
            response = requests.post(
                LM_STUDIO_URL,
                headers={"Content-Type": "application/json"},
                data=json.dumps(plugin_payload),
                timeout=10
            )
            
            plugin_status = response.status_code == 200
            
        except:
            plugin_status = False
            
        embed.add_field(
            name="Plugin Support",
            value="✅ Working" if plugin_status else "❌ Issues detected",
            inline=True
        )
        
    embed.add_field(
        name="LM Studio URL",
        value=LM_STUDIO_URL,
        inline=False
    )
    
    if not basic_health:
        embed.add_field(
            name="🛠️ Troubleshooting",
            value="1. Check if LM Studio is running\n2. Verify the URL is correct\n3. Make sure a model is loaded\n4. Check for plugin conflicts",
            inline=False
        )
    
    await interaction.followup.send(embed=embed)