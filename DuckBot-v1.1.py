# Import the necessary libraries.
# If you don't have these installed, you can install them with `pip install discord requests python-dotenv`
import os
import discord
import requests
import json
from discord.ext import commands
from discord import app_commands
from dotenv import load_dotenv

# Load environment variables from a .env file.
# This is a best practice to keep your Discord token secure.
load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

# --- IMPORTANT: PASTE YOUR SERVER ID HERE ---
# Right-click your server icon in Discord (with Developer Mode on) and "Copy Server ID"
GUILD_ID = 135269011981205504 # <<< YOUR SERVER ID IS NOW HERE

# The URL for your LM Studio server's chat completions API.
# The default is http://localhost:1234/v1/chat/completions.
# If you're running LM Studio on a different machine or port, change this URL.
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"

# Define the intents your bot needs.
# The `message_content` intent is required to read user messages.
# You must enable this in the Discord Developer Portal.
intents = discord.Intents.default()
intents.message_content = True
intents.messages = True
intents.guilds = True

# Create a bot instance.
# We use commands.Bot from the discord.ext library for this.
bot = commands.Bot(command_prefix="!", intents=intents)

# This event runs when the bot successfully connects to Discord.
@bot.event
async def on_ready():
    """
    Log a message when the bot is ready and sync the slash commands to a specific guild.
    """
    print(f'{bot.user} has connected to Discord!')
    # We create a guild object using the ID you provided.
    guild = discord.Object(id=GUILD_ID)
    # This copies the global commands to the guild-specific tree.
    bot.tree.copy_global_to(guild=guild)
    # This syncs the commands to your server instantly.
    try:
        synced = await bot.tree.sync(guild=guild)
        print(f"Synced {len(synced)} command(s) to the guild.")
    except Exception as e:
        print(f"Failed to sync commands: {e}")

# --- DEBUGGING COMMAND ---
# This simple command helps test if public replies are working at all.
@bot.tree.command(name="ping", description="A simple command to test if public replies are working.")
async def ping_command(interaction: discord.Interaction):
    """
    Responds with 'Pong!' publicly.
    """
    await interaction.response.send_message("Pong!", ephemeral=False)


# Define a slash command using the new tree command method.
@bot.tree.command(name="ask", description="Ask a question to the local LLM via LM Studio.")
async def ask_command(interaction: discord.Interaction, prompt: str):
    """
    This function is triggered when a user uses the /ask command.
    It takes the user's prompt, sends it to the LM Studio server,
    and then sends the AI's response back to the Discord channel.
    """
    # Defer the response to make it public
    await interaction.response.defer(ephemeral=False)

    # The payload for the LM Studio API call
    payload = {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful AI assistant DiscordBot Named DuckBot. Respond concisely.",
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "temperature": 0.7,
        "max_tokens": 512,
        "stream": False,
        "model": "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF",
    }

    try:
        # Send the POST request to the LM Studio server.
        response = requests.post(LM_STUDIO_URL, headers={"Content-Type": "application/json"}, data=json.dumps(payload))
        response.raise_for_status()

        # Parse the JSON response from the server.
        data = response.json()

        # Extract the AI's message from the response.
        if "choices" in data and len(data["choices"]) > 0:
            ai_response = data["choices"][0]["message"]["content"]
        else:
            ai_response = "I couldn't get a response from the AI. The response format was unexpected."
        
        await interaction.followup.send(ai_response)
        
    except requests.exceptions.RequestException as e:
        error_message = f"An error occurred while connecting to LM Studio: {e}\nPlease check if your LM Studio server is running and the URL is correct."
        await interaction.followup.send(error_message)
    except Exception as e:
        error_message = f"An unexpected error occurred: {e}"
        await interaction.followup.send(error_message)

# Run the bot with the token from the environment variable.
bot.run(DISCORD_TOKEN)
