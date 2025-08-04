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
    Log a message when the bot is ready and sync the slash commands.
    """
    print(f'{bot.user} has connected to Discord!')
    # This is the correct way to register and sync slash commands globally.
    # It might take up to an hour for the command to appear everywhere.
    try:
        synced = await bot.tree.sync()
        print(f"Synced {len(synced)} command(s).")
    except Exception as e:
        print(f"Failed to sync commands: {e}")

# --- DEBUGGING COMMAND ---
# This simple command helps test if public replies are working at all.
# If /ping works publicly but /ask doesn't, the issue is likely Discord caching the /ask command's old behavior.
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
    # Defer the response to let Discord know the bot is working.
    # This shows a "Bot is thinking..." message, which is necessary because
    # the LLM's response might take longer than 3 seconds.
    # Set ephemeral=False to make the response visible to everyone.
    await interaction.response.defer(ephemeral=False)

    # The payload for the LM Studio API call, which follows the OpenAI format.
    # We include the user's message with a "user" role.
    # You can also add a system message here to provide instructions to the model.
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
        # LM Studio can use the model name. Set this to the model you have loaded.
        "model": "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF",
    }

    try:
        # Send the POST request to the LM Studio server.
        response = requests.post(LM_STUDIO_URL, headers={"Content-Type": "application/json"}, data=json.dumps(payload))
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

        # Parse the JSON response from the server.
        data = response.json()

        # Extract the AI's message from the response.
        # This structure is specific to the OpenAI-compatible API.
        if "choices" in data and len(data["choices"]) > 0:
            ai_response = data["choices"][0]["message"]["content"]
        else:
            ai_response = "I couldn't get a response from the AI. The response format was unexpected."
        
        # Send the final response back to the Discord channel.
        # The `followup` method is used after a `defer`.
        await interaction.followup.send(ai_response)
        
    except requests.exceptions.RequestException as e:
        # Handle network-related errors, such as a connection refusal.
        error_message = f"An error occurred while connecting to LM Studio: {e}\nPlease check if your LM Studio server is running and the URL is correct."
        await interaction.followup.send(error_message)
    except Exception as e:
        # Handle any other unexpected errors.
        error_message = f"An unexpected error occurred: {e}"
        await interaction.followup.send(error_message)

# Run the bot with the token from the environment variable.
bot.run(DISCORD_TOKEN)
