# ==============================================================================
# DUCK BOT - MULTI-SERVER VERSION
# ==============================================================================

# --- 1. IMPORTS ---
import os
import json
import uuid
import urllib.request
import urllib.parse
from io import BytesIO

# Third-party libraries
import discord
import requests
import websockets
import torch
from discord.ext import commands
from discord import app_commands
from dotenv import load_dotenv


# --- 2. INITIAL SETUP & CONFIGURATION ---
load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

# --- API URLs ---
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
COMFYUI_SERVER_ADDRESS = "127.0.0.1:8188"

# A unique ID for our ComfyUI client
CLIENT_ID = str(uuid.uuid4())


# --- 3. BOT CLASS DEFINITION ---
# We define a custom Bot class to use the recommended setup_hook for syncing.
class DuckBot(commands.Bot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def setup_hook(self):
        """This now performs a GLOBAL command sync."""
        print("Running setup_hook to sync global commands...")
        try:
            synced = await self.tree.sync()
            print("--- GLOBAL COMMAND SYNC SUCCESS ---")
            print(f"Synced {len(synced)} command(s) globally: {[c.name for c in synced]}")
            print("Note: It may take up to an hour for commands to appear on all servers.")
            print("-----------------------------------")
        except Exception as e:
            print(f"!!! FAILED TO SYNC GLOBAL COMMANDS: {e} !!!")

# Define the bot's intents
intents = discord.Intents.default()
intents.message_content = True

# Create an instance of our custom bot
bot = DuckBot(command_prefix="!", intents=intents)


# --- 4. BOT EVENTS ---
@bot.event
async def on_ready():
    """This now only prints a confirmation message on successful login."""
    print(f'Logged in as {bot.user} (ID: {bot.user.id})')
    print('Bot is ready and online.')
    print('------')


# --- 5. COMFYUI API HELPER FUNCTIONS (No Changes) ---
def queue_prompt(prompt_workflow):
    try:
        p = {"prompt": prompt_workflow, "client_id": CLIENT_ID}
        data = json.dumps(p).encode('utf-8')
        req = urllib.request.Request(f"http://{COMFYUI_SERVER_ADDRESS}/prompt", data=data)
        return json.loads(urllib.request.urlopen(req).read())
    except Exception as e:
        print(f"Error queueing prompt: {e}")
        return None

def get_image(filename, subfolder, folder_type):
    try:
        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        url_values = urllib.parse.urlencode(data)
        with urllib.request.urlopen(f"http://{COMFYUI_SERVER_ADDRESS}/view?{url_values}") as response:
            return response.read()
    except Exception as e:
        print(f"Error getting image: {e}")
        return None

def get_history(prompt_id):
    try:
        with urllib.request.urlopen(f"http://{COMFYUI_SERVER_ADDRESS}/history/{prompt_id}") as response:
            return json.loads(response.read())
    except Exception as e:
        print(f"Error getting history: {e}")
        return None

async def get_images_from_comfyui(ws, prompt_workflow):
    prompt_response = queue_prompt(prompt_workflow)
    if not prompt_response or 'prompt_id' not in prompt_response:
        print("Failed to queue prompt.")
        return {}
    prompt_id = prompt_response['prompt_id']
    output_images = {}
    while True:
        out = await ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing' and message['data']['node'] is None and message['data']['prompt_id'] == prompt_id:
                break
    history = get_history(prompt_id)[prompt_id]
    for node_id in history['outputs']:
        node_output = history['outputs'][node_id]
        if 'images' in node_output:
            images_output = [get_image(img['filename'], img['subfolder'], img['type']) for img in node_output['images']]
            output_images[node_id] = [img for img in images_output if img is not None]
    return output_images


# --- 6. SLASH COMMAND DEFINITIONS (No Changes) ---
@bot.tree.command(name="ping", description="A simple command to test if the bot is responsive.")
async def ping_command(interaction: discord.Interaction):
    await interaction.response.send_message("Pong!", ephemeral=False)

@bot.tree.command(name="ask", description="Ask a question to the local LLM via LM Studio.")
async def ask_command(interaction: discord.Interaction, prompt: str):
    await interaction.response.defer(ephemeral=False)
    payload = {"messages": [{"role": "system", "content": "You are DuckBot."}, {"role": "user", "content": prompt}]}
    try:
        response = requests.post(LM_STUDIO_URL, headers={"Content-Type": "application/json"}, data=json.dumps(payload))
        response.raise_for_status()
        data = response.json()
        ai_response = data["choices"][0]["message"]["content"] if "choices" in data and data["choices"] else "No response."
        await interaction.followup.send(ai_response)
    except Exception as e:
        await interaction.followup.send(f"An error occurred with the LLM: {e}")

@bot.tree.command(name="generate", description="Generate an image using ComfyUI.")
@app_commands.describe(prompt="The prompt for the image you want to generate.")
async def generate_command(interaction: discord.Interaction, prompt: str):
    await interaction.response.defer(ephemeral=False)
    try:
        with open("workflow_api.json", "r") as f:
            prompt_workflow = json.load(f)
    except Exception as e:
        await interaction.followup.send(f"Error loading `workflow_api.json`: {e}")
        return

    prompt_workflow["6"]["inputs"]["seed"] = torch.randint(1, 1125899906842624, (1,)).item()
    prompt_workflow["3"]["inputs"]["text"] = prompt
    
    try:
        async with websockets.connect(f"ws://{COMFYUI_SERVER_ADDRESS}/ws?clientId={CLIENT_ID}") as ws:
            images = await get_images_from_comfyui(ws, prompt_workflow)
            if not images:
                await interaction.followup.send("Image generation failed on the ComfyUI server.")
                return
            
            image_files = [discord.File(fp=BytesIO(img_data), filename=f"output_{uuid.uuid4()}.png") for node_id in images for img_data in images[node_id]]
            
            if image_files:
                await interaction.followup.send(f"Prompt: `{prompt}`", files=image_files)
            else:
                await interaction.followup.send("ComfyUI ran, but couldn't retrieve the final images.")
    except Exception as e:
        await interaction.followup.send(f"An error occurred during image generation: {e}")


# --- 7. RUN THE BOT ---
if __name__ == "__main__":
    if not DISCORD_TOKEN:
        print("CRITICAL ERROR: DISCORD_TOKEN not found in .env file. Please set it.")
    else:
        print("Starting bot...")
        bot.run(DISCORD_TOKEN)
