# ==============================================================================
# DUCK BOT v2.1 - MULTI-SERVER EDITION
# Features: All v2.0 features + Multi-server support with data isolation
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
import aiohttp
import websockets
import torch
import cv2
import tempfile
from PIL import Image
from discord.ext import commands
from discord import app_commands
from dotenv import load_dotenv
import datetime
import random
import hashlib
import re
import time
import asyncio
import websockets
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from neo4j import GraphDatabase


# --- 2. INITIAL SETUP & CONFIGURATION ---
load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

# --- API URLs ---
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
COMFYUI_SERVER_ADDRESS = "127.0.0.1:8188"

# --- NEO4J DATABASE CONFIGURATION ---
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
NEO4J_ENABLED = os.getenv("NEO4J_ENABLED", "false").lower() == "true"

# Multi-server configuration
GLOBAL_ADMIN_IDS = [int(x) for x in os.getenv("GLOBAL_ADMIN_IDS", "").split(",") if x.strip()]
MAX_SERVERS_PER_INSTANCE = int(os.getenv("MAX_SERVERS_PER_INSTANCE", "100"))

# --- QUEUE SYSTEM AND COMFYUI FUNCTIONS ---

from collections import deque

# Old QueueItem definition removed - using the multi-server version below

# Server-specific queues
SERVER_QUEUES = {}

def get_server_queue(server_id: int):
    """Get or create a server-specific queue."""
    if server_id not in SERVER_QUEUES:
        SERVER_QUEUES[server_id] = {
            'queue': deque(),
            'currently_processing': False,
            'lock': asyncio.Lock()
        }
    return SERVER_QUEUES[server_id]

# Average processing times for wait estimation
AVERAGE_TIMES = {
    "image": 15.0,  # seconds
    "video": 45.0   # seconds
}

def update_average_time(generation_type: str, actual_time: float):
    """Update average processing time using exponential moving average."""
    alpha = 0.3  # Learning rate
    AVERAGE_TIMES[generation_type] = (alpha * actual_time + 
                                    (1 - alpha) * AVERAGE_TIMES[generation_type])

def calculate_estimated_wait(position: int, generation_type: str) -> float:
    """Calculate estimated wait time based on queue position."""
    if position <= 1:
        return 0
    return (position - 1) * AVERAGE_TIMES[generation_type]

def format_time(seconds: float) -> str:
    """Format seconds into human-readable time."""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = int(seconds % 60)
        return f"{minutes}m {remaining_seconds}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"

async def run_comfyui_workflow(workflow_data: dict, is_video: bool = False) -> list:
    """Execute a ComfyUI workflow and return the generated images/video frames."""
    print(f"🔧 Starting ComfyUI workflow execution...")
    print(f"🌐 Connecting to ComfyUI at: {COMFYUI_SERVER_ADDRESS}")
    
    try:
        # Test basic connectivity first
        try:
            test_response = requests.get(f"http://{COMFYUI_SERVER_ADDRESS}/", timeout=5)
            print(f"✅ ComfyUI server responding: {test_response.status_code}")
        except Exception as test_e:
            print(f"❌ ComfyUI server not responding: {test_e}")
            return []
            
        # Connect to ComfyUI WebSocket
        uri = f"ws://{COMFYUI_SERVER_ADDRESS}/ws"
        print(f"🔌 Connecting to WebSocket: {uri}")
        
        async with websockets.connect(uri) as websocket:
            print("✅ WebSocket connected successfully")
            
            # Queue the workflow
            queue_url = f"http://{COMFYUI_SERVER_ADDRESS}/prompt"
            payload = {"prompt": workflow_data}
            
            print(f"📤 Queuing workflow to: {queue_url}")
            response = requests.post(queue_url, json=payload, timeout=10)
            print(f"📥 Queue response: {response.status_code}")
            
            if response.status_code != 200:
                print(f"❌ Failed to queue workflow: {response.status_code} - {response.text}")
                return []
            
            result = response.json()
            prompt_id = result.get("prompt_id")
            print(f"🆔 Received prompt_id: {prompt_id}")
            
            if not prompt_id:
                print("❌ No prompt_id received in response")
                return []
            
            # Wait for completion with progress tracking
            print("⏳ Waiting for workflow completion...")
            start_time = time.time()
            
            while True:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=60.0)
                    data = json.loads(message)
                    
                    # Log progress messages
                    if data.get("type") == "progress":
                        progress_data = data.get("data", {})
                        print(f"⚡ Progress: {progress_data.get('value', 0)}/{progress_data.get('max', 'unknown')}")
                    elif data.get("type") == "executed":
                        executed_prompt_id = data.get("data", {}).get("prompt_id")
                        print(f"✅ Workflow executed for prompt_id: {executed_prompt_id}")
                        if executed_prompt_id == prompt_id:
                            break
                    
                    # Safety timeout (5 minutes total)
                    if time.time() - start_time > 300:
                        print("⏰ Overall timeout reached (5 minutes)")
                        return []
                        
                except asyncio.TimeoutError:
                    elapsed = time.time() - start_time
                    print(f"⏰ Timeout waiting for ComfyUI response (elapsed: {elapsed:.1f}s)")
                    return []
                except websockets.exceptions.ConnectionClosed:
                    print("❌ WebSocket connection closed unexpectedly")
                    return []
            
            print("🎯 Workflow completed, retrieving files...")
            # Get the generated files
            return await get_output_files(prompt_id, is_video)
            
    except ConnectionRefusedError:
        print("❌ ComfyUI WebSocket connection refused - is ComfyUI running?")
        return []
    except Exception as e:
        print(f"❌ Error in ComfyUI workflow: {type(e).__name__}: {e}")
        return []

async def get_output_files(prompt_id: str, is_video: bool = False) -> list:
    """Retrieve output files from ComfyUI."""
    try:
        print(f"📁 Getting output files for prompt_id: {prompt_id}")
        history_url = f"http://{COMFYUI_SERVER_ADDRESS}/history/{prompt_id}"
        response = requests.get(history_url)
        
        if response.status_code != 200:
            print(f"❌ Failed to get history: {response.status_code}")
            return []
        
        print(f"✅ Got history response: {response.status_code}")
        
        history = response.json()
        if prompt_id not in history:
            print(f"❌ Prompt ID {prompt_id} not found in history")
            return []
        
        print(f"✅ Found prompt in history")
        outputs = history[prompt_id].get("outputs", {})
        print(f"📊 Found {len(outputs)} output nodes")
        file_data_list = []
        
        # Look for output nodes (usually SaveImage nodes)
        for node_id, node_output in outputs.items():
            print(f"🔍 Checking node {node_id}")
            if "images" in node_output:
                print(f"📷 Node {node_id} has {len(node_output['images'])} images")
                for i, image_info in enumerate(node_output["images"]):
                    filename = image_info["filename"]
                    subfolder = image_info.get("subfolder", "")
                    print(f"📥 Downloading image {i+1}: {filename}")
                    
                    # Download the file
                    if subfolder:
                        file_url = f"http://{COMFYUI_SERVER_ADDRESS}/view?filename={filename}&subfolder={subfolder}"
                    else:
                        file_url = f"http://{COMFYUI_SERVER_ADDRESS}/view?filename={filename}"
                    
                    file_response = requests.get(file_url)
                    if file_response.status_code == 200:
                        print(f"✅ Downloaded {filename}: {len(file_response.content)} bytes")
                        file_data_list.append(file_response.content)
                    else:
                        print(f"❌ Failed to download {filename}: {file_response.status_code}")
            else:
                print(f"ℹ️ Node {node_id} has no images")
        
        return file_data_list
        
    except Exception as e:
        print(f"Error getting output files: {e}")
        return []

async def add_to_queue(interaction: discord.Interaction, prompt: str, generation_type: str):
    """Add a generation request to the server-specific queue."""
    server_queue = get_server_queue(interaction.guild.id)
    queue_item = QueueItem(interaction, prompt, generation_type, interaction.guild.id)
    
    async with server_queue['lock']:
        server_queue['queue'].append(queue_item)
        position = len(server_queue['queue'])
        
        if position == 1 and not server_queue['currently_processing']:
            # Start processing immediately
            queue_item.status_message = await interaction.followup.send(
                f"🚀 **Starting {generation_type} generation**\n"
                f"Prompt: `{prompt}`\n"
                f"⚡ Processing now..."
            )
        else:
            # Show queue position and estimated wait time
            wait_time = calculate_estimated_wait(position, generation_type)
            wait_str = f"\n⏱️ Estimated wait: {format_time(wait_time)}" if wait_time > 0 else ""
            
            queue_item.status_message = await interaction.followup.send(
                f"📍 **{generation_type.title()} generation queued**\n"
                f"Prompt: `{prompt}`\n"
                f"Position {position} in queue{wait_str}"
            )
    
    # Start processing if not already running
    if not server_queue['currently_processing']:
        asyncio.create_task(process_server_queue(interaction.guild.id))

async def process_server_queue(server_id: int):
    """Process the generation queue for a specific server."""
    server_queue = get_server_queue(server_id)
    
    async with server_queue['lock']:
        if server_queue['currently_processing'] or not server_queue['queue']:
            return
        server_queue['currently_processing'] = True
    
    while server_queue['queue']:
        async with server_queue['lock']:
            if not server_queue['queue']:
                break
            current_item = server_queue['queue'].popleft()
        
        try:
            if current_item.generation_type == "image":
                await process_image_generation(current_item)
            elif current_item.generation_type == "video":
                await process_video_generation(current_item)
        except Exception as e:
            await current_item.status_message.edit(
                content=f"❌ **{current_item.generation_type.title()} generation error**\n"
                       f"Prompt: `{current_item.prompt}`\n"
                       f"Error: {str(e)[:100]}..."
            )
    
    server_queue['currently_processing'] = False

async def process_image_generation(queue_item):
    """Process an image generation request."""
    start_time = time.time()
    
    try:
        # Load the workflow
        with open("workflow_api.json", "r") as f:
            prompt_workflow = json.load(f)
        
        # Update the prompt in the workflow
        prompt_workflow["3"]["inputs"]["text"] = queue_item.prompt
        
        # Generate random seed
        prompt_workflow["6"]["inputs"]["seed"] = torch.randint(1, 1125899906842624, (1,)).item()
        
        # Update status
        await queue_item.status_message.edit(
            content=f"🎨 **Generating image**\nPrompt: `{queue_item.prompt}`\n⚡ Processing..."
        )
        
        # Generate image
        print(f"🎨 Starting ComfyUI image generation for: {queue_item.prompt}")
        images_data = await run_comfyui_workflow(prompt_workflow, is_video=False)
        print(f"📊 ComfyUI returned {len(images_data) if images_data else 0} images")
        
        if not images_data:
            await queue_item.status_message.edit(
                content=f"❌ **Image generation failed**\nPrompt: `{queue_item.prompt}`"
            )
            return
        
        # Create Discord files (simplified like working version)
        image_files = [discord.File(fp=BytesIO(data), filename=f"image_{uuid.uuid4()}.png") for data in images_data]
        
        # Send the generated images FIRST (before editing status)
        await queue_item.interaction.followup.send(files=image_files)
        
        # Then update status message 
        await queue_item.status_message.edit(content=f"✅ **Image generation complete!**\nPrompt: `{queue_item.prompt}`")
        
        # Update average time
        actual_time = time.time() - start_time
        update_average_time("image", actual_time)
        
    except Exception as e:
        await queue_item.status_message.edit(
            content=f"❌ **Image generation error**\nPrompt: `{queue_item.prompt}`\nError: {str(e)[:100]}..."
        )

async def process_video_generation(queue_item):
    """Process a video generation request."""
    start_time = time.time()
    
    try:
        # Load the video workflow
        with open("workflow_video_api.json", "r") as f:
            prompt_workflow = json.load(f)
        
        # Update the prompt in the workflow
        prompt_workflow["3"]["inputs"]["text"] = queue_item.prompt
        
        # Generate random seed
        prompt_workflow["6"]["inputs"]["seed"] = torch.randint(1, 1125899906842624, (1,)).item()
        
        # Update status
        await queue_item.status_message.edit(
            content=f"🎬 **Generating video**\nPrompt: `{queue_item.prompt}`\n⚡ Processing... (this may take a while)"
        )
        
        # Generate video frames
        frame_data_list = await run_comfyui_workflow(prompt_workflow, is_video=True)
        
        if not frame_data_list:
            await queue_item.status_message.edit(
                content=f"❌ **Video generation failed**\nPrompt: `{queue_item.prompt}`"
            )
            return
        
        # Create video from frames
        video_data = await create_video_from_frames(frame_data_list, fps=24)
        
        if video_data:
            # Send the video
            video_filename = f"generated_video_{uuid.uuid4()}.mp4"
            video_file = discord.File(fp=BytesIO(video_data), filename=video_filename)
            
            await queue_item.status_message.edit(
                content=f"✅ **Video generation complete!**\nPrompt: `{queue_item.prompt}`\n🎬 Duration: 10 seconds"
            )
            await queue_item.interaction.followup.send(files=[video_file])
        else:
            await queue_item.status_message.edit(
                content=f"❌ **Video processing failed**\nPrompt: `{queue_item.prompt}`"
            )
        
        # Update average time
        actual_time = time.time() - start_time
        update_average_time("video", actual_time)
        
    except Exception as e:
        await queue_item.status_message.edit(
            content=f"❌ **Video generation error**\nPrompt: `{queue_item.prompt}`\nError: {str(e)[:100]}..."
        )

async def create_video_from_frames(frame_data_list: List[bytes], fps: int = 24) -> bytes:
    """Create an MP4 video from a list of frame data."""
    try:
        import cv2
        import numpy as np
        
        if not frame_data_list:
            return None
        
        # Create temporary file for video output
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
            temp_video_path = temp_video.name
        
        # Get dimensions from first frame
        first_frame = cv2.imdecode(np.frombuffer(frame_data_list[0], np.uint8), cv2.IMREAD_COLOR)
        height, width, _ = first_frame.shape
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
        
        # Add frames to video
        for frame_data in frame_data_list:
            frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
            video_writer.write(frame)
        
        video_writer.release()
        
        # Read the video file and return as bytes
        with open(temp_video_path, 'rb') as f:
            video_bytes = f.read()
        
        # Clean up temporary file
        os.unlink(temp_video_path)
        
        return video_bytes
        
    except Exception as e:
        print(f"Error creating video: {e}")
        return None

print("✅ Queue system and ComfyUI functions loaded!")

# --- BOT SETUP ---

# Discord bot setup
intents = discord.Intents.default()
intents.message_content = True
intents.members = True
intents.guilds = True

bot = commands.Bot(command_prefix='!', intents=intents)

# Neo4j driver instance
neo4j_driver = None

@bot.event
async def on_ready():
    print(f"✅ {bot.user} has connected to Discord!")
    print(f"🌐 Connected to {len(bot.guilds)} servers")
    
    # Sync slash commands with enhanced error handling
    try:
        print("🔄 Starting command sync...")
        
        # Force global sync to all servers
        synced = await bot.tree.sync()
        print(f"✅ Successfully synced {len(synced)} slash commands globally")
        
        # List all available commands for verification
        print("📋 Available slash commands:")
        for command in synced:
            print(f"   /{command.name} - {command.description}")
            
        print("⏱️ Commands may take 1-5 minutes to appear in Discord")
        
    except discord.HTTPException as e:
        print(f"❌ HTTP error during command sync: {e}")
        print("🔄 Retrying command sync in 10 seconds...")
        await asyncio.sleep(10)
        try:
            synced = await bot.tree.sync()
            print(f"✅ Retry successful: {len(synced)} commands synced")
        except Exception as retry_e:
            print(f"❌ Retry failed: {retry_e}")
    except Exception as e:
        print(f"❌ Failed to sync commands: {e}")
        print("⚠️ Bot will continue running, but slash commands may not work")
    
    # Initialize Neo4j connection
    await initialize_neo4j()
    
    print("🦆 DuckBot v2.1 Multi-Server is ready!")

@bot.event
async def on_guild_join(guild):
    """Handle bot joining a new server."""
    print(f"🆕 Joined new server: {guild.name} (ID: {guild.id})")
    
    # Initialize server configuration
    config = get_server_config(guild.id, guild.name)
    print(f"📋 Initialized config for {guild.name}")
    
    # Store server info in Neo4j if enabled
    if NEO4J_ENABLED and neo4j_driver:
        try:
            with neo4j_driver.session() as session:
                session.run("""
                    MERGE (s:Server {id: $server_id})
                    SET s.name = $server_name,
                        s.joined_date = datetime()
                """, server_id=guild.id, server_name=guild.name)
        except Exception as e:
            print(f"Error storing server info: {e}")

@bot.event
async def on_message(message):
    """Handle message events for Neo4j storage."""
    if message.author.bot:
        return
    
    # Store message data if Neo4j is enabled
    if NEO4J_ENABLED and neo4j_driver and message.guild:
        try:
            message_data = {
                "message_id": message.id,
                "user_id": message.author.id,
                "username": message.author.name,
                "display_name": message.author.display_name,
                "content": message.content[:1000],  # Limit content length
                "channel_id": message.channel.id,
                "channel_name": message.channel.name,
                "server_id": message.guild.id,
                "timestamp": message.created_at.isoformat(),
                "mentions": [user.id for user in message.mentions]
            }
            
            await store_message_data_server_isolated(message_data)
        except Exception as e:
            print(f"Error storing message: {e}")
    
    # Process commands
    await bot.process_commands(message)

# --- NEO4J INITIALIZATION ---

async def initialize_neo4j():
    """Initialize Neo4j database connection."""
    global neo4j_driver
    
    if not NEO4J_ENABLED:
        print("📊 Neo4j disabled - analytics and memory features unavailable")
        return
    
    try:
        neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        
        # Test connection
        with neo4j_driver.session() as session:
            result = session.run("RETURN 1 as test")
            test_value = result.single()["test"]
            
        print(f"✅ Neo4j connected successfully to {NEO4J_URI}")
        print("📊 Analytics, memory, and social features enabled!")
        
    except Exception as e:
        print(f"❌ Neo4j connection failed: {e}")
        print("📊 Analytics and memory features will be disabled")
        neo4j_driver = None

async def store_message_data_server_isolated(message_data):
    """Store message data with server isolation."""
    if not NEO4J_ENABLED or not neo4j_driver:
        return False
        
    try:
        with neo4j_driver.session() as session:
            query = """
            MERGE (u:User {id: $user_id})
            SET u.username = $username,
                u.display_name = $display_name
            
            MERGE (s:Server {id: $server_id})
            MERGE (c:Channel {id: $channel_id})
            SET c.name = $channel_name
            MERGE (c)-[:IN_SERVER]->(s)
            
            CREATE (m:Message {
                id: $message_id,
                content: $content,
                timestamp: datetime($timestamp),
                user_id: $user_id,
                channel_id: $channel_id,
                server_id: $server_id
            })
            
            MERGE (u)-[:SENT]->(m)
            MERGE (m)-[:IN_CHANNEL]->(c)
            MERGE (m)-[:BELONGS_TO]->(s)
            
            WITH m, $mentions as mention_ids
            UNWIND mention_ids as mention_id
            MERGE (mentioned:User {id: mention_id})
            MERGE (m)-[:MENTIONS]->(mentioned)
            """
            
            session.run(query, message_data)
            return True
            
    except Exception as e:
        print(f"❌ Error storing message data: {e}")
        return False

# A unique ID for our ComfyUI client
CLIENT_ID = str(uuid.uuid4())

# --- MULTI-SERVER DATA STRUCTURES ---
server_configs = {}  # Store per-server configurations
server_queues = {}   # Per-server generation queues
server_locks = {}    # Per-server async locks

# --- QUEUE SYSTEM (Updated for Multi-Server) ---
import asyncio
from collections import deque

@dataclass
class QueueItem:
    interaction: discord.Interaction
    prompt: str
    generation_type: str  # 'image' or 'video'
    server_id: int
    status_message: Optional[discord.Message] = None
    model_id: str = None  # For enhanced image generation

@dataclass
class ServerConfig:
    server_id: int
    server_name: str
    admin_ids: List[int] = field(default_factory=list)
    features_enabled: Dict[str, bool] = field(default_factory=dict)
    custom_settings: Dict[str, Any] = field(default_factory=dict)
    created_date: datetime.datetime = field(default_factory=datetime.datetime.now)

def get_server_queue(server_id: int):
    """Get or create server-specific generation queue."""
    if server_id not in server_queues:
        server_queues[server_id] = {
            'queue': deque(),
            'currently_processing': False,
            'lock': asyncio.Lock()
        }
    return server_queues[server_id]

def get_server_config(server_id: int, server_name: str = None) -> ServerConfig:
    """Get or create server configuration."""
    if server_id not in server_configs:
        server_configs[server_id] = ServerConfig(
            server_id=server_id,
            server_name=server_name or f"Server {server_id}",
            features_enabled={
                'knowledge_management': True,
                'personal_memory': True,
                'adventures': True,
                'art_tracking': True,
                'idea_system': True,
                'social_analytics': True
            }
        )
    return server_configs[server_id]

# Average generation times (in seconds) - global defaults
average_times = {
    'image': 30,    # 30 seconds average
    'video': 900    # 15 minutes average (900 seconds)
}

import time

# --- MULTI-SERVER NEO4J FUNCTIONS ---

def initialize_neo4j():
    """Initialize Neo4j connection and create database schema."""
    global neo4j_driver
    
    if not NEO4J_ENABLED:
        print("⚠️  Neo4j disabled - enhanced features unavailable")
        return False
        
    try:
        neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        
        # Test connection and create initial schema
        with neo4j_driver.session() as session:
            # Create constraints and indexes for better performance
            schema_queries = [
                # Multi-server support - all nodes include server_id
                "CREATE CONSTRAINT IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (m:Message) REQUIRE (m.id, m.server_id) IS UNIQUE", 
                "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Channel) REQUIRE (c.id, c.server_id) IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Server) REQUIRE s.id IS UNIQUE",
                "CREATE INDEX IF NOT EXISTS FOR (m:Message) ON (m.server_id, m.timestamp)",
                "CREATE INDEX IF NOT EXISTS FOR (u:User) ON u.username",
                "CREATE INDEX IF NOT EXISTS FOR ()-[r:REACTS_TO]-() ON r.timestamp",
                
                # Enhanced schema for v2.1 multi-server features
                "CREATE CONSTRAINT IF NOT EXISTS FOR (k:Knowledge) REQUIRE (k.id, k.server_id) IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE",
                "CREATE INDEX IF NOT EXISTS FOR (k:Knowledge) ON (k.server_id, k.content)",
                "CREATE INDEX IF NOT EXISTS FOR (c:Concept) ON c.category",
                
                # Personal AI Memory (user-specific, not server-specific)
                "CREATE CONSTRAINT IF NOT EXISTS FOR (m:Memory) REQUIRE m.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Project) REQUIRE (p.id, p.user_id) IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (g:Goal) REQUIRE (g.id, g.user_id) IS UNIQUE",
                "CREATE INDEX IF NOT EXISTS FOR (m:Memory) ON (m.user_id, m.timestamp)",
                
                # Gaming Systems (server-specific)
                "CREATE CONSTRAINT IF NOT EXISTS FOR (scene:Scene) REQUIRE (scene.id, scene.server_id) IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (char:Character) REQUIRE (char.id, char.server_id) IS UNIQUE", 
                "CREATE CONSTRAINT IF NOT EXISTS FOR (item:Item) REQUIRE item.name IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (quest:Quest) REQUIRE (quest.id, quest.server_id) IS UNIQUE",
                
                # Content Management (user-specific but can be server-contextualized)
                "CREATE CONSTRAINT IF NOT EXISTS FOR (art:Artwork) REQUIRE art.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (story:Story) REQUIRE story.id IS UNIQUE",
                "CREATE INDEX IF NOT EXISTS FOR (art:Artwork) ON (art.user_id, art.created_date)",
                
                # Ideas and Creativity (user-specific)
                "CREATE CONSTRAINT IF NOT EXISTS FOR (idea:Idea) REQUIRE idea.id IS UNIQUE",
                "CREATE INDEX IF NOT EXISTS FOR (idea:Idea) ON (idea.user_id, idea.tags)"
            ]
            
            for query in schema_queries:
                session.run(query)
                
        print("✅ Neo4j multi-server database initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize Neo4j: {e}")
        return False

def close_neo4j():
    """Close Neo4j connection."""
    global neo4j_driver
    if neo4j_driver:
        neo4j_driver.close()
        print("Neo4j connection closed")

# --- SERVER MANAGEMENT FUNCTIONS ---

async def register_server(guild: discord.Guild):
    """Register a new server in the database."""
    if not NEO4J_ENABLED or not neo4j_driver:
        return False
        
    try:
        with neo4j_driver.session() as session:
            query = """
            MERGE (s:Server {id: $server_id})
            SET s.name = $server_name,
                s.member_count = $member_count,
                s.created_date = datetime($created_date),
                s.registered_date = datetime()
            RETURN s.id as server_id
            """
            
            result = session.run(query,
                server_id=guild.id,
                server_name=guild.name,
                member_count=guild.member_count,
                created_date=guild.created_at.isoformat()
            )
            
            # Initialize server config
            get_server_config(guild.id, guild.name)
            
            print(f"✅ Registered server: {guild.name} (ID: {guild.id})")
            return True
            
    except Exception as e:
        print(f"❌ Error registering server {guild.name}: {e}")
        return False

async def get_server_stats():
    """Get statistics about all servers using the bot."""
    if not NEO4J_ENABLED or not neo4j_driver:
        return None
        
    try:
        with neo4j_driver.session() as session:
            query = """
            MATCH (s:Server)
            OPTIONAL MATCH (s)<-[:IN_SERVER]-(c:Channel)<-[:IN_CHANNEL]-(m:Message)
            OPTIONAL MATCH (s)<-[:IN_SERVER]-(ch:Channel)<-[:ACTIVE_IN]-(u:User)
            
            RETURN s.id as server_id,
                   s.name as server_name,
                   s.member_count as member_count,
                   count(DISTINCT m) as total_messages,
                   count(DISTINCT u) as active_users,
                   s.registered_date as registered
            ORDER BY total_messages DESC
            """
            
            result = session.run(query)
            return [dict(record) for record in result]
            
    except Exception as e:
        print(f"❌ Error getting server stats: {e}")
        return None

# --- MULTI-SERVER DATA ISOLATION FUNCTIONS ---

async def store_message_data(message_data):
    """Store message data with server isolation."""
    if not NEO4J_ENABLED or not neo4j_driver:
        return False
        
    try:
        with neo4j_driver.session() as session:
            query = """
            MERGE (u:User {id: $user_id})
            SET u.username = $username
            
            MERGE (c:Channel {id: $channel_id, server_id: $server_id})
            SET c.name = $channel_name
            
            MERGE (s:Server {id: $server_id})
            
            CREATE (m:Message {
                id: $message_id,
                server_id: $server_id,
                content_length: $content_length,
                timestamp: datetime($timestamp),
                reactions_count: $reactions
            })
            
            MERGE (u)-[:SENT]->(m)
            MERGE (m)-[:IN_CHANNEL]->(c)
            MERGE (c)-[:IN_SERVER]->(s)
            
            // Track user activity in channel (server-specific)
            MERGE (u)-[a:ACTIVE_IN]->(c)
            SET a.last_active = datetime($timestamp),
                a.message_count = coalesce(a.message_count, 0) + 1
            
            // Create mention relationships
            WITH m
            UNWIND $mentions as mention_id
            MERGE (mentioned:User {id: mention_id})
            MERGE (m)-[:MENTIONS]->(mentioned)
            """
            
            session.run(query, message_data)
            return True
            
    except Exception as e:
        print(f"❌ Error storing message data: {e}")
        return False

async def store_knowledge_server_isolated(entry, server_id: int):
    """Store knowledge with server isolation."""
    if not NEO4J_ENABLED or not neo4j_driver:
        return False
        
    try:
        with neo4j_driver.session() as session:
            knowledge_id = hashlib.md5(f"{entry.content}{server_id}".encode()).hexdigest()
            
            query = """
            MERGE (k:Knowledge {id: $knowledge_id, server_id: $server_id})
            SET k.content = $content,
                k.category = $category,
                k.user_id = $user_id,
                k.created_date = datetime(),
                k.sources = $sources
            
            MERGE (u:User {id: $user_id})
            MERGE (s:Server {id: $server_id})
            MERGE (u)-[:CONTRIBUTED]->(k)
            MERGE (k)-[:BELONGS_TO]->(s)
            
            // Create concept nodes and relationships
            WITH k
            UNWIND $concepts as concept_name
            MERGE (c:Concept {name: concept_name})
            SET c.category = $category
            MERGE (k)-[:RELATES_TO]->(c)
            """
            
            session.run(query, 
                knowledge_id=knowledge_id,
                server_id=server_id,
                content=entry.content,
                category=entry.category, 
                user_id=entry.user_id,
                concepts=entry.concepts,
                sources=entry.sources
            )
            return knowledge_id
            
    except Exception as e:
        print(f"❌ Error storing server-isolated knowledge: {e}")
        return None

async def query_knowledge_server_isolated(query_text: str, server_id: int, user_id: int = None):
    """Query knowledge base with server isolation."""
    if not NEO4J_ENABLED or not neo4j_driver:
        return []
        
    try:
        with neo4j_driver.session() as session:
            keywords = [word.lower() for word in query_text.split() if len(word) > 3]
            
            cypher_query = """
            MATCH (k:Knowledge {server_id: $server_id})
            WHERE any(keyword in $keywords WHERE toLower(k.content) CONTAINS keyword)
            
            OPTIONAL MATCH (k)-[:RELATES_TO]->(c:Concept)
            OPTIONAL MATCH (u:User)-[:CONTRIBUTED]->(k)
            
            RETURN k.content as content, 
                   k.category as category,
                   collect(DISTINCT c.name) as concepts,
                   u.username as contributor,
                   k.created_date as created
            ORDER BY k.created_date DESC
            LIMIT 10
            """
            
            result = session.run(cypher_query, keywords=keywords, server_id=server_id)
            return [dict(record) for record in result]
            
    except Exception as e:
        print(f"❌ Error querying server-isolated knowledge: {e}")
        return []

# --- ENHANCED BOT CLASS FOR MULTI-SERVER ---

class MultiServerDuckBot(commands.Bot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.server_count = 0
        self.total_users = 0

    async def setup_hook(self):
        """Enhanced setup for multi-server support."""
        print("Running multi-server setup hook...")
        
        # Initialize Neo4j
        if NEO4J_ENABLED:
            print("Initializing Neo4j database...")
            initialize_neo4j()
        
        # Sync commands globally
        try:
            synced = await self.tree.sync()
            print("--- GLOBAL COMMAND SYNC SUCCESS ---")
            print(f"Synced {len(synced)} command(s) globally: {[c.name for c in synced]}")
            print("-----------------------------------")
        except Exception as e:
            print(f"!!! FAILED TO SYNC GLOBAL COMMANDS: {e} !!!")

    async def on_guild_join(self, guild):
        """Handle bot joining a new server."""
        print(f"🆕 Joined new server: {guild.name} (ID: {guild.id}, Members: {guild.member_count})")
        
        # Check server limit
        if len(self.guilds) > MAX_SERVERS_PER_INSTANCE:
            print(f"⚠️ Server limit exceeded ({MAX_SERVERS_PER_INSTANCE}). Consider scaling to multiple instances.")
        
        # Register server in database
        await register_server(guild)
        
        # Send welcome message to the server
        if guild.system_channel:
            embed = discord.Embed(
                title="🦆 DuckBot v2.1 Multi-Server Edition",
                description="Thanks for adding me! I'm an advanced AI assistant with knowledge management, personal memory, and creative tools.",
                color=0x00ff88
            )
            embed.add_field(
                name="🚀 Quick Start",
                value="• `/ping` - Test connectivity\n• `/learn` - Teach me knowledge\n• `/generate` - AI image generation\n• `/help_server` - See all commands",
                inline=False
            )
            embed.add_field(
                name="🛠️ Server Setup",
                value="• Admins can use `/server_config` to customize features\n• Each server has isolated data and settings\n• Enable Neo4j for advanced features",
                inline=False
            )
            
            try:
                await guild.system_channel.send(embed=embed)
            except:
                pass  # If we can't send, that's okay

    async def on_guild_remove(self, guild):
        """Handle bot leaving a server."""
        print(f"👋 Left server: {guild.name} (ID: {guild.id})")
        
        # Clean up server data if configured to do so
        # (You might want to keep data for a while in case they re-add the bot)
        
        # Remove from local configs  
        if guild.id in server_configs:
            del server_configs[guild.id]
        if guild.id in server_queues:
            del server_queues[guild.id]

intents = discord.Intents.default()
intents.message_content = True
intents.messages = True
intents.guilds = True
intents.reactions = True
intents.members = True
bot = MultiServerDuckBot(command_prefix="!", intents=intents)

# --- MULTI-SERVER EVENT HANDLERS ---

@bot.event
async def on_ready():
    print(f'🚀 DuckBot v2.1 Multi-Server Edition Online!')
    print(f'Logged in as {bot.user} (ID: {bot.user.id})')
    print(f'Connected to {len(bot.guilds)} servers')
    print(f'Total users across servers: {sum(guild.member_count for guild in bot.guilds)}')
    print('------')

@bot.event
async def on_message(message):
    """Enhanced message handler with server isolation."""
    # Skip bot messages
    if message.author.bot:
        return
    
    # Only process if in a guild (not DMs)
    if not message.guild:
        return
    
    server_id = message.guild.id
    
    # Check if features are enabled for this server
    config = get_server_config(server_id, message.guild.name)
    if not config.features_enabled.get('social_analytics', True):
        return
    
    # Store message data with server isolation
    message_data = {
        "message_id": message.id,
        "user_id": message.author.id,
        "username": message.author.name,
        "channel_id": message.channel.id,
        "channel_name": message.channel.name,
        "server_id": server_id,
        "content_length": len(message.content),
        "timestamp": message.created_at.isoformat(),
        "mentions": [user.id for user in message.mentions],
        "reactions": len(message.reactions) if message.reactions else 0
    }
    await store_message_data(message_data)

# --- MULTI-SERVER COMMANDS ---

@bot.tree.command(name="server_info", description="Get information about this server's DuckBot setup")
async def server_info_command(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=False)
    
    server_id = interaction.guild.id
    config = get_server_config(server_id, interaction.guild.name)
    
    embed = discord.Embed(
        title=f"🏠 Server Info: {interaction.guild.name}",
        color=0x3498db
    )
    
    # Basic server info
    embed.add_field(name="Server ID", value=server_id, inline=True)
    embed.add_field(name="Members", value=interaction.guild.member_count, inline=True)
    embed.add_field(name="Channels", value=len(interaction.guild.channels), inline=True)
    
    # Feature status
    features_status = []
    for feature, enabled in config.features_enabled.items():
        status = "✅" if enabled else "❌"
        features_status.append(f"{status} {feature.replace('_', ' ').title()}")
    
    embed.add_field(
        name="📋 Features Status",
        value="\n".join(features_status),
        inline=False
    )
    
    # Neo4j status
    neo4j_status = "✅ Connected" if NEO4J_ENABLED else "❌ Disabled"
    embed.add_field(name="🗄️ Database", value=neo4j_status, inline=True)
    
    # Queue status
    queue_info = get_server_queue(server_id)
    queue_length = len(queue_info['queue'])
    processing = "Yes" if queue_info['currently_processing'] else "No"
    embed.add_field(name="🎨 Generation Queue", value=f"{queue_length} items", inline=True)
    embed.add_field(name="Currently Processing", value=processing, inline=True)
    
    await interaction.followup.send(embed=embed)

@bot.tree.command(name="server_config", description="Configure server settings (Admin only)")
@app_commands.describe(
    feature="Feature to toggle",
    enabled="Enable or disable the feature"
)
@app_commands.choices(feature=[
    app_commands.Choice(name="Knowledge Management", value="knowledge_management"),
    app_commands.Choice(name="Personal Memory", value="personal_memory"),
    app_commands.Choice(name="Adventures", value="adventures"),
    app_commands.Choice(name="Art Tracking", value="art_tracking"),
    app_commands.Choice(name="Idea System", value="idea_system"),
    app_commands.Choice(name="Social Analytics", value="social_analytics"),
])
async def server_config_command(interaction: discord.Interaction, feature: str, enabled: bool):
    # Check admin permissions
    if not interaction.user.guild_permissions.administrator and interaction.user.id not in GLOBAL_ADMIN_IDS:
        await interaction.response.send_message("❌ Admin permissions required", ephemeral=True)
        return
    
    await interaction.response.defer(ephemeral=False)
    
    server_id = interaction.guild.id
    config = get_server_config(server_id, interaction.guild.name)
    
    # Update feature setting
    config.features_enabled[feature] = enabled
    
    status = "enabled" if enabled else "disabled"
    feature_name = feature.replace('_', ' ').title()
    
    embed = discord.Embed(
        title="⚙️ Server Configuration Updated",
        description=f"**{feature_name}** has been **{status}** for this server.",
        color=0x00ff88 if enabled else 0xe74c3c
    )
    
    await interaction.followup.send(embed=embed)

@bot.tree.command(name="global_stats", description="View DuckBot statistics across all servers (Global Admin only)")
async def global_stats_command(interaction: discord.Interaction):
    if interaction.user.id not in GLOBAL_ADMIN_IDS:
        await interaction.response.send_message("❌ Global admin permissions required", ephemeral=True)
        return
    
    await interaction.response.defer(ephemeral=True)
    
    # Get server statistics
    server_stats = await get_server_stats()
    
    embed = discord.Embed(
        title="🌐 DuckBot Global Statistics",
        color=0x9b59b6
    )
    
    # Basic stats
    total_servers = len(bot.guilds)
    total_users = sum(guild.member_count for guild in bot.guilds)
    
    embed.add_field(name="Total Servers", value=total_servers, inline=True)
    embed.add_field(name="Total Users", value=f"{total_users:,}", inline=True)
    embed.add_field(name="Database Status", value="✅ Connected" if NEO4J_ENABLED else "❌ Disabled", inline=True)
    
    # Top servers by activity
    if server_stats:
        top_servers = sorted(server_stats, key=lambda x: x['total_messages'] or 0, reverse=True)[:5]
        
        server_list = []
        for server in top_servers:
            name = server['server_name'][:20] if server['server_name'] else "Unknown"
            messages = server['total_messages'] or 0
            users = server['active_users'] or 0
            server_list.append(f"**{name}**: {messages:,} messages, {users} users")
        
        embed.add_field(
            name="📊 Top Servers by Activity",
            value="\n".join(server_list) if server_list else "No data available",
            inline=False
        )
    
    await interaction.followup.send(embed=embed)

# [Continue with all the v2.0 commands but with server isolation...]
# I'll add the key multi-server enhanced commands here

@bot.tree.command(name="learn", description="Teach DuckBot something new (Server-specific)")
@app_commands.describe(
    content="What do you want to teach me?",
    category="Category (science, programming, art, etc.)",
    concepts="Related concepts (comma-separated)"
)
async def learn_command(interaction: discord.Interaction, content: str, category: str = "general", concepts: str = None):
    await interaction.response.defer(ephemeral=False)
    
    # Check if feature is enabled for this server
    config = get_server_config(interaction.guild.id, interaction.guild.name)
    if not config.features_enabled.get('knowledge_management', True):
        await interaction.followup.send("❌ Knowledge management is disabled for this server. Ask an admin to enable it with `/server_config`")
        return
    
    if not NEO4J_ENABLED:
        await interaction.followup.send("❌ Knowledge system requires Neo4j. Set NEO4J_ENABLED=true in .env")
        return
    
    concept_list = []
    if concepts:
        concept_list = [c.strip() for c in concepts.split(",")]
    
    # Create knowledge entry
    from dataclasses import dataclass, field
    
    @dataclass
    class KnowledgeEntry:
        content: str
        category: str
        user_id: int
        concepts: List[str] = field(default_factory=list)
        sources: List[str] = field(default_factory=list)
    
    entry = KnowledgeEntry(
        content=content,
        category=category,
        user_id=interaction.user.id,
        concepts=concept_list
    )
    
    # Store with server isolation
    knowledge_id = await store_knowledge_server_isolated(entry, interaction.guild.id)
    
    if knowledge_id:
        embed = discord.Embed(
            title="🧠 Knowledge Learned!",
            description=f"I've learned about **{category}** for **{interaction.guild.name}**",
            color=0x00ff88
        )
        embed.add_field(name="Content", value=content[:1000], inline=False)
        if concepts:
            embed.add_field(name="Related Concepts", value=concepts, inline=False)
        embed.set_footer(text=f"Knowledge ID: {knowledge_id[:8]}... | Server-specific knowledge")
        
        await interaction.followup.send(embed=embed)
    else:
        await interaction.followup.send("❌ Failed to store knowledge")

@bot.tree.command(name="ask_knowledge", description="Query this server's knowledge base")
@app_commands.describe(query="What do you want to know about?")
async def ask_knowledge_command(interaction: discord.Interaction, query: str):
    await interaction.response.defer(ephemeral=False)
    
    # Check if feature is enabled for this server
    config = get_server_config(interaction.guild.id, interaction.guild.name)
    if not config.features_enabled.get('knowledge_management', True):
        await interaction.followup.send("❌ Knowledge management is disabled for this server")
        return
    
    if not NEO4J_ENABLED:
        await interaction.followup.send("❌ Knowledge system requires Neo4j")
        return
    
    # Query with server isolation
    results = await query_knowledge_server_isolated(query, interaction.guild.id, interaction.user.id)
    
    if not results:
        await interaction.followup.send(f"🤔 I don't know anything about '{query}' in **{interaction.guild.name}** yet. Try `/learn` to teach me!")
        return
    
    embed = discord.Embed(
        title=f"🔍 {interaction.guild.name}'s Knowledge: '{query}'",
        color=0x3498db
    )
    
    for i, result in enumerate(results[:5]):  # Show top 5 results
        concepts_str = ", ".join(result['concepts'][:3]) if result['concepts'] else "None"
        
        embed.add_field(
            name=f"📚 {result['category'].title()}",
            value=f"{result['content'][:200]}{'...' if len(result['content']) > 200 else ''}\n"
                  f"*Concepts: {concepts_str}*\n"
                  f"*Contributed by: {result['contributor'] or 'Unknown'}*",
            inline=False
        )
    
    embed.set_footer(text=f"Showing server-specific knowledge for {interaction.guild.name}")
    await interaction.followup.send(embed=embed)

# [The rest of the commands would follow the same pattern with server isolation...]

# --- ENHANCED IMAGE GENERATION SYSTEM ---

# Enhanced Image Models Configuration
IMAGE_MODELS = {
    "flux": {
        "name": "FLUX.1 Schnell",
        "description": "⚡ Ultra-fast, highest quality photorealistic images",
        "workflow": "workflow_flux_api.json",
        "steps": 4,
        "cfg": 1.0,
        "resolution": "1024x1024",
        "speed": "Very Fast (2-4s)",
        "best_for": "Photorealism, portraits, detailed scenes",
        "model_file": "flux1-schnell.safetensors",
        "clip_file": "t5xxl_fp16.safetensors"
    },
    "sdxl": {
        "name": "Stable Diffusion XL",
        "description": "🎨 High-quality versatile generation with refiner",
        "workflow": "workflow_sdxl_api.json", 
        "steps": 25,
        "cfg": 7.0,
        "resolution": "1024x1024",
        "speed": "Medium (10-15s)",
        "best_for": "Art, creative styles, general purpose",
        "model_file": "sd_xl_base_1.0.safetensors",
        "refiner_file": "sd_xl_refiner_1.0.safetensors"
    },
    "sd15": {
        "name": "Stable Diffusion 1.5",
        "description": "🚀 Fast, reliable, wide style compatibility",
        "workflow": "workflow_api.json",
        "steps": 20,
        "cfg": 8.0,
        "resolution": "512x512",
        "speed": "Fast (5-8s)",
        "best_for": "Quick generation, artistic styles",
        "model_file": "v1-5-pruned-emaonly-fp16.safetensors"  # Updated to match your ComfyUI
    }
}

# Model priority order (best to worst) - SD15 first since it's most commonly available
MODEL_PRIORITY = ["sd15", "sdxl", "flux"]

def get_available_models() -> List[str]:
    """Check which model workflows and model files are available."""
    available = []
    
    # Check SD 1.5 - most basic model
    if os.path.exists("workflow_api.json"):
        # Check if SD 1.5 model exists in ComfyUI
        sd15_paths = [
            "ComfyUI_windows_portable/ComfyUI/models/checkpoints/v1-5-pruned-emaonly.ckpt",
            "ComfyUI_windows_portable/ComfyUI/models/checkpoints/v1-5-pruned-emaonly-fp16.safetensors", 
            "models/checkpoints/v1-5-pruned-emaonly.ckpt",
            "models/checkpoints/v1-5-pruned-emaonly-fp16.safetensors"
        ]
        if any(os.path.exists(path) for path in sd15_paths):
            available.append("sd15")
    
    # Check SDXL - requires both base and refiner
    if os.path.exists("workflow_sdxl_api.json"):
        sdxl_base_paths = [
            "ComfyUI_windows_portable/ComfyUI/models/checkpoints/sd_xl_base_1.0.safetensors",
            "models/checkpoints/sd_xl_base_1.0.safetensors"
        ]
        if any(os.path.exists(path) for path in sdxl_base_paths):
            available.append("sdxl")
    
    # Check FLUX.1 - requires both model and clip
    if os.path.exists("workflow_flux_api.json"):
        flux_model_paths = [
            "ComfyUI_windows_portable/ComfyUI/models/checkpoints/flux1-schnell.safetensors",
            "models/checkpoints/flux1-schnell.safetensors"
        ]
        flux_clip_paths = [
            "ComfyUI_windows_portable/ComfyUI/models/clip/t5xxl_fp16.safetensors",
            "models/clip/t5xxl_fp16.safetensors"
        ]
        if (any(os.path.exists(path) for path in flux_model_paths) and 
            any(os.path.exists(path) for path in flux_clip_paths)):
            available.append("flux")
    
    # If no models detected, assume SD 1.5 is available (most common)
    if not available:
        available.append("sd15")
    
    return available

def get_best_available_model() -> str:
    """Get the best available model based on priority."""
    available = get_available_models()
    for model_id in MODEL_PRIORITY:
        if model_id in available:
            return model_id
    return "sd15"  # Fallback

async def load_workflow_for_model(model_id: str) -> Optional[Dict]:
    """Load and return the workflow for a specific model."""
    if model_id not in IMAGE_MODELS:
        return None
        
    workflow_path = IMAGE_MODELS[model_id]["workflow"]
    
    try:
        with open(workflow_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading workflow for {model_id}: {e}")
        return None

def update_workflow_prompt(workflow: Dict, prompt: str, model_id: str) -> Dict:
    """Update workflow with new prompt, handling different model structures."""
    
    if model_id == "flux":
        # FLUX uses node 3 for positive prompt, node 7 for negative
        workflow["3"]["inputs"]["text"] = prompt
        # Enhanced negative prompt for FLUX
        workflow["7"]["inputs"]["text"] = "blurry, low quality, watermark, text, bad anatomy, deformed, noise, artifacts, oversaturated"
        
    elif model_id == "sdxl":
        # SDXL uses node 3 for positive prompt, node 7 for negative
        enhanced_prompt = f"{prompt}, masterpiece, best quality, highly detailed, sharp focus"
        workflow["3"]["inputs"]["text"] = enhanced_prompt
        workflow["7"]["inputs"]["text"] = "blurry, low quality, watermark, text, bad anatomy, deformed, pixelated, noise, artifacts, oversaturated, cartoon, anime"
        
    elif model_id == "sd15":
        # SD 1.5 uses node 3 for positive prompt, node 7 for negative
        workflow["3"]["inputs"]["text"] = prompt
        workflow["7"]["inputs"]["text"] = "text, watermark, blurry, low quality"
    
    return workflow

async def enhanced_process_image_generation(queue_item, model_id: str = None):
    """Enhanced image generation with model selection."""
    start_time = time.time()
    
    # Determine which model to use
    if not model_id:
        model_id = get_best_available_model()
    
    model_info = IMAGE_MODELS.get(model_id, IMAGE_MODELS["sd15"])
    
    try:
        # Load workflow for the selected model
        prompt_workflow = await load_workflow_for_model(model_id)
        
        if not prompt_workflow:
            await queue_item.status_message.edit(
                content=f"❌ Error loading {model_info['name']} workflow"
            )
            return
        
        # Update workflow with prompt and model-specific enhancements
        prompt_workflow = update_workflow_prompt(prompt_workflow, queue_item.prompt, model_id)
        
        # Update seed
        seed_node = "6"  # Most models use node 6 for KSampler
        prompt_workflow[seed_node]["inputs"]["seed"] = torch.randint(1, 1125899906842624, (1,)).item()
        
        # Update status with model info
        await queue_item.status_message.edit(
            content=f"🎨 **Generating with {model_info['name']}**\n"
                   f"Prompt: `{queue_item.prompt}`\n"
                   f"⚡ {model_info['speed']} | 📐 {model_info['resolution']}"
        )
        
        # Generate image using existing ComfyUI function
        images_data = await run_comfyui_workflow(prompt_workflow, is_video=False)
        
        if not images_data:
            await queue_item.status_message.edit(
                content=f"❌ **{model_info['name']} generation failed**\nPrompt: `{queue_item.prompt}`"
            )
            return
        
        # Create Discord files (simple working version)
        image_files = [discord.File(fp=BytesIO(data), filename=f"{model_id}_image_{uuid.uuid4()}.png") for data in images_data]
        
        # Send images FIRST (before editing status)
        await queue_item.interaction.followup.send(files=image_files)
        
        # Then update status message
        await queue_item.status_message.edit(content=f"✅ **{model_info['name']} generation complete!**\nPrompt: `{queue_item.prompt}`")
        
        # Update average time
        actual_time = time.time() - start_time
        update_average_time("image", actual_time)
        
    except Exception as e:
        await queue_item.status_message.edit(
            content=f"❌ **{model_info['name']} generation error**\n"
                   f"Prompt: `{queue_item.prompt}`\n"
                   f"Error: {str(e)[:100]}..."
        )

# Enhanced generate command with model selection
@bot.tree.command(name="generate_advanced", description="Generate images with model selection")
@app_commands.describe(
    prompt="The prompt for the image",
    model="Image generation model to use"
)
@app_commands.choices(model=[
    app_commands.Choice(name="🌟 FLUX.1 (Best Quality)", value="flux"),
    app_commands.Choice(name="🎨 SDXL (Versatile)", value="sdxl"), 
    app_commands.Choice(name="🚀 SD 1.5 (Fast)", value="sd15"),
    app_commands.Choice(name="🎯 Auto (Best Available)", value="auto")
])
async def generate_advanced_command(interaction: discord.Interaction, prompt: str, model: str = "auto"):
    await interaction.response.defer(ephemeral=False)
    
    # Determine model to use
    if model == "auto":
        selected_model = get_best_available_model()
    else:
        selected_model = model
    
    # Check if model is available
    available_models = get_available_models()
    if selected_model not in available_models:
        embed = discord.Embed(
            title="❌ Model Not Available",
            description=f"The {IMAGE_MODELS[selected_model]['name']} model is not available.",
            color=0xe74c3c
        )
        embed.add_field(
            name="📋 Available Models",
            value="\n".join([f"• {IMAGE_MODELS[m]['name']}" for m in available_models]),
            inline=False
        )
        embed.add_field(
            name="💡 Solution",
            value="Download the model files and place them in your ComfyUI models folder:\n"
                  f"• **FLUX.1:** `{IMAGE_MODELS['flux']['model_file']}` + `{IMAGE_MODELS['flux']['clip_file']}`\n"
                  f"• **SDXL:** `{IMAGE_MODELS['sdxl']['model_file']}` + `{IMAGE_MODELS['sdxl']['refiner_file']}`\n"
                  f"• **SD 1.5:** `{IMAGE_MODELS['sd15']['model_file']}`",
            inline=False
        )
        await interaction.followup.send(embed=embed)
        return
    
    # Create queue item with model specification
    queue_item = QueueItem(interaction, prompt, "image", interaction.guild.id)
    queue_item.model_id = selected_model
    
    # Add to server queue with model-specific processing
    server_queue = get_server_queue(interaction.guild.id)
    async with server_queue['lock']:
        server_queue['queue'].append(queue_item)
        position = len(server_queue['queue'])
        
        model_info = IMAGE_MODELS[selected_model]
        
        if position == 1 and not server_queue['currently_processing']:
            queue_item.status_message = await interaction.followup.send(
                f"🎨 **Starting {model_info['name']} generation**\n"
                f"Prompt: `{prompt}`\n"
                f"🚀 Processing now... ({model_info['speed']})"
            )
        else:
            wait_time = calculate_estimated_wait(position, "image")
            wait_str = f"\n⏱️ Estimated wait: {format_time(wait_time)}" if wait_time > 0 else ""
            queue_item.status_message = await interaction.followup.send(
                f"🎨 **{model_info['name']} generation queued**\n"
                f"Prompt: `{prompt}`\n"
                f"📍 Position {position} in queue{wait_str}"
            )
    
    # Start processing if not already running
    if not server_queue['currently_processing']:
        asyncio.create_task(process_enhanced_queue(interaction.guild.id))

@bot.tree.command(name="model_info", description="View available image generation models")
async def model_info_command(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=False)
    
    available_models = get_available_models()
    
    embed = discord.Embed(
        title="🎨 Available Image Generation Models",
        description="Choose the best model for your needs!",
        color=0x3498db
    )
    
    for model_id in MODEL_PRIORITY:
        if model_id in IMAGE_MODELS:
            model_info = IMAGE_MODELS[model_id]
            status = "✅ Available" if model_id in available_models else "❌ Not Installed"
            
            embed.add_field(
                name=f"{model_info['name']} {status}",
                value=f"{model_info['description']}\n"
                      f"**Speed:** {model_info['speed']}\n"
                      f"**Resolution:** {model_info['resolution']}\n"
                      f"**Best For:** {model_info['best_for']}",
                inline=False
            )
    
    embed.add_field(
        name="🚀 Usage",
        value="Use `/generate_advanced` to select a model, or `/generate` for auto-selection",
        inline=False
    )
    
    if len(available_models) == 0:
        embed.add_field(
            name="📥 Download Models",
            value="Visit [Hugging Face](https://huggingface.co) to download:\n"
                  "• **FLUX.1 Schnell** - Best quality, fastest\n"
                  "• **SDXL 1.0** - Great versatility\n"
                  "• **SD 1.5** - Reliable fallback",
            inline=False
        )
    
    await interaction.followup.send(embed=embed)

# Style preset command for enhanced prompting
@bot.tree.command(name="generate_style", description="Generate images with predefined artistic styles")
@app_commands.describe(
    prompt="Your base prompt",
    style="Artistic style to apply"
)
@app_commands.choices(style=[
    app_commands.Choice(name="📸 Photorealistic", value="photorealistic"),
    app_commands.Choice(name="🎨 Digital Art", value="digital_art"),
    app_commands.Choice(name="🖼️ Oil Painting", value="oil_painting"),
    app_commands.Choice(name="✏️ Pencil Sketch", value="sketch"),
    app_commands.Choice(name="🌸 Anime Style", value="anime"),
    app_commands.Choice(name="🏛️ Classical Art", value="classical"),
    app_commands.Choice(name="🌟 Fantasy", value="fantasy"),
    app_commands.Choice(name="🤖 Cyberpunk", value="cyberpunk")
])
async def generate_style_command(interaction: discord.Interaction, prompt: str, style: str):
    # Style enhancement prompts
    style_enhancements = {
        "photorealistic": "photorealistic, ultra realistic, high resolution, detailed, professional photography, DSLR quality",
        "digital_art": "digital art, concept art, artstation trending, highly detailed, digital painting, smooth",
        "oil_painting": "oil painting, classical art style, painterly, brushstrokes, artistic, renaissance style",
        "sketch": "pencil sketch, hand drawn, black and white, detailed lineart, crosshatching, artistic sketch",
        "anime": "anime style, manga, cel shading, vibrant colors, anime art, japanese animation style",
        "classical": "classical painting, museum quality, old master style, fine art, historical painting",
        "fantasy": "fantasy art, magical, mystical, fantasy style, enchanted, otherworldly, epic fantasy",
        "cyberpunk": "cyberpunk style, neon lights, futuristic, sci-fi, dark atmosphere, high tech low life"
    }
    
    # Enhance the prompt with style
    enhanced_prompt = f"{prompt}, {style_enhancements.get(style, '')}"
    
    # Replicate generate_advanced_command logic with auto model selection
    await interaction.response.defer(ephemeral=False)
    
    # Determine model to use (auto selection)
    selected_model = get_best_available_model()
    
    # Check if model is available
    available_models = get_available_models()
    if selected_model not in available_models:
        embed = discord.Embed(
            title="❌ Model Not Available",
            description=f"No image generation models are currently available.",
            color=0xe74c3c
        )
        await interaction.followup.send(embed=embed)
        return
    
    # Create queue item with model specification
    queue_item = QueueItem(interaction, enhanced_prompt, "image", interaction.guild.id)
    queue_item.model_id = selected_model
    
    # Add to server queue with model-specific processing
    server_queue = get_server_queue(interaction.guild.id)
    async with server_queue['lock']:
        server_queue['queue'].append(queue_item)
        position = len(server_queue['queue'])
        
        model_info = IMAGE_MODELS[selected_model]
        
        if position == 1 and not server_queue['currently_processing']:
            queue_item.status_message = await interaction.followup.send(
                f"🎨 **Starting {model_info['name']} generation with {style} style**\n"
                f"Prompt: `{enhanced_prompt}`\n"
                f"🚀 Processing now... ({model_info['speed']})"
            )
        else:
            wait_time = calculate_estimated_wait(position, "image")
            wait_str = f"\n⏱️ Estimated wait: {format_time(wait_time)}" if wait_time > 0 else ""
            queue_item.status_message = await interaction.followup.send(
                f"🎨 **{model_info['name']} generation queued with {style} style**\n"
                f"Prompt: `{enhanced_prompt}`\n"
                f"📍 Position {position} in queue{wait_str}"
            )
    
    # Start processing if not already running
    if not server_queue['currently_processing']:
        asyncio.create_task(process_enhanced_queue(interaction.guild.id))

async def process_enhanced_queue(server_id: int):
    """Enhanced queue processing with model support."""
    server_queue = get_server_queue(server_id)
    
    async with server_queue['lock']:
        if server_queue['currently_processing'] or not server_queue['queue']:
            return
        server_queue['currently_processing'] = True
    
    while server_queue['queue']:
        async with server_queue['lock']:
            if not server_queue['queue']:
                break
            current_item = server_queue['queue'].popleft()
        
        try:
            # Check if it's an enhanced generation with model selection
            if hasattr(current_item, 'model_id'):
                await enhanced_process_image_generation(current_item, current_item.model_id)
            else:
                # Use existing generation function for regular items
                if current_item.generation_type == "image":
                    await process_image_generation(current_item)
                elif current_item.generation_type == "video":
                    await process_video_generation(current_item)
        except Exception as e:
            model_name = IMAGE_MODELS.get(getattr(current_item, 'model_id', 'sd15'), {}).get('name', 'Unknown')
            await current_item.status_message.edit(
                content=f"❌ **{model_name} generation failed**\n"
                       f"Prompt: `{current_item.prompt}`\n"
                       f"Error: {str(e)[:100]}..."
            )
    
    server_queue['currently_processing'] = False

print("✅ Enhanced image generation system loaded!")
print("📋 New commands available:")
print("   • /generate_advanced - Choose specific models")
print("   • /model_info - View available models")
print("   • /generate_style - Apply artistic styles")
print("🔄 Your existing /generate command will continue to work as before!")

# --- ENHANCED ASK SYSTEM ---

# LM Studio Configuration
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
LM_STUDIO_TIMEOUT = 60  # seconds
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

# Plugin configurations (safe defaults - no plugins for maximum compatibility)
SAFE_PLUGIN_CONFIG = {}

# Conservative plugin config - try minimal plugins first, then none
MINIMAL_PLUGIN_CONFIG = {
    "plugins": {
        "dice": {"enabled": True}
    },
    "max_plugin_calls": 1,
    "plugin_timeout": 5
}

FULL_PLUGIN_CONFIG = {
    "plugins": {
        "duckduckgo": {"enabled": True, "priority": 1},
        "wikipedia": {"enabled": True, "priority": 2}
    },
    "max_plugin_calls": 2,
    "plugin_timeout": 10
}

async def check_lm_studio_health() -> bool:
    """Check if LM Studio is responding with fast async call."""
    try:
        print(f"🏥 Testing connection to {LM_STUDIO_URL}")
        health_payload = {
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 5,
            "temperature": 0
        }
        
        timeout = aiohttp.ClientTimeout(total=3)  # Very short timeout for health check
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                LM_STUDIO_URL,
                headers={"Content-Type": "application/json"},
                json=health_payload
            ) as response:
                print(f"🏥 Health check response status: {response.status}")
                return response.status == 200
    except Exception as e:
        print(f"🏥 Health check failed with error: {e}")
        return False

async def call_lm_studio_with_retry(payload: Dict[str, Any], max_retries: int = 1) -> Optional[Dict[str, Any]]:
    """Call LM Studio with async requests - no plugins since they consistently fail."""
    
    # Clean payload from the start - remove any plugin configs
    clean_payload = {k: v for k, v in payload.items() 
                    if k not in ["plugins", "max_plugin_calls", "plugin_timeout"]}
    
    # Single attempt with clean payload
    try:
        print(f"🔄 Attempting LM Studio call (no plugins)...")
        timeout = aiohttp.ClientTimeout(total=60)  # Extended timeout for slower responses
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                LM_STUDIO_URL,
                headers={"Content-Type": "application/json"},
                json=clean_payload
            ) as response:
                print(f"📊 LM Studio response: {response.status}")
                
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ LM Studio success: {len(str(data))} chars")
                    return data
                else:
                    print(f"❌ LM Studio failed: {response.status}")
                    
    except asyncio.TimeoutError:
        print("⏰ LM Studio timeout (60s)")
    except Exception as e:
        print(f"❌ LM Studio error: {e}")
    
    print("💥 LM Studio call failed")
    return None

async def call_lm_studio_async(prompt: str, user_name: str, system_role: str = "helpful assistant") -> Optional[str]:
    """Simple async wrapper for LM Studio calls - returns just the text content."""
    payload = {
        "messages": [
            {"role": "system", "content": f"You are a {system_role}. Be creative and engaging."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 500,
        "temperature": 0.8
    }
    
    try:
        response_data = await call_lm_studio_with_retry(payload)
        if response_data and "choices" in response_data:
            return response_data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Error in call_lm_studio_async: {e}")
    
    return None

def roll_dice(sides: int = 20, count: int = 1, modifier: int = 0) -> dict:
    """Roll dice and return detailed results for D&D mechanics."""
    rolls = [random.randint(1, sides) for _ in range(count)]
    total = sum(rolls) + modifier
    
    return {
        "rolls": rolls,
        "modifier": modifier,
        "total": total,
        "sides": sides,
        "count": count,
        "natural_20": any(roll == 20 for roll in rolls) and sides == 20,
        "natural_1": any(roll == 1 for roll in rolls) and sides == 20
    }

def get_stat_modifier(stat_value: int) -> int:
    """Get D&D stat modifier from stat value."""
    return (stat_value - 10) // 2

def determine_basic_outcome(action: str, character_data: dict) -> dict:
    """Determine outcome for basic (non-D&D) adventures using simple dice mechanics."""
    action_lower = action.lower()
    
    # Determine difficulty and which stat to use
    if any(word in action_lower for word in ["attack", "fight", "combat", "break", "force"]):
        base_roll = roll_dice(6, 1, 0)
        difficulty = 4
        stat_used = "skill"
        action_type = "Action"
    elif any(word in action_lower for word in ["sneak", "hide", "avoid", "dodge", "careful"]):
        base_roll = roll_dice(6, 1, character_data.get("luck", 3) - 3)
        difficulty = 4
        stat_used = "luck"
        action_type = "Luck"
    elif any(word in action_lower for word in ["search", "examine", "investigate", "study", "think"]):
        base_roll = roll_dice(6, 1, character_data.get("skill", 3) - 3)
        difficulty = 3
        stat_used = "skill"
        action_type = "Skill"
    else:
        # General action
        base_roll = roll_dice(6, 1, 0)
        difficulty = 4
        stat_used = "general"
        action_type = "General"
    
    # Determine success level
    if base_roll["rolls"][0] == 6:
        success_level = "critical_success"
    elif base_roll["rolls"][0] == 1 and base_roll["total"] <= 1:
        success_level = "critical_failure"
    elif base_roll["total"] >= difficulty + 2:
        success_level = "great_success"
    elif base_roll["total"] >= difficulty:
        success_level = "success"
    else:
        success_level = "failure"
    
    return {
        "roll_result": base_roll,
        "stat_used": stat_used,
        "difficulty": difficulty,
        "action_type": action_type,
        "success_level": success_level
    }

def determine_dnd_outcome(action: str, dnd_data: dict) -> dict:
    """Determine D&D outcome based on action and character stats."""
    action_lower = action.lower()
    
    # Determine which stat to use based on action keywords
    if any(word in action_lower for word in ["attack", "fight", "hit", "strike", "combat"]):
        stat = "strength"
        difficulty = 12
        action_type = "Attack"
    elif any(word in action_lower for word in ["sneak", "hide", "stealth", "dodge", "acrobatics"]):
        stat = "dexterity" 
        difficulty = 13
        action_type = "Stealth"
    elif any(word in action_lower for word in ["investigate", "search", "examine", "study", "recall"]):
        stat = "intelligence"
        difficulty = 11
        action_type = "Investigation"
    elif any(word in action_lower for word in ["perceive", "listen", "spot", "notice", "sense"]):
        stat = "wisdom"
        difficulty = 12
        action_type = "Perception"
    elif any(word in action_lower for word in ["persuade", "convince", "charm", "intimidate", "deceive"]):
        stat = "charisma"
        difficulty = 13
        action_type = "Persuasion"
    else:
        # General action
        stat = "dexterity"
        difficulty = 12
        action_type = "General"
    
    # Get stat modifier and roll
    stat_value = dnd_data.get("stats", {}).get(stat, 10)
    modifier = get_stat_modifier(stat_value)
    roll_result = roll_dice(20, 1, modifier)
    
    # Determine success level
    if roll_result["natural_20"]:
        success_level = "critical_success"
    elif roll_result["natural_1"]:
        success_level = "critical_failure"
    elif roll_result["total"] >= difficulty + 5:
        success_level = "great_success"
    elif roll_result["total"] >= difficulty:
        success_level = "success"
    else:
        success_level = "failure"
    
    return {
        "roll_result": roll_result,
        "stat_used": stat,
        "difficulty": difficulty,
        "action_type": action_type,
        "success_level": success_level
    }

async def get_fallback_response(prompt: str, user_name: str) -> str:
    """Generate a fallback response when LM Studio is unavailable."""
    fallback_responses = [
        f"I'm having trouble connecting to my AI brain right now, {user_name}! 🤖 Could you try asking again in a moment?",
        f"Oops! My AI circuits are a bit scrambled at the moment, {user_name}. Give me a second to reboot! ⚡",
        f"Sorry {user_name}, I'm experiencing some technical difficulties. Try your question again shortly! 🛠️",
        f"My AI assistant seems to be taking a coffee break, {user_name}! ☕ Please retry in a moment.",
        f"Houston, we have a problem! 🚀 My AI systems are temporarily offline, {user_name}. Try again soon!"
    ]
    
    base_response = random.choice(fallback_responses)
    
    # Add helpful suggestions based on the prompt
    if any(word in prompt.lower() for word in ["generate", "image", "picture", "art"]):
        base_response += "\n\n💡 *In the meantime, you can try `/generate_advanced` for AI image creation!*"
    elif any(word in prompt.lower() for word in ["animate", "video", "movie"]):
        base_response += "\n\n💡 *You might want to try `/animate` for AI video generation!*"
    elif any(word in prompt.lower() for word in ["learn", "teach", "knowledge"]):
        base_response += "\n\n💡 *You can use `/learn` to teach me new information while I'm recovering!*"
    
    return base_response

@bot.tree.command(name="ask", description="Ask a question to the local LLM via LM Studio (Enhanced)")
@app_commands.describe(prompt="Your question or request")
async def enhanced_ask_command_multi_server(interaction: discord.Interaction, prompt: str):
    """Enhanced ask command with robust error handling and server isolation."""
    # CRITICAL: Defer immediately to prevent interaction timeout
    try:
        await interaction.response.defer(ephemeral=False)
    except Exception as defer_error:
        print(f"⚠️ Discord defer failed: {defer_error}")
        return  # Exit if we can't defer - interaction is likely expired
    
    user_name = interaction.user.display_name
    server_name = interaction.guild.name if interaction.guild else "DM"
    
    # Check LM Studio health first - but don't block for too long
    try:
        print(f"🏥 Starting health check for user: {user_name}")
        health_check = await asyncio.wait_for(check_lm_studio_health(), timeout=5.0)
        print(f"🏥 Health check result: {health_check}")
        if not health_check:
            print("❌ Health check failed - sending fallback message")
            fallback_msg = await get_fallback_response(prompt, user_name)
            await interaction.followup.send(fallback_msg)
            return
        else:
            print("✅ Health check passed - proceeding with LM Studio call")
    except asyncio.TimeoutError:
        print("⏰ Health check timed out - proceeding with LM Studio call anyway")
        # Continue anyway - let the actual call handle the failure
    except Exception as e:
        print(f"❌ Health check error: {e} - proceeding anyway")
    
    # Build the payload with progressive fallback strategy
    base_payload = {
        "messages": [
            {
                "role": "system",
                "content": f"""You are DuckBot, a helpful AI assistant for Discord. 
                
The user's name is {user_name} and they're on the "{server_name}" server.
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
    
    # Start with NO plugins since they consistently fail with 500 errors
    full_payload = base_payload.copy()  # No plugins at all
    
    try:
        # Attempt to call LM Studio
        print(f"🤖 Starting LM Studio call for prompt: {prompt[:50]}...")
        data = await call_lm_studio_with_retry(full_payload)
        print(f"🤖 LM Studio call completed. Data received: {bool(data)}")
        
        if data and "choices" in data and len(data["choices"]) > 0:
            ai_response = data["choices"][0]["message"]["content"]
            print(f"✅ LM Studio response received (length: {len(ai_response)}): {ai_response[:100]}...")
            
            # Check if response is empty or whitespace
            if ai_response and ai_response.strip():
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
                print("⚠️ LM Studio returned empty response")
                fallback_msg = await get_fallback_response(prompt, user_name)
                await interaction.followup.send(fallback_msg)
                
        else:
            print(f"❌ LM Studio response invalid or missing. Data: {data}")
            # Fallback response
            fallback_msg = await get_fallback_response(prompt, user_name)
            await interaction.followup.send(fallback_msg)
            
    except Exception as e:
        print(f"Enhanced ask command error: {e}")
        fallback_msg = await get_fallback_response(prompt, user_name)
        await interaction.followup.send(fallback_msg)

@bot.tree.command(name="ask_simple", description="Ask a question without plugins (more reliable)")
@app_commands.describe(prompt="Your question or request")
async def simple_ask_command_multi_server(interaction: discord.Interaction, prompt: str):
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
async def lm_health_command_multi_server(interaction: discord.Interaction):
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
    
    # Test with plugins - multiple approaches
    plugin_status = "Not tested"
    if basic_health:
        # Test plugin support with progressive fallback
        try:
            # Test 1: Try with dice plugin only
            dice_payload = {
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 10,
                **MINIMAL_PLUGIN_CONFIG
            }
            
            response = requests.post(
                LM_STUDIO_URL,
                headers={"Content-Type": "application/json"},
                data=json.dumps(dice_payload),
                timeout=10
            )
            
            if response.status_code == 200:
                plugin_status = "✅ Basic plugins working (dice)"
            else:
                # Test 2: Try without any plugins at all
                basic_payload = {
                    "messages": [{"role": "user", "content": "test"}],
                    "max_tokens": 10
                }
                
                response2 = requests.post(
                    LM_STUDIO_URL,
                    headers={"Content-Type": "application/json"},
                    data=json.dumps(basic_payload),
                    timeout=10
                )
                
                if response2.status_code == 200:
                    plugin_status = "⚠️ No plugins - API calls work without plugins"
                else:
                    plugin_status = f"❌ All plugin tests failed (Status: {response.status_code})"
            
        except Exception as e:
            plugin_status = f"❌ Error: {str(e)[:50]}..."
            
        embed.add_field(
            name="Plugin Status",
            value=plugin_status,
            inline=True
        )
        
    embed.add_field(
        name="LM Studio URL",
        value=LM_STUDIO_URL,
        inline=False
    )
    
    # Add troubleshooting info
    if not basic_health:
        embed.add_field(
            name="🛠️ Basic Connection Issues",
            value="1. Check if LM Studio is running\n2. Verify the URL is correct\n3. Make sure a model is loaded\n4. Check firewall settings",
            inline=False
        )
    elif "❌" in plugin_status or "⚠️" in plugin_status:
        embed.add_field(
            name="🔧 Plugin Issues",
            value="**Recommendation:** Use `/ask_simple` for reliable chat without plugins.\n\n**Plugin fixes:**\n• Disable problematic plugins in LM Studio\n• Only enable basic plugins like 'dice'\n• Check LM Studio plugin manager",
            inline=False
        )
    
    await interaction.followup.send(embed=embed)

print("✅ Enhanced ask system loaded!")
print("📋 LM Studio commands available:")
print("   • /ask - Enhanced ask with plugins and retry logic")
print("   • /ask_simple - Simple ask without plugins")
print("   • /lm_health - Check LM Studio connection")

@bot.tree.command(name="comfyui_health", description="Check ComfyUI connection and model availability")
async def comfyui_health_command(interaction: discord.Interaction):
    """Check the health of ComfyUI connection."""
    await interaction.response.defer(ephemeral=True)
    
    embed = discord.Embed(
        title="🎨 ComfyUI Health Check",
        color=0x9b59b6
    )
    
    # Test basic HTTP connection
    try:
        response = requests.get(f"http://{COMFYUI_SERVER_ADDRESS}/", timeout=5)
        http_status = "✅ Connected" if response.status_code == 200 else f"⚠️ HTTP {response.status_code}"
        http_color = 0x00ff00 if response.status_code == 200 else 0xff9900
    except Exception as e:
        http_status = f"❌ Failed: {str(e)[:50]}..."
        http_color = 0xff0000
    
    embed.color = http_color
    embed.add_field(name="HTTP Connection", value=http_status, inline=True)
    
    # Test WebSocket connection  
    try:
        uri = f"ws://{COMFYUI_SERVER_ADDRESS}/ws"
        
        async def test_websocket():
            try:
                async with websockets.connect(uri) as websocket:
                    return "✅ Connected"
            except Exception as e:
                return f"❌ Failed: {str(e)[:30]}..."
        
        ws_status = await test_websocket()
    except Exception as e:
        ws_status = f"❌ Error: {str(e)[:30]}..."
    
    embed.add_field(name="WebSocket", value=ws_status, inline=True)
    
    # Check available models (if HTTP is working)
    model_status = "Not checked"
    if "✅" in http_status:
        try:
            models_response = requests.get(f"http://{COMFYUI_SERVER_ADDRESS}/object_info", timeout=5)
            if models_response.status_code == 200:
                model_info = models_response.json()
                # Count checkpoints
                checkpoints = model_info.get("CheckpointLoaderSimple", {}).get("input", {}).get("required", {}).get("ckpt_name", [])
                if isinstance(checkpoints, list) and len(checkpoints) > 0:
                    model_status = f"✅ {len(checkpoints)} models available"
                else:
                    model_status = "⚠️ No models found"
            else:
                model_status = f"❌ API Error {models_response.status_code}"
        except Exception as e:
            model_status = f"❌ Failed: {str(e)[:30]}..."
    
    embed.add_field(name="Models", value=model_status, inline=True)
    
    # Add server info
    embed.add_field(name="Server Address", value=f"`{COMFYUI_SERVER_ADDRESS}`", inline=False)
    
    # Add troubleshooting tips if there are issues
    if "❌" in http_status or "❌" in ws_status:
        embed.add_field(
            name="🔧 Troubleshooting",
            value="• Check if ComfyUI is running\n• Verify server address and port\n• Ensure no firewall blocking\n• Try restarting ComfyUI",
            inline=False
        )
    
    await interaction.followup.send(embed=embed)

print("📋 ComfyUI commands available:")
print("   • /comfyui_health - Check ComfyUI connection and models")

# --- MISSING COMMANDS FROM ENHANCED VERSION ---

@bot.tree.command(name="ping", description="A simple command to test if the bot is responsive.")
async def ping_command(interaction: discord.Interaction):
    await interaction.response.send_message("Pong! 🦆 DuckBot v2.1 Multi-Server is running!", ephemeral=False)

@bot.tree.command(name="generate", description="Generate an image using ComfyUI.")
@app_commands.describe(prompt="The prompt for the image.")
async def generate_command(interaction: discord.Interaction, prompt: str):
    await interaction.response.defer(ephemeral=False)
    await add_to_queue(interaction, prompt, "image")

@bot.tree.command(name="animate", description="Generate a video using ComfyUI and W.A.N 2.2.")
@app_commands.describe(prompt="The prompt for the video.")
async def animate_command(interaction: discord.Interaction, prompt: str):
    await interaction.response.defer(ephemeral=False)
    await add_to_queue(interaction, prompt, "video")

@bot.tree.command(name="server_stats", description="Get social analytics about this Discord server")
async def server_stats_command(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=False)
    
    if not NEO4J_ENABLED:
        await interaction.followup.send("❌ Analytics require Neo4j database")
        return
    
    server_id = interaction.guild.id
    server_name = interaction.guild.name
    
    try:
        with neo4j_driver.session() as session:
            # Get server statistics with server isolation
            stats_query = """
            MATCH (s:Server {id: $server_id})
            OPTIONAL MATCH (s)<-[:BELONGS_TO]-(m:Message)
            OPTIONAL MATCH (s)<-[:JOINED]-(u:User)
            OPTIONAL MATCH (s)<-[:BELONGS_TO]-(k:Knowledge)
            RETURN 
                count(DISTINCT m) as total_messages,
                count(DISTINCT u) as total_users,
                count(DISTINCT k) as knowledge_entries
            """
            
            result = session.run(stats_query, server_id=server_id)
            record = result.single()
            
            if not record:
                await interaction.followup.send(f"📊 No data found for **{server_name}** yet!")
                return
            
            # Get most active users
            active_users_query = """
            MATCH (u:User)-[:SENT]->(m:Message)-[:BELONGS_TO]->(s:Server {id: $server_id})
            RETURN u.username as username, count(m) as message_count
            ORDER BY message_count DESC
            LIMIT 5
            """
            
            active_result = session.run(active_users_query, server_id=server_id)
            active_users = [{"username": r["username"] or "Unknown", "count": r["message_count"]} for r in active_result]
            
            # Create embed
            embed = discord.Embed(
                title=f"📊 Server Analytics: {server_name}",
                color=0x3498db
            )
            
            embed.add_field(name="💬 Total Messages", value=f"{record['total_messages']:,}", inline=True)
            embed.add_field(name="👥 Total Users", value=f"{record['total_users']:,}", inline=True)
            embed.add_field(name="🧠 Knowledge Entries", value=f"{record['knowledge_entries']:,}", inline=True)
            
            if active_users:
                top_users = "\n".join([f"{i+1}. {user['username']}: {user['count']} messages" 
                                     for i, user in enumerate(active_users)])
                embed.add_field(name="🏆 Most Active Users", value=top_users, inline=False)
            
            embed.set_footer(text=f"Server-specific analytics for {server_name}")
            await interaction.followup.send(embed=embed)
    
    except Exception as e:
        await interaction.followup.send(f"❌ Error getting server stats: {e}")

@bot.tree.command(name="my_connections", description="Find users with similar interests or activity patterns")
async def my_connections_command(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)
    
    if not NEO4J_ENABLED:
        await interaction.followup.send("❌ Connection analysis requires Neo4j database")
        return
    
    user_id = interaction.user.id
    server_id = interaction.guild.id
    
    try:
        with neo4j_driver.session() as session:
            # Find users with similar interests (server-isolated)
            query = """
            MATCH (u1:User {id: $user_id})-[:INTERESTED_IN]->(concept:Concept)<-[:INTERESTED_IN]-(u2:User)
            MATCH (u2)-[:JOINED]->(s:Server {id: $server_id})
            WHERE u1 <> u2
            RETURN u2.username as username, count(concept) as shared_interests
            ORDER BY shared_interests DESC
            LIMIT 10
            """
            
            result = session.run(query, user_id=user_id, server_id=server_id)
            connections = [{"username": r["username"] or "Unknown", "shared": r["shared_interests"]} 
                         for r in result]
            
            if not connections:
                await interaction.followup.send("🤝 No connection data found yet. Try using `/learn` or `/remember` to build your profile!")
                return
            
            embed = discord.Embed(
                title="🤝 Your Server Connections",
                description="Users with similar interests in this server:",
                color=0xe74c3c
            )
            
            connection_text = "\n".join([f"• **{conn['username']}**: {conn['shared']} shared interests" 
                                       for conn in connections])
            embed.add_field(name="Similar Users", value=connection_text, inline=False)
            
            embed.set_footer(text=f"Based on server-specific activity in {interaction.guild.name}")
            await interaction.followup.send(embed=embed)
    
    except Exception as e:
        await interaction.followup.send(f"❌ Error finding connections: {e}")

@bot.tree.command(name="channel_insights", description="Get insights about channel activity and user patterns")
async def channel_insights_command(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=False)
    
    if not NEO4J_ENABLED:
        await interaction.followup.send("❌ Channel insights require Neo4j database")
        return
    
    channel_id = interaction.channel.id
    server_id = interaction.guild.id
    
    try:
        with neo4j_driver.session() as session:
            # Get channel activity (server-isolated)
            query = """
            MATCH (c:Channel {id: $channel_id})-[:IN_SERVER]->(s:Server {id: $server_id})
            OPTIONAL MATCH (m:Message)-[:IN_CHANNEL]->(c)
            OPTIONAL MATCH (u:User)-[:SENT]->(m)
            RETURN 
                count(DISTINCT m) as total_messages,
                count(DISTINCT u) as unique_users,
                c.name as channel_name
            """
            
            result = session.run(query, channel_id=channel_id, server_id=server_id)
            record = result.single()
            
            if not record or record['total_messages'] == 0:
                await interaction.followup.send(f"📈 No activity data for #{interaction.channel.name} yet!")
                return
            
            embed = discord.Embed(
                title=f"📈 Channel Insights: #{record['channel_name'] or interaction.channel.name}",
                color=0x9b59b6
            )
            
            embed.add_field(name="💬 Total Messages", value=f"{record['total_messages']:,}", inline=True)
            embed.add_field(name="👥 Unique Users", value=f"{record['unique_users']:,}", inline=True)
            
            if record['unique_users'] > 0:
                avg_messages = record['total_messages'] / record['unique_users']
                embed.add_field(name="📊 Avg Messages/User", value=f"{avg_messages:.1f}", inline=True)
            
            embed.set_footer(text=f"Server-specific data for {interaction.guild.name}")
            await interaction.followup.send(embed=embed)
    
    except Exception as e:
        await interaction.followup.send(f"❌ Error getting channel insights: {e}")

# Add storage and memory management commands
@bot.tree.command(name="storage_status", description="Check Neo4j database storage usage and health")
async def storage_status_command(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)
    
    if not NEO4J_ENABLED:
        await interaction.followup.send("❌ Storage status requires Neo4j database")
        return
    
    try:
        with neo4j_driver.session() as session:
            # Get basic node and relationship counts
            stats_query = """
            MATCH (n) 
            OPTIONAL MATCH ()-[r]->()
            RETURN count(DISTINCT n) as total_nodes, count(r) as total_relationships
            """
            
            result = session.run(stats_query)
            record = result.single()
            
            total_nodes = record["total_nodes"] if record else 0
            total_relationships = record["total_relationships"] if record else 0
            
            # Estimate storage usage (rough calculation)
            estimated_mb = (total_nodes * 0.1 + total_relationships * 0.05)  # Very rough estimate
            estimated_gb = estimated_mb / 1024
            max_gb = 10  # Your configured limit
            usage_percentage = (estimated_gb / max_gb) * 100
            
            embed = discord.Embed(
                title="💾 Database Storage Status",
                color=0x00ff00 if usage_percentage < 80 else 0xff9900 if usage_percentage < 95 else 0xff0000
            )
            
            embed.add_field(name="📊 Nodes", value=f"{total_nodes:,}", inline=True)
            embed.add_field(name="🔗 Relationships", value=f"{total_relationships:,}", inline=True)
            embed.add_field(name="💽 Est. Usage", value=f"{estimated_gb:.2f} GB / {max_gb} GB", inline=True)
            embed.add_field(name="📈 Usage %", value=f"{usage_percentage:.1f}%", inline=True)
            
            status_icon = "🟢" if usage_percentage < 80 else "🟡" if usage_percentage < 95 else "🔴"
            embed.add_field(name="Status", value=f"{status_icon} {'Healthy' if usage_percentage < 80 else 'High Usage' if usage_percentage < 95 else 'Critical'}", inline=True)
            
            if usage_percentage > 90:
                embed.add_field(name="⚠️ Warning", value="Storage usage is high. Consider using `/force_cleanup` if you're an admin.", inline=False)
            
            await interaction.followup.send(embed=embed)
    
    except Exception as e:
        await interaction.followup.send(f"❌ Error checking storage status: {e}")

@bot.tree.command(name="force_cleanup", description="Manually trigger Neo4j database cleanup (Admin only)")
async def force_cleanup_command(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)
    
    # Check if user is admin or has manage server permissions
    if not interaction.user.guild_permissions.manage_guild:
        await interaction.followup.send("❌ You need 'Manage Server' permissions to use this command")
        return
    
    if not NEO4J_ENABLED:
        await interaction.followup.send("❌ Database cleanup requires Neo4j database")
        return
    
    try:
        with neo4j_driver.session() as session:
            # Clean up old messages (older than 90 days)
            cleanup_query = """
            MATCH (m:Message)
            WHERE m.timestamp < datetime() - duration({days: 90})
            DETACH DELETE m
            RETURN count(m) as deleted_messages
            """
            
            result = session.run(cleanup_query)
            record = result.single()
            deleted_count = record["deleted_messages"] if record else 0
            
            embed = discord.Embed(
                title="🧹 Database Cleanup Complete",
                color=0x00ff00
            )
            
            embed.add_field(name="🗑️ Deleted Messages", value=f"{deleted_count:,}", inline=True)
            embed.add_field(name="📅 Cleanup Criteria", value="Messages older than 90 days", inline=True)
            
            if deleted_count > 0:
                embed.add_field(name="✅ Result", value="Database cleaned successfully!", inline=False)
            else:
                embed.add_field(name="ℹ️ Result", value="No old data found to clean", inline=False)
            
            await interaction.followup.send(embed=embed)
    
    except Exception as e:
        await interaction.followup.send(f"❌ Error during cleanup: {e}")

print("✅ Basic commands and analytics loaded!")

# --- PERSONAL MEMORY AND CREATIVE COMMANDS ---

@bot.tree.command(name="remember", description="Store something for DuckBot to remember about you")
@app_commands.describe(memory="What should I remember about you?")
async def remember_command(interaction: discord.Interaction, memory: str):
    await interaction.response.defer(ephemeral=True)
    
    if not NEO4J_ENABLED:
        await interaction.followup.send("❌ Memory storage requires Neo4j database")
        return
    
    user_id = interaction.user.id
    server_id = interaction.guild.id
    
    try:
        with neo4j_driver.session() as session:
            query = """
            MERGE (u:User {id: $user_id})
            MERGE (s:Server {id: $server_id})
            CREATE (m:Memory {
                id: randomUUID(),
                content: $memory,
                user_id: $user_id,
                server_id: $server_id,
                created_date: datetime()
            })
            MERGE (u)-[:HAS_MEMORY]->(m)
            MERGE (m)-[:BELONGS_TO]->(s)
            RETURN m.id as memory_id
            """
            
            result = session.run(query, user_id=user_id, server_id=server_id, memory=memory)
            record = result.single()
            
            if record:
                embed = discord.Embed(
                    title="🧠 Memory Stored!",
                    description=f"I'll remember: *{memory}*",
                    color=0x3498db
                )
                embed.set_footer(text=f"Memory ID: {record['memory_id'][:8]}... | Server-specific memory")
                await interaction.followup.send(embed=embed)
            else:
                await interaction.followup.send("❌ Failed to store memory")
    
    except Exception as e:
        await interaction.followup.send(f"❌ Error storing memory: {e}")

@bot.tree.command(name="my_context", description="See what DuckBot remembers about you")
async def my_context_command(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)
    
    if not NEO4J_ENABLED:
        await interaction.followup.send("❌ Memory access requires Neo4j database")
        return
    
    user_id = interaction.user.id
    server_id = interaction.guild.id
    
    try:
        with neo4j_driver.session() as session:
            query = """
            MATCH (u:User {id: $user_id})-[:HAS_MEMORY]->(m:Memory)-[:BELONGS_TO]->(s:Server {id: $server_id})
            RETURN m.content as memory, m.created_date as date
            ORDER BY m.created_date DESC
            LIMIT 10
            """
            
            result = session.run(query, user_id=user_id, server_id=server_id)
            memories = [{"content": r["memory"], "date": r["date"]} for r in result]
            
            if not memories:
                await interaction.followup.send("🤔 I don't have any memories about you in this server yet. Use `/remember` to tell me something!")
                return
            
            embed = discord.Embed(
                title="🧠 What I Remember About You",
                color=0x9b59b6
            )
            
            for i, mem in enumerate(memories):
                date_str = mem["date"].strftime("%Y-%m-%d") if mem["date"] else "Unknown"
                embed.add_field(
                    name=f"Memory {i+1} ({date_str})",
                    value=mem["content"][:200] + "..." if len(mem["content"]) > 200 else mem["content"],
                    inline=False
                )
            
            embed.set_footer(text=f"Showing server-specific memories for {interaction.guild.name}")
            await interaction.followup.send(embed=embed)
    
    except Exception as e:
        await interaction.followup.send(f"❌ Error retrieving memories: {e}")

@bot.tree.command(name="start_adventure", description="Begin an interactive text adventure")
@app_commands.describe(
    theme="Adventure theme (fantasy, sci-fi, mystery, cyberpunk, horror, steampunk, dnd)",
    party_mode="Allow other players to join this adventure (default: false)"
)
async def start_adventure_command(interaction: discord.Interaction, theme: str = "fantasy", party_mode: bool = False):
    await interaction.response.defer(ephemeral=False)
    
    if not NEO4J_ENABLED:
        await interaction.followup.send("❌ Adventures require Neo4j database for state management")
        return
    
    user_id = interaction.user.id
    server_id = interaction.guild.id
    user_name = interaction.user.display_name
    
    try:
        # Generate dynamic adventure opening using LM Studio
        opening_prompt = f"""You are a master storyteller creating an interactive {theme} adventure game. 

Create an engaging opening scene for a player named {user_name}. Include:
- Rich, immersive description of the setting
- An intriguing situation or conflict
- The character's background/role
- 3-4 specific, meaningful choices for what to do next
- Each choice should lead to different story paths

Theme: {theme}
Format: Write in second person ("You are..."). End with "What do you choose?" followed by numbered options.
Length: 150-200 words maximum.
Tone: Engaging, immersive, and action-oriented."""

        # Call LM Studio for dynamic story generation
        try:
            response = await call_lm_studio_async(opening_prompt, user_name, "Adventure Master")
            if response and len(response.strip()) > 50:
                initial_story = response.strip()
            else:
                # Fallback to rich themed openings
                fallback_openings = {
                    "fantasy": f"🏰 You are {user_name}, a skilled adventurer who has just arrived at the mysterious village of Ravenshollow. Dark clouds gather overhead as villagers whisper of an ancient curse that has awakened something terrible in the nearby Whispering Woods. The village elder approaches you desperately.\n\n**Choices:**\n1. 🗡️ Immediately head to the Whispering Woods to investigate\n2. 🍺 Visit the tavern to gather information from locals first\n3. 🔮 Seek out the village's hedge witch for magical insight\n4. 💰 Negotiate your fee with the village elder before helping",
                    
                    "sci-fi": f"🚀 You are {user_name}, a deep space salvage operator whose ship has intercepted a distress beacon from the research vessel 'Prometheus.' As you dock with the eerily silent ship, your sensors detect multiple life signs... but something's wrong. The ship's AI is speaking in an unknown language, and there are scorch marks on the walls that don't match any known weapons.\n\n**Choices:**\n1. 🔫 Enter armed and proceed directly to the bridge\n2. 🔬 Head to the science labs to investigate the research data\n3. 🤖 Try to interface with the ship's AI system first\n4. 📡 Scan for survivors using your ship's sensors before boarding",
                    
                    "mystery": f"🕵️ You are {user_name}, a private investigator called to the grand Ashford Estate after the mysterious disappearance of the wealthy patriarch, Edmund Ashford. The butler who called you seems nervous, and you notice fresh tire tracks leading away from the manor. The family members each have alibis, but their stories don't quite add up.\n\n**Choices:**\n1. 🔍 Search Edmund's private study for clues\n2. 🗣️ Interview the nervous butler privately\n3. 🚗 Follow the tire tracks while they're still fresh\n4. 👥 Gather all family members for group questioning",
                    
                    "dnd": f"🎲 **Welcome to the Realm, {user_name}!**\n\nYou are a Level 1 adventurer who has just arrived at the bustling frontier town of Goldbrook. Your coin purse jingles with your last 15 gold pieces, and your starting equipment feels reassuring at your side. The town crier announces that the mayor is offering a 100 gold reward for brave souls willing to investigate strange disappearances near the Whispering Caverns.\n\n**Your Stats:** STR: 12, DEX: 14, CON: 13, INT: 15, WIS: 11, CHA: 10\n**Starting Equipment:** Leather armor, shortsword, shortbow, adventurer's pack\n\n**What's your first move?**\n1. ⚔️ Visit the mayor immediately to accept the quest\n2. 🍺 Gather information at the Prancing Pony tavern first\n3. 🛡️ Browse the general store to upgrade your equipment\n4. 🧙‍♂️ Seek out the local wizard for magical advice"
                }
                initial_story = fallback_openings.get(theme, fallback_openings["fantasy"])
        except Exception as e:
            print(f"LM Studio failed for adventure opening: {e}")
            # Use fallback opening
            initial_story = f"🌟 You are {user_name}, beginning a {theme} adventure filled with mystery and excitement. Your journey starts now..."

        with neo4j_driver.session() as session:
            # Create new adventure with rich state tracking
            story_id = str(uuid.uuid4())
            
            # Character stats initialization (for all themes now)
            if theme.lower() == "dnd":
                # Full D&D stats
                character_stats = {
                    "strength": random.randint(8, 15),
                    "dexterity": random.randint(8, 15), 
                    "constitution": random.randint(8, 15),
                    "intelligence": random.randint(8, 15),
                    "wisdom": random.randint(8, 15),
                    "charisma": random.randint(8, 15)
                }
                character_data = {
                    "level": 1,
                    "hp": 10 + ((character_stats["constitution"] - 10) // 2),
                    "gold": random.randint(10, 25),
                    "stats": character_stats,
                    "class": "Adventurer",
                    "theme_type": "dnd"
                }
            else:
                # Basic stats for other themes (simpler system)
                character_data = {
                    "luck": random.randint(1, 6),  # 1d6 luck stat
                    "skill": random.randint(1, 6),  # 1d6 skill stat  
                    "health": 3,  # Simple health system
                    "theme_type": "basic",
                    "special_abilities": []
                }
            
            query = """
            MERGE (u:User {id: $user_id})
            MERGE (s:Server {id: $server_id})
            CREATE (story:Story {
                id: $story_id,
                creator_id: $user_id,
                server_id: $server_id, 
                theme: $theme,
                current_scene: $initial_story,
                choices_made: 0,
                story_state: 'opening',
                party_mode: $party_mode,
                world_knowledge: [],
                created_date: datetime(),
                last_updated: datetime()
            })
            CREATE (player:Player {
                id: $player_id,
                user_id: $user_id,
                player_name: $player_name,
                character_data: $character_data,
                character_traits: [],
                inventory: [],
                joined_date: datetime()
            })
            MERGE (u)-[:PLAYING]->(story)
            MERGE (player)-[:IN_STORY]->(story)
            MERGE (story)-[:BELONGS_TO]->(s)
            RETURN story.id as story_id
            """
            
            player_id = str(uuid.uuid4())
            
            session.run(query, 
                       user_id=user_id, 
                       server_id=server_id,
                       story_id=story_id,
                       player_id=player_id,
                       player_name=user_name,
                       theme=theme,
                       initial_story=initial_story,
                       party_mode=party_mode,
                       character_data=character_data)
            
            embed = discord.Embed(
                title=f"🎮 {theme.title()} Adventure: {user_name}'s Journey",
                description=initial_story,
                color=0x8b4513 if theme == "fantasy" else 0x00ced1 if theme == "sci-fi" else 0x8b0000 if theme == "mystery" else 0xff6b35 if theme == "dnd" else 0x9b59b6
            )
            embed.add_field(name="🎯 Your Move", value="Use `/continue_adventure <your choice>` to proceed!", inline=False)
            if party_mode:
                embed.add_field(name="👥 Party Mode", value="Other players can join with `/join_adventure`!", inline=False)
            embed.add_field(name="💡 Tip", value="Be specific! Instead of 'look around', try 'examine the mysterious book on the table'", inline=False)
            embed.set_footer(text=f"Adventure ID: {story_id[:8]}... • {'Party Mode' if party_mode else 'Solo'} • Created by {user_name}")
            
            await interaction.followup.send(embed=embed)
    
    except Exception as e:
        await interaction.followup.send(f"❌ Error starting adventure: {e}")

@bot.tree.command(name="join_adventure", description="Join an existing party adventure")
async def join_adventure_command(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=False)
    
    if not NEO4J_ENABLED:
        await interaction.followup.send("❌ Adventures require Neo4j database")
        return
    
    user_id = interaction.user.id
    server_id = interaction.guild.id
    user_name = interaction.user.display_name
    
    try:
        with neo4j_driver.session() as session:
            # Check if user is already in an adventure
            existing_query = """
            MATCH (u:User {id: $user_id})-[:PLAYING]->(story:Story)-[:BELONGS_TO]->(s:Server {id: $server_id})
            RETURN story.id as story_id
            """
            existing_result = session.run(existing_query, user_id=user_id, server_id=server_id)
            if existing_result.single():
                await interaction.followup.send("❌ You're already in an adventure! Use `/end_adventure` first to join a new one.")
                return
            
            # Find available party adventures
            party_query = """
            MATCH (story:Story {party_mode: true})-[:BELONGS_TO]->(s:Server {id: $server_id})
            WHERE story.story_state <> 'ended'
            OPTIONAL MATCH (p:Player)-[:IN_STORY]->(story)
            WITH story, count(p) as player_count
            WHERE player_count < 6  // Max 6 players per party
            RETURN story.id as story_id, story.theme as theme, story.creator_id as creator_id, 
                   story.current_scene as current_scene, player_count
            ORDER BY story.created_date DESC
            LIMIT 1
            """
            
            party_result = session.run(party_query, server_id=server_id)
            party_record = party_result.single()
            
            if not party_record:
                await interaction.followup.send("🤔 No available party adventures found. Ask someone to start one with party mode enabled!")
                return
            
            # Join the adventure
            story_id = party_record["story_id"]
            theme = party_record["theme"]
            player_count = party_record["player_count"]
            
            # Generate character stats based on theme
            if theme.lower() == "dnd":
                character_stats = {
                    "strength": random.randint(8, 15),
                    "dexterity": random.randint(8, 15), 
                    "constitution": random.randint(8, 15),
                    "intelligence": random.randint(8, 15),
                    "wisdom": random.randint(8, 15),
                    "charisma": random.randint(8, 15)
                }
                character_data = {
                    "level": 1,
                    "hp": 10 + ((character_stats["constitution"] - 10) // 2),
                    "gold": random.randint(10, 25),
                    "stats": character_stats,
                    "class": "Adventurer",
                    "theme_type": "dnd"
                }
            else:
                character_data = {
                    "luck": random.randint(1, 6),
                    "skill": random.randint(1, 6),  
                    "health": 3,
                    "theme_type": "basic",
                    "special_abilities": []
                }
            
            # Add player to story
            player_id = str(uuid.uuid4())
            join_query = """
            MATCH (u:User {id: $user_id})
            MATCH (story:Story {id: $story_id})
            CREATE (player:Player {
                id: $player_id,
                user_id: $user_id,
                player_name: $player_name,
                character_data: $character_data,
                character_traits: [],
                inventory: [],
                joined_date: datetime()
            })
            MERGE (u)-[:PLAYING]->(story)
            MERGE (player)-[:IN_STORY]->(story)
            RETURN story.theme as theme
            """
            
            session.run(join_query, 
                       user_id=user_id,
                       story_id=story_id,
                       player_id=player_id,
                       player_name=user_name,
                       character_data=character_data)
            
            embed = discord.Embed(
                title=f"🎉 Joined {theme.title()} Adventure!",
                description=f"**{user_name}** has joined the party adventure!\n\n**Current Scene:**\n{party_record['current_scene'][:300]}{'...' if len(party_record['current_scene']) > 300 else ''}",
                color=0x00ff88
            )
            
            embed.add_field(name="👥 Party Size", value=f"{player_count + 1} players", inline=True)
            embed.add_field(name="🎭 Your Character", value=f"**{user_name}** the Adventurer", inline=True)
            embed.add_field(name="🎯 Next Action", value="Use `/continue_adventure <action>` to participate!", inline=False)
            embed.set_footer(text=f"Adventure ID: {story_id[:8]}... • Party Adventure")
            
            await interaction.followup.send(embed=embed)
            
    except Exception as e:
        await interaction.followup.send(f"❌ Error joining adventure: {e}")

@bot.tree.command(name="continue_adventure", description="Continue your adventure")
@app_commands.describe(action="What do you want to do? Be specific for better results!")
async def continue_adventure_command(interaction: discord.Interaction, action: str):
    await interaction.response.defer(ephemeral=False)
    
    if not NEO4J_ENABLED:
        await interaction.followup.send("❌ Adventures require Neo4j database")
        return
    
    user_id = interaction.user.id
    server_id = interaction.guild.id
    
    try:
        with neo4j_driver.session() as session:
            # Get current adventure with full context
            query = """
            MATCH (u:User {id: $user_id})-[:PLAYING]->(story:Story)-[:BELONGS_TO]->(s:Server {id: $server_id})
            RETURN story.id as story_id, story.theme as theme, story.choices_made as choices,
                   story.current_scene as current_scene, story.player_name as player_name,
                   story.story_state as story_state, story.character_traits as character_traits,
                   story.inventory as inventory, story.world_knowledge as world_knowledge,
                   story.dnd_data as dnd_data
            ORDER BY story.created_date DESC
            LIMIT 1
            """
            
            result = session.run(query, user_id=user_id, server_id=server_id)
            record = result.single()
            
            if not record:
                await interaction.followup.send("🤔 You don't have an active adventure. Use `/start_adventure` to begin!")
                return
            
            # Extract adventure context
            choices = record["choices"] + 1
            theme = record["theme"]
            player_name = record["player_name"]
            current_scene = record["current_scene"]
            story_state = record["story_state"] or "beginning"
            character_traits = record.get("character_traits", [])
            inventory = record.get("inventory", [])
            world_knowledge = record.get("world_knowledge", [])
            dnd_data = record.get("dnd_data", {})
            
            # Handle D&D mechanics if it's a D&D adventure
            dnd_outcome = None
            if theme.lower() == "dnd" and dnd_data:
                dnd_outcome = determine_dnd_outcome(action, dnd_data)
            
            # Generate dynamic story continuation using LM Studio
            if theme.lower() == "dnd" and dnd_data:
                # D&D-specific prompt with dice mechanics
                story_prompt = f"""You are an expert D&D Dungeon Master. Continue this D&D adventure based on the player's action and dice roll result.

CURRENT STORY CONTEXT:
{current_scene}

PLAYER ACTION: {action}

DICE ROLL RESULT:
- Action Type: {dnd_outcome['action_type']}
- Stat Used: {dnd_outcome['stat_used'].title()} 
- Roll: {dnd_outcome['roll_result']['rolls'][0]} + {dnd_outcome['roll_result']['modifier']} = {dnd_outcome['roll_result']['total']}
- Difficulty: {dnd_outcome['difficulty']}
- Result: {dnd_outcome['success_level'].replace('_', ' ').title()}

CHARACTER STATS:
- Level: {dnd_data.get('level', 1)} {dnd_data.get('class', 'Adventurer')}
- HP: {dnd_data.get('hp', 10)}, Gold: {dnd_data.get('gold', 15)}
- STR: {dnd_data.get('stats', {}).get('strength', 10)}, DEX: {dnd_data.get('stats', {}).get('dexterity', 10)}, CON: {dnd_data.get('stats', {}).get('constitution', 10)}
- INT: {dnd_data.get('stats', {}).get('intelligence', 10)}, WIS: {dnd_data.get('stats', {}).get('wisdom', 10)}, CHA: {dnd_data.get('stats', {}).get('charisma', 10)}
- Inventory: {', '.join(inventory) if inventory else "Leather armor, shortsword, shortbow, adventurer's pack"}

INSTRUCTIONS:
1. Narrate the outcome based on the dice roll result
2. Critical successes should be amazing, critical failures should be dramatic but not game-ending  
3. Include vivid D&D-style descriptions and combat if applicable
4. Show consequences appropriate to the success level
5. End with a new challenge or choice
6. Format: Write in second person ("You..." or "As you...")
7. Length: 120-180 words
8. Include D&D emojis (⚔️🛡️🎲🏰🐉)

Make this feel like a real D&D session with meaningful consequences."""
            else:
                # Regular adventure prompt
                story_prompt = f"""You are an expert {theme} adventure game master. Continue this interactive story based on the player's action.

CURRENT STORY CONTEXT:
{current_scene}

PLAYER ACTION: {action}

ADVENTURE STATE:
- Player Name: {player_name}
- Theme: {theme}
- Choices Made: {choices}
- Story Phase: {story_state}
- Character Traits: {', '.join(character_traits) if character_traits else 'None yet'}
- Inventory: {', '.join(inventory) if inventory else 'Empty'}
- World Knowledge: {', '.join(world_knowledge) if world_knowledge else 'Just started'}

INSTRUCTIONS:
1. Write what happens as a result of the player's action
2. Create consequences that feel meaningful and connected to previous choices
3. Include vivid descriptions and immersive details
4. Introduce new elements that advance the plot
5. End with a compelling situation requiring a new choice
6. Format: Write in second person ("You..." or "As you...")
7. Length: 100-150 words
8. Include emojis that match the theme

Make this feel like a real story progression, not a random event. Build on what came before."""

            # Try to get AI-generated response
            try:
                ai_response = await call_lm_studio_async(story_prompt, player_name, f"{theme} Adventure Master")
                if ai_response and len(ai_response.strip()) > 50:
                    next_scene = ai_response.strip()
                    
                    # Extract any new character developments, items, or knowledge
                    new_traits = []
                    new_items = []
                    new_knowledge = []
                    
                    # Simple keyword detection for character development
                    if any(word in action.lower() for word in ["brave", "bold", "heroic", "courage"]):
                        new_traits.append("brave")
                    if any(word in action.lower() for word in ["sneak", "hide", "stealth", "quiet"]):
                        new_traits.append("stealthy")
                    if any(word in action.lower() for word in ["study", "read", "learn", "investigate"]):
                        new_traits.append("scholarly")
                    
                    # Update story state based on choices
                    if choices >= 3 and story_state == "opening":
                        new_story_state = "developing"
                    elif choices >= 7 and story_state == "developing":
                        new_story_state = "climax"
                    elif choices >= 10:
                        new_story_state = "resolution"
                    else:
                        new_story_state = story_state
                        
                else:
                    # Fallback to improved themed responses
                    next_scene = generate_fallback_adventure_scene(action, theme, choices, story_state, dnd_outcome)
                    new_story_state = story_state
                    new_traits = []
                    new_items = []
                    new_knowledge = []
                    
            except Exception as e:
                print(f"Adventure AI generation failed: {e}")
                next_scene = generate_fallback_adventure_scene(action, theme, choices, story_state, dnd_outcome)
                new_story_state = story_state
                new_traits = []
                new_items = []
                new_knowledge = []
            
            # Update adventure with rich state tracking
            update_query = """
            MATCH (story:Story {id: $story_id})
            SET story.current_scene = $next_scene,
                story.choices_made = $choices,
                story.story_state = $story_state,
                story.character_traits = story.character_traits + $new_traits,
                story.inventory = story.inventory + $new_items,
                story.world_knowledge = story.world_knowledge + $new_knowledge,
                story.last_updated = datetime()
            """
            
            session.run(update_query, 
                       story_id=record["story_id"], 
                       next_scene=next_scene, 
                       choices=choices,
                       story_state=new_story_state,
                       new_traits=new_traits,
                       new_items=new_items,
                       new_knowledge=new_knowledge)
            
            # Create rich embed with adventure state
            embed = discord.Embed(
                title=f"🎮 {theme.title()} Adventure: {player_name}",
                description=next_scene,
                color=0x8b4513 if theme == "fantasy" else 0x00ced1 if theme == "sci-fi" else 0x8b0000 if theme == "mystery" else 0xff6b35 if theme == "dnd" else 0x9b59b6
            )
            
            # Add character progression info
            total_traits = len(set(character_traits + new_traits))
            total_items = len(set(inventory + new_items))
            
            if theme.lower() == "dnd" and dnd_data:
                # D&D-specific display with stats and dice roll result
                stats = dnd_data.get('stats', {})
                embed.add_field(
                    name="📊 Character Stats", 
                    value=f"**Level {dnd_data.get('level', 1)} {dnd_data.get('class', 'Adventurer')}** • **HP:** {dnd_data.get('hp', 10)} • **Gold:** {dnd_data.get('gold', 15)}\n**STR:** {stats.get('strength', 10)} **DEX:** {stats.get('dexterity', 10)} **CON:** {stats.get('constitution', 10)} **INT:** {stats.get('intelligence', 10)} **WIS:** {stats.get('wisdom', 10)} **CHA:** {stats.get('charisma', 10)}", 
                    inline=False
                )
                if dnd_outcome:
                    embed.add_field(
                        name="🎲 Last Roll", 
                        value=f"**{dnd_outcome['action_type']}** ({dnd_outcome['stat_used'].upper()}): {dnd_outcome['roll_result']['rolls'][0]} + {dnd_outcome['roll_result']['modifier']} = **{dnd_outcome['roll_result']['total']}**", 
                        inline=True
                    )
            else:
                embed.add_field(
                    name="📊 Adventure Progress", 
                    value=f"**Choices:** {choices} • **Phase:** {new_story_state.title()}\n**Character Traits:** {total_traits} • **Items:** {total_items}", 
                    inline=False
                )
            
            if new_traits:
                embed.add_field(name="✨ New Trait", value=f"You've developed: **{', '.join(new_traits)}**", inline=True)
            if new_items:
                embed.add_field(name="🎒 New Item", value=f"You obtained: **{', '.join(new_items)}**", inline=True)
            
            embed.add_field(name="🎯 Your Move", value="Use `/continue_adventure <action>` to proceed!", inline=False)
            embed.set_footer(text=f"Adventure ID: {record['story_id'][:8]}... • Be creative with your actions!")
            
            await interaction.followup.send(embed=embed)
    
    except Exception as e:
        await interaction.followup.send(f"❌ Error continuing adventure: {e}")

def generate_fallback_adventure_scene(action: str, theme: str, choices: int, story_state: str, dnd_outcome: dict = None) -> str:
    """Generate themed adventure scenes when AI is unavailable."""
    
    # Progressive story elements based on choices made
    if choices <= 3:
        phase = "opening"
    elif choices <= 7:
        phase = "development" 
    elif choices <= 10:
        phase = "climax"
    else:
        phase = "resolution"
    
    # Handle D&D-specific responses with dice mechanics
    if theme.lower() == "dnd" and dnd_outcome:
        success_level = dnd_outcome['success_level']
        roll_info = dnd_outcome['roll_result']
        stat_used = dnd_outcome['stat_used']
        action_type = dnd_outcome['action_type']
        
        roll_text = f"🎲 **{action_type} Check:** {roll_info['rolls'][0]} + {roll_info['modifier']} = **{roll_info['total']}**"
        
        if success_level == "critical_success":
            return f"{roll_text}\n\n🌟 **CRITICAL SUCCESS!** Your attempt to {action} exceeds all expectations! You achieve remarkable results that will be remembered in tavern tales. The path ahead opens with new opportunities and you feel your confidence soar."
        elif success_level == "great_success":
            return f"{roll_text}\n\n✨ **Great Success!** Your {stat_used}-based action to {action} works excellently. You accomplish exactly what you intended and perhaps a bit more. The situation improves significantly in your favor."
        elif success_level == "success":
            return f"{roll_text}\n\n✅ **Success!** Your attempt to {action} works as planned. You make steady progress and feel satisfied with the outcome. The adventure continues with you in a good position."
        elif success_level == "failure":
            return f"{roll_text}\n\n⚠️ **Failure.** Your attempt to {action} doesn't go as planned. While not disastrous, you face a setback that complicates your situation. Time to try a different approach or prepare for consequences."
        else:  # critical_failure
            return f"{roll_text}\n\n💥 **CRITICAL FAILURE!** Your attempt to {action} goes spectacularly wrong! Though not fatal, this dramatic mishap creates unexpected complications. The dice gods demand a different path forward."
    
    themed_elements = {
        "fantasy": {
            "opening": [
                f"⚔️ As you {action}, ancient runes begin to glow on nearby stones, revealing a hidden path deeper into the mystical realm.",
                f"🧙‍♂️ Your action draws the attention of a mysterious figure in robes who steps from the shadows, eyes gleaming with ancient knowledge.",
                f"🏰 The action causes the ground to rumble, and a long-buried tower begins to rise from the earth before you."
            ],
            "development": [
                f"🐉 Your decision to {action} awakens something powerful in the distance - you hear mighty wings beating against the storm clouds.",
                f"⚡ Dark magic responds to your actions, and reality itself seems to shift around you as new possibilities unfold.",
                f"🗡️ Your brave choice to {action} proves your worth, and a legendary weapon materializes in your hands, humming with power."
            ],
            "climax": [
                f"💥 As you {action}, the final confrontation begins! Ancient powers clash around you as fate hangs in the balance.",
                f"🌟 Your heroic decision to {action} triggers a cascade of magical energy that will determine the realm's destiny.",
                f"⚔️ The moment you {action}, time seems to slow as you face the ultimate test of your journey."
            ]
        },
        "sci-fi": {
            "opening": [
                f"🛸 Your action triggers advanced sensors, and alien technology responds by projecting a holographic star map with coordinates to unknown worlds.",
                f"🤖 As you {action}, dormant AI systems come online, and synthetic voices begin discussing your arrival in languages you've never heard.",
                f"⚛️ The quantum field fluctuates wildly in response to your choice, opening rifts in spacetime that reveal glimpses of parallel realities."
            ],
            "development": [
                f"🚀 Your decision to {action} activates an ancient alien defense system, and massive structures begin emerging from the planet's surface.",
                f"👽 Strange life forms react to your actions, and you realize you've stumbled into a complex galactic political situation.",
                f"🔬 Your scientific approach to {action} unlocks advanced technology that could change the course of human civilization."
            ],
            "climax": [
                f"💫 As you {action}, the fate of entire star systems hangs in the balance, and cosmic forces align for the final confrontation.",
                f"🌌 Your crucial decision to {action} triggers a chain reaction that will either save or doom countless worlds.",
                f"⚡ The moment you {action}, reality fractures as you face the ultimate test of your resolve across multiple dimensions."
            ]
        },
        "mystery": {
            "opening": [
                f"🔍 Your careful action to {action} reveals a crucial clue that connects to a much larger conspiracy than you initially suspected.",
                f"📜 As you {action}, you discover a hidden message that suggests the victim knew their fate and left behind vital information.",
                f"🕵️ Your investigative approach to {action} uncovers evidence that someone has been watching your every move."
            ],
            "development": [
                f"💰 Your decision to {action} exposes corruption at the highest levels, and you realize you can trust no one.",
                f"🎭 The truth behind your choice to {action} reveals that nothing is as it seemed, and the real villain has been hiding in plain sight.",
                f"🔐 Your persistence in {action} unlocks a secret that powerful people would kill to keep buried."
            ],
            "climax": [
                f"⚖️ As you {action}, the final pieces of the puzzle fall into place, but the truth is more dangerous than you ever imagined.",
                f"🚨 Your bold decision to {action} triggers the endgame, and now you must survive long enough to expose the conspiracy.",
                f"💀 The moment you {action}, you realize you're not just solving a mystery - you're fighting for your life."
            ]
        }
    }
    
    # Get appropriate responses for theme and phase
    if theme in themed_elements and phase in themed_elements[theme]:
        responses = themed_elements[theme][phase]
        return random.choice(responses)
    else:
        # Generic fallback
        return f"🎯 Your decision to {action} sets new events in motion. The story continues to unfold before you, filled with unexpected possibilities."

@bot.tree.command(name="adventure_status", description="View your current adventure progress and character")
async def adventure_status_command(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)
    
    if not NEO4J_ENABLED:
        await interaction.followup.send("❌ Adventures require Neo4j database")
        return
    
    user_id = interaction.user.id
    server_id = interaction.guild.id
    
    try:
        with neo4j_driver.session() as session:
            query = """
            MATCH (u:User {id: $user_id})-[:PLAYING]->(story:Story)-[:BELONGS_TO]->(s:Server {id: $server_id})
            RETURN story.id as story_id, story.theme as theme, story.choices_made as choices,
                   story.player_name as player_name, story.story_state as story_state,
                   story.character_traits as character_traits, story.inventory as inventory,
                   story.world_knowledge as world_knowledge, story.created_date as created,
                   story.current_scene as current_scene, story.dnd_data as dnd_data
            ORDER BY story.created_date DESC
            LIMIT 1
            """
            
            result = session.run(query, user_id=user_id, server_id=server_id)
            record = result.single()
            
            if not record:
                embed = discord.Embed(
                    title="🎮 No Active Adventure",
                    description="You don't have an active adventure. Use `/start_adventure` to begin one!",
                    color=0x95a5a6
                )
                embed.add_field(name="Available Themes", value="fantasy, sci-fi, mystery, cyberpunk, horror, steampunk, **dnd**", inline=False)
                await interaction.followup.send(embed=embed)
                return
            
            # Extract adventure info
            theme = record["theme"]
            choices = record["choices"]
            player_name = record["player_name"]
            story_state = record["story_state"] or "beginning"
            character_traits = record.get("character_traits", [])
            inventory = record.get("inventory", [])
            world_knowledge = record.get("world_knowledge", [])
            current_scene = record["current_scene"]
            dnd_data = record.get("dnd_data", {})
            
            embed = discord.Embed(
                title=f"🎮 {theme.title()} Adventure: {player_name}",
                description=f"**Current Scene Preview:**\n{current_scene[:200]}{'...' if len(current_scene) > 200 else ''}",
                color=0x8b4513 if theme == "fantasy" else 0x00ced1 if theme == "sci-fi" else 0x8b0000 if theme == "mystery" else 0xff6b35 if theme == "dnd" else 0x9b59b6
            )
            
            if theme.lower() == "dnd" and dnd_data:
                # D&D-specific status display
                stats = dnd_data.get('stats', {})
                embed.add_field(
                    name="🎲 D&D Character",
                    value=f"**Level {dnd_data.get('level', 1)} {dnd_data.get('class', 'Adventurer')}**\n**HP:** {dnd_data.get('hp', 10)} • **Gold:** {dnd_data.get('gold', 15)} pieces",
                    inline=True
                )
                embed.add_field(
                    name="📊 Ability Scores",
                    value=f"**STR:** {stats.get('strength', 10)} **DEX:** {stats.get('dexterity', 10)} **CON:** {stats.get('constitution', 10)}\n**INT:** {stats.get('intelligence', 10)} **WIS:** {stats.get('wisdom', 10)} **CHA:** {stats.get('charisma', 10)}",
                    inline=True
                )
                embed.add_field(
                    name="🗡️ Progress",
                    value=f"**Choices Made:** {choices}\n**Adventure Phase:** {story_state.title()}",
                    inline=True
                )
            else:
                # Regular adventure status display
                embed.add_field(
                    name="📊 Progress",
                    value=f"**Choices Made:** {choices}\n**Story Phase:** {story_state.title()}",
                    inline=True
                )
                
                embed.add_field(
                    name="🎭 Character",
                    value=f"**Traits:** {', '.join(character_traits) if character_traits else 'None yet'}\n**Items:** {len(inventory)} items",
                    inline=True
                )
            
            if inventory:
                embed.add_field(name="🎒 Inventory", value=", ".join(inventory[:5]) + ("..." if len(inventory) > 5 else ""), inline=False)
            
            if world_knowledge:
                embed.add_field(name="🧠 Knowledge", value=", ".join(world_knowledge[:3]) + ("..." if len(world_knowledge) > 3 else ""), inline=False)
            
            embed.add_field(name="🎯 Continue", value="Use `/continue_adventure <action>` to proceed!", inline=False)
            embed.set_footer(text=f"Adventure ID: {record['story_id'][:8]}... • Started: {record['created'].strftime('%Y-%m-%d')}")
            
            await interaction.followup.send(embed=embed)
            
    except Exception as e:
        await interaction.followup.send(f"❌ Error getting adventure status: {e}")

@bot.tree.command(name="end_adventure", description="End your current adventure and start fresh")
async def end_adventure_command(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)
    
    if not NEO4J_ENABLED:
        await interaction.followup.send("❌ Adventures require Neo4j database")
        return
    
    user_id = interaction.user.id
    server_id = interaction.guild.id
    
    try:
        with neo4j_driver.session() as session:
            # Mark current adventure as ended
            query = """
            MATCH (u:User {id: $user_id})-[r:PLAYING]->(story:Story)-[:BELONGS_TO]->(s:Server {id: $server_id})
            DELETE r
            SET story.ended_date = datetime()
            RETURN story.theme as theme, story.choices_made as choices
            """
            
            result = session.run(query, user_id=user_id, server_id=server_id)
            record = result.single()
            
            if record:
                embed = discord.Embed(
                    title="🏁 Adventure Ended",
                    description=f"Your **{record['theme']}** adventure has ended after **{record['choices']}** choices.\n\nReady to start a new adventure? Use `/start_adventure` with a new theme!",
                    color=0xf39c12
                )
            else:
                embed = discord.Embed(
                    title="🤔 No Active Adventure",
                    description="You don't have an active adventure to end. Use `/start_adventure` to begin one!",
                    color=0x95a5a6
                )
            
            await interaction.followup.send(embed=embed)
            
    except Exception as e:
        await interaction.followup.send(f"❌ Error ending adventure: {e}")

@bot.tree.command(name="save_idea", description="Store a creative idea")
@app_commands.describe(
    idea="Your creative idea",
    category="Category (art, story, game, invention, etc.)"
)
async def save_idea_command(interaction: discord.Interaction, idea: str, category: str = "general"):
    await interaction.response.defer(ephemeral=True)
    
    if not NEO4J_ENABLED:
        await interaction.followup.send("❌ Idea storage requires Neo4j database")
        return
    
    user_id = interaction.user.id
    server_id = interaction.guild.id
    
    try:
        with neo4j_driver.session() as session:
            idea_id = str(uuid.uuid4())
            tags = [word.lower() for word in idea.split() if len(word) > 3][:5]  # Extract key terms as tags
            
            query = """
            MERGE (u:User {id: $user_id})
            MERGE (s:Server {id: $server_id})
            CREATE (idea:Idea {
                id: $idea_id,
                content: $idea,
                category: $category,
                user_id: $user_id,
                server_id: $server_id,
                created_date: datetime()
            })
            MERGE (u)-[:CREATED]->(idea)
            MERGE (idea)-[:BELONGS_TO]->(s)
            
            WITH idea
            UNWIND $tags as tag
            WITH idea, tag
            WHERE size(tag) > 3
            MERGE (t:Tag {name: tag})
            MERGE (idea)-[:TAGGED_WITH]->(t)
            
            RETURN idea.id as idea_id
            """
            
            result = session.run(query, 
                               user_id=user_id, 
                               server_id=server_id,
                               idea_id=idea_id,
                               idea=idea,
                               category=category,
                               tags=tags)
            
            embed = discord.Embed(
                title="💡 Idea Saved!",
                description=f"**Category:** {category.title()}\n**Idea:** {idea}",
                color=0xf1c40f
            )
            if tags:
                embed.add_field(name="🏷️ Tags", value=", ".join(tags), inline=True)
            embed.set_footer(text=f"Idea ID: {idea_id[:8]}... | Server-specific storage")
            
            await interaction.followup.send(embed=embed)
    
    except Exception as e:
        await interaction.followup.send(f"❌ Error saving idea: {e}")

@bot.tree.command(name="random_idea_combo", description="Get a random combination of concepts for inspiration")
async def random_idea_combo_command(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=False)
    
    # Creative inspiration generator
    themes = ["magical", "futuristic", "ancient", "mysterious", "colorful", "dark", "bright", "ethereal"]
    objects = ["castle", "robot", "forest", "crystal", "sword", "book", "mirror", "door", "key", "star"]
    actions = ["dancing", "flying", "glowing", "singing", "transforming", "exploding", "melting", "growing"]
    moods = ["peaceful", "chaotic", "dreamy", "intense", "playful", "serious", "whimsical", "dramatic"]
    
    combo = f"{random.choice(themes)} {random.choice(objects)} {random.choice(actions)} in a {random.choice(moods)} way"
    
    embed = discord.Embed(
        title="🎲 Random Creative Inspiration",
        description=f"**Your inspiration combo:** *{combo}*",
        color=0xe67e22
    )
    embed.add_field(name="💡 What could this become?", 
                   value="• A story or poem\n• An art piece\n• A game concept\n• An invention idea\n• A character design", 
                   inline=False)
    embed.add_field(name="💾 Save it!", value="Use `/save_idea` to store your creation!", inline=False)
    
    await interaction.followup.send(embed=embed)

# Missing creative and analysis commands
@bot.tree.command(name="art_journey", description="View your AI art evolution over time")
@app_commands.describe(user="User to analyze (optional, defaults to you)")
async def art_journey_command(interaction: discord.Interaction, user: discord.Member = None):
    await interaction.response.defer(ephemeral=True)
    
    if not NEO4J_ENABLED:
        await interaction.followup.send("❌ Art journey requires Neo4j database")
        return
    
    target_user = user or interaction.user
    server_id = interaction.guild.id
    
    try:
        with neo4j_driver.session() as session:
            query = """
            MATCH (u:User {id: $user_id})-[:CREATED]->(art:Artwork)-[:BELONGS_TO]->(s:Server {id: $server_id})
            RETURN art.prompt as prompt, art.style as style, art.created_date as date
            ORDER BY art.created_date ASC
            LIMIT 20
            """
            
            result = session.run(query, user_id=target_user.id, server_id=server_id)
            artworks = [{"prompt": r["prompt"], "style": r["style"], "date": r["date"]} for r in result]
            
            if not artworks:
                await interaction.followup.send(f"🎨 No AI art found for {target_user.display_name} in this server yet!")
                return
            
            embed = discord.Embed(
                title=f"🎨 {target_user.display_name}'s Art Journey",
                description=f"Artistic evolution through {len(artworks)} creations",
                color=0xe91e63
            )
            
            # Show first, middle, and latest artworks
            if len(artworks) >= 3:
                first = artworks[0]
                middle = artworks[len(artworks)//2]
                latest = artworks[-1]
                
                embed.add_field(
                    name=f"🌱 First Creation ({first['date'].strftime('%Y-%m-%d') if first['date'] else 'Unknown'})",
                    value=f"*{first['prompt'][:100]}...*\nStyle: {first['style']}",
                    inline=False
                )
                
                embed.add_field(
                    name=f"🌿 Middle Journey ({middle['date'].strftime('%Y-%m-%d') if middle['date'] else 'Unknown'})",
                    value=f"*{middle['prompt'][:100]}...*\nStyle: {middle['style']}",
                    inline=False
                )
                
                embed.add_field(
                    name=f"🌟 Latest Creation ({latest['date'].strftime('%Y-%m-%d') if latest['date'] else 'Unknown'})",
                    value=f"*{latest['prompt'][:100]}...*\nStyle: {latest['style']}",
                    inline=False
                )
            else:
                for i, art in enumerate(artworks):
                    date_str = art['date'].strftime('%Y-%m-%d') if art['date'] else 'Unknown'
                    embed.add_field(
                        name=f"🎨 Creation {i+1} ({date_str})",
                        value=f"*{art['prompt'][:100]}...*\nStyle: {art['style']}",
                        inline=False
                    )
            
            embed.set_footer(text=f"Server-specific art journey for {interaction.guild.name}")
            await interaction.followup.send(embed=embed)
    
    except Exception as e:
        await interaction.followup.send(f"❌ Error retrieving art journey: {e}")

@bot.tree.command(name="idea_connections", description="Find connections between your ideas")
@app_commands.describe(current_idea="Current idea to find connections for (optional)")
async def idea_connections_command(interaction: discord.Interaction, current_idea: str = None):
    await interaction.response.defer(ephemeral=True)
    
    if not NEO4J_ENABLED:
        await interaction.followup.send("❌ Idea connections require Neo4j database")
        return
    
    user_id = interaction.user.id
    server_id = interaction.guild.id
    
    try:
        with neo4j_driver.session() as session:
            if current_idea:
                # Find ideas connected to the specified idea
                query = """
                MATCH (u:User {id: $user_id})-[:CREATED]->(idea1:Idea)-[:BELONGS_TO]->(s:Server {id: $server_id})
                WHERE toLower(idea1.content) CONTAINS toLower($current_idea)
                MATCH (idea1)-[:TAGGED_WITH]->(tag:Tag)<-[:TAGGED_WITH]-(idea2:Idea)
                WHERE idea1 <> idea2 AND (u)-[:CREATED]->(idea2)
                RETURN idea2.content as connected_idea, idea2.category as category, 
                       collect(tag.name)[0..3] as shared_tags
                ORDER BY size(shared_tags) DESC
                LIMIT 10
                """
                
                result = session.run(query, user_id=user_id, server_id=server_id, current_idea=current_idea)
                connections = [{"idea": r["connected_idea"], "category": r["category"], "tags": r["shared_tags"]} for r in result]
                
                title = f"🔗 Ideas Connected to: '{current_idea}'"
                
            else:
                # Find all idea connections for the user
                query = """
                MATCH (u:User {id: $user_id})-[:CREATED]->(idea1:Idea)-[:BELONGS_TO]->(s:Server {id: $server_id})
                MATCH (idea1)-[:TAGGED_WITH]->(tag:Tag)<-[:TAGGED_WITH]-(idea2:Idea)
                WHERE idea1 <> idea2 AND (u)-[:CREATED]->(idea2)
                RETURN idea1.content as idea1, idea2.content as idea2, 
                       collect(tag.name)[0..2] as shared_tags
                ORDER BY size(shared_tags) DESC
                LIMIT 10
                """
                
                result = session.run(query, user_id=user_id, server_id=server_id)
                connections = [{"idea1": r["idea1"], "idea2": r["idea2"], "tags": r["shared_tags"]} for r in result]
                
                title = "🔗 Your Idea Connections"
            
            if not connections:
                await interaction.followup.send("💡 No idea connections found yet. Create more ideas with `/save_idea` to see patterns!")
                return
            
            embed = discord.Embed(
                title=title,
                color=0x9c27b0
            )
            
            if current_idea:
                for i, conn in enumerate(connections):
                    embed.add_field(
                        name=f"💡 Connected Idea {i+1}",
                        value=f"**{conn['category'].title()}:** {conn['idea'][:100]}...\n🏷️ Shared: {', '.join(conn['tags'])}",
                        inline=False
                    )
            else:
                for i, conn in enumerate(connections):
                    embed.add_field(
                        name=f"🔗 Connection {i+1}",
                        value=f"**Idea A:** {conn['idea1'][:80]}...\n**Idea B:** {conn['idea2'][:80]}...\n🏷️ Shared: {', '.join(conn['tags'])}",
                        inline=False
                    )
            
            embed.set_footer(text=f"Server-specific idea connections for {interaction.guild.name}")
            await interaction.followup.send(embed=embed)
    
    except Exception as e:
        await interaction.followup.send(f"❌ Error finding idea connections: {e}")

@bot.tree.command(name="ask_enhanced", description="Ask DuckBot with personal memory context")
@app_commands.describe(prompt="Your question or request")
async def ask_enhanced_command(interaction: discord.Interaction, prompt: str):
    """Enhanced ask command with personal memory context."""
    await interaction.response.defer(ephemeral=False)
    
    if not NEO4J_ENABLED:
        await interaction.followup.send("❌ Enhanced ask requires Neo4j database for memory context")
        return
    
    user_id = interaction.user.id
    server_id = interaction.guild.id
    user_name = interaction.user.display_name
    server_name = interaction.guild.name
    
    # Check LM Studio health first
    if not await check_lm_studio_health():
        fallback_msg = await get_fallback_response(prompt, user_name)
        await interaction.followup.send(fallback_msg)
        return
    
    try:
        # Get user's memory context from Neo4j
        with neo4j_driver.session() as session:
            memory_query = """
            MATCH (u:User {id: $user_id})-[:HAS_MEMORY]->(m:Memory)-[:BELONGS_TO]->(s:Server {id: $server_id})
            RETURN m.content as memory
            ORDER BY m.created_date DESC
            LIMIT 5
            """
            
            result = session.run(memory_query, user_id=user_id, server_id=server_id)
            memories = [r["memory"] for r in result]
            
            # Build context string
            memory_context = ""
            if memories:
                memory_context = "\n\nWhat I remember about this user:\n" + "\n".join([f"- {mem}" for mem in memories])
        
        # Build enhanced payload with personal context
        base_payload = {
            "messages": [
                {
                    "role": "system",
                    "content": f"""You are DuckBot, a helpful AI assistant for Discord with personal memory.

User: {user_name} on "{server_name}" server
{memory_context}

Be helpful, personalized, and engaging. Reference their stored memories when relevant. Use tools when needed.
Keep responses under 1500 characters when possible."""
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
        
        # Try with full plugin configuration
        full_payload = {**base_payload, **FULL_PLUGIN_CONFIG}
        
        # Attempt to call LM Studio with retry logic
        data = await call_lm_studio_with_retry(full_payload)
        
        if data and "choices" in data and len(data["choices"]) > 0:
            ai_response = data["choices"][0]["message"]["content"]
            
            # Handle long responses
            if len(ai_response) > 2000:
                chunks = [ai_response[i:i+1900] for i in range(0, len(ai_response), 1900)]
                await interaction.followup.send(chunks[0])
                for chunk in chunks[1:]:
                    await interaction.followup.send(chunk)
            else:
                await interaction.followup.send(ai_response)
        else:
            fallback_msg = await get_fallback_response(prompt, user_name)
            await interaction.followup.send(fallback_msg)
    
    except Exception as e:
        print(f"Enhanced ask command error: {e}")
        fallback_msg = await get_fallback_response(prompt, user_name)
        await interaction.followup.send(fallback_msg)

print("✅ ALL enhanced commands loaded!")
print("📋 Total commands: 28")
print("   • Basic: ping")
print("   • AI Chat: ask, ask_simple, ask_enhanced, lm_health")
print("   • Image Gen: generate, generate_advanced, generate_style, model_info")
print("   • Video Gen: animate")
print("   • Analytics: server_stats, my_connections, channel_insights")
print("   • Knowledge: learn, ask_knowledge")
print("   • Memory: remember, my_context")
print("   • Adventures: start_adventure, continue_adventure")
print("   • Creative: save_idea, idea_connections, random_idea_combo, art_journey")
print("   • Management: server_info, server_config, global_stats, storage_status, force_cleanup")

# --- RUN THE BOT ---
if __name__ == "__main__":
    if not DISCORD_TOKEN:
        print("CRITICAL ERROR: DISCORD_TOKEN not found in .env file.")
    else:
        print("Starting DuckBot v2.1 Multi-Server Edition...")
        try:
            bot.run(DISCORD_TOKEN)
        except KeyboardInterrupt:
            print("Bot shutting down...")
        finally:
            close_neo4j()