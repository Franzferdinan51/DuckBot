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

# Queue system for managing generation requests
@dataclass
class QueueItem:
    interaction: discord.Interaction
    prompt: str
    generation_type: str  # "image" or "video"
    status_message: discord.Message = None
    model_id: str = None  # For enhanced image generation

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
    try:
        # Connect to ComfyUI WebSocket
        uri = f"ws://{COMFYUI_SERVER_ADDRESS}/ws"
        async with websockets.connect(uri) as websocket:
            # Queue the workflow
            queue_url = f"http://{COMFYUI_SERVER_ADDRESS}/prompt"
            payload = {"prompt": workflow_data}
            
            response = requests.post(queue_url, json=payload)
            if response.status_code != 200:
                print(f"Failed to queue workflow: {response.status_code}")
                return []
            
            result = response.json()
            prompt_id = result.get("prompt_id")
            
            if not prompt_id:
                print("No prompt_id received")
                return []
            
            # Wait for completion
            while True:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=120.0)
                    data = json.loads(message)
                    
                    if data.get("type") == "executed" and data.get("data", {}).get("prompt_id") == prompt_id:
                        break
                except asyncio.TimeoutError:
                    print("Timeout waiting for ComfyUI response")
                    return []
                except websockets.exceptions.ConnectionClosed:
                    print("WebSocket connection closed")
                    return []
            
            # Get the generated files
            return await get_output_files(prompt_id, is_video)
            
    except Exception as e:
        print(f"Error in ComfyUI workflow: {e}")
        return []

async def get_output_files(prompt_id: str, is_video: bool = False) -> list:
    """Retrieve output files from ComfyUI."""
    try:
        history_url = f"http://{COMFYUI_SERVER_ADDRESS}/history/{prompt_id}"
        response = requests.get(history_url)
        
        if response.status_code != 200:
            print(f"Failed to get history: {response.status_code}")
            return []
        
        history = response.json()
        if prompt_id not in history:
            print("Prompt ID not found in history")
            return []
        
        outputs = history[prompt_id].get("outputs", {})
        file_data_list = []
        
        # Look for output nodes (usually SaveImage nodes)
        for node_id, node_output in outputs.items():
            if "images" in node_output:
                for image_info in node_output["images"]:
                    filename = image_info["filename"]
                    subfolder = image_info.get("subfolder", "")
                    
                    # Download the file
                    if subfolder:
                        file_url = f"http://{COMFYUI_SERVER_ADDRESS}/view?filename={filename}&subfolder={subfolder}"
                    else:
                        file_url = f"http://{COMFYUI_SERVER_ADDRESS}/view?filename={filename}"
                    
                    file_response = requests.get(file_url)
                    if file_response.status_code == 200:
                        file_data_list.append(file_response.content)
        
        return file_data_list
        
    except Exception as e:
        print(f"Error getting output files: {e}")
        return []

async def add_to_queue(interaction: discord.Interaction, prompt: str, generation_type: str):
    """Add a generation request to the server-specific queue."""
    server_queue = get_server_queue(interaction.guild.id)
    queue_item = QueueItem(interaction, prompt, generation_type)
    
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
        images_data = await run_comfyui_workflow(prompt_workflow, is_video=False)
        
        if not images_data:
            await queue_item.status_message.edit(
                content=f"❌ **Image generation failed**\nPrompt: `{queue_item.prompt}`"
            )
            return
        
        # Create Discord files
        image_files = []
        for i, data in enumerate(images_data):
            filename = f"generated_image_{uuid.uuid4()}.png"
            image_files.append(discord.File(fp=BytesIO(data), filename=filename))
        
        # Send the generated images
        await queue_item.status_message.edit(
            content=f"✅ **Image generation complete!**\nPrompt: `{queue_item.prompt}`"
        )
        await queue_item.interaction.followup.send(files=image_files)
        
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
    
    # Sync slash commands
    try:
        synced = await bot.tree.sync()
        print(f"🔄 Synced {len(synced)} slash commands")
        
        # List all available commands
        print("📋 Available slash commands:")
        for command in synced:
            print(f"   /{command.name} - {command.description}")
            
    except Exception as e:
        print(f"❌ Failed to sync commands: {e}")
    
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
    task_type: str  # 'image' or 'video'
    server_id: int
    status_message: Optional[discord.Message] = None

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
        "model_file": "v1-5-pruned-emaonly.ckpt"
    }
}

# Model priority order (best to worst)
MODEL_PRIORITY = ["flux", "sdxl", "sd15"]

def get_available_models() -> List[str]:
    """Check which model workflows are available."""
    available = []
    for model_id, model_info in IMAGE_MODELS.items():
        workflow_path = model_info["workflow"]
        if os.path.exists(workflow_path):
            available.append(model_id)
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
        
        # Create Discord files
        image_files = []
        for i, data in enumerate(images_data):
            filename = f"{model_id}_image_{uuid.uuid4()}.png"
            image_files.append(discord.File(fp=BytesIO(data), filename=filename))
        
        # Success message with model info
        embed = discord.Embed(
            title="✅ Image Generation Complete!",
            description=f"**Model:** {model_info['name']}\n**Prompt:** {queue_item.prompt}",
            color=0x00ff88
        )
        embed.add_field(name="⚡ Generation Time", value=f"{time.time() - start_time:.1f}s", inline=True)
        embed.add_field(name="📐 Resolution", value=model_info['resolution'], inline=True)
        embed.add_field(name="🎯 Best For", value=model_info['best_for'], inline=False)
        
        await queue_item.status_message.edit(content="", embed=embed)
        await queue_item.interaction.followup.send(files=image_files)
        
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
    queue_item = QueueItem(interaction, prompt, "image")
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
    
    # Use the advanced generation with auto model selection
    await generate_advanced_command(interaction, enhanced_prompt, "auto")

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
    await interaction.response.defer(ephemeral=False)
    
    user_name = interaction.user.display_name
    server_name = interaction.guild.name if interaction.guild else "DM"
    
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
        # Test 1: Try with minimal plugin config
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
            
            if response.status_code == 200:
                plugin_status = "✅ Minimal plugins working"
            else:
                # Test 2: Try without any plugins
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
                    plugin_status = "⚠️ Plugins disabled, basic working"
                else:
                    plugin_status = f"❌ Failed (Status: {response.status_code})"
            
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
@app_commands.describe(theme="Adventure theme (fantasy, sci-fi, mystery, etc.)")
async def start_adventure_command(interaction: discord.Interaction, theme: str = "fantasy"):
    await interaction.response.defer(ephemeral=False)
    
    if not NEO4J_ENABLED:
        await interaction.followup.send("❌ Adventures require Neo4j database for state management")
        return
    
    user_id = interaction.user.id
    server_id = interaction.guild.id
    
    try:
        with neo4j_driver.session() as session:
            # Create new adventure story
            story_id = str(uuid.uuid4())
            initial_story = f"🌟 **{theme.title()} Adventure Begins!** 🌟\n\nYou find yourself at the beginning of an epic {theme} adventure. The world awaits your choices!\n\n*What would you like to do first?*"
            
            query = """
            MERGE (u:User {id: $user_id})
            MERGE (s:Server {id: $server_id})
            CREATE (story:Story {
                id: $story_id,
                user_id: $user_id,
                server_id: $server_id,
                theme: $theme,
                current_scene: $initial_story,
                choices_made: 0,
                created_date: datetime()
            })
            MERGE (u)-[:PLAYING]->(story)
            MERGE (story)-[:BELONGS_TO]->(s)
            RETURN story.id as story_id
            """
            
            session.run(query, 
                       user_id=user_id, 
                       server_id=server_id,
                       story_id=story_id,
                       theme=theme,
                       initial_story=initial_story)
            
            embed = discord.Embed(
                title=f"🎮 Adventure Started: {theme.title()}",
                description=initial_story,
                color=0xf39c12
            )
            embed.add_field(name="🎯 Next Step", value="Use `/continue_adventure` with your choice!", inline=False)
            embed.set_footer(text=f"Adventure ID: {story_id[:8]}...")
            
            await interaction.followup.send(embed=embed)
    
    except Exception as e:
        await interaction.followup.send(f"❌ Error starting adventure: {e}")

@bot.tree.command(name="continue_adventure", description="Continue your adventure")
@app_commands.describe(action="What do you want to do?")
async def continue_adventure_command(interaction: discord.Interaction, action: str):
    await interaction.response.defer(ephemeral=False)
    
    if not NEO4J_ENABLED:
        await interaction.followup.send("❌ Adventures require Neo4j database")
        return
    
    user_id = interaction.user.id
    server_id = interaction.guild.id
    
    try:
        with neo4j_driver.session() as session:
            # Get current adventure
            query = """
            MATCH (u:User {id: $user_id})-[:PLAYING]->(story:Story)-[:BELONGS_TO]->(s:Server {id: $server_id})
            RETURN story.id as story_id, story.theme as theme, story.choices_made as choices
            ORDER BY story.created_date DESC
            LIMIT 1
            """
            
            result = session.run(query, user_id=user_id, server_id=server_id)
            record = result.single()
            
            if not record:
                await interaction.followup.send("🤔 You don't have an active adventure. Use `/start_adventure` to begin!")
                return
            
            # Generate next scene (simple adventure logic)
            choices = record["choices"] + 1
            theme = record["theme"]
            
            adventure_responses = {
                "fantasy": [
                    f"🗡️ You {action}. A mysterious path opens before you, leading to an ancient castle.",
                    f"🧙‍♂️ Your action to {action} attracts the attention of a wise old wizard who offers guidance.",
                    f"🐉 As you {action}, you hear the distant roar of a dragon echoing through the mountains."
                ],
                "sci-fi": [
                    f"🚀 You {action}. The spaceship's AI responds with new coordinates to an unexplored galaxy.",
                    f"🤖 Your decision to {action} activates a dormant android that becomes your companion.",
                    f"⚡ As you {action}, the quantum field fluctuates, opening a portal to another dimension."
                ],
                "mystery": [
                    f"🔍 You {action}. A new clue appears that deepens the mystery even further.",
                    f"🕵️ Your choice to {action} reveals a secret passage in the old mansion.",
                    f"📜 As you {action}, you discover a cryptic message left by the previous investigator."
                ]
            }
            
            responses = adventure_responses.get(theme, adventure_responses["fantasy"])
            next_scene = random.choice(responses)
            
            # Update adventure
            update_query = """
            MATCH (story:Story {id: $story_id})
            SET story.current_scene = $next_scene,
                story.choices_made = $choices
            """
            
            session.run(update_query, story_id=record["story_id"], next_scene=next_scene, choices=choices)
            
            embed = discord.Embed(
                title=f"🎮 Adventure Continues: {theme.title()}",
                description=next_scene,
                color=0xf39c12
            )
            embed.add_field(name="📊 Progress", value=f"Choices made: {choices}", inline=True)
            embed.add_field(name="🎯 Continue", value="Use `/continue_adventure` again!", inline=True)
            
            await interaction.followup.send(embed=embed)
    
    except Exception as e:
        await interaction.followup.send(f"❌ Error continuing adventure: {e}")

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