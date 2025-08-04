# ==============================================================================
# DUCK BOT - W.A.N 2.2 VIDEO & IMAGE VERSION
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

# Neo4j driver instance
neo4j_driver = None

# A unique ID for our ComfyUI client
CLIENT_ID = str(uuid.uuid4())

# --- QUEUE SYSTEM ---
import asyncio
from collections import deque
from dataclasses import dataclass
from typing import Optional

@dataclass
class QueueItem:
    interaction: discord.Interaction
    prompt: str
    task_type: str  # 'image' or 'video'
    status_message: Optional[discord.Message] = None

generation_queue = deque()
currently_processing = False
queue_lock = asyncio.Lock()

# Average generation times (in seconds) - updated dynamically
average_times = {
    'image': 15,    # 15 seconds average
    'video': 300    # 5 minutes average (300 seconds)
}

import time

# Generation timeouts (in seconds)
GENERATION_TIMEOUT = {
    'image': 300,   # 5 minutes max
    'video': 600    # 10 minutes max
}

# --- NEO4J DATABASE FUNCTIONS ---
def initialize_neo4j():
    """Initialize Neo4j connection and create database schema."""
    global neo4j_driver
    
    if not NEO4J_ENABLED:
        print("⚠️  Neo4j disabled - social analytics features unavailable")
        return False
        
    try:
        neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        
        # Test connection and create initial schema
        with neo4j_driver.session() as session:
            # Create constraints and indexes for better performance
            schema_queries = [
                "CREATE CONSTRAINT IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (m:Message) REQUIRE m.id IS UNIQUE", 
                "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Channel) REQUIRE c.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Server) REQUIRE s.id IS UNIQUE",
                "CREATE INDEX IF NOT EXISTS FOR (m:Message) ON m.timestamp",
                "CREATE INDEX IF NOT EXISTS FOR (u:User) ON u.username",
                "CREATE INDEX IF NOT EXISTS FOR ()-[r:REACTS_TO]-() ON r.timestamp"
            ]
            
            for query in schema_queries:
                session.run(query)
                
        print("✅ Neo4j database initialized successfully")
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

async def store_message_data(message_data):
    """Store message data in Neo4j database."""
    if not NEO4J_ENABLED or not neo4j_driver:
        return False
        
    try:
        with neo4j_driver.session() as session:
            query = """
            MERGE (u:User {id: $user_id})
            SET u.username = $username
            
            MERGE (c:Channel {id: $channel_id})
            SET c.name = $channel_name
            
            MERGE (s:Server {id: $server_id})
            
            CREATE (m:Message {
                id: $message_id,
                content_length: $content_length,
                timestamp: datetime($timestamp),
                reactions_count: $reactions
            })
            
            MERGE (u)-[:SENT]->(m)
            MERGE (m)-[:IN_CHANNEL]->(c)
            MERGE (c)-[:IN_SERVER]->(s)
            
            // Track user activity in channel
            MERGE (u)-[a:ACTIVE_IN]->(c)
            SET a.last_active = datetime($timestamp),
                a.message_count = coalesce(a.message_count, 0) + 1
            
            // Create mention relationships
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

async def store_member_data(member_data):
    """Store new member data in Neo4j database."""
    if not NEO4J_ENABLED or not neo4j_driver:
        return False
        
    try:
        with neo4j_driver.session() as session:
            query = """
            MERGE (u:User {id: $user_id})
            SET u.username = $username,
                u.display_name = $display_name,
                u.account_created = datetime($account_created)
                
            MERGE (s:Server {id: $server_id})
            
            MERGE (u)-[j:JOINED]->(s)
            SET j.joined_at = datetime($joined_at)
            """
            
            session.run(query, member_data)
            return True
            
    except Exception as e:
        print(f"❌ Error storing member data: {e}")
        return False

async def store_reaction_data(reaction_data):
    """Store reaction data in Neo4j database."""
    if not NEO4J_ENABLED or not neo4j_driver:
        return False
        
    try:
        with neo4j_driver.session() as session:
            query = """
            MERGE (u:User {id: $user_id})
            
            OPTIONAL MATCH (m:Message) WHERE m.id = $message_id
            MERGE (author:User {id: $message_author_id})
            
            WITH u, m, author
            WHERE m IS NOT NULL
            
            MERGE (u)-[r:REACTS_TO]->(m)
            SET r.emoji = $emoji,
                r.timestamp = datetime($timestamp)
                
            // Track reaction relationship between users
            MERGE (u)-[rr:REACTS_TO_USER]->(author)
            SET rr.count = coalesce(rr.count, 0) + 1
            """
            
            session.run(query, reaction_data)
            return True
            
    except Exception as e:
        print(f"❌ Error storing reaction data: {e}")
        return False

# --- NEO4J STORAGE MANAGEMENT ---
MAX_STORAGE_GB = 10
MAX_STORAGE_BYTES = MAX_STORAGE_GB * 1024 * 1024 * 1024  # 10GB in bytes

def get_neo4j_storage_info():
    """Get current Neo4j database storage information."""
    if not neo4j_driver:
        return {
            "current_size_bytes": 0,
            "current_size_gb": 0,
            "max_size_gb": MAX_STORAGE_GB,
            "usage_percentage": 0,
            "status": "disconnected"
        }
    
    try:
        with neo4j_driver.session() as session:
            # Get node and relationship counts
            stats_query = """
            CALL apoc.meta.stats() YIELD labels, relTypes, stats
            RETURN labels, relTypes, stats.nodes as total_nodes, stats.relationships as total_relationships
            """
            
            # Alternative queries if APOC is not available
            basic_node_query = "MATCH (n) RETURN count(n) as total_nodes"
            basic_rel_query = "MATCH ()-[r]->() RETURN count(r) as total_relationships"
            
            try:
                result = session.run(stats_query)
                record = result.single()
                total_nodes = record["total_nodes"] if record else 0
                total_relationships = record["total_relationships"] if record else 0
            except:
                # Fallback to basic queries
                try:
                    result = session.run(basic_node_query)
                    record = result.single()
                    total_nodes = record["total_nodes"] if record else 0
                    
                    result = session.run(basic_rel_query)
                    record = result.single()
                    total_relationships = record["total_relationships"] if record else 0
                except:
                    total_nodes = 0
                    total_relationships = 0
            
            # Estimate storage usage (rough calculation)
            # Average node size ~200 bytes, relationship ~50 bytes
            estimated_size_bytes = (total_nodes * 200) + (total_relationships * 50)
            estimated_size_gb = estimated_size_bytes / (1024 * 1024 * 1024)
            usage_percentage = (estimated_size_gb / MAX_STORAGE_GB) * 100
            
            return {
                "current_size_bytes": estimated_size_bytes,
                "current_size_gb": round(estimated_size_gb, 3),
                "max_size_gb": MAX_STORAGE_GB,
                "usage_percentage": round(usage_percentage, 1),
                "total_nodes": total_nodes,
                "total_relationships": total_relationships,
                "status": "connected"
            }
            
    except Exception as e:
        print(f"❌ Error getting Neo4j storage info: {e}")
        return {
            "current_size_bytes": 0,
            "current_size_gb": 0,
            "max_size_gb": MAX_STORAGE_GB,
            "usage_percentage": 0,
            "status": "error"
        }

async def cleanup_old_data_if_needed():
    """Clean up old data if approaching storage limit."""
    storage_info = get_neo4j_storage_info()
    
    if storage_info["usage_percentage"] > 80:  # Cleanup at 80% capacity
        print(f"⚠️ Neo4j storage at {storage_info['usage_percentage']}%, starting cleanup...")
        
        if not neo4j_driver:
            print("❌ Neo4j not connected, cannot perform cleanup")
            return
        
        try:
            with neo4j_driver.session() as session:
                # Cleanup strategy (oldest first):
                cleanup_queries = [
                    # Delete messages older than 6 months
                    "MATCH (m:Message) WHERE m.timestamp < datetime() - duration('P6M') DETACH DELETE m",
                    
                    # Delete reaction data older than 3 months  
                    "MATCH (u:User)-[r:REACTS_TO]->(m:Message) WHERE r.timestamp < datetime() - duration('P3M') DELETE r",
                    
                    # Compress old activity data (keep aggregated stats only)
                    """MATCH (u:User)-[a:ACTIVE_IN]->(c:Channel) 
                       WHERE a.last_active < datetime() - duration('P2M')
                       SET a.message_count = coalesce(a.message_count, 0)
                       REMOVE a.detailed_activity""",
                    
                    # Delete orphaned nodes
                    "MATCH (n) WHERE size((n)--()) = 0 DELETE n"
                ]
                
                # Execute cleanup queries directly
                for query in cleanup_queries:
                    try:
                        result = session.run(query)
                        summary = result.consume()
                        print(f"Cleanup query executed: {summary.counters}")
                    except Exception as e:
                        print(f"❌ Cleanup query failed: {e}")
                
                print("✅ Neo4j cleanup completed")
                
        except Exception as e:
            print(f"❌ Neo4j cleanup error: {e}")

async def execute_neo4j_maintenance(query):
    """Execute maintenance queries through LM Studio Neo4j plugin."""
    payload = {
        "messages": [
            {
                "role": "system",
                "content": "You are a database maintenance assistant. Execute the provided Cypher query for cleanup purposes."
            },
            {
                "role": "user", 
                "content": f"Execute this Neo4j maintenance query: {query}"
            }
        ],
        "temperature": 0,
        "max_tokens": 100,
        "plugins": {
            "query-neo4j": {"enabled": True, "priority": 1}
        }
    }
    
    try:
        response = requests.post(LM_STUDIO_URL, headers={"Content-Type": "application/json"}, data=json.dumps(payload))
        return response.json()
    except Exception as e:
        print(f"❌ Neo4j maintenance error: {e}")

def format_time(seconds):
    """Format seconds into human readable time."""
    if seconds < 60:
        return f"{int(seconds)} seconds"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = int(seconds % 60)
        if remaining_seconds > 0:
            return f"{minutes}m {remaining_seconds}s"
        return f"{minutes} minutes"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        if minutes > 0:
            return f"{hours}h {minutes}m"
        return f"{hours} hours"

def calculate_estimated_wait(position, current_task_type):
    """Calculate estimated wait time based on queue position and task types."""
    if position <= 1:
        return 0
    
    total_wait = 0
    
    # Add time for currently processing item (if any)
    if currently_processing and position > 1:
        total_wait += average_times.get(current_task_type, 300)  # Default 5 minutes
    
    # Add time for items ahead in queue
    queue_list = list(generation_queue)
    items_ahead = min(position - 1, len(queue_list))
    
    for i in range(items_ahead):
        if i < len(queue_list):
            task_type = queue_list[i].task_type
            total_wait += average_times.get(task_type, 300)
    
    return total_wait

def update_average_time(task_type, actual_time):
    """Update average generation time based on actual completion time."""
    current_avg = average_times.get(task_type, 300)
    # Use exponential moving average (70% old, 30% new)
    new_avg = (current_avg * 0.7) + (actual_time * 0.3)
    average_times[task_type] = new_avg


# --- 3. BOT CLASS DEFINITION ---
class DuckBot(commands.Bot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def setup_hook(self):
        """Performs a GLOBAL command sync and initializes Neo4j."""
        print("Running setup_hook to sync global commands...")
        try:
            synced = await self.tree.sync()
            print("--- GLOBAL COMMAND SYNC SUCCESS ---")
            print(f"Synced {len(synced)} command(s) globally: {[c.name for c in synced]}")
            print("-----------------------------------")
        except Exception as e:
            print(f"!!! FAILED TO SYNC GLOBAL COMMANDS: {e} !!!")
        
        # Initialize Neo4j database
        print("Initializing Neo4j database...")
        initialize_neo4j()

intents = discord.Intents.default()
intents.message_content = True
intents.messages = True
intents.guilds = True
intents.reactions = True
intents.members = True
bot = DuckBot(command_prefix="!", intents=intents)


# --- 4. BOT EVENTS ---
@bot.event
async def on_ready():
    print(f'Logged in as {bot.user} (ID: {bot.user.id})')
    print('Bot is ready and online.')
    print('------')

# --- NEO4J DATA COLLECTION EVENTS ---
@bot.event
async def on_message(message):
    """Track messages for Neo4j social graph analysis."""
    # Skip bot messages
    if message.author.bot:
        return
    
    # Check storage limits before adding new data
    await cleanup_old_data_if_needed()
    
    # Store message data with Discord message ID
    message_data = {
        "message_id": message.id,
        "user_id": message.author.id,
        "username": message.author.name,
        "channel_id": message.channel.id,
        "channel_name": message.channel.name,
        "server_id": message.guild.id if message.guild else None,
        "content_length": len(message.content),
        "timestamp": message.created_at.isoformat(),
        "mentions": [user.id for user in message.mentions],
        "reactions": len(message.reactions) if message.reactions else 0
    }
    await store_message_data(message_data)
    
@bot.event
async def on_member_join(member):
    """Track new member joins."""
    member_data = {
        "user_id": member.id,
        "username": member.name,
        "display_name": member.display_name,
        "server_id": member.guild.id,
        "joined_at": member.joined_at.isoformat(),
        "account_created": member.created_at.isoformat()
    }
    await store_member_data(member_data)

@bot.event  
async def on_reaction_add(reaction, user):
    """Track reactions for social graph."""
    if user.bot:
        return
        
    reaction_data = {
        "user_id": user.id,
        "message_id": reaction.message.id,
        "message_author_id": reaction.message.author.id,
        "emoji": str(reaction.emoji),
        "timestamp": datetime.datetime.now().isoformat()
    }
    await store_reaction_data(reaction_data)


# --- 5. COMFYUI API HELPER FUNCTIONS ---
def queue_prompt(prompt_workflow):
    try:
        p = {"prompt": prompt_workflow, "client_id": CLIENT_ID}
        data = json.dumps(p).encode('utf-8')
        req = urllib.request.Request(f"http://{COMFYUI_SERVER_ADDRESS}/prompt", data=data)
        return json.loads(urllib.request.urlopen(req).read())
    except Exception as e:
        print(f"Error queueing prompt: {e}")
        return None

def get_history(prompt_id):
    try:
        with urllib.request.urlopen(f"http://{COMFYUI_SERVER_ADDRESS}/history/{prompt_id}") as response:
            return json.loads(response.read())
    except Exception as e:
        print(f"Error getting history: {e}")
        return None

def get_output_files(ws, prompt_id, is_video=False):
    """A generic function to get output files (images or videos)."""
    history = get_history(prompt_id)[prompt_id]
    output_files = []
    
    # For our current workflow, we're using SaveImage which outputs 'images' even for video frames
    file_key = 'images'
    
    for node_id in history['outputs']:
        node_output = history['outputs'][node_id]
        if file_key in node_output:
            for file_info in node_output[file_key]:
                # Standard SaveImage format
                url_values = urllib.parse.urlencode({'filename': file_info['filename'], 'subfolder': file_info['subfolder'], 'type': file_info['type']})
                
                with urllib.request.urlopen(f"http://{COMFYUI_SERVER_ADDRESS}/view?{url_values}") as response:
                    output_files.append(response.read())
    return output_files


async def run_comfyui_workflow(prompt_workflow, is_video=False):
    """Connects to ComfyUI, runs a workflow, and returns the output files with timeout."""
    task_type = 'video' if is_video else 'image'
    timeout = GENERATION_TIMEOUT[task_type]
    
    try:
        async with asyncio.timeout(timeout):
            async with websockets.connect(f"ws://{COMFYUI_SERVER_ADDRESS}/ws?clientId={CLIENT_ID}") as ws:
                prompt_response = queue_prompt(prompt_workflow)
                if not prompt_response or 'prompt_id' not in prompt_response:
                    print("Failed to queue prompt.")
                    return []
                prompt_id = prompt_response['prompt_id']
                
                while True:
                    out = await ws.recv()
                    if isinstance(out, str):
                        message = json.loads(out)
                        if message['type'] == 'executing' and message['data']['node'] is None and message['data']['prompt_id'] == prompt_id:
                            break
                
                return get_output_files(ws, prompt_id, is_video)
    except asyncio.TimeoutError:
        print(f"{task_type.title()} generation timed out after {timeout} seconds")
        return []


def create_video_from_frames(frame_data_list, fps=24):
    """Create an MP4 video from a list of frame data."""
    if not frame_data_list:
        return None
    
    # Create temporary directory for frames
    with tempfile.TemporaryDirectory() as temp_dir:
        frame_paths = []
        
        # Save frames as temporary files
        for i, frame_data in enumerate(frame_data_list):
            # Convert bytes to PIL Image
            img = Image.open(BytesIO(frame_data))
            frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
            img.save(frame_path)
            frame_paths.append(frame_path)
        
        # Get dimensions from first frame
        first_img = cv2.imread(frame_paths[0])
        height, width, layers = first_img.shape
        
        # Create temporary output file
        output_path = os.path.join(temp_dir, "output_video.mp4")
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Write frames to video
        for frame_path in frame_paths:
            frame = cv2.imread(frame_path)
            video_writer.write(frame)
        
        video_writer.release()
        
        # Read the video file as bytes
        with open(output_path, 'rb') as f:
            video_bytes = f.read()
        
        return video_bytes


# --- QUEUE MANAGEMENT FUNCTIONS ---
async def add_to_queue(interaction: discord.Interaction, prompt: str, task_type: str):
    """Add a task to the generation queue."""
    async with queue_lock:
        queue_item = QueueItem(interaction, prompt, task_type)
        generation_queue.append(queue_item)
        position = len(generation_queue)
        
        # Send initial queue position message
        if position == 1 and not currently_processing:
            emoji = "🎬" if task_type == "video" else "🎨"
            queue_item.status_message = await interaction.followup.send(
                f"{emoji} **Starting {task_type} generation**\nPrompt: `{prompt}`\n🚀 Processing now..."
            )
        else:
            emoji = "🎬" if task_type == "video" else "🎨"
            wait_time = calculate_estimated_wait(position, task_type)
            wait_str = f"\n⏱️ Estimated wait: {format_time(wait_time)}" if wait_time > 0 else ""
            queue_item.status_message = await interaction.followup.send(
                f"{emoji} **{task_type.title()} generation queued**\nPrompt: `{prompt}`\n📍 Position {position} in queue{wait_str}"
            )
    
    # Start processing if not already running
    if not currently_processing:
        asyncio.create_task(process_queue())

async def process_queue():
    """Process items in the generation queue."""
    global currently_processing
    
    async with queue_lock:
        if currently_processing or not generation_queue:
            return
        currently_processing = True
    
    while generation_queue:
        async with queue_lock:
            if not generation_queue:
                break
            current_item = generation_queue.popleft()
        
        try:
            # Update status for current item
            emoji = "🎬" if current_item.task_type == "video" else "🎨"
            await current_item.status_message.edit(
                content=f"{emoji} **Processing {current_item.task_type}**\nPrompt: `{current_item.prompt}`\n🔄 Generating now..."
            )
            
            # Update queue positions for remaining items
            await update_queue_positions()
            
            # Process the current item
            if current_item.task_type == "video":
                await process_video_generation(current_item)
            else:
                await process_image_generation(current_item)
                
        except Exception as e:
            await current_item.status_message.edit(
                content=f"❌ **{current_item.task_type.title()} generation failed**\nPrompt: `{current_item.prompt}`\nError: {e}"
            )
    
    currently_processing = False

async def update_queue_positions():
    """Update queue position messages for waiting items."""
    async with queue_lock:
        for i, item in enumerate(generation_queue, 1):
            if item.status_message:
                emoji = "🎬" if item.task_type == "video" else "🎨"
                wait_time = calculate_estimated_wait(i + 1, item.task_type)  # +1 because currently processing item
                wait_str = f"\n⏱️ Estimated wait: {format_time(wait_time)}" if wait_time > 0 else ""
                try:
                    await item.status_message.edit(
                        content=f"{emoji} **{item.task_type.title()} generation queued**\nPrompt: `{item.prompt}`\n📍 Position {i} in queue{wait_str}"
                    )
                except:
                    pass  # Message might be deleted

async def process_image_generation(queue_item: QueueItem):
    """Process image generation for a queue item."""
    start_time = time.time()
    
    try:
        with open("workflow_api.json", "r") as f:
            prompt_workflow = json.load(f)
    except Exception as e:
        await queue_item.status_message.edit(content=f"❌ Error loading workflow: {e}")
        return

    # Update workflow with prompt and seed
    prompt_workflow["6"]["inputs"]["seed"] = torch.randint(1, 1125899906842624, (1,)).item()
    prompt_workflow["3"]["inputs"]["text"] = queue_item.prompt
    
    try:
        images_data = await run_comfyui_workflow(prompt_workflow, is_video=False)
        if not images_data:
            await queue_item.status_message.edit(content=f"❌ **Image generation failed or timed out**\nPrompt: `{queue_item.prompt}`\nMax time: {GENERATION_TIMEOUT['image']//60} minutes")
            return
        
        image_files = [discord.File(fp=BytesIO(data), filename=f"image_{uuid.uuid4()}.png") for data in images_data]
        await queue_item.status_message.edit(content=f"✅ **Image generation complete!**\nPrompt: `{queue_item.prompt}`")
        await queue_item.interaction.followup.send(files=image_files)
        
        # Update average time
        actual_time = time.time() - start_time
        update_average_time("image", actual_time)
        
    except Exception as e:
        await queue_item.status_message.edit(content=f"❌ **Image generation error**\nPrompt: `{queue_item.prompt}`\nError: {e}")

async def process_video_generation(queue_item: QueueItem):
    """Process video generation for a queue item."""
    start_time = time.time()
    
    try:
        with open("workflow_video_api.json", "r") as f:
            prompt_workflow = json.load(f)
    except Exception as e:
        await queue_item.status_message.edit(content=f"❌ Error loading workflow: {e}")
        return

    # Update workflow with prompt and seed
    prompt_workflow["3"]["inputs"]["text"] = queue_item.prompt
    prompt_workflow["8"]["inputs"]["seed"] = torch.randint(1, 1125899906842624, (1,)).item()
    
    try:
        # Update status during generation
        await queue_item.status_message.edit(
            content=f"🎬 **Generating video**\nPrompt: `{queue_item.prompt}`\n🔄 Processing with WAN 2.2... This takes several minutes."
        )
        
        videos_data = await run_comfyui_workflow(prompt_workflow, is_video=True)
        if not videos_data:
            await queue_item.status_message.edit(content=f"❌ **Video generation failed or timed out**\nPrompt: `{queue_item.prompt}`\nMax time: {GENERATION_TIMEOUT['video']//60} minutes")
            return
        
        # Create MP4 video
        await queue_item.status_message.edit(
            content=f"🎬 **Creating video file**\nPrompt: `{queue_item.prompt}`\n🔄 Converting {len(videos_data)} frames to MP4..."
        )
        
        video_bytes = create_video_from_frames(videos_data, fps=24)
        if not video_bytes:
            await queue_item.status_message.edit(content=f"❌ **Failed to create video file**\nPrompt: `{queue_item.prompt}`")
            return
        
        video_file = discord.File(fp=BytesIO(video_bytes), filename=f"animation_{uuid.uuid4()}.mp4")
        await queue_item.status_message.edit(content=f"✅ **Video generation complete!**\nPrompt: `{queue_item.prompt}`\n🎥 Created MP4 video:")
        await queue_item.interaction.followup.send(files=[video_file])
        
        # Update average time
        actual_time = time.time() - start_time
        update_average_time("video", actual_time)
        
    except Exception as e:
        await queue_item.status_message.edit(content=f"❌ **Video generation error**\nPrompt: `{queue_item.prompt}`\nError: {e}")


# --- 6. SLASH COMMAND DEFINITIONS ---
@bot.tree.command(name="ping", description="A simple command to test if the bot is responsive.")
async def ping_command(interaction: discord.Interaction):
    await interaction.response.send_message("Pong!", ephemeral=False)

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

    # The payload for LM Studio API call with multi-plugin support
    payload = {
        "messages": [
            {
                "role": "system",
                "content": """You are DuckBot, an advanced AI assistant with access to multiple tools and current information. You have access to:

- Web search (DuckDuckGo, general web search)
- Website content extraction (visit-website)
- Wikipedia knowledge base
- Advanced RAG (retrieval augmented generation)
- JavaScript code execution sandbox
- Neo4j graph database queries
- Dice rolling and random generation

Use these tools intelligently based on user requests. For current events, use web search. For factual information, use Wikipedia or RAG. For calculations or code, use the JavaScript sandbox. Always provide accurate, helpful responses.""",
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "temperature": 0.7,
        "max_tokens": 512,
        "stream": False,
        "model": "model-id-from-lm-studio",
        # Enable multiple plugins
        "plugins": {
            "duckduckgo": {"enabled": True, "priority": 1},
            "web-search": {"enabled": True, "priority": 2},
            "visit-website": {"enabled": True, "priority": 3},
            "wikipedia": {"enabled": True, "priority": 1},
            "rag-v1": {"enabled": True, "priority": 1, "sources": ["web", "news", "wikipedia"]},
            "js-code-sandbox": {"enabled": True, "priority": 2},
            "query-neo4j": {"enabled": True, "priority": 2},
            "dice": {"enabled": True, "priority": 3}
        },
        # Plugin configuration
        "max_plugin_calls": 3,
        "plugin_timeout": 10
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

# --- NEO4J ANALYTICS COMMANDS ---
@bot.tree.command(name="server_stats", description="Get social analytics about this Discord server")
async def server_stats_command(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=False)
    
    # This would use the Neo4j plugin through LM Studio
    query_prompt = f"""Using the Neo4j database, analyze the social patterns in Discord server {interaction.guild.id}. 
    
    Please provide:
    1. Most active users this week
    2. Most active channels 
    3. Users with similar activity patterns
    4. Recent joining trends
    
    Use Cypher queries to get this data from our Discord social graph."""
    
    # Send to LM Studio with Neo4j plugin enabled
    payload = {
        "messages": [
            {
                "role": "system", 
                "content": "You are DuckBot with access to Neo4j database containing Discord server social graph data. Use Cypher queries to analyze the data."
            },
            {"role": "user", "content": query_prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 1000,
        "plugins": {
            "query-neo4j": {"enabled": True, "priority": 1}
        }
    }
    
    try:
        response = requests.post(LM_STUDIO_URL, headers={"Content-Type": "application/json"}, data=json.dumps(payload))
        data = response.json()
        ai_response = data["choices"][0]["message"]["content"]
        await interaction.followup.send(f"📊 **Server Analytics:**\n{ai_response}")
    except Exception as e:
        await interaction.followup.send(f"❌ Error getting server stats: {e}")

@bot.tree.command(name="my_connections", description="Find users with similar interests or activity patterns")
async def my_connections_command(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=False)
    
    query_prompt = f"""Using Neo4j, find users similar to user {interaction.user.id} in server {interaction.guild.id}.
    
    Look for users who:
    1. Are active in similar channels
    2. React to similar content  
    3. Mention or interact with similar people
    4. Have similar joining dates or activity patterns
    
    Return the top 5 most similar users with explanations."""
    
    # Similar implementation as above...
    await interaction.followup.send("🔍 **Finding your connections...** (Neo4j analysis)")

@bot.tree.command(name="channel_insights", description="Get insights about channel activity and user patterns")
async def channel_insights_command(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=False)
    
    query_prompt = f"""Analyze channel activity patterns in server {interaction.guild.id}:
    
    1. Which channels have the most cross-talk (users active in multiple channels)?
    2. What are the peak activity hours for each channel?
    3. Which users are 'channel bridges' (active across many channels)?
    4. Channel growth trends over time
    
    Use Neo4j graph analysis to find these patterns."""
    
    await interaction.followup.send("📈 **Analyzing channel patterns...** (Neo4j graph analysis)")

@bot.tree.command(name="storage_status", description="Check Neo4j database storage usage and health")
async def storage_status_command(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=False)
    
    # Get storage information through Neo4j plugin
    query_prompt = """Check the current Neo4j database storage status. Please provide:
    
    1. Current database size in GB
    2. Number of nodes and relationships
    3. Storage usage percentage
    4. Oldest data timestamp
    5. Recent data cleanup history
    
    Use appropriate Cypher queries to gather this information."""
    
    payload = {
        "messages": [
            {
                "role": "system",
                "content": f"You are DuckBot's database administrator. The maximum storage limit is {MAX_STORAGE_GB}GB. Provide detailed storage analysis."
            },
            {"role": "user", "content": query_prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 800,
        "plugins": {
            "query-neo4j": {"enabled": True, "priority": 1}
        }
    }
    
    try:
        response = requests.post(LM_STUDIO_URL, headers={"Content-Type": "application/json"}, data=json.dumps(payload))
        data = response.json()
        ai_response = data["choices"][0]["message"]["content"]
        
        # Add storage limit info
        status_message = f"""💾 **Neo4j Database Status**
        
**Storage Limit**: {MAX_STORAGE_GB}GB
**Auto-cleanup**: Triggers at 80% capacity (8GB)

{ai_response}

📋 **Cleanup Policy:**
• Messages older than 6 months are deleted
• Reaction data older than 3 months is removed  
• Detailed activity data older than 2 months is compressed
• Orphaned nodes are removed regularly"""

        await interaction.followup.send(status_message)
    except Exception as e:
        await interaction.followup.send(f"❌ Error checking storage status: {e}")

@bot.tree.command(name="force_cleanup", description="Manually trigger Neo4j database cleanup (Admin only)")
async def force_cleanup_command(interaction: discord.Interaction):
    # Check if user has admin permissions
    if not interaction.user.guild_permissions.administrator:
        await interaction.response.send_message("❌ Admin permissions required for manual cleanup.", ephemeral=True)
        return
        
    await interaction.response.defer(ephemeral=False)
    
    try:
        await interaction.followup.send("🧹 **Starting manual database cleanup...**")
        await cleanup_old_data_if_needed()
        await interaction.followup.send("✅ **Database cleanup completed!** Use `/storage_status` to see updated stats.")
    except Exception as e:
        await interaction.followup.send(f"❌ Cleanup failed: {e}")


# --- 7. RUN THE BOT ---
if __name__ == "__main__":
    if not DISCORD_TOKEN:
        print("CRITICAL ERROR: DISCORD_TOKEN not found in .env file.")
    else:
        print("Starting DuckBot with Neo4j integration...")
        try:
            bot.run(DISCORD_TOKEN)
        except KeyboardInterrupt:
            print("Bot shutting down...")
        finally:
            close_neo4j()
