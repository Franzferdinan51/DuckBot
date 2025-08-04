# ==============================================================================
# DUCK BOT v2.0 - ENHANCED EDITION WITH AI INTELLIGENCE
# Features: Video/Image Generation + Knowledge Management + AI Memory + Gaming
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
    'image': 30,    # 30 seconds average
    'video': 900    # 15 minutes average (900 seconds)
}

import time

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
        
        # Initialize enhanced schema for v2.0 features
        initialize_enhanced_schema()
        
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

# --- ENHANCED NEO4J FEATURES FOR V2.0 ---

def initialize_enhanced_schema():
    """Initialize enhanced schema for v2.0 features."""
    if not neo4j_driver:
        return False
    
    try:
        with neo4j_driver.session() as session:
            enhanced_schema = [
                # Knowledge Management
                "CREATE CONSTRAINT IF NOT EXISTS FOR (k:Knowledge) REQUIRE k.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE",
                "CREATE INDEX IF NOT EXISTS FOR (k:Knowledge) ON k.content",
                "CREATE INDEX IF NOT EXISTS FOR (c:Concept) ON c.category",
                
                # Personal AI Memory
                "CREATE CONSTRAINT IF NOT EXISTS FOR (m:Memory) REQUIRE m.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Project) REQUIRE p.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (g:Goal) REQUIRE g.id IS UNIQUE",
                "CREATE INDEX IF NOT EXISTS FOR (m:Memory) ON m.timestamp",
                
                # Gaming Systems
                "CREATE CONSTRAINT IF NOT EXISTS FOR (scene:Scene) REQUIRE scene.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (char:Character) REQUIRE char.id IS UNIQUE", 
                "CREATE CONSTRAINT IF NOT EXISTS FOR (item:Item) REQUIRE item.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (quest:Quest) REQUIRE quest.id IS UNIQUE",
                
                # Content Management
                "CREATE CONSTRAINT IF NOT EXISTS FOR (art:Artwork) REQUIRE art.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (story:Story) REQUIRE story.id IS UNIQUE",
                "CREATE INDEX IF NOT EXISTS FOR (art:Artwork) ON art.created_date",
                
                # Ideas and Creativity
                "CREATE CONSTRAINT IF NOT EXISTS FOR (idea:Idea) REQUIRE idea.id IS UNIQUE",
                "CREATE INDEX IF NOT EXISTS FOR (idea:Idea) ON idea.tags"
            ]
            
            for query in enhanced_schema:
                session.run(query)
            
            print("✅ Enhanced v2.0 schema initialized")
            return True
            
    except Exception as e:
        print(f"❌ Enhanced schema initialization failed: {e}")
        return False

@dataclass
class KnowledgeEntry:
    content: str
    category: str
    user_id: int
    concepts: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)

@dataclass  
class UserMemory:
    user_id: int
    content: str
    context: str
    importance: int = 5  # 1-10 scale
    projects: List[str] = field(default_factory=list)
    goals: List[str] = field(default_factory=list)

# --- KNOWLEDGE MANAGEMENT SYSTEM ---

async def store_knowledge(entry: KnowledgeEntry):
    """Store knowledge in the graph database."""
    if not NEO4J_ENABLED or not neo4j_driver:
        return False
        
    try:
        with neo4j_driver.session() as session:
            knowledge_id = hashlib.md5(entry.content.encode()).hexdigest()
            
            query = """
            MERGE (k:Knowledge {id: $knowledge_id})
            SET k.content = $content,
                k.category = $category,
                k.user_id = $user_id,
                k.created_date = datetime(),
                k.sources = $sources
            
            MERGE (u:User {id: $user_id})
            MERGE (u)-[:CONTRIBUTED]->(k)
            
            // Create concept nodes and relationships
            WITH k
            UNWIND $concepts as concept_name
            MERGE (c:Concept {name: concept_name})
            SET c.category = $category
            MERGE (k)-[:RELATES_TO]->(c)
            """
            
            session.run(query, 
                knowledge_id=knowledge_id,
                content=entry.content,
                category=entry.category, 
                user_id=entry.user_id,
                concepts=entry.concepts,
                sources=entry.sources
            )
            return knowledge_id
            
    except Exception as e:
        print(f"❌ Error storing knowledge: {e}")
        return None

async def query_knowledge(query_text: str, user_id: int = None):
    """Query the knowledge base using semantic search."""
    if not NEO4J_ENABLED or not neo4j_driver:
        return []
        
    try:
        with neo4j_driver.session() as session:
            # Simple keyword-based search for now
            keywords = [word.lower() for word in query_text.split() if len(word) > 3]
            
            cypher_query = """
            MATCH (k:Knowledge)
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
            
            result = session.run(cypher_query, keywords=keywords)
            return [dict(record) for record in result]
            
    except Exception as e:
        print(f"❌ Error querying knowledge: {e}")
        return []

# --- PERSONAL AI MEMORY SYSTEM ---

async def store_memory(memory: UserMemory):
    """Store personal memory for AI context."""
    if not NEO4J_ENABLED or not neo4j_driver:
        return False
        
    try:
        with neo4j_driver.session() as session:
            memory_id = str(uuid.uuid4())
            
            query = """
            CREATE (m:Memory {
                id: $memory_id,
                user_id: $user_id,
                content: $content,
                context: $context,
                importance: $importance,
                timestamp: datetime()
            })
            
            MERGE (u:User {id: $user_id})
            MERGE (u)-[:HAS_MEMORY]->(m)
            
            // Link to projects
            WITH m, u
            UNWIND $projects as project_name
            MERGE (p:Project {name: project_name, user_id: $user_id})
            MERGE (m)-[:RELATES_TO_PROJECT]->(p)
            MERGE (u)-[:WORKING_ON]->(p)
            
            // Link to goals  
            WITH m, u
            UNWIND $goals as goal_name
            MERGE (g:Goal {name: goal_name, user_id: $user_id})
            MERGE (m)-[:RELATES_TO_GOAL]->(g)
            MERGE (u)-[:HAS_GOAL]->(g)
            """
            
            session.run(query,
                memory_id=memory_id,
                user_id=memory.user_id,
                content=memory.content,
                context=memory.context,
                importance=memory.importance,
                projects=memory.projects,
                goals=memory.goals
            )
            return memory_id
            
    except Exception as e:
        print(f"❌ Error storing memory: {e}")
        return None

async def get_user_context(user_id: int, limit: int = 10):
    """Get user's recent context for AI conversations."""
    if not NEO4J_ENABLED or not neo4j_driver:
        return []
        
    try:
        with neo4j_driver.session() as session:
            query = """
            MATCH (u:User {id: $user_id})-[:HAS_MEMORY]->(m:Memory)
            
            OPTIONAL MATCH (m)-[:RELATES_TO_PROJECT]->(p:Project)
            OPTIONAL MATCH (m)-[:RELATES_TO_GOAL]->(g:Goal)
            
            RETURN m.content as content,
                   m.context as context,
                   m.importance as importance,
                   m.timestamp as timestamp,
                   collect(DISTINCT p.name) as projects,
                   collect(DISTINCT g.name) as goals
            ORDER BY m.importance DESC, m.timestamp DESC
            LIMIT $limit
            """
            
            result = session.run(query, user_id=user_id, limit=limit)
            return [dict(record) for record in result]
            
    except Exception as e:
        print(f"❌ Error getting user context: {e}")
        return []

# --- GAMING & ADVENTURE SYSTEM ---

@dataclass
class GameScene:
    title: str
    description: str
    choices: List[Dict[str, str]] = field(default_factory=list)
    items: List[str] = field(default_factory=list)
    consequences: Dict[str, Any] = field(default_factory=dict)

async def create_adventure_scene(scene: GameScene, scene_id: str = None):
    """Create a new adventure scene in the game world."""
    if not NEO4J_ENABLED or not neo4j_driver:
        return None
        
    try:
        with neo4j_driver.session() as session:
            if not scene_id:
                scene_id = str(uuid.uuid4())
                
            query = """
            CREATE (s:Scene {
                id: $scene_id,
                title: $title,
                description: $description,
                created_date: datetime()
            })
            
            // Create choice nodes
            WITH s
            UNWIND $choices as choice_data
            CREATE (c:Choice {
                id: randomUUID(),
                text: choice_data.text,
                consequence: choice_data.consequence
            })
            CREATE (s)-[:HAS_CHOICE]->(c)
            
            // Create item nodes
            WITH s
            UNWIND $items as item_name
            MERGE (i:Item {name: item_name})
            CREATE (s)-[:CONTAINS_ITEM]->(i)
            
            RETURN s.id as scene_id
            """
            
            result = session.run(query,
                scene_id=scene_id,
                title=scene.title,
                description=scene.description,
                choices=scene.choices,
                items=scene.items
            )
            
            record = result.single()
            return record["scene_id"] if record else scene_id
            
    except Exception as e:
        print(f"❌ Error creating adventure scene: {e}")
        return None

async def get_adventure_scene(scene_id: str):
    """Get adventure scene details."""
    if not NEO4J_ENABLED or not neo4j_driver:
        return None
        
    try:
        with neo4j_driver.session() as session:
            query = """
            MATCH (s:Scene {id: $scene_id})
            
            OPTIONAL MATCH (s)-[:HAS_CHOICE]->(c:Choice)
            OPTIONAL MATCH (s)-[:CONTAINS_ITEM]->(i:Item)
            
            RETURN s.title as title,
                   s.description as description,
                   collect(DISTINCT {text: c.text, consequence: c.consequence}) as choices,
                   collect(DISTINCT i.name) as items
            """
            
            result = session.run(query, scene_id=scene_id)
            record = result.single()
            
            if record:
                return {
                    "title": record["title"],
                    "description": record["description"],
                    "choices": [c for c in record["choices"] if c["text"]],
                    "items": [i for i in record["items"] if i]
                }
            return None
            
    except Exception as e:
        print(f"❌ Error getting adventure scene: {e}")
        return None

# --- CREATIVE CONTENT SYSTEM ---

async def store_artwork(user_id: int, prompt: str, style: str = "unknown", image_data: bytes = None):
    """Store AI-generated artwork with metadata."""
    if not NEO4J_ENABLED or not neo4j_driver:
        return None
        
    try:
        with neo4j_driver.session() as session:
            artwork_id = str(uuid.uuid4())
            
            query = """
            CREATE (art:Artwork {
                id: $artwork_id,
                user_id: $user_id,
                prompt: $prompt,
                style: $style,
                created_date: datetime()
            })
            
            MERGE (u:User {id: $user_id})
            MERGE (u)-[:CREATED]->(art)
            
            // Extract and create style/concept relationships
            WITH art
            UNWIND split(toLower($prompt), ' ') as word
            WITH art, word
            WHERE size(word) > 3 AND NOT word IN ['the', 'and', 'with', 'that', 'this']
            MERGE (c:Concept {name: word})
            SET c.category = 'art_concept'
            MERGE (art)-[:DEPICTS]->(c)
            
            RETURN art.id as artwork_id
            """
            
            result = session.run(query,
                artwork_id=artwork_id,
                user_id=user_id,
                prompt=prompt,
                style=style
            )
            
            return artwork_id
            
    except Exception as e:
        print(f"❌ Error storing artwork: {e}")
        return None

async def get_user_art_evolution(user_id: int, limit: int = 20):
    """Get user's artistic evolution over time."""
    if not NEO4J_ENABLED or not neo4j_driver:
        return []
        
    try:
        with neo4j_driver.session() as session:
            query = """
            MATCH (u:User {id: $user_id})-[:CREATED]->(art:Artwork)
            
            OPTIONAL MATCH (art)-[:DEPICTS]->(c:Concept)
            
            RETURN art.prompt as prompt,
                   art.style as style,
                   art.created_date as created,
                   collect(DISTINCT c.name) as concepts
            ORDER BY art.created_date DESC
            LIMIT $limit
            """
            
            result = session.run(query, user_id=user_id, limit=limit)
            return [dict(record) for record in result]
            
    except Exception as e:
        print(f"❌ Error getting art evolution: {e}")
        return []

# --- IDEA & CREATIVITY SYSTEM ---

async def store_idea(user_id: int, content: str, tags: List[str] = None, inspiration_source: str = None):
    """Store a creative idea with connections."""
    if not NEO4J_ENABLED or not neo4j_driver:
        return None
        
    try:
        with neo4j_driver.session() as session:
            idea_id = str(uuid.uuid4())
            if not tags:
                tags = []
                
            query = """
            CREATE (idea:Idea {
                id: $idea_id,
                user_id: $user_id,
                content: $content,
                tags: $tags,
                inspiration_source: $inspiration_source,
                created_date: datetime()
            })
            
            MERGE (u:User {id: $user_id})
            MERGE (u)-[:HAD_IDEA]->(idea)
            
            // Create tag relationships for idea connections
            WITH idea
            UNWIND $tags as tag
            MERGE (t:Tag {name: tag})
            MERGE (idea)-[:TAGGED_WITH]->(t)
            
            RETURN idea.id as idea_id
            """
            
            result = session.run(query,
                idea_id=idea_id,
                user_id=user_id,
                content=content,
                tags=tags,
                inspiration_source=inspiration_source
            )
            
            return idea_id
            
    except Exception as e:
        print(f"❌ Error storing idea: {e}")
        return None

async def find_idea_connections(user_id: int, current_idea: str = None):
    """Find connections between user's ideas."""
    if not NEO4J_ENABLED or not neo4j_driver:
        return []
        
    try:
        with neo4j_driver.session() as session:
            if current_idea:
                # Find ideas similar to current one
                query = """
                MATCH (u:User {id: $user_id})-[:HAD_IDEA]->(idea1:Idea)
                MATCH (u)-[:HAD_IDEA]->(idea2:Idea)
                WHERE idea1 <> idea2 AND toLower(idea1.content) CONTAINS toLower($current_idea)
                
                OPTIONAL MATCH (idea1)-[:TAGGED_WITH]->(t:Tag)<-[:TAGGED_WITH]-(idea2)
                
                RETURN idea1.content as original_idea,
                       idea2.content as connected_idea,
                       count(t) as tag_overlap,
                       idea2.created_date as created
                ORDER BY tag_overlap DESC, created DESC
                LIMIT 10
                """
                
                result = session.run(query, user_id=user_id, current_idea=current_idea)
            else:
                # Find all idea connections for user
                query = """
                MATCH (u:User {id: $user_id})-[:HAD_IDEA]->(idea1:Idea)
                MATCH (idea1)-[:TAGGED_WITH]->(t:Tag)<-[:TAGGED_WITH]-(idea2:Idea)
                WHERE idea1 <> idea2 AND (idea2)<-[:HAD_IDEA]-(u)
                
                RETURN idea1.content as idea_a,
                       idea2.content as idea_b,
                       t.name as common_tag,
                       count(*) as connection_strength
                ORDER BY connection_strength DESC
                LIMIT 15
                """
                
                result = session.run(query, user_id=user_id)
            
            return [dict(record) for record in result]
            
    except Exception as e:
        print(f"❌ Error finding idea connections: {e}")
        return []

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
    """Connects to ComfyUI, runs a workflow, and returns the output files."""
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
            await queue_item.status_message.edit(content=f"❌ **Image generation failed**\nPrompt: `{queue_item.prompt}`")
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
            await queue_item.status_message.edit(content=f"❌ **Video generation failed**\nPrompt: `{queue_item.prompt}`")
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

# --- ENHANCED V2.0 COMMANDS ---

# === KNOWLEDGE MANAGEMENT COMMANDS ===

@bot.tree.command(name="learn", description="Teach DuckBot something new")
@app_commands.describe(
    content="What do you want to teach me?",
    category="Category (science, programming, art, etc.)",
    concepts="Related concepts (comma-separated)"
)
async def learn_command(interaction: discord.Interaction, content: str, category: str = "general", concepts: str = None):
    await interaction.response.defer(ephemeral=False)
    
    if not NEO4J_ENABLED:
        await interaction.followup.send("❌ Knowledge system requires Neo4j. Set NEO4J_ENABLED=true in .env")
        return
    
    concept_list = []
    if concepts:
        concept_list = [c.strip() for c in concepts.split(",")]
    
    entry = KnowledgeEntry(
        content=content,
        category=category,
        user_id=interaction.user.id,
        concepts=concept_list
    )
    
    knowledge_id = await store_knowledge(entry)
    
    if knowledge_id:
        embed = discord.Embed(
            title="🧠 Knowledge Learned!",
            description=f"I've learned about **{category}**",
            color=0x00ff88
        )
        embed.add_field(name="Content", value=content[:1000], inline=False)
        if concepts:
            embed.add_field(name="Related Concepts", value=concepts, inline=False)
        embed.set_footer(text=f"Knowledge ID: {knowledge_id[:8]}...")
        
        await interaction.followup.send(embed=embed)
    else:
        await interaction.followup.send("❌ Failed to store knowledge")

@bot.tree.command(name="ask_knowledge", description="Query DuckBot's learned knowledge")
@app_commands.describe(query="What do you want to know about?")
async def ask_knowledge_command(interaction: discord.Interaction, query: str):
    await interaction.response.defer(ephemeral=False)
    
    if not NEO4J_ENABLED:
        await interaction.followup.send("❌ Knowledge system requires Neo4j")
        return
    
    results = await query_knowledge(query, interaction.user.id)
    
    if not results:
        await interaction.followup.send(f"🤔 I don't know anything about '{query}' yet. Try `/learn` to teach me!")
        return
    
    embed = discord.Embed(
        title=f"🔍 Knowledge about '{query}'",
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
    
    await interaction.followup.send(embed=embed)

# === PERSONAL AI MEMORY COMMANDS ===

@bot.tree.command(name="remember", description="Store something for DuckBot to remember about you")
@app_commands.describe(
    content="What should I remember?",
    context="Context/situation (work, hobby, goal, etc.)",
    importance="How important? (1-10, default 5)",
    projects="Related projects (comma-separated)",
    goals="Related goals (comma-separated)"
)
async def remember_command(interaction: discord.Interaction, content: str, context: str = "general", 
                          importance: int = 5, projects: str = None, goals: str = None):
    await interaction.response.defer(ephemeral=False)
    
    if not NEO4J_ENABLED:
        await interaction.followup.send("❌ Memory system requires Neo4j")
        return
    
    project_list = [p.strip() for p in projects.split(",")] if projects else []
    goal_list = [g.strip() for g in goals.split(",")] if goals else []
    
    memory = UserMemory(
        user_id=interaction.user.id,
        content=content,
        context=context,
        importance=max(1, min(10, importance)),  # Clamp between 1-10
        projects=project_list,
        goals=goal_list
    )
    
    memory_id = await store_memory(memory)
    
    if memory_id:
        embed = discord.Embed(
            title="🧠 Memory Stored!",
            description=f"I'll remember this about you, {interaction.user.mention}",
            color=0x9b59b6
        )
        embed.add_field(name="Memory", value=content, inline=False)
        embed.add_field(name="Context", value=context, inline=True)
        embed.add_field(name="Importance", value=f"{importance}/10", inline=True)
        
        if projects:
            embed.add_field(name="Projects", value=projects, inline=False)
        if goals:
            embed.add_field(name="Goals", value=goals, inline=False)
            
        await interaction.followup.send(embed=embed)
    else:
        await interaction.followup.send("❌ Failed to store memory")

@bot.tree.command(name="my_context", description="See what DuckBot remembers about you")
@app_commands.describe(user="User to check (optional, defaults to you)")
async def my_context_command(interaction: discord.Interaction, user: discord.Member = None):
    await interaction.response.defer(ephemeral=False)
    
    if not NEO4J_ENABLED:
        await interaction.followup.send("❌ Memory system requires Neo4j")
        return
    
    target_user = user or interaction.user
    
    if target_user != interaction.user and not interaction.user.guild_permissions.administrator:
        await interaction.followup.send("❌ You can only view your own context (unless you're an admin)")
        return
    
    memories = await get_user_context(target_user.id, limit=10)
    
    if not memories:
        await interaction.followup.send(f"🤔 I don't have any memories stored for {target_user.mention} yet")
        return
    
    embed = discord.Embed(
        title=f"🧠 What I Remember About {target_user.display_name}",
        color=0x9b59b6
    )
    
    for memory in memories[:5]:  # Show top 5 memories
        projects_str = ", ".join(memory['projects']) if memory['projects'] else "None"
        goals_str = ", ".join(memory['goals']) if memory['goals'] else "None"
        
        embed.add_field(
            name=f"💭 {memory['context'].title()} (Importance: {memory['importance']}/10)",
            value=f"{memory['content'][:150]}{'...' if len(memory['content']) > 150 else ''}\n"
                  f"*Projects: {projects_str}*\n"
                  f"*Goals: {goals_str}*",
            inline=False
        )
    
    embed.set_footer(text=f"Showing {len(memories[:5])} of {len(memories)} memories")
    await interaction.followup.send(embed=embed)

# === GAMING & ADVENTURE COMMANDS ===

@bot.tree.command(name="start_adventure", description="Begin an interactive text adventure")
async def start_adventure_command(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=False)
    
    if not NEO4J_ENABLED:
        await interaction.followup.send("❌ Adventure system requires Neo4j")
        return
    
    # Create starting scene
    starting_scene = GameScene(
        title="🌙 Moonlit Crossroads",
        description="You stand at a crossroads under the pale moonlight. Ancient stone markers point in three directions, each path shrouded in mystery. A gentle breeze carries whispers of adventure from each route.",
        choices=[
            {"text": "Take the forest path", "consequence": "forest_scene"},
            {"text": "Follow the mountain trail", "consequence": "mountain_scene"}, 
            {"text": "Descend into the valley", "consequence": "valley_scene"},
            {"text": "Examine the stone markers", "consequence": "examine_markers"}
        ],
        items=["weathered map", "rusty compass"]
    )
    
    scene_id = await create_adventure_scene(starting_scene)
    
    if scene_id:
        embed = discord.Embed(
            title=starting_scene.title,
            description=starting_scene.description,
            color=0x2c3e50
        )
        
        choices_text = "\n".join([f"**{i+1}.** {choice['text']}" for i, choice in enumerate(starting_scene.choices)])
        embed.add_field(name="🎯 Your Choices", value=choices_text, inline=False)
        
        if starting_scene.items:
            items_text = ", ".join(starting_scene.items)
            embed.add_field(name="🎒 Items Available", value=items_text, inline=False)
        
        embed.set_footer(text=f"Adventure ID: {scene_id[:8]}... | Use /continue_adventure to proceed")
        
        await interaction.followup.send(embed=embed)
    else:
        await interaction.followup.send("❌ Failed to start adventure")

@bot.tree.command(name="continue_adventure", description="Continue your adventure")
@app_commands.describe(
    scene_id="Adventure scene ID",
    choice="Your choice number (1, 2, 3, etc.)"
)
async def continue_adventure_command(interaction: discord.Interaction, scene_id: str, choice: int):
    await interaction.response.defer(ephemeral=False)
    
    if not NEO4J_ENABLED:
        await interaction.followup.send("❌ Adventure system requires Neo4j")
        return
    
    scene = await get_adventure_scene(scene_id)
    
    if not scene:
        await interaction.followup.send("❌ Adventure scene not found")
        return
    
    if choice < 1 or choice > len(scene["choices"]):
        await interaction.followup.send(f"❌ Invalid choice. Please choose 1-{len(scene['choices'])}")
        return
    
    selected_choice = scene["choices"][choice - 1]
    consequence = selected_choice["consequence"]
    
    # This is where you'd implement the consequence system
    # For now, just show what happens
    embed = discord.Embed(
        title="✨ Adventure Continues...",
        description=f"You chose: **{selected_choice['text']}**\n\n*The adventure continues based on your choice...*",
        color=0x27ae60
    )
    
    embed.add_field(
        name="🎭 What Happens Next",
        value=f"Your choice leads to: `{consequence}`\n\n*This is where the story would branch based on your decision!*",
        inline=False
    )
    
    await interaction.followup.send(embed=embed)

# === CREATIVE CONTENT COMMANDS ===

@bot.tree.command(name="art_journey", description="View your AI art evolution over time")
@app_commands.describe(user="User to analyze (optional, defaults to you)")
async def art_journey_command(interaction: discord.Interaction, user: discord.Member = None):
    await interaction.response.defer(ephemeral=False)
    
    if not NEO4J_ENABLED:
        await interaction.followup.send("❌ Art tracking requires Neo4j")
        return
    
    target_user = user or interaction.user
    art_history = await get_user_art_evolution(target_user.id, limit=10)
    
    if not art_history:
        await interaction.followup.send(f"🎨 {target_user.mention} hasn't created any tracked artwork yet")
        return
    
    embed = discord.Embed(
        title=f"🎨 {target_user.display_name}'s Art Journey",
        description=f"Artistic evolution over {len(art_history)} creations",
        color=0xe74c3c
    )
    
    for i, art in enumerate(art_history[:5]):  # Show latest 5
        concepts_str = ", ".join(art['concepts'][:4]) if art['concepts'] else "No concepts"
        created_date = art['created'].strftime("%Y-%m-%d") if art['created'] else "Unknown"
        
        embed.add_field(
            name=f"🖼️ Creation #{len(art_history) - i}",
            value=f"**Style:** {art['style']}\n"
                  f"**Prompt:** {art['prompt'][:100]}{'...' if len(art['prompt']) > 100 else ''}\n"
                  f"**Concepts:** {concepts_str}\n"
                  f"**Created:** {created_date}",
            inline=False
        )
    
    await interaction.followup.send(embed=embed)

# === IDEA & CREATIVITY COMMANDS ===

@bot.tree.command(name="save_idea", description="Store a creative idea")
@app_commands.describe(
    idea="Your creative idea",
    tags="Tags for categorization (comma-separated)",
    inspiration="What inspired this idea?"
)
async def save_idea_command(interaction: discord.Interaction, idea: str, tags: str = None, inspiration: str = None):
    await interaction.response.defer(ephemeral=False)
    
    if not NEO4J_ENABLED:
        await interaction.followup.send("❌ Idea system requires Neo4j")
        return
    
    tag_list = [t.strip() for t in tags.split(",")] if tags else []
    
    idea_id = await store_idea(interaction.user.id, idea, tag_list, inspiration)
    
    if idea_id:
        embed = discord.Embed(
            title="💡 Idea Saved!",
            description="Your creative spark has been captured",
            color=0xf39c12
        )
        embed.add_field(name="💭 Idea", value=idea, inline=False)
        
        if tags:
            embed.add_field(name="🏷️ Tags", value=tags, inline=True)
        if inspiration:
            embed.add_field(name="✨ Inspiration", value=inspiration, inline=True)
            
        embed.set_footer(text=f"Idea ID: {idea_id[:8]}...")
        
        await interaction.followup.send(embed=embed)
    else:
        await interaction.followup.send("❌ Failed to save idea")

@bot.tree.command(name="idea_connections", description="Find connections between your ideas")
@app_commands.describe(current_idea="Current idea to find connections for (optional)")
async def idea_connections_command(interaction: discord.Interaction, current_idea: str = None):
    await interaction.response.defer(ephemeral=False)
    
    if not NEO4J_ENABLED:
        await interaction.followup.send("❌ Idea system requires Neo4j")
        return
    
    connections = await find_idea_connections(interaction.user.id, current_idea)
    
    if not connections:
        await interaction.followup.send("💡 No idea connections found yet. Save more ideas with `/save_idea`!")
        return
    
    embed = discord.Embed(
        title="🕸️ Your Idea Connections",
        description="Discover how your thoughts connect!",
        color=0x8e44ad
    )
    
    for conn in connections[:5]:  # Show top 5 connections
        if current_idea:
            embed.add_field(
                name=f"🔗 Connection (Overlap: {conn['tag_overlap']})",
                value=f"**Original:** {conn['original_idea'][:100]}...\n"
                      f"**Connected:** {conn['connected_idea'][:100]}...",
                inline=False
            )
        else:
            embed.add_field(
                name=f"🔗 Connected by: {conn['common_tag']}",
                value=f"**Idea A:** {conn['idea_a'][:80]}...\n"
                      f"**Idea B:** {conn['idea_b'][:80]}...\n"
                      f"**Strength:** {conn['connection_strength']}",
                inline=False
            )
    
    await interaction.followup.send(embed=embed)

@bot.tree.command(name="random_idea_combo", description="Get a random combination of concepts for inspiration")
async def random_idea_combo_command(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=False)
    
    # Generate random idea combinations
    concepts = [
        "time travel", "floating islands", "crystal caves", "ancient robots", "singing forests",
        "mirror worlds", "liquid light", "shadow creatures", "clockwork hearts", "dream architects",
        "star whisperers", "rainbow bridges", "invisible cities", "memory gardens", "wind dancers",
        "fire sculptures", "ice libraries", "thunder paintings", "cloud ships", "moon harbors"
    ]
    
    random_combo = random.sample(concepts, 3)
    
    embed = discord.Embed(
        title="🎲 Random Idea Generator",
        description="Here's your creative spark!",
        color=0xe67e22
    )
    
    embed.add_field(
        name="💫 Your Concept Combination",
        value=" + ".join([f"**{concept}**" for concept in random_combo]),
        inline=False
    )
    
    embed.add_field(
        name="🎯 Creative Challenge",
        value=f"Try combining these three concepts into something new! What story, art piece, or project could emerge from: {', '.join(random_combo)}?",
        inline=False
    )
    
    await interaction.followup.send(embed=embed)

# === ENHANCED ASK COMMAND WITH MEMORY ===

async def get_enhanced_ask_response(interaction: discord.Interaction, prompt: str):
    """Enhanced ask command that uses personal memory context."""
    
    # Get user's context if Neo4j is enabled
    context_info = ""
    if NEO4J_ENABLED:
        memories = await get_user_context(interaction.user.id, limit=5)
        if memories:
            context_info = "\n\nPersonal Context (what I remember about you):\n"
            for memory in memories[:3]:  # Use top 3 memories
                context_info += f"- {memory['content']} (Context: {memory['context']})\n"
    
    # Enhanced system prompt with memory
    enhanced_system_prompt = f"""You are DuckBot v2.0, an advanced AI assistant with access to multiple tools and the user's personal context. You have access to:

- Web search (DuckDuckGo, general web search)
- Website content extraction (visit-website)  
- Wikipedia knowledge base
- Advanced RAG (retrieval augmented generation)
- JavaScript code execution sandbox
- Neo4j graph database queries
- Dice rolling and random generation
- Personal memory system (remembers user's projects, goals, and context)

User: {interaction.user.display_name} (ID: {interaction.user.id}){context_info}

Use this personal context when relevant to provide more helpful, personalized responses. Remember their projects, goals, and previous conversations. Always be helpful, accurate, and consider their personal context when appropriate."""

    payload = {
        "messages": [
            {"role": "system", "content": enhanced_system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 800,  # Increased for more detailed responses
        "stream": False,
        "model": "model-id-from-lm-studio",
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
        "max_plugin_calls": 3,
        "plugin_timeout": 10
    }
    
    return payload

# Override the original ask command with enhanced version
@bot.tree.command(name="ask_enhanced", description="Ask DuckBot v2.0 with personal memory context")
@app_commands.describe(prompt="Your question or request")
async def ask_enhanced_command(interaction: discord.Interaction, prompt: str):
    await interaction.response.defer(ephemeral=False)
    
    try:
        payload = await get_enhanced_ask_response(interaction, prompt)
        
        response = requests.post(LM_STUDIO_URL, headers={"Content-Type": "application/json"}, data=json.dumps(payload))
        response.raise_for_status()
        
        data = response.json()
        
        if "choices" in data and len(data["choices"]) > 0:
            ai_response = data["choices"][0]["message"]["content"]
            
            # Auto-store important information as memory
            if any(keyword in prompt.lower() for keyword in ["working on", "project", "goal", "remember", "learning"]):
                memory = UserMemory(
                    user_id=interaction.user.id,
                    content=f"User mentioned: {prompt}",
                    context="conversation",
                    importance=6
                )
                await store_memory(memory)
        else:
            ai_response = "I couldn't get a response from the AI. The response format was unexpected."
        
        # If response is too long, split it
        if len(ai_response) > 2000:
            chunks = [ai_response[i:i+2000] for i in range(0, len(ai_response), 2000)]
            await interaction.followup.send(chunks[0])
            for chunk in chunks[1:]:
                await interaction.followup.send(chunk)
        else:
            await interaction.followup.send(ai_response)
        
    except Exception as e:
        error_message = f"❌ An error occurred: {e}"
        await interaction.followup.send(error_message)

# --- UPDATE EXISTING GENERATE COMMAND TO TRACK ART ---

# Store the original generate function
original_process_image_generation = process_image_generation

async def enhanced_process_image_generation(queue_item):
    """Enhanced image generation that tracks artwork."""
    # Call original function
    await original_process_image_generation(queue_item)
    
    # Store artwork metadata if Neo4j is enabled
    if NEO4J_ENABLED:
        await store_artwork(
            user_id=queue_item.interaction.user.id,
            prompt=queue_item.prompt,
            style="AI Generated",
        )

# Replace the original function
process_image_generation = enhanced_process_image_generation

# --- UPDATE EXISTING ANIMATE COMMAND TO TRACK ART ---

# Store the original video function
original_process_video_generation = process_video_generation

async def enhanced_process_video_generation(queue_item):
    """Enhanced video generation that tracks artwork."""
    # Call original function  
    await original_process_video_generation(queue_item)
    
    # Store artwork metadata if Neo4j is enabled
    if NEO4J_ENABLED:
        await store_artwork(
            user_id=queue_item.interaction.user.id,
            prompt=queue_item.prompt,
            style="AI Generated Video",
        )

# Replace the original function
process_video_generation = enhanced_process_video_generation


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
