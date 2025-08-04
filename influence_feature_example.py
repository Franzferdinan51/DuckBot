# Example: User Influence Score Feature for DuckBot
# This would be added to DuckBot-v1.2VID.py

async def calculate_influence_score(user_id, server_id):
    """Calculate user influence based on Neo4j social graph data."""
    if not neo4j_driver:
        return None
    
    try:
        with neo4j_driver.session() as session:
            # Complex influence calculation using multiple metrics
            query = """
            MATCH (u:User {id: $user_id})-[:ACTIVE_IN]->(c:Channel)-[:IN_SERVER]->(s:Server {id: $server_id})
            
            // Get user's message metrics
            OPTIONAL MATCH (u)-[:SENT]->(m:Message)-[:IN_CHANNEL]->(c)
            WITH u, count(m) as message_count, sum(m.reactions_count) as total_reactions
            
            // Get mentions received (popularity indicator)
            OPTIONAL MATCH (m2:Message)-[:MENTIONS]->(u)
            WITH u, message_count, total_reactions, count(m2) as mentions_received
            
            // Get network centrality (how connected they are)
            OPTIONAL MATCH (u)-[r:REACTS_TO_USER]->(other:User)
            WITH u, message_count, total_reactions, mentions_received, 
                 count(DISTINCT other) as unique_connections
            
            // Calculate weighted influence score
            WITH u,
                 (message_count * 1.0) as message_score,
                 (total_reactions * 2.0) as reaction_score, 
                 (mentions_received * 3.0) as mention_score,
                 (unique_connections * 1.5) as network_score
            
            RETURN u.username,
                   message_score + reaction_score + mention_score + network_score as influence_score,
                   message_count, total_reactions, mentions_received, unique_connections
            """
            
            result = session.run(query, user_id=user_id, server_id=server_id)
            record = result.single()
            
            if record:
                return {
                    "username": record["username"],
                    "influence_score": round(record["influence_score"], 2),
                    "breakdown": {
                        "messages": record["message_count"],
                        "reactions_received": record["total_reactions"], 
                        "mentions_received": record["mentions_received"],
                        "unique_connections": record["unique_connections"]
                    }
                }
            return None
            
    except Exception as e:
        print(f"❌ Error calculating influence: {e}")
        return None

@bot.tree.command(name="influence_score", description="Calculate a user's social influence score")
@app_commands.describe(user="User to analyze (optional, defaults to you)")
async def influence_score_command(interaction: discord.Interaction, user: discord.Member = None):
    await interaction.response.defer(ephemeral=False)
    
    target_user = user or interaction.user
    
    if not NEO4J_ENABLED:
        await interaction.followup.send("❌ Neo4j analytics not enabled. Set NEO4J_ENABLED=true in .env")
        return
    
    result = await calculate_influence_score(target_user.id, interaction.guild.id)
    
    if not result:
        await interaction.followup.send(f"❌ No data found for {target_user.mention}")
        return
    
    # Create fancy embed response
    embed = discord.Embed(
        title=f"🎭 Social Influence Score",
        description=f"**{result['username']}**",
        color=0x00ff88
    )
    
    embed.add_field(
        name="📊 Overall Influence", 
        value=f"**{result['influence_score']}** points",
        inline=False
    )
    
    breakdown = result['breakdown']
    embed.add_field(name="💬 Messages", value=breakdown['messages'], inline=True)
    embed.add_field(name="⭐ Reactions Received", value=breakdown['reactions_received'], inline=True) 
    embed.add_field(name="📢 Mentions Received", value=breakdown['mentions_received'], inline=True)
    embed.add_field(name="🔗 Unique Connections", value=breakdown['unique_connections'], inline=True)
    
    # Add influence level
    score = result['influence_score']
    if score >= 1000:
        level = "🌟 Community Legend"
    elif score >= 500:
        level = "👑 High Influencer"
    elif score >= 200:
        level = "🎯 Active Member"
    elif score >= 50:
        level = "🌱 Growing Presence"
    else:
        level = "👋 New Member"
    
    embed.add_field(name="🏆 Influence Level", value=level, inline=False)
    
    await interaction.followup.send(embed=embed)

# Additional cool queries you could add:

async def find_trending_topics(server_id, days=7):
    """Find trending discussion topics in the last N days."""
    query = """
    MATCH (m:Message)-[:IN_CHANNEL]->(c:Channel)-[:IN_SERVER]->(s:Server {id: $server_id})
    WHERE m.timestamp >= datetime() - duration({days: $days})
    
    // Extract keywords and find co-occurrence patterns
    WITH m, split(toLower(m.content), ' ') as words
    UNWIND words as word
    WITH word
    WHERE size(word) > 3 AND NOT word IN ['the', 'and', 'but', 'for', 'with']
    
    WITH word, count(*) as frequency
    WHERE frequency >= 5
    ORDER BY frequency DESC
    LIMIT 20
    
    RETURN word, frequency
    """
    # Implementation here...

async def detect_communities(server_id):
    """Find natural friend groups/communities in the server."""
    query = """
    // Use Graph Data Science library for community detection
    CALL gds.graph.project(
        'user-network',
        'User', 
        'REACTS_TO_USER',
        {relationshipProperties: 'count'}
    )
    
    CALL gds.louvain.stream('user-network')
    YIELD nodeId, communityId
    
    RETURN gds.util.asNode(nodeId).username as username, communityId
    ORDER BY communityId, username
    """
    # Implementation here...