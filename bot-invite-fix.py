# Discord Bot Invite URL Generator and Troubleshooter
# Run this to generate proper invite URLs and check bot configuration

import discord
import os
from urllib.parse import urlencode

def generate_invite_url(client_id, permissions=None):
    """Generate Discord bot invite URL with proper permissions."""
    
    # Required permissions for DuckBot
    required_permissions = [
        'send_messages',        # Send messages
        'use_slash_commands',   # Use application commands  
        'embed_links',          # Embed links
        'attach_files',         # Send files (for images/videos)
        'read_message_history', # Read message history
        'add_reactions',        # Add reactions
        'use_external_emojis',  # Use external emojis
        'manage_messages',      # Manage messages (for cleanup)
        'read_messages',        # Read messages
        'view_channel'          # View channels
    ]
    
    # Calculate permission integer
    permission_value = 0
    permission_map = {
        'view_channel': 1024,
        'send_messages': 2048, 
        'embed_links': 16384,
        'attach_files': 32768,
        'read_message_history': 65536,
        'use_external_emojis': 262144,
        'add_reactions': 64,
        'use_slash_commands': 2147483648,
        'manage_messages': 8192,
        'read_messages': 1024  # Same as view_channel
    }
    
    for perm in required_permissions:
        if perm in permission_map:
            permission_value |= permission_map[perm]
    
    # Alternative: Use a safe permission set (Administrator for simplicity)
    admin_permissions = 8  # Administrator permission
    
    base_url = "https://discord.com/api/oauth2/authorize"
    params = {
        'client_id': client_id,
        'permissions': admin_permissions,  # Using admin for simplicity
        'scope': 'bot applications.commands'  # Both bot and slash commands
    }
    
    return f"{base_url}?{urlencode(params)}"

def check_bot_requirements():
    """Check if all requirements are met for bot invitation."""
    print("🤖 DuckBot Invitation Troubleshooter")
    print("=" * 50)
    
    # Check 1: Discord Token
    token = os.getenv("DISCORD_TOKEN")
    if token:
        print("✅ DISCORD_TOKEN found in environment")
        # Don't print the actual token for security
        print(f"   Token starts with: {token[:10]}...")
    else:
        print("❌ DISCORD_TOKEN not found!")
        print("   → Add DISCORD_TOKEN to your .env file")
        return False
    
    # Check 2: Bot Application Settings
    print("\n📋 Discord Developer Portal Checklist:")
    print("1. Go to https://discord.com/developers/applications")
    print("2. Select your DuckBot application")
    print("3. Go to 'Bot' section:")
    print("   ✅ Bot token generated")
    print("   ✅ MESSAGE CONTENT INTENT enabled")
    print("   ✅ SERVER MEMBERS INTENT enabled") 
    print("   ✅ Bot is set to 'Public' (if you want others to invite)")
    
    print("\n4. Go to 'OAuth2' → 'URL Generator':")
    print("   ✅ Select 'bot' scope")
    print("   ✅ Select 'applications.commands' scope")
    print("   ✅ Select required permissions (or Administrator)")
    
    return True

def main():
    """Main troubleshooting function."""
    
    if not check_bot_requirements():
        return
    
    print(f"\n🔗 Manual Invite URL Generation:")
    print("If you don't have your Client ID, get it from:")
    print("https://discord.com/developers/applications → Your App → Application ID")
    
    client_id = input("\nEnter your Discord Application Client ID: ").strip()
    
    if client_id.isdigit():
        invite_url = generate_invite_url(client_id)
        print(f"\n✅ Generated Invite URL:")
        print(f"🔗 {invite_url}")
        print(f"\n📋 Instructions:")
        print("1. Copy the URL above")
        print("2. Paste it in your browser")
        print("3. Select the server to invite the bot to")
        print("4. Click 'Authorize'")
        print("5. Complete any CAPTCHA if prompted")
        
        print(f"\n🛠️ If invitation still fails:")
        print("• Make sure you have 'Manage Server' permission")
        print("• Check the server doesn't have bot restrictions")
        print("• Try using Administrator permission")
        print("• Verify the Client ID is correct")
        
    else:
        print("❌ Invalid Client ID. Must be numbers only.")

if __name__ == "__main__":
    main()