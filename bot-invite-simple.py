# Simple Discord Bot Invite URL Generator
import os

def generate_invite_url(client_id):
    """Generate Discord bot invite URL with admin permissions."""
    # Using Administrator permission (8) for simplicity
    base_url = "https://discord.com/api/oauth2/authorize"
    return f"{base_url}?client_id={client_id}&permissions=8&scope=bot%20applications.commands"

def main():
    print("DuckBot Invitation URL Generator")
    print("=" * 40)
    
    # Check token
    token = os.getenv("DISCORD_TOKEN")
    if token:
        print("Token found in environment")
    else:
        print("WARNING: DISCORD_TOKEN not found in .env file")
    
    print("\nTo get your Client ID:")
    print("1. Go to https://discord.com/developers/applications")
    print("2. Click your DuckBot application")
    print("3. Copy the 'Application ID' from General Information")
    
    client_id = input("\nEnter your Discord Application Client ID: ").strip()
    
    if client_id.isdigit() and len(client_id) > 10:
        invite_url = generate_invite_url(client_id)
        print(f"\nInvite URL:")
        print(invite_url)
        print("\nInstructions:")
        print("1. Copy the URL above")
        print("2. Paste in browser")
        print("3. Select server and click Authorize")
        print("\nIMPORTANT: Make sure these are enabled in Discord Developer Portal:")
        print("- Bot > MESSAGE CONTENT INTENT")
        print("- Bot > SERVER MEMBERS INTENT")
        print("- Bot > Bot is Public (if others need to invite)")
    else:
        print("Invalid Client ID. Should be 18-19 digits.")

if __name__ == "__main__":
    main()