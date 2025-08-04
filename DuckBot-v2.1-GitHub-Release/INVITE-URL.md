# 🤖 DuckBot Invitation Setup

## Bot Client ID
Based on your Discord token, your Application ID (Client ID) is: **1397321689830002768**

## 🔗 Invitation URL
Copy this URL and paste it in your browser to invite DuckBot to a server:

```
https://discord.com/api/oauth2/authorize?client_id=1397321689830002768&permissions=8&scope=bot%20applications.commands
```

## 📋 Before Inviting - Discord Developer Portal Checklist

1. **Go to:** https://discord.com/developers/applications
2. **Select:** Your DuckBot application (ID: 1397321689830002768)
3. **Navigate to:** Bot section (left sidebar)
4. **CRITICAL - Enable these Intents:**
   - ✅ **MESSAGE CONTENT INTENT** (Required for chat commands)
   - ✅ **SERVER MEMBERS INTENT** (Required for user analytics)
   - ✅ **PRESENCE INTENT** (Optional, for user status)

5. **Check Bot Settings:**
   - ✅ **Public Bot** enabled (if others should be able to invite)
   - ✅ **Require OAuth2 Code Grant** disabled
   - ✅ Bot token exists and matches your .env file

## 🛠️ If Invitation Still Fails

### Common Issues:
1. **Missing Permissions:** You need "Manage Server" permission on the target server
2. **Bot Restrictions:** Server may have restrictions on adding bots
3. **Intents Not Enabled:** The MESSAGE CONTENT INTENT must be enabled
4. **Bot Not Public:** If disabled, only you can invite the bot

### Troubleshooting Steps:
1. Try the invite URL in an incognito/private browser window
2. Make sure you're logged into the correct Discord account
3. Check if the server has any bot verification requirements
4. Verify the Client ID is correct: `1397321689830002768`

### Alternative URL (Specific Permissions):
If Administrator permission causes issues, try this URL with specific permissions:
```
https://discord.com/api/oauth2/authorize?client_id=1397321689830002768&permissions=2147488832&scope=bot%20applications.commands
```

## ✅ After Successfully Inviting

1. The bot should appear in your server's member list
2. Slash commands should be available (type `/` to see them)
3. Use `/ping` to test if the bot is responding
4. Use `/lm_health` to check LM Studio connection

## 🚨 Security Note
Your bot token is visible in your .env file. Keep this file secure and never share it publicly!

---

**Next Steps After Invitation:**
1. Test with `/ping`
2. Check LM Studio with `/lm_health` 
3. Try image generation with `/generate`
4. Verify database with `/server_stats`