# LM Studio 500 Error Troubleshooting Guide

## 🚨 Quick Fixes (Try These First)

### **1. 🔄 Restart LM Studio**
- Close LM Studio completely
- Wait 10 seconds
- Restart and reload your model
- Test with `/lm_health` command

### **2. 🔌 Check Plugin Configuration**
The most common cause is plugin conflicts:

```python
# Problem: Too many plugins or unavailable plugins
"plugins": {
    "query-neo4j": {"enabled": True, "priority": 2},  # ← This might not exist
    "rag-v1": {"enabled": True, "priority": 1},       # ← Might be causing issues
    # ... other plugins
}
```

**Solution:** Use the enhanced ask command with progressive fallback.

### **3. 🚦 Concurrent Request Issues**
LM Studio can struggle with multiple users at once:

- **Symptom:** Works fine when you test alone, fails with multiple users
- **Solution:** Implement request queuing or rate limiting

## 🔍 Detailed Diagnosis Steps

### **Step 1: Test Basic Connectivity**

```bash
# Test if LM Studio is responding at all
curl -X POST http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "test"}],
    "max_tokens": 10
  }'
```

### **Step 2: Test Without Plugins**

```python
# Minimal payload that should always work
{
    "messages": [
        {"role": "user", "content": "Hello"}
    ],
    "max_tokens": 50,
    "temperature": 0.7
}
```

### **Step 3: Test Plugins One by One**

```python
# Test each plugin individually
test_plugins = [
    {"duckduckgo": {"enabled": True}},
    {"wikipedia": {"enabled": True}},
    {"web-search": {"enabled": True}},
    {"rag-v1": {"enabled": True}},
    # etc.
]
```

## 🛠️ Implementation Solutions

### **Solution 1: Replace Your Ask Command**

Copy the code from `enhanced_ask_command.py` and replace your existing `/ask` command. This version:

- ✅ **Retry logic** - Automatically retries failed requests
- ✅ **Progressive fallback** - Tries full plugins → safe plugins → no plugins  
- ✅ **Timeout handling** - Prevents hanging requests
- ✅ **Health checking** - Tests LM Studio before making requests
- ✅ **Graceful errors** - Friendly error messages for users

### **Solution 2: Add Request Queuing**

```python
import asyncio
from asyncio import Semaphore

# Limit concurrent LM Studio requests
lm_studio_semaphore = Semaphore(2)  # Max 2 concurrent requests

async def safe_lm_studio_call(payload):
    async with lm_studio_semaphore:
        # Your LM Studio call here
        return await call_lm_studio_with_retry(payload)
```

### **Solution 3: Plugin Management**

```python
# Dynamic plugin configuration based on availability
AVAILABLE_PLUGINS = []

async def detect_available_plugins():
    """Test which plugins are actually working."""
    test_plugins = ["duckduckgo", "wikipedia", "dice", "web-search", "rag-v1"]
    
    for plugin in test_plugins:
        if await test_plugin_availability(plugin):
            AVAILABLE_PLUGINS.append(plugin)
    
    print(f"Available plugins: {AVAILABLE_PLUGINS}")

def build_safe_payload(prompt, user_name):
    """Build payload with only known-working plugins."""
    payload = {
        "messages": [
            {"role": "system", "content": f"You are DuckBot. User: {user_name}"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 600
    }
    
    # Only add plugins we know work
    if AVAILABLE_PLUGINS:
        payload["plugins"] = {}
        for plugin in AVAILABLE_PLUGINS[:3]:  # Limit to 3 plugins
            payload["plugins"][plugin] = {"enabled": True, "priority": 1}
    
    return payload
```

## 🔧 LM Studio Configuration Tips

### **1. Model Settings**
- **Context Length:** Reduce if you're hitting memory limits
- **GPU Layers:** Adjust based on your hardware
- **Thread Count:** Don't exceed your CPU cores

### **2. Plugin Configuration in LM Studio**
- Go to LM Studio settings
- Check which plugins are actually installed and enabled
- Disable problematic plugins temporarily

### **3. Server Settings**
- **Port:** Make sure 1234 isn't blocked
- **Host:** Should be 127.0.0.1 or localhost
- **CORS:** Enable if needed

## 🚀 Advanced Solutions

### **Solution 1: Multiple LM Studio Instances**

```python
# Load balance across multiple LM Studio instances
LM_STUDIO_URLS = [
    "http://localhost:1234/v1/chat/completions",
    "http://localhost:1235/v1/chat/completions",
    "http://localhost:1236/v1/chat/completions"
]

current_url_index = 0

async def get_next_lm_studio_url():
    global current_url_index
    url = LM_STUDIO_URLS[current_url_index]
    current_url_index = (current_url_index + 1) % len(LM_STUDIO_URLS)
    return url
```

### **Solution 2: External API Fallback**

```python
# Fallback to OpenAI or other API if LM Studio fails
async def call_ai_with_fallback(prompt, user_name):
    # Try LM Studio first
    response = await call_lm_studio_with_retry(payload)
    
    if response:
        return response
    
    # Fallback to OpenAI API
    if OPENAI_API_KEY:
        return await call_openai_api(prompt, user_name)
    
    # Final fallback
    return await get_fallback_response(prompt, user_name)
```

### **Solution 3: Caching Layer**

```python
import hashlib
from datetime import datetime, timedelta

response_cache = {}

async def cached_ai_call(prompt, user_name, cache_duration_minutes=5):
    # Create cache key
    cache_key = hashlib.md5(f"{prompt}{user_name}".encode()).hexdigest()
    
    # Check cache
    if cache_key in response_cache:
        cached_response, timestamp = response_cache[cache_key]
        if datetime.now() - timestamp < timedelta(minutes=cache_duration_minutes):
            return cached_response + "\n\n*📝 (Cached response)*"
    
    # Make fresh call
    response = await call_lm_studio_with_retry(payload)
    
    # Cache the response
    if response:
        response_cache[cache_key] = (response, datetime.now())
    
    return response
```

## 📊 Monitoring & Alerting

### **Add Health Monitoring**

```python
import time
from collections import deque

# Track LM Studio performance
lm_studio_stats = {
    "request_count": 0,
    "error_count": 0,
    "avg_response_time": 0,
    "recent_errors": deque(maxlen=10)
}

async def log_lm_studio_call(success, response_time, error=None):
    lm_studio_stats["request_count"] += 1
    
    if success:
        # Update average response time
        current_avg = lm_studio_stats["avg_response_time"]
        count = lm_studio_stats["request_count"]
        lm_studio_stats["avg_response_time"] = (current_avg * (count-1) + response_time) / count
    else:
        lm_studio_stats["error_count"] += 1
        lm_studio_stats["recent_errors"].append({
            "time": datetime.now(),
            "error": str(error)
        })
        
        # Alert if error rate too high
        error_rate = lm_studio_stats["error_count"] / lm_studio_stats["request_count"]
        if error_rate > 0.3:  # 30% error rate
            print(f"🚨 High LM Studio error rate: {error_rate:.1%}")
```

## 🎯 Quick Implementation

**To fix your current issue immediately:**

1. **Replace your `/ask` command** with the enhanced version from `enhanced_ask_command.py`
2. **Add the `/lm_health` command** to diagnose issues
3. **Test with `/ask_simple`** to bypass plugin issues
4. **Restart LM Studio** and test again

**Long-term improvements:**
1. Implement request queuing for multiple users
2. Add plugin detection and management
3. Set up monitoring and alerting
4. Consider multiple LM Studio instances for scaling

The enhanced ask command should solve your 500 error issues by providing multiple fallback layers and proper error handling! 🛠️✨