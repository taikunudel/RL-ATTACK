
import sys
import os
from openai import OpenAI
import time

print("Starting diagnostic test...", flush=True)

# 1. Test vLLM Connection
server_url = "http://localhost:8002/v1"
print(f"Testing connection to {server_url}...", flush=True)

try:
    client = OpenAI(api_key="EMPTY", base_url=server_url)
    models = client.models.list()
    print(f"Success! Found models: {[m.id for m in models.data]}", flush=True)
except Exception as e:
    print(f"FAILED to connect to vLLM: {e}", flush=True)
    sys.exit(1)

# 2. Test simple generation
print("Testing simple generation...", flush=True)
try:
    response = client.chat.completions.create(
        model="qwen3-8b",
        messages=[{"role": "user", "content": "Say hello"}],
        max_tokens=10
    )
    print(f"Generation success! Output: {response.choices[0].message.content}", flush=True)
except Exception as e:
    print(f"FAILED generation: {e}", flush=True)
    sys.exit(1)

print("Diagnostic test passed!", flush=True)
