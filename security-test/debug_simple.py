#!/usr/bin/env python3

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minions.clients.distributed_inference import DistributedInferenceClient
import json

print("Testing Distributed Inference Client directly...")

# Create client
client = DistributedInferenceClient(
    model_name="llama3.2:1b", 
    temperature=0.0, 
    max_tokens=100,
    base_url="http://localhost:8080"
)

# Simple test message
test_messages = [{
    "role": "user",
    "content": """Respond with valid JSON: {"explanation": "test", "citation": null, "answer": "test"}"""
}]

try:
    print("Calling chat...")
    responses, usage, done_reasons = client.chat(test_messages)
    print(f"Success! Response: {responses[0]}")
    
    # Try to parse as JSON
    try:
        parsed = json.loads(responses[0])
        print("✓ Valid JSON!")
        print(f"Keys: {list(parsed.keys())}")
    except json.JSONDecodeError as e:
        print(f"✗ Invalid JSON: {e}")
        print(f"Raw: {repr(responses[0])}")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc() 