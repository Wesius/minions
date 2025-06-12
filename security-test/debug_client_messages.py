#!/usr/bin/env python3
"""
Debug script to compare what messages are sent to Ollama vs Distributed Inference
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minions.clients.ollama import OllamaClient
from minions.clients.distributed_inference import DistributedInferenceClient

# Create a simple test message like Minions would send
test_messages = [{
    "role": "user", 
    "content": """Here is a document excerpt:

Packet 1: community_id=1:0VPcn3QN8s5A5SJy9gNMANOZbGI=, conn_state=SF, duration=0.123, history=ShADadFf, src_ip_zeek=192.168.1.10, src_port_zeek=12345, dest_ip_zeek=10.0.0.1, dest_port_zeek=22

--------------------------------
And here is your task:

You are a network security expert analyzing network packets for potential malicious activity.

--------------------------------
And here is additional higher-level advice on how to approach the task:

Look for unusual port numbers, suspicious connection patterns, abnormal byte counts, connections to/from unusual IPs, and patterns consistent with scanning, brute force, or data exfiltration.

--------------------------------

IMPORTANT: You MUST respond with a valid JSON object in exactly this format:
{{
    "explanation": "Your reasoning here",
    "citation": null,
    "answer": "Your answer here"
}}

Do not include any text before or after the JSON. Only output the JSON object.
Your response:"""
}]

print("="*80)
print("DEBUGGING CLIENT MESSAGE HANDLING")
print("="*80)

print("\n1. TEST MESSAGES TO BE SENT:")
print("-"*40)
for i, msg in enumerate(test_messages):
    print(f"Message {i+1}:")
    print(f"  Role: {msg['role']}")
    print(f"  Content: {msg['content'][:200]}...")
    print()

# Test with Ollama (if available)
print("\n2. TESTING WITH OLLAMA:")
print("-"*40)
try:
    ollama_client = OllamaClient(model_name="llama3.2", temperature=0.0, max_tokens=100)
    print("✓ Ollama client created successfully")
    
    # Let's monkey patch to see what Ollama actually receives
    original_chat = ollama_client.schat
    
    def debug_ollama_chat(messages, **kwargs):
        print("OLLAMA RECEIVES:")
        print(f"  Number of messages: {len(messages)}")
        for i, msg in enumerate(messages):
            print(f"  Message {i+1}: role={msg.get('role')}, content_length={len(msg.get('content', ''))}")
            print(f"    Content preview: {msg.get('content', '')[:100]}...")
        print()
        return original_chat(messages, **kwargs)
    
    ollama_client.schat = debug_ollama_chat
    
    try:
        responses, usage, done_reasons = ollama_client.chat(test_messages)
        print(f"✓ Ollama response: {responses[0][:200]}...")
        print(f"✓ Done reason: {done_reasons[0]}")
    except Exception as e:
        print(f"✗ Ollama error: {e}")
        
except Exception as e:
    print(f"✗ Could not test Ollama: {e}")

# Test with Distributed Inference
print("\n3. TESTING WITH DISTRIBUTED INFERENCE:")
print("-"*40)
try:
    dist_client = DistributedInferenceClient(
        model_name="llama3.2:1b", 
        temperature=0.0, 
        max_tokens=100,
        base_url="http://localhost:8080"
    )
    print("✓ Distributed inference client created successfully")
    
    # Let's monkey patch to see what gets sent to the API
    original_make_request = dist_client._make_request
    
    def debug_make_request(method, url, **kwargs):
        print("DISTRIBUTED INFERENCE SENDS:")
        if 'params' in kwargs and 'query' in kwargs['params']:
            query = kwargs['params']['query']
            print(f"  Query length: {len(query)}")
            print(f"  Query preview: {query[:200]}...")
            print(f"  Query contains 'JSON': {'JSON' in query}")
            print(f"  Query contains 'explanation': {'explanation' in query}")
        print()
        return original_make_request(method, url, **kwargs)
    
    dist_client._make_request = debug_make_request
    
    try:
        responses, usage, done_reasons = dist_client.chat(test_messages)
        print(f"✓ Distributed response: {responses[0][:200]}...")
        print(f"✓ Done reason: {done_reasons[0]}")
    except Exception as e:
        print(f"✗ Distributed inference error: {e}")
        
except Exception as e:
    print(f"✗ Could not test Distributed Inference: {e}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80) 