#!/usr/bin/env python3

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minions.clients.distributed_inference import DistributedInferenceClient
import json

print("Testing Distributed Inference with Real Security Scenario...")

# Create client exactly like in the security test
client = DistributedInferenceClient(
    model_name="llama3.2:1b", 
    temperature=0.0, 
    max_tokens=4096,  # Increased from 2000 to ensure complete responses
    base_url="http://localhost:8080"
)

# Real test message that was failing
test_messages = [{
    "role": "user",
    "content": """Here is a document excerpt:

Packet 1: community_id=1:0VPcn3QN8s5A5SJy9gNMANOZbGI=, conn_state=SF, duration=0.123, history=ShADadFf, src_ip_zeek=192.168.1.10, src_port_zeek=12345, dest_ip_zeek=10.0.0.1, dest_port_zeek=22

--------------------------------
And here is your task:

You are a network security expert analyzing network packets for potential malicious activity.

Analyze the provided packet data and identify any indicators of malicious activity. Focus on unusual ports, suspicious connection patterns, abnormal byte counts, connections to/from unusual IPs, and patterns consistent with scanning, brute force, or data exfiltration.

--------------------------------
And here is additional higher-level advice on how to approach the task:

Look for unusual port numbers, suspicious connection patterns, abnormal byte counts, connections to/from unusual IPs, and patterns consistent with scanning, brute force, or data exfiltration.

--------------------------------

IMPORTANT: You MUST respond with a valid JSON object in exactly this format:
{
    "explanation": "Your reasoning here",
    "citation": null,
    "answer": "Your answer here"  
}

Do not include any text before or after the JSON. Only output the JSON object.
Your response:"""
}]

try:
    print("Running security analysis with distributed inference...")
    responses, usage, done_reasons = client.chat(test_messages)
    print(f"✓ Chat successful!")
    print(f"Response length: {len(responses[0])} characters")
    print(f"Usage: {usage}")
    print(f"Done reason: {done_reasons[0]}")
    
    print(f"\nRaw response:")
    print(f"'{responses[0]}'")
    
    # Try to parse as JSON
    try:
        parsed = json.loads(responses[0])
        print("\n✅ JSON PARSING SUCCESS!")
        print(f"Keys: {list(parsed.keys())}")
        print(f"Explanation: {parsed.get('explanation', 'N/A')[:100]}...")
        print(f"Answer: {parsed.get('answer', 'N/A')[:100]}...")
    except json.JSONDecodeError as e:
        print(f"\n❌ JSON PARSING FAILED: {e}")
        print(f"Raw response: {repr(responses[0][:500])}")
        
except Exception as e:
    print(f"\n❌ CHAT ERROR: {e}")
    import traceback
    traceback.print_exc() 