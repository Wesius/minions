#!/usr/bin/env python3
"""
Quick test to validate the A2A-Minions server fixes.
"""

import httpx
import asyncio
import json

async def test_server():
    """Test the fixed server."""
    
    base_url = "http://localhost:8001"
    headers = {
        "X-API-Key": "abcd",
        "Content-Type": "application/json"
    }
    
    async with httpx.AsyncClient(timeout=30.0, headers=headers) as client:
        # Test 1: Health check
        print("Testing health check...")
        response = await client.get(f"{base_url}/health")
        print(f"Health check: {response.status_code} - {response.json()}")
        
        # Test 2: Send a simple task
        print("\nTesting task creation...")
        payload = {
            "jsonrpc": "2.0",
            "method": "tasks/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [
                        {
                            "kind": "text",
                            "text": "What is 2 + 2?"
                        }
                    ]
                },
                "metadata": {
                    "skill_id": "minion_query"
                }
            },
            "id": "test-1"
        }
        
        response = await client.post(base_url, json=payload)
        print(f"Task creation: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Result: {json.dumps(result, indent=2)}")
            
            # Get the task ID
            if "result" in result and "id" in result["result"]:
                task_id = result["result"]["id"]
                
                # Test 3: Get task status
                print(f"\nGetting task status for {task_id}...")
                get_payload = {
                    "jsonrpc": "2.0",
                    "method": "tasks/get",
                    "params": {
                        "id": task_id
                    },
                    "id": "test-2"
                }
                
                await asyncio.sleep(2)  # Wait a bit for task to process
                response = await client.post(base_url, json=get_payload)
                print(f"Get task: {response.status_code}")
                if response.status_code == 200:
                    print(f"Task status: {json.dumps(response.json(), indent=2)}")
        else:
            print(f"Error: {response.text}")

if __name__ == "__main__":
    asyncio.run(test_server())