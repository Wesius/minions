#!/usr/bin/env python3
"""
Comprehensive test to validate all A2A-Minions server fixes.
"""

import httpx
import asyncio
import json
import time

async def test_comprehensive():
    """Test all fixed scenarios."""
    
    base_url = "http://localhost:8001"
    headers = {
        "X-API-Key": "abcd",
        "Content-Type": "application/json"
    }
    
    results = []
    
    async with httpx.AsyncClient(timeout=60.0, headers=headers) as client:
        print("🧪 Comprehensive A2A-Minions Server Test")
        print("=" * 50)
        
        # Test 1: Health check
        print("\n1. Testing health check...")
        try:
            response = await client.get(f"{base_url}/health")
            if response.status_code == 200:
                print("   ✅ Health check passed")
                results.append(("Health Check", True))
            else:
                print(f"   ❌ Health check failed: {response.status_code}")
                results.append(("Health Check", False))
        except Exception as e:
            print(f"   ❌ Health check error: {e}")
            results.append(("Health Check", False))
        
        # Test 2: Task creation with proper schema
        print("\n2. Testing task creation (schema validation fix)...")
        try:
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
                "id": "test-schema"
            }
            
            response = await client.post(base_url, json=payload)
            if response.status_code == 200:
                result = response.json()
                task = result.get("result", {})
                # Verify required fields exist
                if all(field in task for field in ["id", "sessionId", "status", "history"]):
                    print("   ✅ Task created with correct schema")
                    results.append(("Task Schema", True))
                else:
                    print("   ❌ Task missing required fields")
                    results.append(("Task Schema", False))
            else:
                print(f"   ❌ Task creation failed: {response.status_code}")
                results.append(("Task Schema", False))
        except Exception as e:
            print(f"   ❌ Task creation error: {e}")
            results.append(("Task Schema", False))
        
        # Test 3: Validation error handling
        print("\n3. Testing validation error handling...")
        try:
            # Send invalid request (missing message parts)
            invalid_payload = {
                "jsonrpc": "2.0",
                "method": "tasks/send",
                "params": {
                    "message": {
                        "role": "user",
                        "parts": []  # Empty parts should fail validation
                    }
                },
                "id": "test-validation"
            }
            
            response = await client.post(base_url, json=invalid_payload)
            if response.status_code == 400:
                error_response = response.json()
                if "error" in error_response and "Invalid params" in error_response["error"].get("message", ""):
                    print("   ✅ Validation errors handled correctly")
                    results.append(("Validation Error Handling", True))
                else:
                    print("   ❌ Unexpected error response format")
                    results.append(("Validation Error Handling", False))
            else:
                print(f"   ❌ Expected 400 but got: {response.status_code}")
                results.append(("Validation Error Handling", False))
        except Exception as e:
            print(f"   ❌ Validation test error: {e}")
            results.append(("Validation Error Handling", False))
        
        # Test 4: Minions query (tests JSON serialization fix)
        print("\n4. Testing Minions parallel query (JSON serialization fix)...")
        try:
            minions_payload = {
                "jsonrpc": "2.0",
                "method": "tasks/send",
                "params": {
                    "message": {
                        "role": "user",
                        "parts": [
                            {
                                "kind": "text",
                                "text": "Explain the benefits of parallel processing in 3 points"
                            }
                        ]
                    },
                    "metadata": {
                        "skill_id": "minions_query",
                        "max_rounds": 1,
                        "max_jobs_per_round": 3
                    }
                },
                "id": "test-minions"
            }
            
            response = await client.post(base_url, json=minions_payload)
            if response.status_code == 200:
                result = response.json()
                task_id = result["result"]["id"]
                print(f"   ✅ Minions task created: {task_id}")
                
                # Wait a bit for processing
                await asyncio.sleep(3)
                
                # Check task status
                get_payload = {
                    "jsonrpc": "2.0",
                    "method": "tasks/get",
                    "params": {"id": task_id},
                    "id": "test-get"
                }
                
                status_response = await client.post(base_url, json=get_payload)
                if status_response.status_code == 200:
                    task_data = status_response.json()["result"]
                    state = task_data["status"]["state"]
                    print(f"   📊 Task state: {state}")
                    if state in ["working", "completed"]:
                        results.append(("Minions Query", True))
                    else:
                        print(f"   ❌ Unexpected state: {state}")
                        results.append(("Minions Query", False))
                else:
                    print(f"   ❌ Failed to get task status")
                    results.append(("Minions Query", False))
            else:
                print(f"   ❌ Minions task creation failed: {response.status_code}")
                print(f"   Response: {response.text}")
                results.append(("Minions Query", False))
        except Exception as e:
            print(f"   ❌ Minions query error: {e}")
            results.append(("Minions Query", False))
        
        # Test 5: Task retrieval (field access fix)
        print("\n5. Testing task retrieval...")
        try:
            # Create a task
            create_payload = {
                "jsonrpc": "2.0",
                "method": "tasks/send",
                "params": {
                    "message": {
                        "role": "user",
                        "parts": [{"kind": "text", "text": "Test retrieval"}]
                    },
                    "metadata": {"skill_id": "minion_query"}
                },
                "id": "test-create-for-retrieval"
            }
            
            create_response = await client.post(base_url, json=create_payload)
            if create_response.status_code == 200:
                task_id = create_response.json()["result"]["id"]
                
                # Retrieve the task
                get_payload = {
                    "jsonrpc": "2.0",
                    "method": "tasks/get",
                    "params": {"id": task_id},
                    "id": "test-retrieval"
                }
                
                get_response = await client.post(base_url, json=get_payload)
                if get_response.status_code == 200:
                    retrieved_task = get_response.json()["result"]
                    # Verify we can access the message from history
                    if (retrieved_task.get("history") and 
                        len(retrieved_task["history"]) > 0 and
                        retrieved_task["history"][0].get("role") == "user"):
                        print("   ✅ Task retrieval working correctly")
                        results.append(("Task Retrieval", True))
                    else:
                        print("   ❌ Task structure incorrect")
                        results.append(("Task Retrieval", False))
                else:
                    print(f"   ❌ Task retrieval failed: {get_response.status_code}")
                    results.append(("Task Retrieval", False))
            else:
                print("   ❌ Failed to create task for retrieval test")
                results.append(("Task Retrieval", False))
        except Exception as e:
            print(f"   ❌ Task retrieval error: {e}")
            results.append(("Task Retrieval", False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:.<30} {status}")
    
    print("=" * 50)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The server is working correctly.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the issues above.")

if __name__ == "__main__":
    asyncio.run(test_comprehensive())