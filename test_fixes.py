#!/usr/bin/env python3
"""
Test script to verify the prompt processing and cancellation fixes.
"""
import asyncio
import aiohttp
import json
import time

BASE_URL = "http://localhost:8000"

async def test_standard_processing():
    """Test standard prompt processing."""
    print("🧪 Testing standard prompt processing...")
    
    async with aiohttp.ClientSession() as session:
        # Test standard processing
        payload = {
            "prompt": "Create a simple hello world function in Python",
            "prompt_type": "raw",
            "return_comparison": True,
            "use_langgraph": False
        }
        
        async with session.post(f"{BASE_URL}/api/process-prompt", json=payload) as resp:
            if resp.status == 200:
                result = await resp.json()
                workflow_id = result.get("workflow_id")
                print(f"✅ Standard processing started: {workflow_id}")
                
                # Poll for completion
                for i in range(30):  # Poll for up to 30 seconds
                    await asyncio.sleep(1)
                    async with session.get(f"{BASE_URL}/api/workflow-status/{workflow_id}") as status_resp:
                        if status_resp.status == 200:
                            status_data = await status_resp.json()
                            print(f"📊 Status: {status_data.get('status')}")
                            
                            if status_data.get('status') == 'completed':
                                print("✅ Standard processing completed successfully!")
                                return True
                            elif status_data.get('status') in ['failed', 'cancelled']:
                                print(f"❌ Standard processing {status_data.get('status')}")
                                return False
                        else:
                            print(f"⚠️  Status check failed: {status_resp.status}")
                
                print("⏰ Standard processing timed out")
                return False
            else:
                print(f"❌ Standard processing failed to start: {resp.status}")
                error_text = await resp.text()
                print(f"Error: {error_text}")
                return False

async def test_memory_processing():
    """Test memory-enhanced prompt processing."""
    print("\n🧪 Testing memory-enhanced prompt processing...")
    
    async with aiohttp.ClientSession() as session:
        # Test memory processing
        payload = {
            "prompt": "What is machine learning?",
            "prompt_type": "raw",
            "return_comparison": True,
            "use_langgraph": False,
            "user_id": "test_user_123",
            "chat_history": [
                {"role": "user", "content": "Hello, I'm learning about AI"},
                {"role": "assistant", "content": "Great! AI is a fascinating field. What would you like to know?"}
            ]
        }
        
        async with session.post(f"{BASE_URL}/api/process-prompt-with-memory", json=payload) as resp:
            if resp.status == 200:
                result = await resp.json()
                workflow_id = result.get("workflow_id")
                print(f"✅ Memory processing started: {workflow_id}")
                
                # Poll for completion
                for i in range(30):  # Poll for up to 30 seconds
                    await asyncio.sleep(1)
                    async with session.get(f"{BASE_URL}/api/workflow-status/{workflow_id}") as status_resp:
                        if status_resp.status == 200:
                            status_data = await status_resp.json()
                            print(f"📊 Status: {status_data.get('status')}")
                            
                            if status_data.get('status') == 'completed':
                                print("✅ Memory processing completed successfully!")
                                return True
                            elif status_data.get('status') in ['failed', 'cancelled']:
                                print(f"❌ Memory processing {status_data.get('status')}")
                                return False
                        else:
                            print(f"⚠️  Status check failed: {status_resp.status}")
                
                print("⏰ Memory processing timed out")
                return False
            else:
                print(f"❌ Memory processing failed to start: {resp.status}")
                error_text = await resp.text()
                print(f"Error: {error_text}")
                return False

async def test_cancellation():
    """Test workflow cancellation."""
    print("\n🧪 Testing workflow cancellation...")
    
    async with aiohttp.ClientSession() as session:
        # Start a workflow
        payload = {
            "prompt": "Write a very long detailed essay about the history of computing",
            "prompt_type": "raw",
            "return_comparison": True,
            "use_langgraph": False
        }
        
        async with session.post(f"{BASE_URL}/api/process-prompt", json=payload) as resp:
            if resp.status == 200:
                result = await resp.json()
                workflow_id = result.get("workflow_id")
                print(f"✅ Workflow started for cancellation test: {workflow_id}")
                
                # Wait a moment then cancel
                await asyncio.sleep(2)
                
                async with session.post(f"{BASE_URL}/api/cancel-workflow/{workflow_id}") as cancel_resp:
                    if cancel_resp.status == 200:
                        cancel_result = await cancel_resp.json()
                        print(f"✅ Cancellation requested: {cancel_result.get('message')}")
                        
                        # Check if it was actually cancelled
                        await asyncio.sleep(1)
                        async with session.get(f"{BASE_URL}/api/workflow-status/{workflow_id}") as status_resp:
                            if status_resp.status == 200:
                                status_data = await status_resp.json()
                                if status_data.get('status') == 'cancelled':
                                    print("✅ Workflow successfully cancelled!")
                                    return True
                                else:
                                    print(f"⚠️  Workflow status after cancellation: {status_data.get('status')}")
                                    return False
                            else:
                                print(f"❌ Failed to check status after cancellation: {status_resp.status}")
                                return False
                    else:
                        print(f"❌ Cancellation failed: {cancel_resp.status}")
                        error_text = await cancel_resp.text()
                        print(f"Error: {error_text}")
                        return False
            else:
                print(f"❌ Failed to start workflow for cancellation test: {resp.status}")
                return False

async def main():
    """Run all tests."""
    print("🚀 Starting comprehensive test suite...\n")
    
    results = []
    
    # Test standard processing
    results.append(await test_standard_processing())
    
    # Test memory processing
    results.append(await test_memory_processing())
    
    # Test cancellation
    results.append(await test_cancellation())
    
    # Summary
    print(f"\n📋 Test Results:")
    print(f"✅ Passed: {sum(results)}/{len(results)}")
    print(f"❌ Failed: {len(results) - sum(results)}/{len(results)}")
    
    if all(results):
        print("\n🎉 All tests passed! The fixes are working correctly.")
    else:
        print("\n⚠️  Some tests failed. Please check the issues above.")

if __name__ == "__main__":
    asyncio.run(main())
