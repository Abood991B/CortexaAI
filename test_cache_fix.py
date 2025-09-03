#!/usr/bin/env python3
"""
Test script to verify the cache clearing fix for cancelled workflows.

This script tests the scenario where:
1. A workflow is started with a prompt
2. The workflow is cancelled during grace period
3. The same prompt is used again - it should process normally, not return cached cancelled result

NOTE: This tests the backend cache clearing. The frontend has its own localStorage cache
that must be cleared separately when a workflow is cancelled.
"""

import asyncio
import aiohttp
import time
import json


async def test_cancelled_workflow_cache_fix():
    """Test that cancelled workflows don't pollute the cache."""
    
    base_url = "http://localhost:8001"
    test_prompt = "Write a function to validate email addresses using regex"
    
    print("Starting cache fix test...")
    print(f"Test prompt: {test_prompt}")
    print("-" * 50)
    
    async with aiohttp.ClientSession() as session:
        # Step 1: Start a workflow
        print("\n1. Starting first workflow...")
        async with session.post(
            f"{base_url}/api/process-prompt",
            json={
                "prompt": test_prompt,
                "prompt_type": "auto",
                "return_comparison": True,
                "use_langgraph": False
            }
        ) as response:
            result1 = await response.json()
            workflow_id1 = result1.get("workflow_id")
            print(f"   Workflow started: {workflow_id1}")
        
        # Step 2: Cancel it within grace period (wait 1 second then cancel)
        await asyncio.sleep(1)
        print("\n2. Cancelling workflow within grace period...")
        async with session.post(f"{base_url}/api/cancel-workflow/{workflow_id1}") as response:
            cancel_result = await response.json()
            print(f"   Cancel result: {cancel_result}")
        
        # Step 3: Check workflow status
        await asyncio.sleep(0.5)
        print("\n3. Checking workflow status...")
        async with session.get(f"{base_url}/api/workflow-status/{workflow_id1}") as response:
            status_result = await response.json()
            print(f"   Status: {status_result.get('status')}")
            assert status_result.get('status') == 'cancelled', "Workflow should be cancelled"
        
        # Step 4: Start the same workflow again (this should not return cached cancelled result)
        print("\n4. Starting second workflow with same prompt...")
        async with session.post(
            f"{base_url}/api/process-prompt",
            json={
                "prompt": test_prompt,
                "prompt_type": "auto",
                "return_comparison": True,
                "use_langgraph": False
            }
        ) as response:
            result2 = await response.json()
            workflow_id2 = result2.get("workflow_id")
            print(f"   Workflow started: {workflow_id2}")
        
        # Step 5: Wait for completion and check result
        print("\n5. Waiting for second workflow to complete...")
        max_wait = 60  # Maximum 60 seconds
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            async with session.get(f"{base_url}/api/workflow-status/{workflow_id2}") as response:
                status_result = await response.json()
                status = status_result.get('status')
                
                if status == 'completed':
                    print(f"   ✓ Workflow completed successfully!")
                    print(f"   Processing time: {status_result.get('result', {}).get('processing_time_seconds', 0):.2f}s")
                    print(f"   Quality score: {status_result.get('result', {}).get('output', {}).get('quality_score', 0)}")
                    
                    # Verify it's not a cached cancelled result
                    assert status != 'cancelled', "Second workflow should not be cancelled"
                    assert 'result' in status_result, "Completed workflow should have results"
                    
                    print("\n✅ TEST PASSED: Cache fix is working correctly!")
                    print("   Cancelled workflows no longer pollute the cache.")
                    return True
                
                elif status == 'cancelled':
                    print(f"\n❌ TEST FAILED: Second workflow returned cached cancelled status!")
                    print("   The cache fix is not working properly.")
                    return False
                
                elif status == 'failed':
                    print(f"\n❌ TEST FAILED: Second workflow failed!")
                    print(f"   Error: {status_result.get('error', 'Unknown error')}")
                    return False
                
                await asyncio.sleep(2)
        
        print(f"\n❌ TEST TIMEOUT: Second workflow did not complete within {max_wait} seconds")
        return False


async def main():
    """Run the test."""
    try:
        success = await test_cancelled_workflow_cache_fix()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ TEST ERROR: {e}")
        exit(1)


if __name__ == "__main__":
    print("Cache Fix Test Script")
    print("=" * 50)
    print("Make sure the backend server is running on http://localhost:8001")
    print("=" * 50)
    
    asyncio.run(main())
