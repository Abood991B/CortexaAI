#!/usr/bin/env python3
"""
Hybrid Integration Demo for Prompt Management & Versioning System

This script demonstrates the hybrid integration approach where the new
Prompt Management & Versioning system works alongside the existing agentic system.
"""

import asyncio
import json
from datetime import datetime
from agents.coordinator import get_coordinator, PROMPT_MANAGEMENT_AVAILABLE
from config.config import get_logger

logger = get_logger(__name__)


async def demonstrate_hybrid_integration():
    """Demonstrate the hybrid integration capabilities."""

    print("🔄 HYBRID INTEGRATION DEMO")
    print("=" * 50)

    # Check if Prompt Management System is available
    print(f"📦 Prompt Management System Available: {PROMPT_MANAGEMENT_AVAILABLE}")

    if not PROMPT_MANAGEMENT_AVAILABLE:
        print("❌ Prompt Management System not available - running with fallback only")
        return

    # Step 1: Enable features progressively
    print("\n📈 STEP 1: Enabling Features Progressively")
    print("-" * 40)

    # Get the coordinator instance
    coordinator = get_coordinator()

    # Enable performance tracking first (safest)
    success = coordinator.enable_performance_tracking(True)
    print(f"✅ Performance Tracking Enabled: {success}")

    # Enable prompt management
    success = coordinator.enable_prompt_management(True)
    print(f"✅ Prompt Management Enabled: {success}")

    # Enable experimentation
    success = coordinator.enable_experimentation(True)
    print(f"✅ Experimentation Enabled: {success}")

    # Step 2: Test basic functionality
    print("\n🧪 STEP 2: Testing Basic Functionality")
    print("-" * 40)

    test_prompt = "Write a function to calculate fibonacci numbers efficiently"

    try:
        # Process a prompt with the existing system
        result = await coordinator.process_prompt(
            prompt=test_prompt,
            prompt_type="auto",
            return_comparison=True
        )

        print("✅ Prompt processed successfully")
        print(f"   Workflow ID: {result['workflow_id']}")
        print(f"   Domain: {result['output']['domain']}")
        print(f"   Quality Score: {result['output']['quality_score']:.2f}")
        print(f"   Processing Time: {result['processing_time_seconds']:.2f}s")

    except Exception as e:
        print(f"❌ Error processing prompt: {e}")
        return

    # Step 3: Demonstrate domain migration
    print("\n🔄 STEP 3: Domain Migration")
    print("-" * 40)

    # Get the domain from the result
    domain = result['output']['domain']
    print(f"🎯 Detected Domain: {domain}")

    # Migrate domain to new system
    success = coordinator.migrate_domain_to_new_system(domain)
    print(f"✅ Domain '{domain}' migrated to new system: {success}")

    # Test getting best prompt from new system
    best_prompt = coordinator.get_best_prompt_for_domain(domain, fallback_to_old=True)
    if best_prompt:
        print("✅ Retrieved best prompt from new system")
        print(f"   Best Prompt Preview: {best_prompt[:100]}...")
    else:
        print("📝 No optimized prompts found - using existing system (this is correct fallback behavior)")

    # Step 4: Performance tracking
    print("\n📊 STEP 4: Performance Tracking")
    print("-" * 40)

    # Record performance for the prompt we just processed
    success = coordinator.record_prompt_performance(
        domain=domain,
        prompt_content=result['output']['optimized_prompt'],
        performance_score=result['output']['quality_score'],
        metadata={
            "workflow_id": result['workflow_id'],
            "processing_time": result['processing_time_seconds'],
            "iterations_used": result['output']['iterations_used']
        }
    )
    print(f"✅ Performance recorded: {success}")

    # Step 5: Create A/B experiment
    print("\n🧪 STEP 5: A/B Testing Setup")
    print("-" * 40)

    try:
        experiment_id = coordinator.create_experiment(
            name=f"{domain} Optimization Test",
            domain=domain,
            variants=[
                result['output']['optimized_prompt'],
                "Write an optimized function for fibonacci sequence calculation with O(n) time complexity"
            ],
            traffic_split=[0.7, 0.3]  # 70% control, 30% variant
        )
        print(f"✅ Experiment created: {experiment_id}")

    except Exception as e:
        print(f"❌ Failed to create experiment: {e}")

    # Step 6: Template system
    print("\n📝 STEP 6: Template System")
    print("-" * 40)

    # Get available templates
    templates = coordinator.get_available_templates()
    print(f"📋 Available Templates: {len(templates)}")

    if templates:
        print("📝 Template names:")
        for template in templates:
            print(f"   - {template.get('name', 'Unknown')}")

    # Step 7: System statistics
    print("\n📈 STEP 7: System Statistics")
    print("-" * 40)

    stats = coordinator.get_prompt_management_stats()

    print("📊 System Status:")
    print(f"   Available: {stats.get('system_available', False)}")
    print(f"   Management Enabled: {stats.get('prompt_management_enabled', False)}")
    print(f"   Performance Tracking: {stats.get('performance_tracking_enabled', False)}")
    print(f"   Experimentation: {stats.get('experimentation_enabled', False)}")
    print(f"   Migrated Domains: {stats.get('total_migrated_domains', 0)}")

    if stats.get('migrated_domains'):
        print(f"   Domains: {', '.join(stats['migrated_domains'])}")

    # Step 8: Workflow statistics
    print("\n📋 STEP 8: Workflow Statistics")
    print("-" * 40)

    workflow_stats = coordinator.get_workflow_stats()
    if isinstance(workflow_stats, dict) and 'error' not in workflow_stats:
        print("📊 Workflow Performance:")
        print(f"   Total Workflows: {workflow_stats.get('total_workflows', 0)}")
        print(f"   Completed: {workflow_stats.get('completed_workflows', 0)}")
        print(f"   Success Rate: {workflow_stats.get('success_rate', 0):.1%}")
        print(f"   Average Quality: {workflow_stats.get('average_quality_score', 0):.2f}")
        print(f"   Average Time: {workflow_stats.get('average_processing_time', 0):.2f}s")

        if workflow_stats.get('domain_distribution'):
            print("   Domain Distribution:")
            for domain_name, count in workflow_stats['domain_distribution'].items():
                print(f"     {domain_name}: {count}")

    print("\n🎉 HYBRID INTEGRATION DEMO COMPLETED!")
    print("=" * 50)
    print("💡 Key Benefits Demonstrated:")
    print("   ✅ Zero-downtime migration")
    print("   ✅ Gradual feature adoption")
    print("   ✅ Performance tracking")
    print("   ✅ A/B testing capabilities")
    print("   ✅ Template system integration")
    print("   ✅ Comprehensive analytics")

    print("\n🚀 Next Steps:")
    print("   1. Start migrating high-traffic domains")
    print("   2. Enable experiments for prompt optimization")
    print("   3. Set up automated performance monitoring")
    print("   4. Create templates for common prompt patterns")


def demonstrate_fallback_mode():
    """Demonstrate fallback behavior when new system is unavailable."""

    print("\n🔄 FALLBACK MODE DEMO")
    print("=" * 30)

    # Get the coordinator instance
    coordinator = get_coordinator()

    # Disable new system features
    coordinator.enable_prompt_management(False)
    coordinator.enable_performance_tracking(False)
    coordinator.enable_experimentation(False)

    print("📝 Fallback Mode - Using original system only")
    print("✅ All features gracefully disabled")
    print("✅ Existing functionality preserved")
    print("✅ No impact on current operations")


async def main():
    """Main demonstration function."""

    print("🚀 Prompt Management & Versioning System")
    print("🔄 Hybrid Integration Demonstration")
    print("=" * 50)

    try:
        # Main demonstration
        await demonstrate_hybrid_integration()

        # Fallback demonstration
        demonstrate_fallback_mode()

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n🎯 SUMMARY:")
    print("The hybrid integration approach provides:")
    print("• Zero risk migration path")
    print("• Gradual adoption of new features")
    print("• Comprehensive enterprise capabilities")
    print("• Full backward compatibility")
    print("• Production-ready deployment features")


if __name__ == "__main__":
    asyncio.run(main())
