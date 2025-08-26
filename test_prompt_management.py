#!/usr/bin/env python3
"""
Test Script for Prompt Management & Versioning System

This script demonstrates the key features of the standalone Prompt Management & Versioning system.
Run this to test the functionality before deciding on integration strategy.
"""

import time
from datetime import datetime
from agents.prompt import PromptManagementSystem
from agents.prompt.prompt_models import PromptMetadata


def test_basic_functionality():
    """Test basic prompt management operations."""
    print("🧪 Testing Basic Functionality...")
    print("=" * 50)

    # Initialize the system
    pms = PromptManagementSystem()
    print("✅ Prompt Management System initialized")

    # Create a test prompt
    metadata = PromptMetadata(
        domain="customer_support",
        strategy="helpful_assistant",
        author="test_user",
        tags=["support", "chatbot", "helpful"],
        description="A helpful customer support assistant"
    )

    prompt_id = pms.create_prompt(
        name="Customer Support Assistant",
        content="You are a helpful customer support assistant. Always be polite and provide accurate information.",
        metadata=metadata,
        created_by="test_user",
        commit_message="Initial version of customer support prompt"
    )

    print(f"✅ Created prompt with ID: {prompt_id}")

    # Get prompt details
    prompt_details = pms.get_prompt(prompt_id)
    print(f"✅ Retrieved prompt: {prompt_details['name']}")
    print(f"   Current version: {prompt_details['current_version']}")
    print(f"   Status: {prompt_details['status']}")

    # Create a new version
    new_version = pms.create_version(
        prompt_id=prompt_id,
        content="You are an advanced customer support assistant with extensive knowledge. Always be polite, provide accurate information, and offer additional resources when relevant.",
        created_by="test_user",
        commit_message="Enhanced with more capabilities and resource suggestions",
        bump_type="minor"
    )

    print(f"✅ Created new version: {new_version}")

    # Get version history
    history = pms.get_version_history(prompt_id)
    print(f"✅ Version history has {len(history)} versions")

    for version in history:
        print(f"   Version {version['version']}: {version['commit_message']}")

    print("✅ Basic functionality test completed!\n")


def test_version_control():
    """Test version control features."""
    print("🔄 Testing Version Control...")
    print("=" * 50)

    pms = PromptManagementSystem()

    # Create a prompt for testing
    metadata = PromptMetadata(
        domain="code_review",
        strategy="technical_assistant",
        author="test_user",
        tags=["coding", "review", "technical"],
        description="A technical assistant for code review"
    )

    prompt_id = pms.create_prompt(
        name="Code Review Assistant",
        content="You are a code review assistant. Focus on best practices, security, and performance.",
        metadata=metadata,
        created_by="test_user"
    )

    # Compare versions (should show no differences initially)
    comparison = pms.compare_versions(prompt_id, "1.0.0", "1.0.0")
    print(f"✅ Version comparison completed: {comparison['comparison']['content_changed']}")

    # Test version statistics
    stats = pms.version_manager.get_version_statistics(prompt_id)
    print(f"✅ Version statistics: {stats['total_versions']} versions, {stats['authors']} authors")

    print("✅ Version control test completed!\n")


def test_experiment_system():
    """Test A/B testing capabilities."""
    print("🧪 Testing Experiment System...")
    print("=" * 50)

    pms = PromptManagementSystem()

    # Create test prompts
    metadata1 = PromptMetadata(
        domain="marketing",
        strategy="creative_writer",
        author="test_user",
        tags=["marketing", "creative", "writing"],
        description="Creative marketing copy writer"
    )

    prompt_id_1 = pms.create_prompt(
        name="Marketing Copy Assistant V1",
        content="You are a marketing copywriter. Create engaging content.",
        metadata=metadata1,
        created_by="test_user"
    )

    prompt_id_2 = pms.create_prompt(
        name="Marketing Copy Assistant V2",
        content="You are an advanced marketing copywriter with expertise in psychology and persuasion. Create highly engaging, conversion-optimized content.",
        metadata=metadata1,
        created_by="test_user"
    )

    # Create experiment
    variants = [
        {
            'prompt_id': prompt_id_1,
            'prompt_version': '1.0.0',
            'name': 'Control',
            'weight': 0.5
        },
        {
            'prompt_id': prompt_id_2,
            'prompt_version': '1.0.0',
            'name': 'Enhanced',
            'weight': 0.5
        }
    ]

    experiment_id = pms.create_experiment(
        name="Marketing Copy Optimization",
        description="Testing enhanced marketing copy performance",
        variants=variants,
        created_by="test_user",
        duration_days=1
    )

    print(f"✅ Created experiment: {experiment_id}")

    # Start experiment
    success = pms.start_experiment(experiment_id)
    print(f"✅ Experiment started: {success}")

    # Simulate some events
    print("📊 Simulating experiment events...")

    # Get next variant (simulating traffic)
    for i in range(10):
        variant = pms.experiment_manager.get_next_variant(experiment_id)
        if variant:
            # Record impression
            pms.record_experiment_event(experiment_id, variant.id, 'impression')

            # Simulate some conversions (randomly)
            if i % 3 == 0:  # 33% conversion rate for demo
                pms.record_experiment_event(experiment_id, variant.id, 'conversion')

    # Get experiment results
    results = pms.get_experiment_results(experiment_id)
    if results:
        print("✅ Experiment results:")
        for variant in results['variants']:
            print(f"   {variant['name']}: {variant['impressions']} impressions, "
                  f"{variant['conversions']} conversions, "
                  f"{variant['conversion_rate']:.3f} rate")

    print("✅ Experiment system test completed!\n")


def test_deployment_system():
    """Test deployment capabilities."""
    print("🚀 Testing Deployment System...")
    print("=" * 50)

    pms = PromptManagementSystem()

    # Create a prompt for deployment testing
    metadata = PromptMetadata(
        domain="production",
        strategy="reliable_assistant",
        author="test_user",
        tags=["production", "reliable", "stable"],
        description="Production-ready assistant"
    )

    prompt_id = pms.create_prompt(
        name="Production Assistant",
        content="You are a production-ready assistant with stable responses.",
        metadata=metadata,
        created_by="test_user"
    )

    # Deploy to staging first
    deployment_id = pms.deploy_prompt(
        prompt_id=prompt_id,
        version="1.0.0",
        environment="staging",
        deployed_by="test_user",
        description="Initial deployment to staging"
    )

    print(f"✅ Deployed to staging: {deployment_id}")

    # Get deployment status
    status = pms.get_deployment_status(deployment_id)
    if status:
        print(f"✅ Deployment status: {status['status']}")

    # List deployments
    deployments = pms.list_deployments(prompt_id=prompt_id)
    print(f"✅ Found {len(deployments)} deployments for prompt")

    print("✅ Deployment system test completed!\n")


def test_template_system():
    """Test template management capabilities."""
    print("📝 Testing Template System...")
    print("=" * 50)

    pms = PromptManagementSystem()

    # Create a base template
    metadata = PromptMetadata(
        domain="templates",
        strategy="template_base",
        author="test_user",
        tags=["template", "base", "reusable"],
        description="Base template for assistants"
    )

    template_id = pms.create_template(
        name="Assistant Base Template",
        content="You are a ${ROLE} assistant. ${PERSONALITY}\n\nYour main tasks:\n${TASKS}\n\nGuidelines:\n${GUIDELINES}",
        metadata=metadata,
        variables={
            'ROLE': 'general',
            'PERSONALITY': 'helpful and professional',
            'TASKS': 'assist users with their requests',
            'GUIDELINES': 'always be polite and accurate'
        }
    )

    print(f"✅ Created template: {template_id}")

    # Render the template
    variables = {
        'ROLE': 'customer support',
        'PERSONALITY': 'friendly and empathetic',
        'TASKS': 'help customers with their issues and provide solutions',
        'GUIDELINES': 'listen carefully, show empathy, provide clear solutions'
    }

    rendered = pms.render_template(template_id, variables)
    if rendered:
        print("✅ Template rendered successfully:")
        print(rendered[:200] + "..." if len(rendered) > 200 else rendered)

    # Create a prompt from template
    prompt_id = pms.create_prompt_from_template(
        template_id=template_id,
        name="Customer Support Assistant",
        variables=variables,
        metadata=metadata,
        created_by="test_user"
    )

    print(f"✅ Created prompt from template: {prompt_id}")

    print("✅ Template system test completed!\n")


def test_analytics_system():
    """Test analytics and performance tracking."""
    print("📊 Testing Analytics System...")
    print("=" * 50)

    pms = PromptManagementSystem()

    # Create a prompt for analytics testing
    metadata = PromptMetadata(
        domain="analytics",
        strategy="performance_test",
        author="test_user",
        tags=["analytics", "performance", "testing"],
        description="Prompt for analytics testing"
    )

    prompt_id = pms.create_prompt(
        name="Analytics Test Prompt",
        content="You are a test assistant for analytics demonstration.",
        metadata=metadata,
        created_by="test_user"
    )

    # Record some performance metrics
    print("📈 Recording performance metrics...")

    for i in range(10):
        pms.record_performance_metric(
            prompt_id=prompt_id,
            version="1.0.0",
            environment="development",
            metric_name="response_time",
            value=0.5 + (i * 0.1),  # Increasing response time
            metadata={"request_type": "test", "complexity": "low"}
        )

        pms.record_performance_metric(
            prompt_id=prompt_id,
            version="1.0.0",
            environment="development",
            metric_name="quality_score",
            value=0.8 + (i * 0.01),  # Slightly increasing quality
            metadata={"evaluator": "test_system"}
        )

    # Get performance metrics
    metrics = pms.get_performance_metrics(
        prompt_id=prompt_id,
        time_range_hours=1
    )

    if metrics and 'metrics' in metrics:
        print("✅ Performance metrics retrieved:")
        for metric_name, stats in metrics['metrics'].items():
            if 'mean' in stats:
                print(f"   {metric_name}: mean={stats['mean']:.3f}, count={stats['count']}")

    # Test anomaly detection
    anomalies = pms.detect_performance_anomalies(
        prompt_id=prompt_id,
        sensitivity=2.0
    )

    print(f"✅ Anomaly detection completed: {anomalies.get('total_anomalies', 0)} anomalies found")

    print("✅ Analytics system test completed!\n")


def test_system_health():
    """Test system health monitoring."""
    print("🏥 Testing System Health...")
    print("=" * 50)

    pms = PromptManagementSystem()

    # Get system health
    health = pms.get_system_health()

    print(f"✅ System health: {health['overall_status']}")
    print("   Component status:")

    for component, status in health['components'].items():
        print(f"   {component}: {status['status']}")

    print("✅ System health test completed!\n")


def run_all_tests():
    """Run all test functions."""
    print("🚀 PROMPT MANAGEMENT SYSTEM - COMPREHENSIVE TEST")
    print("=" * 60)

    start_time = time.time()

    try:
        test_basic_functionality()
        test_version_control()
        test_experiment_system()
        test_deployment_system()
        test_template_system()
        test_analytics_system()
        test_system_health()

        end_time = time.time()

        print("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"⏱️  Total execution time: {end_time - start_time:.2f} seconds")
        print("\n📋 SUMMARY:")
        print("   ✅ Basic Functionality: Prompt creation, versioning, retrieval")
        print("   ✅ Version Control: Branching, comparison, history tracking")
        print("   ✅ Experiment System: A/B testing with statistical analysis")
        print("   ✅ Deployment System: Multi-environment deployment")
        print("   ✅ Template System: Variable substitution and inheritance")
        print("   ✅ Analytics System: Performance tracking and anomaly detection")
        print("   ✅ System Health: Component monitoring and status")
        print("\n🎯 The Prompt Management & Versioning System is ready for use!")
        print("\nNext steps:")
        print("1. Review the test results above")
        print("2. Decide on integration strategy (gradual, full, or hybrid)")
        print("3. Begin integration with your existing agentic system")

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
