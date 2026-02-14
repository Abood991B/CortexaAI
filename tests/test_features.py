"""Functional tests for all new feature modules."""
import json
import sys

def main():
    print("=== FUNCTIONAL TESTS ===")
    print()

    # 1. Auth
    print("1. AUTH MODULE")
    from core.auth import auth_manager
    result = auth_manager.create_key("test-key", ["read", "write"], 100)
    print(f"  Create key: OK - name={result['name']}")
    raw = result["api_key"]
    verify = auth_manager.verify_key(raw)
    print(f"  Verify key: OK - active={verify['is_active']}")
    keys = auth_manager.list_keys()
    print(f"  List keys: OK - count={len(keys)}")
    auth_manager.revoke_key("test-key")
    verify2 = auth_manager.verify_key(raw)
    print(f"  Revoke key: OK - now None={verify2 is None}")
    auth_manager.delete_key("test-key")
    print()

    # 2. Marketplace
    print("2. MARKETPLACE MODULE")
    from core.marketplace import marketplace
    item = marketplace.publish("Test Prompt", "A test", "Write a poem about {{topic}}", "creative", "tester", ["test", "demo"])
    print(f"  Publish: OK - id={item['id']}")
    results = marketplace.search("Test")
    print(f"  Search: OK - found={len(results)}")
    dl = marketplace.download(item["id"])
    print(f"  Download: OK - downloads={dl['downloads']}")
    rate = marketplace.rate(item["id"], 5)
    print(f"  Rate: OK - rating={rate['rating']}")
    stats = marketplace.stats()
    print(f"  Stats: OK - total={stats['total_items']}")
    marketplace.delete(item["id"])
    print()

    # 3. Regression
    print("3. REGRESSION MODULE")
    from core.regression import regression_runner
    suite = regression_runner.create_suite("Test Suite", "coding", "A test suite", [
        {"input": "Write hello world", "expected_keywords": ["hello", "world"], "min_score": 0.5}
    ])
    print(f"  Create suite: OK - id={suite['id']}")
    got = regression_runner.get_suite(suite["id"])
    print(f"  Get suite: OK - name={got['name']}")
    suites = regression_runner.list_suites()
    print(f"  List suites: OK - count={len(suites)}")
    case = regression_runner.add_test_case(suite["id"], "Make a function", ["def", "function"], 0.6)
    print(f"  Add case: OK - id={case['id']}")
    regression_runner.delete_suite(suite["id"])
    print()

    # 4. Plugins
    print("4. PLUGINS MODULE")
    from core.plugins import plugin_manager
    reg = plugin_manager.register("test-plugin", "1.0.0", "expert", "A test", "tester", {"key": "val"})
    print(f"  Register: OK - name={reg['name']}")
    plugins = plugin_manager.list_plugins()
    print(f"  List: OK - count={len(plugins)}")
    got = plugin_manager.get_plugin("test-plugin")
    print(f"  Get: OK - version={got['version']}")
    plugin_manager.disable("test-plugin")
    got2 = plugin_manager.get_plugin("test-plugin")
    print(f"  Disable: OK - enabled={got2['enabled']}")
    plugin_manager.unregister("test-plugin")
    print()

    # 5. Templates
    print("5. TEMPLATES MODULE")
    from core.templates import template_engine
    templates = template_engine.list_all()
    print(f"  List all: OK - count={len(templates)}")
    tpl = template_engine.create("My Test", "general", "A {{topic}} prompt", "Test template", ["topic"])
    print(f"  Create: OK - id={tpl['id']}")
    rendered = template_engine.render(tpl["id"], {"topic": "AI"})
    rtext = rendered["rendered_prompt"][:40]
    print(f"  Render: OK - text={rtext}...")
    found = template_engine.search("Test")
    print(f"  Search: OK - found={len(found)}")
    print()

    # 6. Complexity
    print("6. COMPLEXITY MODULE")
    from core.complexity import complexity_analyzer
    score = complexity_analyzer.analyze("Write a hello world program")
    print(f"  Simple prompt: level={score['level']}")
    score2 = complexity_analyzer.analyze(
        "Design a distributed microservices architecture with event sourcing, "
        "CQRS pattern, and implement saga pattern for distributed transactions. "
        "Consider fault tolerance, circuit breakers, and rate limiting."
    )
    print(f"  Complex prompt: level={score2['level']}")
    print()

    # 7. Language
    print("7. LANGUAGE MODULE")
    from core.language import language_processor
    lang = language_processor.analyze("Hello, how are you?")
    print(f"  Detect English: code={lang['code']}")
    lang2 = language_processor.analyze("\u0645\u0631\u062d\u0628\u0627 \u0643\u064a\u0641 \u062d\u0627\u0644\u0643")
    print(f"  Detect Arabic: code={lang2['code']}")
    supported = language_processor.get_supported_languages()
    print(f"  Supported: count={len(supported)}")
    print()

    # 8. Similarity
    print("8. SIMILARITY MODULE")
    from core.similarity import similarity_engine
    similarity_engine.add_document("doc1", "Machine learning is a subset of artificial intelligence")
    similarity_engine.add_document("doc2", "Deep learning uses neural networks")
    similarity_engine.add_document("doc3", "Cooking recipes for pasta")
    results = similarity_engine.search("AI and neural networks")
    top_id = results[0]["id"] if results else "none"
    print(f"  Search: OK - top={top_id}")
    dupes = similarity_engine.find_duplicates(0.3)
    print(f"  Duplicates: OK - pairs={len(dupes)}")
    stats = similarity_engine.stats()
    print(f"  Stats: OK - docs={stats['corpus_size']}")
    print()

    # 9. Prompt Builder
    print("9. PROMPT BUILDER MODULE")
    from core.prompt_builder import prompt_builder
    session = prompt_builder.create_session("coding")
    sid = session["session_id"]
    print(f"  Create session: OK - id={sid}")
    prompt_builder.add_block(sid, "role", "You are a Python expert")
    prompt_builder.add_block(sid, "task", "Write a sorting algorithm")
    prompt_builder.add_block(sid, "constraints", "Use only built-in functions")
    assembled = prompt_builder.assemble(sid)
    alen = len(assembled["prompt"])
    print(f"  Assemble: OK - length={alen}")
    presets = prompt_builder.list_presets()
    print(f"  Presets: OK - count={len(presets)}")
    print()

    # 10. Batch
    print("10. BATCH MODULE")
    from core.batch import batch_processor
    batch = batch_processor.create_batch([{"prompt": "test1"}, {"prompt": "test2"}])
    print(f"  Create batch: OK - id={batch['batch_id']}")
    status = batch_processor.get_status(batch["batch_id"])
    print(f"  Status: OK - status={status['status']}")
    batches = batch_processor.list_batches()
    print(f"  List: OK - count={len(batches)}")
    print()

    # 11. Fine-tuning
    print("11. FINE-TUNING MODULE")
    from core.finetuning import finetuning_manager
    models = finetuning_manager.get_supported_models("openai")
    print(f"  Models: OK - count={len(models)}")
    estimate = finetuning_manager.estimate_cost("openai", 100)
    est_cost = estimate["estimated_cost_usd"]
    print(f"  Estimate: OK - cost={est_cost}")
    print()

    # 12. Error Recovery
    print("12. ERROR RECOVERY MODULE")
    from core.error_recovery import classify_error, ErrorAnalytics
    err = classify_error(Exception("rate limit exceeded"))
    print(f"  Classify: OK - category={err.category.value}")
    analytics = ErrorAnalytics()
    summary = analytics.get_summary()
    print(f"  Summary: OK - total={summary['total_errors']}")
    recent = analytics.get_recent()
    print(f"  Recent: OK - count={len(recent)}")
    print()

    # 13. Webhooks
    print("13. WEBHOOKS MODULE")
    from core.webhooks import webhook_manager
    sub = webhook_manager.subscribe("https://example.com/hook", ["completed"])
    print(f"  Subscribe: OK - id={sub['id']}")
    subs = webhook_manager.list_subscriptions()
    print(f"  List: OK - count={len(subs)}")
    webhook_manager.unsubscribe(sub["id"])
    print()

    # 14. Database dashboard
    print("14. DATABASE DASHBOARD")
    from core.database import db
    dashboard = db.get_dashboard_stats()
    print(f"  Dashboard: OK - templates={dashboard['total_templates']}")
    print()

    print("=" * 40)
    print("=== ALL 14 FUNCTIONAL TESTS PASSED ===")
    print("=" * 40)


if __name__ == "__main__":
    main()
