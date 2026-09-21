from tutor.metrics import MetricsCollector, render_report


def make_collector() -> MetricsCollector:
    c = MetricsCollector()
    c.record_ttfb("OpenAILLMService#0", 0.5)
    c.record_ttfb("OpenAILLMService#0", 0.7)
    c.record_ttfb("OpenAITTSService#0", 0.3)
    c.record_processing("OpenAILLMService#0", 1.2)
    c.record_llm_usage("gpt-4o", prompt=1000, completion=200, total=1200, cached=100)
    c.record_llm_usage("gpt-4o", prompt=3000, completion=400, total=3400)
    c.record_tts_usage(500)
    c.record_tts_usage(700)
    c.record_stt_usage(30.0)
    c.record_turn_latency(0.9)
    c.record_turn_latency(1.1)
    c.record_event("pause")
    c.record_event("safety_block", 2)
    c.record_tool_call("go_to_slide")
    c.record_user_turn()
    c.record_bot_turn()
    c.record_bot_turn()
    return c


def test_aggregation_and_processor_name_shortening():
    report = make_collector().report()
    assert report["latency"]["ttfb"]["OpenAILLM"] == {"avg": 0.6, "min": 0.5, "max": 0.7, "p95": 0.7, "n": 2}
    assert report["latency"]["ttfb"]["OpenAITTS"]["n"] == 1
    assert report["latency"]["processing"]["OpenAILLM"]["avg"] == 1.2
    assert report["latency"]["user_to_bot_response"]["avg"] == 1.0
    tokens = report["llm_tokens"]["gpt-4o"]
    assert tokens == {"calls": 2, "prompt": 4000, "completion": 600, "total": 4600, "cached": 100}
    assert report["tts"] == {"calls": 2, "characters": 1200}
    assert report["stt"]["audio_seconds"] == 30.0
    assert report["events"] == {"pause": 1, "safety_block": 2}
    assert report["tool_calls"] == {"go_to_slide": 1}
    assert report["turns"] == {"user": 1, "bot": 2}
    assert report["duration_seconds"] >= 0


def test_cost_estimate_uses_pricing_table():
    c = make_collector()
    expected = 4000 / 1e6 * 2.50 + 600 / 1e6 * 10.0 + 1200 / 1e6 * 15.0 + 30.0 * 0.0001
    assert abs(c.estimated_cost_usd() - expected) < 1e-9


def test_cost_estimate_maps_dated_model_snapshots_and_unknown_models():
    c = MetricsCollector()
    c.record_llm_usage("gpt-4o-2024-08-06", prompt=1_000_000, completion=0)
    assert abs(c.estimated_cost_usd() - 2.50) < 1e-9
    c2 = MetricsCollector()
    c2.record_llm_usage("some-future-model", prompt=1_000_000, completion=1_000_000)
    assert c2.estimated_cost_usd() == 0.0


def test_total_tokens_defaults_to_sum():
    c = MetricsCollector()
    c.record_llm_usage("gpt-4o-mini", prompt=10, completion=5)
    assert c.llm_tokens["gpt-4o-mini"].total == 15


def test_empty_report_and_stats():
    report = MetricsCollector().report()
    assert report["latency"]["user_to_bot_response"] == {"avg": 0.0, "min": 0.0, "max": 0.0, "p95": 0.0, "n": 0}
    text = render_report(report)
    assert "no LLM usage recorded" in text


def test_render_report_contains_key_rows():
    text = render_report(make_collector().report())
    assert "SESSION METRICS REPORT" in text
    assert "TTFB OpenAILLM" in text
    assert "gpt-4o" in text and "4000" in text and "600" in text
    assert "TTS: 2 requests, 1200 characters" in text
    assert "TOOL CALLS: go_to_slide=1" in text
    assert "EVENTS: pause=1, safety_block=2" in text
    assert "Estimated cost: $" in text


def test_finish_freezes_duration():
    c = MetricsCollector()
    c.finish()
    d1 = c.duration_seconds
    c.finish()
    assert c.duration_seconds == d1
