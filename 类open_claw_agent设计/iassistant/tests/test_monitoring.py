#!/usr/bin/env python
# -*-coding: utf-8-*-

import os
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

MONITORING_PATH = Path(__file__).resolve().parents[1] / "src" / "mind_zone" / "monitoring.py"
SPEC = spec_from_file_location("monitoring_under_test", MONITORING_PATH)
monitoring = module_from_spec(SPEC)
SPEC.loader.exec_module(monitoring)


def test_monitoring_disabled_by_default(monkeypatch):
    monkeypatch.delenv("MZ_OTEL_ENABLED", raising=False)

    settings = monitoring.get_monitoring_settings()

    assert settings.enabled is False
    assert monitoring.configure_monitoring() is False


def test_monitoring_settings_from_environment(monkeypatch):
    monkeypatch.setenv("MZ_OTEL_ENABLED", "true")
    monkeypatch.setenv("MZ_SVC_NAME", "iassistant-test")
    monkeypatch.setenv("MZ_ENV", "test")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://signoz:4317")

    settings = monitoring.get_monitoring_settings()

    assert settings.enabled is True
    assert settings.service_name == "iassistant-test"
    assert settings.environment == "test"
    assert settings.endpoint == "http://signoz:4317"
