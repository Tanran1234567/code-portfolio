#!/usr/bin/env python
# -*-coding: utf-8-*-

import os
import logging
from dataclasses import dataclass
from typing import Any, Optional


_CONFIGURED = False
_TRUTHY = {"1", "true", "yes", "on"}
Logger = logging.getLogger(os.getenv("MZ_SVC_NAME", "iassistant"))


@dataclass(frozen=True)
class MonitoringSettings:
    enabled: bool
    service_name: str
    endpoint: str
    environment: str


def get_monitoring_settings() -> MonitoringSettings:
    """Read OpenTelemetry settings from environment variables."""
    service_name = os.getenv("OTEL_SERVICE_NAME") or os.getenv("MZ_SVC_NAME") or "iassistant"
    return MonitoringSettings(
        enabled=_is_truthy(os.getenv("MZ_OTEL_ENABLED")),
        service_name=service_name,
        endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"),
        environment=os.getenv("MZ_ENV", "dev"),
    )


def configure_monitoring(app: Optional[Any] = None) -> bool:
    """Configure OpenTelemetry tracing for SigNoz.

    Monitoring is opt-in through MZ_OTEL_ENABLED so local development and tests do
    not need the OpenTelemetry packages unless tracing is explicitly enabled.
    """
    global _CONFIGURED

    settings = get_monitoring_settings()
    if not settings.enabled:
        return False
    if _CONFIGURED:
        return True

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError as exc:
        Logger.warning(f"OpenTelemetry monitoring is enabled but dependencies are missing: {exc}")
        return False

    os.environ.setdefault("OTEL_SERVICE_NAME", settings.service_name)
    os.environ.setdefault("OTEL_EXPORTER_OTLP_ENDPOINT", settings.endpoint)
    os.environ.setdefault(
        "OTEL_RESOURCE_ATTRIBUTES",
        f"service.name={settings.service_name},deployment.environment={settings.environment}",
    )

    resource = Resource.create(
        {
            "service.name": settings.service_name,
            "deployment.environment": settings.environment,
        }
    )
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=settings.endpoint, insecure=_is_insecure_endpoint(settings.endpoint))
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    if app is not None:
        FastAPIInstrumentor.instrument_app(app)
    HTTPXClientInstrumentor().instrument()
    _instrument_llama_index_if_available()

    _CONFIGURED = True
    Logger.info(f"OpenTelemetry monitoring enabled, exporting traces to {settings.endpoint}")
    return True


def _instrument_llama_index_if_available() -> None:
    """Enable LlamaIndex spans when OpenInference instrumentation is installed."""
    try:
        from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
    except ImportError:
        return

    LlamaIndexInstrumentor().instrument()


def _is_truthy(value: Optional[str]) -> bool:
    return (value or "").strip().lower() in _TRUTHY


def _is_insecure_endpoint(endpoint: str) -> bool:
    return endpoint.startswith("http://")
