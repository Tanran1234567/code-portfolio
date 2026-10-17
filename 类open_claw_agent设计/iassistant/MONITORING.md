# Monitoring Prototype

This branch adds optional OpenTelemetry tracing for the FastAPI service. SigNoz
can receive the traces through the OTLP gRPC endpoint.

## Scope

- FastAPI request tracing.
- HTTPX client tracing.
- LlamaIndex tracing when `openinference-instrumentation-llama-index` is installed.
- Disabled by default unless `MZ_OTEL_ENABLED=true`.

## Local SigNoz

Start SigNoz with the official deployment guide, then keep the default OTLP gRPC
endpoint available at:

```powershell
http://localhost:4317
```

## Service Environment

Set these variables before starting the service:

```powershell
$env:MZ_OTEL_ENABLED="true"
$env:OTEL_SERVICE_NAME="iassistant"
$env:OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"
```

Then start the service as usual. After calling `/health` or any API endpoint,
open SigNoz and check whether service `iassistant` has new traces.

## Success Criteria

- `python -m pytest tests/test_monitoring.py -v` reports `2 passed`.
- Service still starts when `MZ_OTEL_ENABLED` is unset.
- With SigNoz running and `MZ_OTEL_ENABLED=true`, requests to the FastAPI app
  appear as traces under service `iassistant`.
