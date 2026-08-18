from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def run_managed_service(
    service_endpoint: str,
    request: dict[str, Any],
    out_path: Path,
    timeout: int = 900,
) -> dict[str, Any]:
    """Run a subscribed Kipu Hub managed service via the official Service SDK.

    Credentials are read from `KIPU_ACCESS_KEY_ID` and `KIPU_SECRET_ACCESS_KEY`.
    We intentionally accept an arbitrary request object instead of inventing Rimay's
    service schema; use the OpenAPI/request schema shown after you subscribe.
    """
    access_key = os.getenv("KIPU_ACCESS_KEY_ID")
    secret = os.getenv("KIPU_SECRET_ACCESS_KEY")
    if not access_key or not secret:
        raise RuntimeError("Set KIPU_ACCESS_KEY_ID and KIPU_SECRET_ACCESS_KEY first")
    try:
        from qhub.service.client import HubServiceClient
    except ImportError as exc:
        raise RuntimeError("Install the Kipu integration extra: pip install -e '.[kipu]'") from exc

    client = HubServiceClient(
        service_endpoint=service_endpoint,
        access_key_id=access_key,
        secret_access_key=secret,
    )
    execution = client.run(request=request)
    execution.wait_for_final_state(timeout=timeout)
    result = execution.result()
    payload = {
        "execution_id": str(execution.id),
        "status": str(execution.status),
        "data": result.data(),
        "files": result.files(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return payload
