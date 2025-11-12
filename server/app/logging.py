from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger("lk-sarvam-rag")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def log_step(step: str, status: str, details: Optional[Dict[str, Any]] = None) -> None:
    payload: Dict[str, Any] = {"timestamp": time.time(), "step": step, "status": status}
    if details is not None:
        payload["details"] = details
    logger.info(json.dumps(payload))
