"""Local LLM client for quality checks inside the machine (owner decision 2026-10-05).

Talks only to a local Ollama server (127.0.0.1) and refuses hosted (`:cloud`) models, so register
text never leaves the machine (CLAUDE.md §2.3, §2.4). Used for QA flags only — never for labels,
training texts or annotations (§2.1).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx

DEFAULT_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen3.5:35b-a3b-q4_K_M"
LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


class NotLocalError(RuntimeError):
    """Raised when the server or model would send data off the machine."""


def check_local(url: str, model: str) -> None:
    host = urlparse(url).hostname or ""
    if host not in LOCAL_HOSTS:
        raise NotLocalError(f"refusing non-local LLM server {host!r}")
    if "cloud" in model.lower():
        raise NotLocalError(f"refusing hosted model {model!r} (Ollama :cloud runs remotely)")


@dataclass
class LocalLLM:
    model: str = DEFAULT_MODEL
    url: str = DEFAULT_URL
    timeout: float = 300.0

    def __post_init__(self) -> None:
        check_local(self.url, self.model)
        self._client = httpx.Client(base_url=self.url, timeout=self.timeout)
        tags = self._client.get("/api/tags").json().get("models", [])
        local = {m["name"] for m in tags if not m.get("remote_host") and "cloud" not in m["name"]}
        if self.model not in local:
            raise NotLocalError(f"{self.model} is not a locally downloaded model")

    def ask_json(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        """One deterministic call with a JSON-schema constrained answer."""
        resp = self._client.post(
            "/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "format": schema,
                "stream": False,
                "think": False,
                "options": {"temperature": 0, "seed": 0, "num_ctx": 8192},
            },
        )
        resp.raise_for_status()
        return json.loads(resp.json()["message"]["content"])

    def close(self) -> None:
        self._client.close()
