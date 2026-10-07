"""Polite, caching HTTP client shared by all register ingesters (CLAUDE.md §2.7, §4).

- identifies itself with a descriptive User-Agent;
- rate-limits to `max_rps` (default 5 req/s) and backs off on 429/5xx, honouring Retry-After;
- caches every response under data/raw/ and never re-downloads a cached URL;
- strips personal IDs (birth dates, national IDs) *before* the body touches disk;
- never logs or raises with response bodies, only URL and status.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from signrule.ingest.pii import SANITIZER_VERSION, sanitize_record, scan_for_personal_ids

log = logging.getLogger(__name__)

RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
CACHEABLE_EMPTY_STATUSES = frozenset({404, 410})


class IngestError(RuntimeError):
    """Raised for unrecoverable fetch errors. Messages never contain response bodies."""


@dataclass
class ClientStats:
    requests: int = 0
    cache_hits: int = 0
    retries: int = 0
    status_counts: dict[int, int] = field(default_factory=dict)


@dataclass(frozen=True)
class CachedResponse:
    status_code: int
    body: Any  # sanitized JSON, or None for cached 404/410
    from_cache: bool


class RegisterClient:
    def __init__(
        self,
        *,
        user_agent: str,
        max_rps: float = 5.0,
        max_retries: int = 6,
        timeout: float = 30.0,
        backoff_base: float = 1.0,
        backoff_cap: float = 120.0,
        transport: httpx.BaseTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        sanitize: Callable[[Any], tuple[Any, int]] = sanitize_record,
    ) -> None:
        if max_rps <= 0 or max_rps > 5.0:
            raise ValueError("max_rps must be in (0, 5] (CLAUDE.md §2.7)")
        self._http = httpx.Client(
            headers={"User-Agent": user_agent},
            timeout=timeout,
            transport=transport,
            follow_redirects=True,
        )
        self._min_interval = 1.0 / max_rps
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._backoff_cap = backoff_cap
        self._clock = clock
        self._sleep = sleep
        self._sanitize = sanitize
        self._next_slot: float | None = None
        self._lock = threading.Lock()
        self.stats = ClientStats()

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> RegisterClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---- public API -------------------------------------------------------------------------

    def get_json(
        self,
        url: str,
        cache_path: Path,
        *,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> CachedResponse:
        """GET a JSON resource through the cache. 200 and 404/410 are cached; others raise."""
        return self.fetch_cached(
            "GET", url, cache_path, params=params, headers=headers, parse=lambda r: r.json()
        )

    def fetch_cached(
        self,
        method: str,
        url: str,
        cache_path: Path,
        *,
        parse: Callable[[httpx.Response], Any],
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        content: bytes | None = None,
        transform: Callable[[Any], Any] | None = None,
        cache_key: Mapping[str, str] | None = None,
    ) -> CachedResponse:
        """Request through the cache: parse -> transform -> strip personal IDs -> scan -> write.

        200 and 404/410 are cached; other statuses raise without the body. `cache_key` replaces
        `params` in the stored envelope (e.g. the SOAP operation and its key, never secrets).
        """
        if cache_path.exists():
            env = json.loads(cache_path.read_text(encoding="utf-8"))
            self._count("cache_hits")
            return CachedResponse(env["status_code"], env["body"], from_cache=True)

        resp = self._request(method, url, params=params, headers=headers, content=content)
        if resp.status_code in CACHEABLE_EMPTY_STATUSES:
            body, removed, digest = None, 0, hashlib.sha256(resp.content).hexdigest()
        elif resp.status_code == 200:
            try:
                parsed = parse(resp)
            except ValueError as e:
                raise IngestError(f"unparseable 200 response from {url}") from e
            if transform is not None:
                parsed = transform(parsed)
            digest = hashlib.sha256(resp.content).hexdigest()
            body, removed = self._sanitize(parsed)
            hits = scan_for_personal_ids(body)
            if hits:
                raise IngestError(
                    f"personal-ID-like values at {len(hits)} path(s) in response from {url} "
                    f"(e.g. {hits[0]}); not written. Extend PERSONAL_ID_KEYS."
                )
        else:
            raise IngestError(f"HTTP {resp.status_code} from {url}")

        envelope = {
            "url": url,
            "params": dict(cache_key if cache_key is not None else (params or {})),
            "status_code": resp.status_code,
            "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "sha256_response": digest,
            "sanitizer": SANITIZER_VERSION,
            "removed_fields": removed,
            "body": body,
        }
        _atomic_write_json(cache_path, envelope)
        return CachedResponse(resp.status_code, body, from_cache=False)

    def get_bytes(self, url: str, *, headers: Mapping[str, str] | None = None) -> httpx.Response:
        """GET a bulk file into memory (not cached: callers persist a sanitized projection)."""
        resp = self._request("GET", url, headers=headers, timeout=600.0)
        if resp.status_code != 200:
            raise IngestError(f"HTTP {resp.status_code} from {url}")
        return resp

    def head(self, url: str) -> httpx.Response:
        return self._request("HEAD", url)

    # ---- internals --------------------------------------------------------------------------

    def _reserve_slot(self, now: float) -> float:
        """Thread-safe: the start time of this request, >= min_interval after the previous one."""
        with self._lock:
            slot = now if self._next_slot is None else max(now, self._next_slot)
            self._next_slot = slot + self._min_interval
            return slot

    def _throttle(self) -> None:
        wait = self._reserve_slot(self._clock()) - self._clock()
        if wait > 0:
            self._sleep(wait)

    def _count(self, field: str, status: int | None = None) -> None:
        with self._lock:
            if status is None:
                setattr(self.stats, field, getattr(self.stats, field) + 1)
            else:
                self.stats.status_counts[status] = self.stats.status_counts.get(status, 0) + 1

    def _backoff(self, attempt: int, retry_after: str | None) -> float:
        if retry_after:
            try:
                return min(float(retry_after), self._backoff_cap)
            except ValueError:
                pass
        delay = self._backoff_base * 2**attempt
        return min(delay + random.uniform(0, delay / 2), self._backoff_cap)

    def _request(self, method: str, url: str, **kw: Any) -> httpx.Response:
        for attempt in range(self._max_retries + 1):
            self._throttle()
            self._count("requests")
            try:
                resp = self._http.request(method, url, **kw)
            except httpx.TransportError as e:
                if attempt == self._max_retries:
                    raise IngestError(f"{type(e).__name__} for {url}") from None
                self._count("retries")
                self._sleep(self._backoff(attempt, None))
                continue
            self._count("status_counts", resp.status_code)
            if resp.status_code in RETRY_STATUSES:
                if attempt == self._max_retries:
                    raise IngestError(f"HTTP {resp.status_code} from {url} after {attempt} retries")
                self._count("retries")
                log.warning("HTTP %s from %s, backing off", resp.status_code, url)
                self._sleep(self._backoff(attempt, resp.headers.get("Retry-After")))
                continue
            return resp
        raise AssertionError("unreachable")


def _atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    try:
        tmp.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
