"""Shared HTTP helpers: rate limiting, retries, JSON fetching.

Uses `requests` when it is available (the recommended install), and falls back to
the standard library so the script still runs on a bare Python install.
"""
from __future__ import annotations

import gzip
import json
import random
import threading
import time
import urllib.error
import urllib.request

try:  # pragma: no cover - depends on the host environment
    import requests  # type: ignore

    HAVE_REQUESTS = True
except Exception:  # noqa: BLE001
    requests = None  # type: ignore
    HAVE_REQUESTS = False

DEFAULT_UA = "polymarket-btc5m-research/1.0 (+research; contact: local)"

RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504, 522, 524}


class HttpError(Exception):
    def __init__(self, status: int, url: str, body: str = "") -> None:
        super().__init__(f"HTTP {status} for {url}")
        self.status = status
        self.url = url
        self.body = body


class RateLimiter:
    """Simple per-host pacing so we stay under public API rate limits."""

    def __init__(self, rps: float) -> None:
        self.min_interval = 0.0 if rps <= 0 else 1.0 / rps
        self._lock = threading.Lock()
        self._next_at: dict[str, float] = {}

    def wait(self, host: str) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            earliest = self._next_at.get(host, 0.0)
            sleep_for = max(0.0, earliest - now)
            self._next_at[host] = max(now, earliest) + self.min_interval
        if sleep_for > 0:
            time.sleep(sleep_for)


class HttpClient:
    def __init__(
        self,
        rps: float = 6.0,
        timeout: float = 25.0,
        retries: int = 4,
        user_agent: str = DEFAULT_UA,
    ) -> None:
        self.limiter = RateLimiter(rps)
        self.timeout = timeout
        self.retries = retries
        self.user_agent = user_agent
        self._session = requests.Session() if HAVE_REQUESTS else None

    # -- internals ---------------------------------------------------------
    def _once(self, url: str, params: dict | None) -> tuple[int, str]:
        if self._session is not None:
            resp = self._session.get(
                url,
                params=params,
                timeout=self.timeout,
                headers={"User-Agent": self.user_agent, "Accept": "application/json"},
            )
            return resp.status_code, resp.text

        if params:
            from urllib.parse import urlencode

            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}{urlencode(params)}"
        req = urllib.request.Request(
            url, headers={"User-Agent": self.user_agent, "Accept-Encoding": "gzip"}
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
                if resp.headers.get("Content-Encoding") == "gzip":
                    raw = gzip.decompress(raw)
                return resp.status, raw.decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:  # non-2xx
            body = exc.read().decode("utf-8", "replace") if exc.fp else ""
            return exc.code, body

    # -- public API --------------------------------------------------------
    def get_json(self, url: str, params: dict | None = None, *, allow_404: bool = False):
        """GET -> parsed JSON. Returns None for 404 when allow_404 is set."""
        from urllib.parse import urlparse

        host = urlparse(url).netloc
        last_err: Exception | None = None
        for attempt in range(self.retries + 1):
            self.limiter.wait(host)
            try:
                status, text = self._once(url, params)
            except Exception as exc:  # noqa: BLE001 - network layer
                last_err = exc
                status, text = 0, str(exc)

            if status == 404 and allow_404:
                return None
            if status == 200:
                try:
                    return json.loads(text)
                except json.JSONDecodeError as exc:
                    last_err = exc
            elif status in RETRY_STATUS or status == 0:
                last_err = HttpError(status, url, text[:200])
            else:
                raise HttpError(status, url, text[:200])

            if attempt < self.retries:
                backoff = min(30.0, 0.6 * (2**attempt)) + random.uniform(0, 0.4)
                time.sleep(backoff)

        raise last_err if last_err else HttpError(0, url, "unknown failure")
