"""Per-key rate limiting with a token bucket, kept in memory.

The bucket holds `rpm` tokens and refills at rpm/60 tokens per second, so a key can
burst up to its per-minute limit and then continues at the steady rate. Run the server
as one process (threads handle concurrency) so every request sees the same buckets.
"""

from __future__ import annotations

import threading
import time
from collections import deque


class WindowLimiter:
    """At most `limit` events per `window` seconds for each key (an IP address, a name)."""

    def __init__(self, limit: int, window: float):
        self.limit = limit
        self.window = window
        self._events: dict[str, deque] = {}
        self._lock = threading.Lock()

    def hit(self, key: str, now: float | None = None) -> bool:
        """Record an event. Returns False if the key is over its limit (the event is not recorded)."""
        now = time.monotonic() if now is None else now
        with self._lock:
            events = self._events.setdefault(key, deque())
            while events and events[0] <= now - self.window:
                events.popleft()
            if len(events) >= self.limit:
                return False
            events.append(now)
            if len(self._events) > 10_000:  # forget idle keys so memory stays bounded
                for k in [k for k, v in self._events.items() if not v or v[-1] <= now - self.window]:
                    del self._events[k]
            return True


class RateLimiter:
    def __init__(self) -> None:
        self._buckets: dict[int, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def check(self, key_id: int, rpm: int, now: float | None = None) -> float | None:
        """Take one token. Returns None if allowed, else the seconds to wait."""
        if rpm <= 0:
            return None
        now = time.monotonic() if now is None else now
        rate = rpm / 60.0
        with self._lock:
            tokens, last = self._buckets.get(key_id, (float(rpm), now))
            tokens = min(float(rpm), tokens + (now - last) * rate)
            if tokens >= 1.0:
                self._buckets[key_id] = (tokens - 1.0, now)
                return None
            self._buckets[key_id] = (tokens, now)
            return (1.0 - tokens) / rate
