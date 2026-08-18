"""In-process TTL memo — the web layer's replacement for @st.cache_data.

Streamlit's cache is per-session and cleared by reruns; a server needs the
opposite shape — one store shared across requests, expired by time. A dict and
a monotonic clock cover that; anything heavier (redis, diskcache) buys nothing
for one process serving one drafter. Disk-level feed caching already lives in
src/cache.py and is untouched by this.
"""

from __future__ import annotations

import functools
import threading
import time
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

_REGISTRY: list[dict[Any, tuple[float, Any]]] = []


def memo(ttl_seconds: float | None = None) -> Callable[[F], F]:
    """Memoize by arguments, expiring entries after `ttl_seconds`.

    `None` means never expire — the value lives until `clear_all()` or a server
    restart, matching the st.cache_data defaults the app was tuned around.
    Arguments must be hashable; the callers here pass profile names and ints.
    """

    def deco(fn: F) -> F:
        cache: dict[Any, tuple[float, Any]] = {}
        lock = threading.Lock()
        _REGISTRY.append(cache)

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            key = (args, tuple(sorted(kwargs.items())))
            now = time.monotonic()
            with lock:
                hit = cache.get(key)
                if hit is not None and (ttl_seconds is None or now - hit[0] < ttl_seconds):
                    return hit[1]
            # Computed outside the lock: a slow board build must not block a
            # request for a page that is already cached. Two concurrent misses
            # computing twice is the cheap side of that trade.
            value = fn(*args, **kwargs)
            with lock:
                cache[key] = (now, value)
            return value

        return wrapper  # type: ignore[return-value]

    return deco


def clear_all() -> int:
    """Drop every memoized value. Returns how many entries were cleared."""
    n = sum(len(c) for c in _REGISTRY)
    for c in _REGISTRY:
        c.clear()
    return n
