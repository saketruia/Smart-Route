"""Random-number helpers.

Two flavours are provided:

* :func:`make_rng` - ordinary seeded ``numpy`` generators, one per named stream.
* Stateless hashing (:func:`hash_uniform`, :func:`hash_normal`) - a pure function of
  ``(seed, keys...)``.  The traffic engine uses it so that the congestion of a
  segment at a given time is identical no matter *which* vehicle asks, in *which* order,
  in historical *or* live mode.
"""

from __future__ import annotations

import numpy as np

_GOLD = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_MASK = (1 << 64) - 1
_LUT = np.frombuffer(b"0123456789abcdef", dtype=np.uint8)
_SHIFTS = np.arange(60, -1, -4, dtype=np.uint64)


def make_rng(seed: int, *stream: int) -> np.random.Generator:
    """Independent, reproducible generator for ``(seed, *stream)``."""
    return np.random.default_rng(np.random.SeedSequence([int(seed), *[int(s) for s in stream]]))


def splitmix64(x: np.ndarray) -> np.ndarray:
    """Vectorised SplitMix64 finaliser (uint64 -> uint64)."""
    x = np.asarray(x, dtype=np.uint64)
    with np.errstate(over="ignore"):
        x = x + _GOLD
        z = (x ^ (x >> np.uint64(30))) * _M1
        z = (z ^ (z >> np.uint64(27))) * _M2
        return z ^ (z >> np.uint64(31))


def hash_keys(seed: int, *keys: np.ndarray | int) -> np.ndarray:
    """Combine ``seed`` and integer key arrays (broadcastable) into uint64 hashes."""
    h = splitmix64(np.uint64(int(seed) & _MASK))
    for key in keys:
        k = np.asarray(key).astype(np.int64).astype(np.uint64)
        h = splitmix64(h ^ splitmix64(k))
    return h


def _to_unit(h: np.ndarray) -> np.ndarray:
    return ((h >> np.uint64(11)).astype(np.float64) + 0.5) * (1.0 / (1 << 53))


def hash_uniform(seed: int, *keys: np.ndarray | int) -> np.ndarray:
    """Deterministic U(0,1) values."""
    return _to_unit(hash_keys(seed, *keys))


def hash_normal(seed: int, *keys: np.ndarray | int) -> np.ndarray:
    """Deterministic standard-normal values (Box-Muller on two hashed uniforms)."""
    h = hash_keys(seed, *keys)
    u1, u2 = _to_unit(h), _to_unit(splitmix64(h))
    return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)


def format_event_ids(h: np.ndarray) -> np.ndarray:
    """Format uint64 hashes as ``EVT-<16 hex>`` (numpy ``S20`` array, no Python loop)."""
    h = np.atleast_1d(np.asarray(h, dtype=np.uint64))
    nibbles = ((h[:, None] >> _SHIFTS[None, :]) & np.uint64(0xF)).astype(np.uint8)
    out = np.empty((h.shape[0], 20), dtype=np.uint8)
    out[:, :4] = np.frombuffer(b"EVT-", dtype=np.uint8)
    out[:, 4:] = _LUT[nibbles]
    return out.view("S20").reshape(h.shape[0])
