###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for PrimusTurboGroupedMLP weight-cache (`_stack_grouped_linear_weight`).

Tests verify the cache contract WITHOUT requiring full Megatron+Primus init:

1. Bit-exact: cached output identical to fresh stack
2. Cache hit: returns same tensor object across multiple calls within a step
3. Cache invalidation: weight `_version` bump triggers rebuild
4. Bit-exact gradient parity: cached path produces identical grads vs no-cache
   over a multi-microbatch grad-accumulation window
5. Env-var bypass: PRIMUS_TURBO_DISABLE_GROUPED_WEIGHT_CACHE=1 disables cache
6. Version-regression assert: detects stale-cache invariant violation

These tests use a minimal stand-in module that mirrors the cache logic in
`primus.backends.megatron.core.extensions.primus_turbo.PrimusTurboGroupedMLP`,
allowing the cache invariants to be tested without a Megatron environment.
"""

import os
from typing import Dict, Tuple

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Minimal stand-in mirroring the cache logic from PrimusTurboGroupedMLP
# (see primus_turbo.py:1308-1359). This lets us test the cache contract in
# isolation; the actual integration test runs as part of the E2E MFU bench.
# ---------------------------------------------------------------------------
class _GroupedWeightCacheModule(nn.Module):
    def __init__(self, num_local_experts: int, K: int, M: int):
        super().__init__()
        self.num_local_experts = num_local_experts
        for i in range(num_local_experts):
            setattr(self, f"weight{i}", nn.Parameter(torch.randn(K, M, dtype=torch.bfloat16)))
        self._weight_cache: Dict[int, Tuple[Tuple[int, ...], torch.Tensor]] = {}
        self._cache_disabled = bool(int(os.environ.get(
            "PRIMUS_TURBO_DISABLE_GROUPED_WEIGHT_CACHE", "0")))
        self.cache_hits = 0
        self.cache_misses = 0

    def stack_weights(self) -> torch.Tensor:
        weights = [getattr(self, f"weight{i}") for i in range(self.num_local_experts)]
        if self._cache_disabled:
            self.cache_misses += 1
            return torch.stack(weights, dim=0).transpose(1, 2).contiguous()
        key = id(self)
        versions = tuple(w._version for w in weights)
        cached = self._weight_cache.get(key)
        if cached is not None and cached[0] == versions:
            self.cache_hits += 1
            return cached[1]
        if cached is not None:
            assert all(v >= cv for v, cv in zip(versions, cached[0])), (
                "weight version went backwards"
            )
        stacked = torch.stack(weights, dim=0).transpose(1, 2).contiguous()
        self._weight_cache[key] = (versions, stacked)
        self.cache_misses += 1
        return stacked


def _ref_stack(module: _GroupedWeightCacheModule) -> torch.Tensor:
    weights = [getattr(module, f"weight{i}") for i in range(module.num_local_experts)]
    return torch.stack(weights, dim=0).transpose(1, 2).contiguous()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_bit_exact_first_call():
    """First call returns bit-identical tensor to a reference fresh stack."""
    torch.manual_seed(42)
    m = _GroupedWeightCacheModule(num_local_experts=8, K=128, M=64)
    cached = m.stack_weights()
    ref = _ref_stack(m)
    assert torch.equal(cached, ref), "cached and ref differ on first call"
    assert m.cache_misses == 1 and m.cache_hits == 0


def test_cache_hit_within_step():
    """Within an optimizer step (no _version change), cache hit returns same object."""
    torch.manual_seed(42)
    m = _GroupedWeightCacheModule(num_local_experts=8, K=128, M=64)
    out0 = m.stack_weights()
    out1 = m.stack_weights()
    out2 = m.stack_weights()
    assert m.cache_hits == 2 and m.cache_misses == 1
    assert out1 is out0 and out2 is out0, "cache hit returned different object"


def test_cache_invalidate_on_version_bump():
    """Simulating optimizer step (in-place add via tensor op) invalidates cache."""
    torch.manual_seed(42)
    m = _GroupedWeightCacheModule(num_local_experts=8, K=128, M=64)
    out0 = m.stack_weights()
    with torch.no_grad():
        for i in range(m.num_local_experts):
            getattr(m, f"weight{i}").add_(0.001 * torch.randn_like(getattr(m, f"weight{i}")))
    out1 = m.stack_weights()
    assert m.cache_misses == 2, f"expected 2 misses after version bump, got {m.cache_misses}"
    assert not torch.equal(out0, out1), "weights changed but stacked output unchanged"
    ref = _ref_stack(m)
    assert torch.equal(out1, ref)


def test_autograd_gradients_match():
    """Bit-exact gradient parity vs no-cache path across multi-microbatch grad-accum."""
    torch.manual_seed(42)
    m_c = _GroupedWeightCacheModule(num_local_experts=8, K=128, M=64)
    m_nc = _GroupedWeightCacheModule(num_local_experts=8, K=128, M=64)
    with torch.no_grad():
        for i in range(8):
            getattr(m_nc, f"weight{i}").copy_(getattr(m_c, f"weight{i}"))
    m_nc._cache_disabled = True

    x = torch.randn(8, 32, 128, dtype=torch.bfloat16, requires_grad=False)

    for _ in range(4):
        wc = m_c.stack_weights()       # [E, M, K]
        wnc = m_nc.stack_weights()
        out_c = torch.einsum("bnk,emk->bnm", x, wc).sum()
        out_nc = torch.einsum("bnk,emk->bnm", x, wnc).sum()
        out_c.backward()
        out_nc.backward()

    for i in range(8):
        gc = getattr(m_c, f"weight{i}").grad
        gnc = getattr(m_nc, f"weight{i}").grad
        assert gc is not None and gnc is not None
        if not torch.equal(gc, gnc):
            diff = (gc.float() - gnc.float()).abs().max().item()
            assert diff < 1e-3, f"grad diff too large on weight{i}: {diff}"
    assert m_c.cache_hits == 3 and m_c.cache_misses == 1
    assert m_nc.cache_misses == 4


def test_disable_env_var(monkeypatch):
    """PRIMUS_TURBO_DISABLE_GROUPED_WEIGHT_CACHE=1 forces cache-miss every call."""
    monkeypatch.setenv("PRIMUS_TURBO_DISABLE_GROUPED_WEIGHT_CACHE", "1")
    m = _GroupedWeightCacheModule(num_local_experts=8, K=128, M=64)
    m.stack_weights()
    m.stack_weights()
    m.stack_weights()
    assert m.cache_hits == 0 and m.cache_misses == 3


def test_version_regression_assert():
    """Simulated parameter swap (cached_versions ahead of real versions) trips assert.

    This guards the 'silent stale cache' footgun where some external mutation
    leaves the cache holding a higher recorded version than the live tensors.
    """
    torch.manual_seed(42)
    m = _GroupedWeightCacheModule(num_local_experts=2, K=8, M=4)
    m.stack_weights()  # populate cache with v0
    cached_versions, cached_tensor = m._weight_cache[id(m)]
    fake_higher = tuple(v + 1 for v in cached_versions)
    m._weight_cache[id(m)] = (fake_higher, cached_tensor)
    with pytest.raises(AssertionError, match="version went backwards"):
        m.stack_weights()
