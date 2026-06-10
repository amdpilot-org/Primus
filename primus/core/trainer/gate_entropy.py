"""Gate entropy helper for MoE routing validation."""
from __future__ import annotations

import math
from typing import Iterable


def gate_entropy(probabilities: Iterable[float]) -> float:
    """Return Shannon entropy for router probabilities, ignoring zero mass."""
    total = 0.0
    values = [float(p) for p in probabilities]
    mass = sum(values)
    if mass <= 0:
        return 0.0
    for p in values:
        if p > 0:
            q = p / mass
            total -= q * math.log(q)
    return total


def gate_entropy_within_tolerance(probabilities: Iterable[float], expected: float = 2.08, tolerance: float = 0.15) -> bool:
    return abs(gate_entropy(probabilities) - expected) <= tolerance
