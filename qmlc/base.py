"""Base interfaces for the complexity metric library.

Every metric subclasses BaseMetric and returns a MetricResult that carries
value, uncertainty, elapsed time, and an estimator tag. The estimator tag
is what lets us put exact and proxy variants of the same quantity on the
same axis when benchmarking.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class MetricResult:
    name: str
    value: float
    uncertainty: Optional[float] = None
    elapsed_s: float = 0.0
    estimator: str = "exact"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        u = f" ± {self.uncertainty:.4g}" if self.uncertainty is not None else ""
        return (
            f"{self.name}[{self.estimator}] = {self.value:.6g}{u}"
            f"  ({self.elapsed_s * 1000:.2f} ms)"
        )

    @property
    def abs_error(self) -> Optional[float]:
        """Convenience accessor when ground truth is stored in metadata."""
        gt = self.metadata.get("ground_truth")
        if gt is None:
            return None
        return abs(self.value - gt)


class BaseMetric(ABC):
    """Abstract metric. Subclasses implement `_compute`; `compute` adds timing."""

    name: str = "metric"
    estimator: str = "exact"

    @abstractmethod
    def _compute(self, data: Any) -> tuple[float, Optional[float], dict]:
        """Return (value, uncertainty_or_None, metadata)."""
        raise NotImplementedError

    def compute(self, data: Any, ground_truth: Optional[float] = None) -> MetricResult:
        t0 = time.perf_counter()
        value, unc, meta = self._compute(data)
        elapsed = time.perf_counter() - t0
        if ground_truth is not None:
            meta = {**meta, "ground_truth": ground_truth}
        return MetricResult(
            name=self.name,
            value=float(value),
            uncertainty=unc,
            elapsed_s=elapsed,
            estimator=self.estimator,
            metadata=meta,
        )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r} estimator={self.estimator!r}>"
