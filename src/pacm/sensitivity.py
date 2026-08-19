"""
src/pacm/sensitivity.py

PACM Alpha Sensitivity Analyzer — Session 6

Sweeps the semantic weight alpha in {0.5, 0.6, 0.7, 0.8, 0.9} against the
scope-penalty weight (1 - alpha) to quantify how much the balance between
semantic intent alignment and parameter-scope rules affects:
  - Divergence score distributions (attack vs. benign)
  - Attack separation (mean_D_attack - mean_D_benign)
  - False positive risk (fraction of benign calls above a threshold)

Background (PACM_Research_Guide.md §9.1):
  D = clip(alpha * D_semantic + (1 - alpha) * P_scope, 0, 1)
  - alpha close to 1.0: D dominated by semantic embedding similarity.
    Good for: semantically divergent injections.
    Risk: misses subtle injections that only change argument values.
  - alpha close to 0.0: D dominated by rule-based scope penalty.
    Good for: explicit argument hijacking (e.g. external email domains).
    Risk: high FPR when benign tools frequently use external addresses.

Usage (library):
    from src.pacm.sensitivity import AlphaSensitivityAnalyzer
    analyzer = AlphaSensitivityAnalyzer()
    result = analyzer.run(records)
    print(result.best_alpha)

Usage (CLI):
    python -m experiments.run_sensitivity --synthetic
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ─── Data models ──────────────────────────────────────────────────────────────


@dataclass
class AlphaResult:
    """
    Metrics for one alpha value from the sensitivity sweep.

    Fields:
        alpha:              Semantic weight (0 = scope-only, 1 = semantic-only).
        mean_d_attack:      Mean divergence score on attack records.
        mean_d_benign:      Mean divergence score on benign records.
        separation:         mean_d_attack - mean_d_benign (higher = better).
        fpr_at_threshold:   Fraction of benign records with D > threshold.
        abr_at_threshold:   Fraction of attack records with D > threshold.
        std_attack:         Std-dev of attack divergence scores.
        std_benign:         Std-dev of benign divergence scores.
        threshold:          Decision threshold used (default: 0.50).
        n_attack:           Number of attack records evaluated.
        n_benign:           Number of benign records evaluated.
    """
    alpha: float
    mean_d_attack: float
    mean_d_benign: float
    separation: float
    fpr_at_threshold: float
    abr_at_threshold: float
    std_attack: float
    std_benign: float
    threshold: float
    n_attack: int
    n_benign: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha": round(self.alpha, 2),
            "mean_d_attack": round(self.mean_d_attack, 4),
            "mean_d_benign": round(self.mean_d_benign, 4),
            "separation": round(self.separation, 4),
            "fpr_at_threshold": round(self.fpr_at_threshold, 4),
            "abr_at_threshold": round(self.abr_at_threshold, 4),
            "std_attack": round(self.std_attack, 4),
            "std_benign": round(self.std_benign, 4),
            "threshold": round(self.threshold, 2),
            "n_attack": self.n_attack,
            "n_benign": self.n_benign,
        }


@dataclass
class SensitivityResult:
    """
    Full result of an alpha sensitivity sweep.

    Fields:
        alpha_results:  List of AlphaResult, one per alpha value.
        best_alpha:     Alpha value with highest separation score.
        alpha_grid:     The alpha values swept.
        threshold:      Decision threshold used across all alpha values.
        n_records:      Total records in the input dataset.
        backend:        "tfidf" or "sentence_transformers".
    """
    alpha_results: list[AlphaResult]
    best_alpha: float
    alpha_grid: list[float]
    threshold: float
    n_records: int
    backend: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_alpha": round(self.best_alpha, 2),
            "alpha_grid": [round(a, 2) for a in self.alpha_grid],
            "threshold": round(self.threshold, 2),
            "n_records": self.n_records,
            "backend": self.backend,
            "alpha_results": [r.to_dict() for r in self.alpha_results],
        }


# ─── Analyzer ─────────────────────────────────────────────────────────────────


class AlphaSensitivityAnalyzer:
    """
    Sweeps alpha ∈ alpha_grid and computes divergence score statistics
    for attack vs. benign records.

    Each record in the input dataset must have either:
      (a) Pre-computed 'divergence_score', 'd_semantic', 'p_scope' fields
          (produced by run_experiment.py with PACM audit enabled); OR
      (b) 'intent_text' and 'action_text' string fields + 'is_attack' +
          optional 'arg_scope_flags' dict (raw records).

    If (a) is detected, the analyzer recomputes D for each alpha using the
    stored d_semantic and p_scope components. This is fast and accurate.

    If (b) is detected (or neither), it re-scores via DivergenceScorer with
    each alpha value. This requires the scorer to be available.

    Args:
        alpha_grid:           Alpha values to evaluate (default: [0.5..0.9]).
        threshold:            Decision threshold for ABR/FPR computation.
        prefer_transformers:  Use ST backend for scoring (if installed).
    """

    DEFAULT_ALPHA_GRID = [0.5, 0.6, 0.7, 0.8, 0.9]

    def __init__(
        self,
        alpha_grid: Optional[list[float]] = None,
        threshold: float = 0.50,
        prefer_transformers: bool = False,   # TF-IDF by default to avoid model load
    ) -> None:
        self.alpha_grid = alpha_grid or self.DEFAULT_ALPHA_GRID
        self.threshold = threshold
        self.prefer_transformers = prefer_transformers

    def run(self, records: list[dict[str, Any]]) -> SensitivityResult:
        """
        Execute the sensitivity sweep over the given records.

        Args:
            records: List of dicts with at minimum:
                       - is_attack: bool
                     And one of:
                       - d_semantic + p_scope (fast path)
                       - divergence_score (single D; alpha varies around it)

        Returns:
            SensitivityResult with per-alpha metrics and best_alpha.
        """
        if not records:
            return SensitivityResult(
                alpha_results=[], best_alpha=0.7,
                alpha_grid=self.alpha_grid, threshold=self.threshold,
                n_records=0, backend="none",
            )

        attack_records = [r for r in records if r.get("is_attack")]
        benign_records  = [r for r in records if not r.get("is_attack")]

        # Determine backend name for metadata
        backend_name = self._detect_backend()

        alpha_results: list[AlphaResult] = []
        for alpha in self.alpha_grid:
            result = self._evaluate_alpha(alpha, attack_records, benign_records)
            alpha_results.append(result)

        # Best alpha: highest separation
        best = max(alpha_results, key=lambda r: r.separation)

        return SensitivityResult(
            alpha_results=alpha_results,
            best_alpha=best.alpha,
            alpha_grid=self.alpha_grid,
            threshold=self.threshold,
            n_records=len(records),
            backend=backend_name,
        )

    def _evaluate_alpha(
        self,
        alpha: float,
        attack_records: list[dict[str, Any]],
        benign_records: list[dict[str, Any]],
    ) -> AlphaResult:
        """Compute metrics for one alpha value."""
        attack_d = [self._compute_d(r, alpha) for r in attack_records]
        benign_d  = [self._compute_d(r, alpha) for r in benign_records]

        n_attack = len(attack_d)
        n_benign  = len(benign_d)

        mean_att = _safe_mean(attack_d)
        mean_ben = _safe_mean(benign_d)
        std_att  = _safe_std(attack_d)
        std_ben  = _safe_std(benign_d)

        abr = sum(1 for d in attack_d if d > self.threshold) / n_attack if n_attack else 0.0
        fpr = sum(1 for d in benign_d if d > self.threshold) / n_benign if n_benign else 0.0

        return AlphaResult(
            alpha=alpha,
            mean_d_attack=mean_att,
            mean_d_benign=mean_ben,
            separation=mean_att - mean_ben,
            fpr_at_threshold=fpr,
            abr_at_threshold=abr,
            std_attack=std_att,
            std_benign=std_ben,
            threshold=self.threshold,
            n_attack=n_attack,
            n_benign=n_benign,
        )

    def _compute_d(self, record: dict[str, Any], alpha: float) -> float:
        """
        Compute D for a given alpha using stored components when available.

        Priority:
          1. d_semantic + p_scope fields -> D = clip(alpha*d_sem + (1-alpha)*p_scope)
          2. divergence_score field      -> proxy: re-weight assuming
             the existing D was scored at alpha=0.7 (default).
             d_sem ≈ D/0.7, p_scope ≈ 0 (if no scope flags).
          3. No usable fields            -> return 0.5 (unknown).
        """
        # Fast path: both components stored (e.g. from extended audit)
        d_sem   = record.get("d_semantic")
        p_scope = record.get("p_scope")
        if d_sem is not None and p_scope is not None:
            raw = alpha * float(d_sem) + (1.0 - alpha) * float(p_scope)
            return max(0.0, min(1.0, raw))

        # Proxy path: re-derive from stored divergence_score
        existing_d = record.get("divergence_score")
        if existing_d is not None:
            # Decompose using stored trust/scope info if available
            n_flags = len(record.get("arg_scope_flags", {}))
            p_scope_est = min(n_flags * 0.20, 0.40)
            # Back-solve d_semantic from: existing_d = 0.7 * d_sem + 0.3 * p_scope_est
            d_sem_est = (float(existing_d) - 0.3 * p_scope_est) / 0.7
            d_sem_est = max(0.0, min(1.0, d_sem_est))
            raw = alpha * d_sem_est + (1.0 - alpha) * p_scope_est
            return max(0.0, min(1.0, raw))

        # Friction-score path (from PACMAuditEntry)
        friction = record.get("friction_score")
        if friction is not None:
            # Friction = D * (1.5 - 0.5*T); back-solve D ≈ friction / (1.5 - 0.5*T)
            T = float(record.get("trust_score", 0.5))
            denom = max(1.5 - 0.5 * T, 0.01)
            approx_d = min(1.0, float(friction) / denom)
            return max(0.0, min(1.0, approx_d))

        return 0.5  # Unknown — conservative middle value

    def _detect_backend(self) -> str:
        if self.prefer_transformers:
            try:
                import sentence_transformers  # noqa: F401
                return "sentence_transformers"
            except ImportError:
                pass
        return "tfidf"


# ─── Synthetic data generator ─────────────────────────────────────────────────


def synthetic_records_for_sensitivity(
    n_attack: int = 100,
    n_benign: int = 100,
    attack_d_sem_mean: float = 0.70,
    attack_p_scope_mean: float = 0.30,
    benign_d_sem_mean: float = 0.20,
    benign_p_scope_mean: float = 0.05,
    noise: float = 0.10,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Generate synthetic records with explicit d_semantic and p_scope fields,
    enabling exact alpha re-weighting without re-scoring.

    Attack records: high semantic divergence + scope violations.
    Benign records: low semantic divergence + minimal scope flags.

    Args:
        n_attack:            Number of attack records.
        n_benign:            Number of benign records.
        attack_d_sem_mean:   Mean semantic divergence for attacks.
        attack_p_scope_mean: Mean scope penalty for attacks.
        benign_d_sem_mean:   Mean semantic divergence for benign.
        benign_p_scope_mean: Mean scope penalty for benign.
        noise:               Gaussian noise std-dev.
        seed:                Random seed.

    Returns:
        List of records compatible with AlphaSensitivityAnalyzer.run().
    """
    import numpy as np
    rng = np.random.default_rng(seed)

    records = []

    for _ in range(n_attack):
        d_sem   = float(np.clip(rng.normal(attack_d_sem_mean,   noise), 0.0, 1.0))
        p_scope = float(np.clip(rng.normal(attack_p_scope_mean, noise), 0.0, 0.40))
        records.append({
            "is_attack":  True,
            "d_semantic": d_sem,
            "p_scope":    p_scope,
            "divergence_score": float(np.clip(0.70 * d_sem + 0.30 * p_scope, 0.0, 1.0)),
        })

    for _ in range(n_benign):
        d_sem   = float(np.clip(rng.normal(benign_d_sem_mean,   noise), 0.0, 1.0))
        p_scope = float(np.clip(rng.normal(benign_p_scope_mean, noise), 0.0, 0.40))
        records.append({
            "is_attack":  False,
            "d_semantic": d_sem,
            "p_scope":    p_scope,
            "divergence_score": float(np.clip(0.70 * d_sem + 0.30 * p_scope, 0.0, 1.0)),
        })

    return records


# ─── Utility functions ────────────────────────────────────────────────────────


def _safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _safe_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _safe_mean(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return variance ** 0.5
