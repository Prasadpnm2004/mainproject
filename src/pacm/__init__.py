"""
pacm — Plan-Action Consistency Monitoring core package.

Session 3: Core modules (extractor, action_capture, scorer, middleware).
Session 4: Trust-weighted friction policy (policy.py).
Session 5: Threshold calibration engine (calibration.py).
Session 6: Alpha sensitivity analysis (sensitivity.py).
Session 7: Adaptive attack stress test (adaptive_attack.py);
           error analysis & failure taxonomy (error_analysis.py).
  - pacm.extractor      — plan/intent extraction from reasoning traces
  - pacm.action_capture — action capture and arg scope checks
  - pacm.scorer         — divergence scoring (embedding + rule-based)
  - pacm.policy         — trust-weighted friction policy (F = D × (1.5 − 0.5 × T))
  - pacm.middleware     — pre_tool_hook integration (PACMMiddleware + FrictionPolicy)

Planned (Sessions 5+):
  - pacm.trust          — dynamic trust scoring (registry-backed)

References (PACM_Research_Guide.md §9):
  - Divergence scoring: Section 9.1 step 3
  - Trust augmentation: Section 9.1 step 4
"""

__version__ = "1.0.0"

from src.pacm.extractor import DeclaredIntent, IntentExtractor
from src.pacm.action_capture import CapturedAction, ActionCapturer
from src.pacm.scorer import DivergenceScorer
from src.pacm.policy import (
    FrictionDecision, FrictionPolicy, PolicyResult,
    STANDARD_POLICY, CONSERVATIVE_POLICY, AGGRESSIVE_POLICY,
)
from src.pacm.calibration import ThresholdCalibrator, CalibrationResult
from src.pacm.middleware import PACMMiddleware, PACMAuditEntry

__all__ = [
    "DeclaredIntent", "IntentExtractor",
    "CapturedAction", "ActionCapturer",
    "DivergenceScorer",
    "FrictionDecision", "FrictionPolicy", "PolicyResult",
    "STANDARD_POLICY", "CONSERVATIVE_POLICY", "AGGRESSIVE_POLICY",
    "ThresholdCalibrator", "CalibrationResult",
    "PACMMiddleware", "PACMAuditEntry",
]

