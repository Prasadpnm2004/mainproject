"""
src/pacm/scorer.py

PACM Divergence Scorer — Session 3

Computes a continuous divergence score D in [0.0, 1.0] between a
DeclaredIntent and a CapturedAction.

Background (PACM_Research_Guide.md §9.1 step 3):
  D=0.0 → intent and action are perfectly aligned (no injection signal).
  D=1.0 → maximum divergence (intent completely inconsistent with action).

Scoring strategy (two components combined):

  1. Semantic similarity (primary signal):
       Cosine similarity between sentence embeddings of:
         - intent_text = declared_goal + " using " + target_tool
         - action_text = action_description
       sim ∈ [−1, 1] → normalized to [0, 1].
       D_semantic = 1.0 − sim_normalized

       Embedding backend priority:
         a. sentence-transformers (if installed): all-MiniLM-L6-v2
         b. Lightweight TF-IDF + cosine fallback (no extra deps)

  2. Rule-based scope penalty (supplementary signal):
       If arg_scope_flags is non-empty (e.g. external_domain detected),
       add a fixed penalty P_scope per flag (default: +0.20 per flag, cap 0.40).
       Captures tool-argument hijacking that semantic similarity may miss
       when the injection is subtle (e.g. only the 'to' address changed).

  Final score:
       D = clip(alpha * D_semantic + (1-alpha) * P_scope, 0.0, 1.0)
       Default alpha = 0.70 (semantic dominates; scope is supplementary).

  Tool name mismatch (fast path):
       If captured tool_name != declared target_tool AND target_tool != "":
       set D_semantic = 0.80 immediately (tool name mismatch is a strong signal).

Usage:
    from src.pacm.scorer import DivergenceScorer
    score = DivergenceScorer().score(intent, action)
    print(score)   # 0.0 = aligned, 1.0 = fully divergent
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

from src.pacm.extractor import DeclaredIntent
from src.pacm.action_capture import CapturedAction

logger = logging.getLogger(__name__)


# ─── Embedding backends ───────────────────────────────────────────────────────


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _tfidf_embed(text: str, vocab: dict[str, int]) -> list[float]:
    """
    Minimal TF-IDF bag-of-words vector for a text against a shared vocab.
    No external libraries required.
    """
    tokens = text.lower().split()
    vec = [0.0] * len(vocab)
    for tok in tokens:
        if tok in vocab:
            vec[vocab[tok]] += 1.0
    # L2 normalize
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


class _TFIDFBackend:
    """
    Zero-dependency TF-IDF cosine similarity backend.
    Vocabulary is built lazily from the first pair of texts.
    """

    def similarity(self, text_a: str, text_b: str) -> float:
        tokens_a = set(text_a.lower().split())
        tokens_b = set(text_b.lower().split())
        vocab_words = sorted(tokens_a | tokens_b)
        vocab = {w: i for i, w in enumerate(vocab_words)}
        if not vocab:
            return 1.0  # both empty → treat as identical
        vec_a = _tfidf_embed(text_a, vocab)
        vec_b = _tfidf_embed(text_b, vocab)
        return _cosine(vec_a, vec_b)


class _SentenceTransformersBackend:
    """
    sentence-transformers backend using all-MiniLM-L6-v2.
    Loaded lazily so tests without the library still run via fallback.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model: Any = None

    def _load(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self._model = SentenceTransformer(self._model_name)
            logger.info("Loaded sentence-transformer: %s", self._model_name)

    def similarity(self, text_a: str, text_b: str) -> float:
        self._load()
        embeddings = self._model.encode([text_a, text_b], normalize_embeddings=True)
        # Clamp to [0.0, 1.0]: cosine similarity can be negative for dissimilar
        # texts even with unit-norm embeddings. Negative values are semantically
        # "maximally dissimilar" — we treat them as 0.0 (D_semantic will be 1.0).
        raw = float(_cosine(embeddings[0].tolist(), embeddings[1].tolist()))
        return max(0.0, min(1.0, raw))


def _get_embedding_backend(prefer_transformers: bool = True) -> Any:
    """
    Return the best available embedding backend.
    Falls back to TF-IDF if sentence-transformers is not installed.
    """
    if prefer_transformers:
        try:
            import sentence_transformers  # noqa: F401
            return _SentenceTransformersBackend()
        except ImportError:
            logger.info(
                "sentence-transformers not installed — using TF-IDF fallback scorer. "
                "Install with: pip install sentence-transformers"
            )
    return _TFIDFBackend()


# ─── Divergence scorer ────────────────────────────────────────────────────────


class DivergenceScorer:
    """
    Computes plan-action divergence score D ∈ [0.0, 1.0].

    Args:
        alpha: Weight of semantic similarity vs. rule-based scope penalty.
               0.70 by default (semantic dominates).
        scope_penalty_per_flag: Added to D per scope flag (default: 0.20).
        max_scope_penalty: Cap on total scope penalty (default: 0.40).
        tool_mismatch_score: D_semantic value when tool names diverge (default: 0.80).
        prefer_transformers: Try sentence-transformers first; fallback to TF-IDF.
    """

    def __init__(
        self,
        alpha: float = 0.70,
        scope_penalty_per_flag: float = 0.20,
        max_scope_penalty: float = 0.40,
        tool_mismatch_score: float = 0.80,
        prefer_transformers: bool = True,
    ) -> None:
        self.alpha = alpha
        self.scope_penalty_per_flag = scope_penalty_per_flag
        self.max_scope_penalty = max_scope_penalty
        self.tool_mismatch_score = tool_mismatch_score
        self._backend: Any = _get_embedding_backend(prefer_transformers)

    def score(
        self,
        intent: DeclaredIntent,
        action: CapturedAction,
    ) -> float:
        """
        Compute divergence score between intent and action.

        Args:
            intent: DeclaredIntent from IntentExtractor.
            action: CapturedAction from ActionCapturer.

        Returns:
            float in [0.0, 1.0]. 0=aligned, 1=fully divergent.
        """
        # ── Fast path: tool name mismatch ─────────────────────────────────────
        if (
            intent.target_tool
            and intent.target_tool.lower() != action.tool_name.lower()
        ):
            d_semantic = self.tool_mismatch_score
            logger.debug(
                "Tool name mismatch: intent=%s action=%s → D_semantic=%.2f",
                intent.target_tool,
                action.tool_name,
                d_semantic,
            )
        else:
            # ── Semantic similarity ────────────────────────────────────────────
            intent_text = self._build_intent_text(intent)
            action_text = action.action_description

            if not intent_text.strip() or not action_text.strip():
                # No text to compare → cannot determine divergence
                d_semantic = 0.5
            else:
                sim = self._backend.similarity(intent_text, action_text)
                # sim in [-1,1] → normalize to [0,1]
                sim_norm = (sim + 1.0) / 2.0
                d_semantic = 1.0 - sim_norm

            logger.debug(
                "Semantic D=%.3f | intent=%s... | action=%s...",
                d_semantic,
                intent_text[:40],
                action_text[:40],
            )

        # ── Rule-based scope penalty ───────────────────────────────────────────
        n_flags = len(action.arg_scope_flags)
        p_scope = min(
            n_flags * self.scope_penalty_per_flag,
            self.max_scope_penalty,
        )
        if n_flags:
            logger.debug(
                "Scope flags: %s → P_scope=%.2f", action.arg_scope_flags, p_scope
            )

        # ── Combined score ─────────────────────────────────────────────────────
        d_raw = self.alpha * d_semantic + (1.0 - self.alpha) * p_scope
        d_final = max(0.0, min(1.0, d_raw))

        logger.debug(
            "D_final=%.3f (alpha=%.2f * D_sem=%.3f + %.2f * P_scope=%.2f)",
            d_final, self.alpha, d_semantic, 1 - self.alpha, p_scope,
        )
        return d_final

    def score_from_texts(
        self,
        intent_text: str,
        action_text: str,
        scope_flags: Optional[dict[str, str]] = None,
    ) -> float:
        """
        Low-level scoring from plain text strings (for testing / debugging).

        Args:
            intent_text: Natural language description of intended action.
            action_text: Natural language description of proposed action.
            scope_flags: Optional scope flag dict from ActionCapturer.

        Returns:
            float in [0.0, 1.0].
        """
        sim = self._backend.similarity(intent_text, action_text)
        sim_norm = (sim + 1.0) / 2.0
        d_semantic = 1.0 - sim_norm

        n_flags = len(scope_flags) if scope_flags else 0
        p_scope = min(n_flags * self.scope_penalty_per_flag, self.max_scope_penalty)

        return max(0.0, min(1.0, self.alpha * d_semantic + (1.0 - self.alpha) * p_scope))

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_intent_text(self, intent: DeclaredIntent) -> str:
        """Build a single text string representing the declared intent."""
        parts = []
        if intent.declared_goal:
            parts.append(intent.declared_goal)
        if intent.target_tool:
            parts.append(f"using {intent.target_tool}")
        if intent.expected_params_summary:
            parts.append(f"with {intent.expected_params_summary}")
        return " ".join(parts)
