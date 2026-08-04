# Research Guide & Proposal

## Plan–Action Consistency Monitoring (PACM): A Lightweight, Protocol-Agnostic Runtime Defense Against Context-Aware Prompt Injection in Tool-Using LLM Agents

*A security-threat-modeling research track for agentic AI systems (MCP / tool-calling agents)*
*Compiled July 2026*

---

## Contents

1. [Executive Summary](#1-executive-summary)
2. [Detailed Problem Statement](#2-detailed-problem-statement)
3. [Why This Problem, Why Now](#3-why-this-problem-why-now)
4. [Research Questions](#4-research-questions)
5. [What Already Exists — The Landscape](#5-what-already-exists--the-landscape)
6. [Base Papers to Study (with live links)](#6-base-papers-to-study-with-live-links)
7. [The Gap: What Nobody Has Solved Yet](#7-the-gap-what-nobody-has-solved-yet)
8. [Your Unique Contribution (Novelty Statement)](#8-your-unique-contribution-novelty-statement)
9. [Proposed Approach & System Design](#9-proposed-approach--system-design)
10. [Evaluation Plan & Metrics](#10-evaluation-plan--metrics)
11. [Week-by-Week Execution Plan (12 Weeks)](#11-week-by-week-execution-plan-12-weeks)
12. [Tooling & Tech Stack](#12-tooling--tech-stack)
13. [Risks, Pitfalls & Mitigations](#13-risks-pitfalls--mitigations)
14. [Target Venues & Publication Strategy](#14-target-venues--publication-strategy)
15. [Full Reference List (Live Links)](#15-full-reference-list-live-links)

---

## 1. Executive Summary

You are going to work on runtime security for tool-using / MCP-connected LLM agents — specifically, detecting and stopping prompt-injection-induced behavioral hijacking while the agent is executing a multi-step task, without needing to retrain a model or rewrite the agent's execution engine.

The specific gap you will target: existing defenses and benchmarks overwhelmingly assume the agent's task is fully specified up front and attacks are context-independent. In real deployments, agents make decisions based on dynamic runtime context (tool outputs, retrieved documents, other agents' messages), and current defenses collapse in this setting — while also generating too many false positives on legitimate context-dependent tasks, which is precisely why real deployments disable these defenses in production.

Your proposed system — **Plan–Action Consistency Monitoring (PACM)** — sits as middleware between an agent's reasoning step and its tool-call execution. It does not classify text for "malicious content" (brittle and adversarially gameable). Instead, it measures whether the agent's own declared intent (extracted from its reasoning trace right before a tool call) is consistent with the tool call it is about to issue, and escalates friction only when a divergence appears immediately after the agent has ingested untrusted external content. This is combined with a lightweight trust score for the MCP server/tool involved, informed by registry-level reputation signals.

This is a genuinely under-explored angle: current work either (a) builds attack/defense benchmarks, (b) proposes heavy architectural rewrites (capability-based execution isolation), or (c) proposes training-time fixes (instruction hierarchies). Almost nobody has proposed a lightweight, model-agnostic, cross-framework runtime monitor evaluated explicitly on the false-positive/utility axis that the newest surveys (2026) flag as the field's biggest blind spot.

---

## 2. Detailed Problem Statement

Large Language Model (LLM) agents are increasingly deployed with the ability to call external tools — web search, email, file systems, code execution, databases — often standardized through the Model Context Protocol (MCP) or similar tool-calling interfaces. This tool access is what makes agents useful, and it is also what makes them dangerous: any content an agent reads from the outside world (a webpage, an email, a tool's output, another agent's message, or even a tool's own name/description) becomes part of its context and can therefore influence its next action.

This is the **indirect prompt injection** problem: an attacker does not need access to the user's prompt at all. They only need to place malicious instructions somewhere the agent will read — inside a document the agent is asked to summarize, inside a webpage the agent visits, inside the metadata of a tool the agent has access to ("tool poisoning"), or inside the response of another agent in a multi-agent system. When the agent reads that content, it may silently adopt the injected instructions as if they came from the user, and then act on them using its real tool access — exfiltrating data, sending unauthorized messages, deleting files, or manipulating financial transactions.

Three things make this problem urgent and still open in 2026:

1. **Scale of exposure** — MCP has become a de facto industry standard for connecting agents to tools; large-scale audits of public MCP registries have found tens of thousands of servers with weak vetting, and thousands with exploitable or misleading tool metadata.
2. **Defenses don't generalize to realistic settings** — almost all published defenses (filtering, delimiters, hierarchical instruction-following, spotlighting) are benchmarked on context-independent tasks where the correct action does not depend on runtime data. The newest 2026 surveys explicitly show these defenses degrade sharply once tasks require the agent to legitimately act on context — which is the common case in real deployments (e.g., "reply to whichever of these three emails is most urgent").
3. **Utility vs. security tradeoff is under-measured** — papers report attack-block rate but rarely report how often a defense breaks a legitimate task. A defense with a 95% block rate that also fails 40% of benign context-dependent tasks is not deployable — yet this axis is barely benchmarked.

**In one sentence:** we do not yet have a lightweight, generalizable, deployment-ready way to detect when an agent's next tool call has been hijacked by injected content, that works across arbitrary agent frameworks and MCP tool combinations, without destroying the agent's ability to legitimately act on the context it was designed to use.

---

## 3. Why This Problem, Why Now

- Agent task horizons are lengthening rapidly (from single tool calls to multi-hour autonomous workstreams), which means more opportunities for an attacker to plant an injection somewhere in a long session, and more compounding damage if one is missed.
- MCP and similar tool-protocols are consolidating as the standard integration layer across the industry — a defense that works at the protocol/middleware level (rather than being tied to one vendor's agent framework) has real, immediate practical value and a clear audience of practitioners who would care about your results.
- 2026 surveys and SoKs (Section 6) explicitly call out context-dependent evaluation and the utility–security tradeoff as open problems — reviewers in this space already recognize this exact gap, which makes your contribution easier to position and defend.
- This topic gives you deep, transferable technical experience: you will build an actual agent harness, instrument tool-calling pipelines, design an evaluation methodology, and reason adversarially — skills equally valuable for research, applied AI engineering, or AI security roles.

---

## 4. Research Questions

**RQ1 — Diagnostic:** How do existing prompt-injection defenses (filtering, spotlighting, hierarchical instruction-following, tool isolation) actually perform when tasks are context-dependent versus context-independent, across multiple agent frameworks and multiple MCP tool combinations, measured jointly on attack-block rate and false-positive rate?

**RQ2 — Core method:** Can a lightweight signal derived purely from an agent's own reasoning trace — the divergence between its declared next-step intent and the tool call it actually issues — reliably flag injection-induced behavioral hijacking, without a separately trained classifier or a rewritten execution engine?

**RQ3 — Trust augmentation:** Does incorporating a lightweight MCP server/tool trust score (derived from registry reputation signals such as publisher verification, metadata consistency, and update history) improve detection precision and reduce false positives compared to using the divergence signal alone?

**RQ4 — Cost/utility tradeoff:** What is the latency, token-cost, and task-completion-rate overhead of this approach compared to heavier existing defenses (e.g., capability-based execution isolation), and where does it sit on the security–utility Pareto frontier?

**RQ5 (stretch) — Adversarial robustness:** Can an adaptive attacker who knows about plan–action consistency monitoring craft injections that keep the agent's stated reasoning consistent with its hijacked action (i.e., attack the monitor itself)? What does this imply for the monitor's own robustness?

---

## 5. What Already Exists — The Landscape

Before you design anything, you need a mental map of the field. Broadly, existing work clusters into five buckets:

**A. Attack taxonomy / foundational attacks** — Establishes that prompt injection (direct and indirect) is a first-class vulnerability class for LLM-integrated applications. Foundational, must-read, but mostly pre-agentic or early-agentic.

**B. Benchmarks & evaluation environments** — AgentDojo, InjecAgent, ToolHijacker, and the newer 2026 context-dependent benchmarks (AgentLure/AgentPI) build environments to measure attack success and defense performance. These are your evaluation infrastructure — you will likely build on top of or alongside one of these rather than from scratch.

**C. Defenses** — Range from lightweight (prompting-based defenses, spotlighting, delimiters — shown to be weak), to structural (CaMeL's capability-based control-flow isolation, hierarchical instruction-following trained into the model), to detection-based (classifiers over tool outputs). Each has a cost: prompting defenses are weak, structural defenses require rewriting the agent's execution model, trained defenses require retraining or fine-tuning access you likely won't have.

**D. MCP / protocol-specific security** — A 2026 wave of papers specifically threat-models MCP: registry-level trust issues, tool poisoning, client-side validation gaps. This is your most directly relevant, most recent literature — read this bucket most carefully.

**E. Multi-agent / cross-agent protocol threats** — Newer work extends the threat model to agent-to-agent (A2A) communication and multi-agent systems, where one compromised or malicious agent can inject into another. Relevant for your stretch goals and future-work framing.

---

## 6. Base Papers to Study (with Live Links)

Read in this order: Bucket A and B first (understand the problem + evaluation methodology), then D (your most direct foundations), then C (understand what "heavy" defenses look like and why you're avoiding that path), then E as time allows.

### 6.1 Foundational Attack Papers (read first)

**Ignore Previous Prompt: Attack Techniques for Language Models** — *Perez & Ribeiro, 2022*
https://arxiv.org/abs/2211.09527
The original framing of prompt injection as an attack class. Short, foundational, read in one sitting.

**Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection** — *Greshake, Abdelnabi, Mishra, Endres, Holz & Fritz, ACM AISec 2023*
https://dl.acm.org/doi/10.1145/3605764.3623985
The paper that defines indirect prompt injection — the exact attack class your whole project is about. Non-negotiable read.

**Prompt Injection Attack Against LLM-Integrated Applications** — *Liu et al., 2023*
https://arxiv.org/abs/2306.05499
Systematizes injection attacks against real deployed LLM-integrated apps; good source of real-world attack patterns.

### 6.2 Benchmarks & Evaluation Environments (your evaluation backbone)

**AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents** — *Debenedetti, Zhang, Balunović, Beurer-Kellner, Fischer & Tramèr, NeurIPS 2024*
https://arxiv.org/abs/2406.13352
THE reference benchmark: 97 realistic tool-calling tasks, 629 security test cases, extensible to new attacks/defenses. You will very likely build your evaluation harness on top of this or use it as a direct baseline comparison.

**InjecAgent: Benchmarking Indirect Prompt Injections in Tool-Integrated Large Language Model Agents** — *Zhan, Liang, Ying & Kang, 2024*
https://arxiv.org/abs/2403.02691
Another core benchmark specifically for tool-integrated agents; good complementary attack surface to AgentDojo.

**Prompt Injection Attack to Tool Selection in LLM Agents (ToolHijacker)** — *2025*
https://arxiv.org/abs/2504.19793
Shows that both prevention- and detection-based defenses fail against attacks targeting the tool-selection step specifically — directly relevant to your plan–action divergence idea, since tool selection is exactly the action you're monitoring.

**The Landscape of Prompt Injection Threats in LLM Agents: From Taxonomy to Analysis (SoK) + AgentPI benchmark** — *Wang et al., 2026*
https://arxiv.org/abs/2602.10453
Critical 2026 paper: a systematization-of-knowledge that explicitly identifies context-dependent tasks as the field's blind spot and introduces AgentPI to address it. Read this closely — it is the closest existing work to your framing, and you must clearly differentiate from it (see Section 8).

**ARGUS: Defending LLM Agents Against Context-Aware Prompt Injection (+ AgentLure benchmark)** — *Weng et al., 2026*
https://arxiv.org/abs/2605.03378
The most recent, most directly competing paper: introduces context-aware attacks and a defense. Read this in full and articulate precisely how PACM differs (runtime/plan-trace based vs. their approach, lighter-weight, cross-framework).

### 6.3 MCP-Specific Security (your most recent, most relevant bucket)

**Model Context Protocol Threat Modeling and Analyzing Vulnerabilities to Prompt Injection with Tool Poisoning** — *Huang et al., 2026*
https://arxiv.org/abs/2603.22489
STRIDE/DREAD threat model across 5 MCP components; empirically tests 7 major MCP clients against tool-poisoning attacks. Gives you a ready-made threat taxonomy to adopt and cite.

**Model Context Protocol (MCP): Landscape, Security Threats, and Future Research Directions** — *Hou, Zhao, Wang & Wang, 2025*
https://arxiv.org/abs/2503.23278
Comprehensive MCP threat taxonomy across the server lifecycle (creation, operation, update) and 16 threat scenarios across 4 attacker types. Good structural reference for your threat model chapter.

**A First Look at the Security Issues in the Model Context Protocol Ecosystem** — *Li & Gao, 2025/2026*
https://arxiv.org/abs/2510.16558
Empirical study of 67,057 real MCP servers across six registries; quantifies registry-level trust failures. This is your direct evidence base for the "server trust score" component of PACM — cite this to justify why trust scoring matters.

**Model Context Protocol (MCP) at First Glance: Studying the Security and Maintainability of MCP Servers** — *Hasan et al., 2026*
https://arxiv.org/abs/2506.13538
Large-scale empirical study of 1,899 open-source MCP servers; useful for building your own "server reputation" dataset/features.

**Enterprise-Grade Security for the Model Context Protocol (MCP): Frameworks and Mitigation Strategies** — *2025*
https://arxiv.org/abs/2504.08623
Defense-in-depth framework proposal; useful for positioning — shows the field is currently proposing heavy, multi-layered enterprise controls, which is the "heavy" alternative your lightweight middleware should compare against.

### 6.4 Defenses (understand the alternatives you are positioning against)

**CaMeL: Defeating Prompt Injections by Design** — *Debenedetti, Carlini, Nasr & Tramèr, 2025*
https://arxiv.org/abs/2503.18813
The strongest structural defense in the literature — uses a capability-based control-flow architecture that separates the LLM's reasoning from tool execution. Very effective but requires rewriting the agent's execution engine. Key point of contrast: PACM aims to be a middleware layer that doesn't require this level of architectural rewrite.

**The Instruction Hierarchy: Training LLMs to Prioritize Privileged Instructions** — *Wallace, Xiao, Leike, Weng, Heidecke & Beutel, 2024*
https://arxiv.org/abs/2404.13208
OpenAI's training-time defense — teaches the model itself to weigh system/user instructions above tool/content instructions. Relevant contrast: this needs model-level training access, which you likely won't have; your defense should work purely at the orchestration layer, model-agnostic.

**Attention Is All You Need to Defend Against Indirect Prompt Injection Attacks in LLMs** — *2026*
https://arxiv.org/abs/2512.08417
A mechanistic, attention-based detection approach. Good to understand as a white-box alternative to your black-box, trace-based approach.

### 6.5 Surveys / SoKs (read for structure and citation coverage)

**A Survey on Agentic Security: Applications, Threats and Defenses** — *2025/2026*
https://arxiv.org/abs/2510.06445
Broad, recent survey — use its taxonomy and reference list to make sure your related-work section is comprehensive.

**A Systematic Survey of Security Threats and Defenses in LLM-Based AI Agents: A Layered Attack Surface Framework** — *2026*
https://arxiv.org/abs/2604.23338
Layered attack-surface framework covering the full agent stack; useful structural template for positioning where in the stack PACM sits (orchestration/middleware layer).

**From Prompt Injections to Protocol Exploits: Threats in LLM-Powered AI Agent Workflows** — *Ferrag et al., 2025/2026*
https://arxiv.org/abs/2506.23260
Unifies host-to-tool and agent-to-agent threat models across 30+ attack techniques. Good for the multi-agent stretch-goal framing (RQ5).

**Agent-Fence: Mapping Security Vulnerabilities Across Deep Research Agents** — *2026*
https://arxiv.org/abs/2602.07652
Applies threat modeling specifically to "deep research" style long-horizon agents — relevant since your motivating scenario overlaps heavily with this.

---

## 7. The Gap: What Nobody Has Solved Yet

After reading the above, you will notice a consistent pattern across the field:

- Benchmark papers (AgentDojo, InjecAgent, AgentPI, AgentLure) measure the problem extremely well but are not, themselves, deployable defenses.
- Structural defenses (CaMeL) are strong but require rewriting the agent's execution engine into a constrained, capability-based interpreter — a large engineering lift, hard to retrofit onto an arbitrary existing agent framework or MCP client.
- Training-time defenses (instruction hierarchy) require model fine-tuning access most researchers and almost all developers don't have, and don't transfer across model providers.
- MCP-specific papers are overwhelmingly diagnostic (threat modeling, empirical prevalence studies) rather than proposing and evaluating a concrete, lightweight runtime mitigation.
- Almost no paper jointly optimizes for and reports both attack-block rate AND false-positive rate on legitimate context-dependent tasks — the exact axis the 2026 SoK (Section 6.2) flags as critically under-evaluated.

**The gap, stated precisely:** there is no lightweight, model-agnostic, protocol-level middleware that (a) works across arbitrary agent frameworks without rewriting their execution engine, (b) requires no model fine-tuning, (c) is explicitly evaluated on the security–utility tradeoff for context-dependent tasks, and (d) incorporates external, empirically-grounded trust signals about the tools/servers involved (rather than only looking at the text content of a single interaction).

---

## 8. Your Unique Contribution (Novelty Statement)

This is the paragraph you should be able to say out loud, unprompted, in a lab meeting or to a reviewer. Refine it as you go, but this is your starting position:

> "I propose Plan–Action Consistency Monitoring (PACM): a runtime, model-agnostic middleware that sits between an agent's reasoning step and its tool-call execution. Instead of classifying content for malicious text (brittle and gameable) or rewriting the agent's execution engine into a constrained interpreter (powerful but heavy and hard to retrofit), PACM computes a divergence score between the agent's own declared next-step intent and the tool call it is about to issue, with heightened sensitivity in the window immediately following ingestion of untrusted external content. This signal is combined with a lightweight, empirically-derived trust score for the specific MCP server/tool involved. Unlike prior work, I evaluate this jointly on attack-block rate AND task-completion rate on legitimate context-dependent tasks, across multiple agent frameworks and multiple LLM backbones — directly targeting the utility–security tradeoff that current literature under-reports."

### What makes each piece defensible as novel

- **Signal source:** prior detection defenses inspect tool outputs / external content for malicious text. PACM instead inspects the agent's own reasoning-to-action transition — a signal that is largely unexplored as a primary detection mechanism at the middleware layer.
- **Deployment model:** no fine-tuning, no execution-engine rewrite — designed explicitly to be a drop-in wrapper compatible with multiple agent frameworks (LangGraph, a custom agent loop, AutoGen) and MCP clients, which none of the structural defenses target.
- **Trust augmentation:** grounding the trust score in the empirical registry findings from Li & Gao (2510.16558) and Hasan et al. (2506.13538) rather than inventing an ungrounded heuristic — gives your defense a defensible empirical basis.
- **Evaluation axis:** explicit, joint reporting of attack-block rate and false-positive/utility-loss rate on context-dependent tasks — directly answering the gap the AgentPI/ARGUS papers (2602.10453, 2605.03378) identify but do not themselves fully close with a lightweight solution.

**Important:** read ARGUS and the AgentPI SoK very carefully before finalizing your framing — they are the closest prior work and you must be precise about what they do differently from you (they focus on benchmark construction and, in ARGUS's case, a specific defense; your contribution should be positioned as complementary or a lighter-weight alternative with a different evaluation emphasis, not a restatement of their work).

---

## 9. Proposed Approach & System Design

### 9.1 High-level architecture

PACM sits as an interceptor between the agent's LLM reasoning output and the tool-execution layer. For every proposed tool call, it performs four steps before the call is allowed to execute:

1. **Plan extraction** — parse the agent's most recent reasoning trace (its stated "next step" / "why I'm calling this tool") into a structured intent representation (target tool, target parameters, expected effect).
2. **Action capture** — capture the actual tool call the agent is about to issue (tool name, arguments).
3. **Divergence scoring** — compute a divergence score between declared intent and actual action using (a) embedding similarity between stated goal and action effect, and (b) rule-based constraint checks (e.g., does the tool/argument fall outside the scope of the original user task; did the divergence appear immediately after ingesting new untrusted content).
4. **Trust-weighted friction** — combine the divergence score with the MCP server/tool trust score to decide the action: allow, require re-confirmation (human-in-the-loop or a secondary LLM check), sandbox execution, or block.

### 9.2 Why this design avoids the weaknesses of prior approaches

- No text classifier to train — avoids the adversarial-gameability problem that plagues prompt-based and classifier-based filters.
- No execution-engine rewrite — avoids the heavy engineering lift and poor portability of capability-based isolation like CaMeL.
- No fine-tuning — avoids requiring model-provider-level access, keeping the method usable by any developer on any model.
- Adaptive friction, not blanket blocking — directly targets the utility-loss problem that makes existing defenses undeployable in practice.

### 9.3 Testbed you will need to build

You will need a controlled agent environment with real tool/MCP calls to run experiments in. Recommended approach: extend AgentDojo's task suite (Section 6.2) with your own MCP-wrapped tool servers (email, file system, calendar, search) so you can inject at the protocol layer, not just the text layer. Build 2–3 different agent "hosts" (e.g., a LangGraph-based agent, a custom minimal agent loop, and if time allows an AutoGen multi-agent setup) so you can demonstrate cross-framework generalizability — this is one of your key differentiators, so don't skip it even under time pressure.

---

## 10. Evaluation Plan & Metrics

| Metric | Definition / Purpose |
|---|---|
| Attack Block Rate (ABR) | % of injected attack attempts successfully prevented from executing their malicious tool call. |
| Benign Task Success Rate (BTSR) | % of legitimate, context-dependent tasks that still complete correctly with PACM enabled — your key differentiator metric. |
| False Positive Rate (FPR) | % of legitimate tool calls incorrectly flagged/blocked/escalated. |
| Latency Overhead | Added wall-clock time per tool call vs. an undefended baseline. |
| Token/Cost Overhead | Additional LLM tokens consumed (if using a secondary LLM check step) vs. baseline. |
| Cross-Framework Consistency | Variance in ABR/BTSR/FPR across the 2–3 different agent frameworks tested — evidence of generalizability. |
| Comparative Baselines | Undefended agent; prompting-based defenses (spotlighting/delimiters); tool-filter isolation (AgentDojo baseline); CaMeL (if reproducible) — report ABR/BTSR/FPR for each. |

Run all experiments on both context-independent tasks (to confirm you're not regressing on the "easy" case) and context-dependent tasks (AgentPI/AgentLure-style, or your own extensions) so you can explicitly show the gap other defenses have on the latter and how PACM narrows it.

---

## 11. Week-by-Week Execution Plan (12 Weeks)

| Phase | Deliverables |
|---|---|
| Weeks 1–2 | Deep-read Section 6 papers (order given). Set up environment: Python, an agent framework (LangGraph), AgentDojo installed and running end-to-end. Reproduce 2–3 existing attacks/defenses from AgentDojo as a sanity check. |
| Weeks 3–4 | Build your own MCP-wrapped toolset (email/file/calendar/search) and 2nd agent framework (custom loop or AutoGen). Design and implement your attack corpus: context-independent + context-dependent injections, tool-poisoning payloads, registry-metadata poisoning. |
| Weeks 5–6 | Implement PACM core: plan extraction module, action capture hooks, divergence scoring (start with embedding similarity + simple rules). |
| Week 7 | Implement trust-scoring module using features inspired by Li & Gao (2510.16558) and Hasan et al. (2506.13538): publisher verification, metadata consistency, update recency, etc. (Synthesize a small trust-labeled dataset since a live registry crawl may be out of scope.) |
| Week 8 | Integrate divergence + trust into the friction policy (allow / confirm / sandbox / block). Tune thresholds on a held-out validation split. |
| Weeks 9–10 | Full evaluation run across both agent frameworks, both task types (context-dependent/independent), and all baselines. Collect ABR, BTSR, FPR, latency, cost. |
| Week 11 | Ablations (divergence-only vs. trust-only vs. combined), adaptive-attacker stress test (RQ5, best-effort), error analysis on false positives/negatives. |
| Week 12+ | Write-up: paper draft, figures, repo cleanup, README, prepare arXiv preprint and target-venue submission (Section 14). |

---

## 12. Tooling & Tech Stack

- Agent frameworks: LangGraph, and one of AutoGen / a hand-rolled minimal agent loop (for cross-framework generalizability claims).
- Protocol layer: Model Context Protocol (MCP) Python/TypeScript SDKs for building your own toy MCP servers to control the attack surface precisely.
- Evaluation backbone: AgentDojo (extend its task/attack registry rather than building from zero).
- Embeddings/similarity: any open embedding model (e.g., sentence-transformers) for the divergence scoring module — keep this lightweight and swappable.
- LLM backbones to test generalizability across: at least 2 different model providers, since injection susceptibility and reasoning-trace style differ by model.
- Experiment tracking: a simple structured logging setup (JSON lines per run) is enough — don't over-engineer this.

---

## 13. Risks, Pitfalls & Mitigations

| Risk | Mitigation |
|---|---|
| Scope creep — trying to also fix CaMeL-level guarantees | Explicitly scope PACM as a lightweight complement, not a replacement, for structural defenses. State this in your framing early. |
| Reproducing baselines takes longer than expected | Timebox baseline reproduction to 1.5 weeks; if a baseline is unreproducible from the paper alone, cite the reported numbers and clearly mark them as "reported, not reproduced." |
| Your divergence signal turns out weak/noisy on its own | This is a valid and publishable negative/partial result if well analyzed — pivot framing toward "when does trace-based divergence work and when does it fail" rather than abandoning the project. |
| Adaptive attacker breaks your monitor easily (RQ5) | Treat this as a contribution, not a failure — characterizing the monitor's own attack surface is valuable and expected in security research; report it honestly. |
| Novelty overlap with ARGUS/AgentPI if you don't read them closely enough | Re-read Section 6.2 papers before finalizing your final paper framing; make the differentiation paragraph in Section 8 airtight and cite them generously as concurrent/prior work. |

---

## 14. Target Venues & Publication Strategy

Post a preprint to arXiv (cs.CR / cs.AI) as soon as you have a complete draft — this timestamps your contribution and is standard practice in this fast-moving subfield.

**Primary targets:**
- IEEE S&P, USENIX Security, ACM CCS, NDSS — top security venues; strong fit given the security-empirical nature of the work, though competitive with longer review cycles.
- NeurIPS / ICML Datasets & Benchmarks track or a dedicated workshop (e.g., agent-security or agentic-AI workshops at ICLR/NeurIPS/ICSE) — excellent fit, faster cycle, and the field is actively running such workshops right now (see the ICSE 2026 AGENT workshop referenced in Section 6).
- ACM AISec Workshop (co-located with CCS) — the same venue that published the foundational Greshake et al. indirect prompt injection paper; a very natural home for this exact topic.

**Secondary / fallback:**
- arXiv preprint alone is a legitimate first outcome for a 2–3 month project — don't let "must get into a top-tier venue" block you from finishing and publishing something citable on your timeline.
- Domain-specific workshops on LLM agent safety/security attached to major ML conferences typically have rolling or workshop-specific deadlines — track these actively as you approach Week 10–12.

---

## 15. Full Reference List (Live Links)

- Perez & Ribeiro (2022) — Ignore Previous Prompt — https://arxiv.org/abs/2211.09527
- Greshake et al. (2023) — Not What You've Signed Up For — https://dl.acm.org/doi/10.1145/3605764.3623985
- Liu et al. (2023) — Prompt Injection Attack Against LLM-Integrated Applications — https://arxiv.org/abs/2306.05499
- Debenedetti et al. (2024) — AgentDojo — https://arxiv.org/abs/2406.13352
- Zhan et al. (2024) — InjecAgent — https://arxiv.org/abs/2403.02691
- ToolHijacker (2025) — https://arxiv.org/abs/2504.19793
- Wang et al. (2026) — Landscape of Prompt Injection Threats (SoK) + AgentPI — https://arxiv.org/abs/2602.10453
- Weng et al. (2026) — ARGUS + AgentLure — https://arxiv.org/abs/2605.03378
- Huang et al. (2026) — MCP Threat Modeling (STRIDE/DREAD) — https://arxiv.org/abs/2603.22489
- Hou et al. (2025) — MCP Landscape, Security Threats — https://arxiv.org/abs/2503.23278
- Li & Gao (2025/2026) — A First Look at MCP Ecosystem Security — https://arxiv.org/abs/2510.16558
- Hasan et al. (2026) — MCP at First Glance — https://arxiv.org/abs/2506.13538
- Enterprise-Grade Security for MCP (2025) — https://arxiv.org/abs/2504.08623
- Debenedetti et al. (2025) — CaMeL — https://arxiv.org/abs/2503.18813
- Wallace et al. (2024) — The Instruction Hierarchy — https://arxiv.org/abs/2404.13208
- Attention Is All You Need to Defend (2026) — https://arxiv.org/abs/2512.08417
- A Survey on Agentic Security (2025/2026) — https://arxiv.org/abs/2510.06445
- Layered Attack Surface Framework (2026) — https://arxiv.org/abs/2604.23338
- Ferrag et al. (2025/2026) — From Prompt Injections to Protocol Exploits — https://arxiv.org/abs/2506.23260
- Agent-Fence (2026) — https://arxiv.org/abs/2602.07652

*Note: read every abstract yourself before committing — arXiv IDs and versions update frequently, and a couple of these are recent enough (2026) that they may still be under revision. Always pull the latest version from the abstract page link, not a cached PDF.*
