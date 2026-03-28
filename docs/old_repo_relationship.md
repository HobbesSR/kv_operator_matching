# Relationship to kv_compaction_experiment

This document records what `kv_compaction_experiment` did, what framing it used, and what carries forward into this repo vs. what is new.

---

## What kv_compaction_experiment Did

`kv_compaction_experiment` was an exploration of KV cache compression for transformer inference, with the following focus areas:

- **Qwen 2 experiments**: ran inference with Qwen 2 (7B Instruct) and measured the effect of KV cache compression on output quality
- **Online evidence collection**: collected attention patterns and query statistics during live inference to inform compression decisions
- **Attention Matching direction**: the primary framing was matching the compressed attention distribution to the full-cache distribution — i.e., find a small set of KV pairs such that softmax(q K_hat^T) approximates softmax(q K^T) for live queries
- **Baselines**: recency selection (keep most recent tokens), attention-mass selection (keep highest-attention tokens), and uniform random selection

The repo produced working experiment infrastructure and baseline comparisons, and identified the online evidence framing as the most promising direction. It did not yet have a principled mathematical framework for the compression objective.

---

## Old Framing vs. New Framing

| Aspect | kv_compaction_experiment | kv_operator_matching |
|--------|--------------------------|----------------------|
| Core object | Attention distribution softmax(q K^T) | Attention response operator A_mu(q) = N_mu(q) / Z_mu(q) |
| Factorization | None explicit | N/Z factorization: separate Z (partition) and N (numerator) |
| Composition across blocks | Not clean; softmax re-normalizes | Exact: A_{P||F} = (N_P + N_F) / (Z_P + Z_F) |
| Support restriction | Always drawn from original KV cache | Can be synthetic/merged; beta weighting |
| Objective | Match softmax attention weights | Match response operator A_mu(q) over query bank |
| Surrogate losses | Ad hoc | L_Z, L_N, L_lin (convex in beta for fixed support) |
| Theoretical grounding | Informal | N/Z framing has formal concatenation property; beta-refit is NNLS |
| Scope | Compression as token selection | Compression as measure approximation |

The key conceptual shift: old framing treated KV compression as selecting a subset of tokens. New framing treats it as approximating a measure (operator), where the support can be anything and the weights (betas) are fit coefficients, not indicators.

---

## What We Carry Forward

- **Experiment structure**: the overall pattern of (load model, run prompts, collect evidence, evaluate baselines) is the same
- **Qwen 2 as the first test model**: same model, same basic setup
- **Online evidence collection**: the idea of collecting live query vectors during inference is central to both repos
- **Recency and attention-mass baselines**: both are used as comparison points; their selection logic can be adapted directly
- **Evaluation methodology**: response-error metrics (L2 distance between full-cache and compressed-cache outputs) carry forward

---

## What Is New

- **N/Z factorization**: the explicit decomposition of attention into partition function Z and value numerator N, as separate objects
- **Concatenation property**: the formal proof that N and Z compose additively across KV blocks, enabling streaming updates without refitting
- **Beta weighting**: support points have explicit nonneg coefficients beta_j that are fit to minimize the surrogate objective, rather than indicator weights
- **Surrogate objectives L_Z, L_N, L_lin**: convex in beta for fixed support; enable principled optimization
- **Query bank as a first-class object**: the empirical query bank is now a typed, maintained data structure, not just an ad hoc collection
- **Verification gate**: explicit held-out verification before deploying a compressed representation
- **Operator framing**: the compression target is an operator (function from query to output), not a set of tokens

---

## TODO: Code Patterns to Consider Importing

The following patterns from `kv_compaction_experiment` may be worth adapting when wiring up the Qwen 2 experiment in Phase 2:

- [ ] **Inference loop hook structure**: how attention hook callbacks were registered on the Qwen 2 model to intercept KV cache state and query vectors
- [ ] **Prompt loading and batching**: the prompt loading utilities and batching logic for the evaluation corpus
- [ ] **KV cache extraction**: the pattern for extracting layer-by-layer KV tensors from Hugging Face model outputs (past_key_values handling)
- [ ] **Attention mass computation**: the attention weight accumulation logic used for attention-mass baseline selection
- [ ] **Results logging**: the CSV/JSON logging pattern for per-head, per-layer, per-prompt metrics
- [ ] **Baseline comparison harness**: the side-by-side evaluation loop that runs multiple baselines on the same prompt set

These are patterns to port or adapt, not copy verbatim. The new types (`HeadState`, `CompactRepresentation`, `QueryBank`) will require interface changes throughout.
