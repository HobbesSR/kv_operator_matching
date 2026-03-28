# Relationship to kv_compaction_experiment and the Attention Matching Paper

This document records what `kv_compaction_experiment` did, what framing it used, and what carries forward into this repo vs. what is new. It also captures key details from the Attention Matching paper that directly inform this work.

---

## The Attention Matching Paper

**Citation**: "Fast KV Compaction via Attention Matching", Zweiger, Fu, Guo, Kim (MIT, 2025). arXiv:2602.16284. Code: https://github.com/adamzweiger/compaction.

**Reference implementation location in old repo**: `third_party/compaction_paper/`

**Reference digest in old repo**: `docs/reference/paper_digest.md`

### What the paper proves and does

The paper establishes that KV cache compaction can be cast as an optimization over $(C_k, \beta, C_v)$ — compact keys, per-key scalar biases (in log space), and compact values — such that the compacted representation preserves:
1. **Attention mass** (= our $Z$): $\sum_j \exp(q (C_k)^\top_j + \beta_j) \approx \sum_i \exp(q K_i^\top)$
2. **Attention output** (= our $N/Z$): the local softmax-weighted value sum

The key insight (Appendix A.2): matching these two quantities is **sufficient for preservation under arbitrary future concatenation**. This is the same concatenation property at the heart of this repo's N/Z framing.

### The paper's decomposition

Given a fixed subset of keys $C_k \subseteq K$, the joint problem decomposes into two closed-form steps:

- **$\beta$ fit (NNLS on mass)**: solve $\min_{w \geq 0} \| A w - m \|_2^2$ where $A_{ij} = \exp(q_i (C_k)_j^\top)$ and $m_i = Z_\mu(q_i)$. Set $\beta_j = \log(w_j)$. In our notation: $w_j = \beta_j^{\text{ours}}$.
- **$C_v$ fit (LS on output)**: with $C_k$ and $\beta$ fixed, solve ordinary least squares for $C_v$ minimizing $\| X C_v - Y \|_F^2$ where $Y_i = A_\mu(q_i)$.

Key selection methods:
- **HighestAttnKeys**: retain keys with highest RMS attention mass under reference queries. Fast and effective.
- **OMP**: greedy residual minimization on the mass objective. Equivalent to greedy support selection under $L_Z$. Stronger but ~100–500× slower.

### Reference query strategies

The paper considers three strategies for constructing the reference query set (the "query bank" in our framing):

| Strategy | How | Runtime (60k tokens, H200) | Notes |
|---|---|---|---|
| `context-prefill` | Prefill context alone; extract query vectors | Fast | Slightly weaker than repeat-prefill |
| `repeat-prefill` | Prefill "Context. Repeat it. Context." | ~8s | Paper default; good balance |
| `self-study` | Sample synthetic Q&A pairs; extract queries | ~139s | Best quality; expensive |

The paper uses `repeat-prefill` as the primary source and `self-study` to supplement. Random Gaussian queries work but are worse. This is directly relevant to our query bank design: `repeat-prefill` is the lowest-cost baseline strategy worth implementing first.

### Key empirical findings

- Query generation is the dominant runtime cost (7–139s), not fitting (4s total for $\beta$ + $C_v$).
- OMP is strongest but ~100× slower than HighestAttnKeys; OMP-fast (batch key selection) recovers most of the gap.
- Nonuniform head budgets (precomputed per model) improve results substantially at no inference-time cost.
- Per-layer "on-policy" queries (recomputing queries with earlier layers already compacted) give consistent small improvements.
- The paper achieves up to 50× compaction with little quality loss on QuALITY and LongHealth.

### What this repo is not trying to do

This repo is **not** trying to replicate the paper's best results (AM-OMP with self-study + nonuniform budgets). That is explicitly a non-goal for Phase 1 and 2.

The paper is used as:
- An objective definition and mathematical framing (we generalize it)
- A source of baseline methodology (HighestAttnKeys, NNLS $\beta$ fit, LS $C_v$ fit)
- A performance reference for the "paper method" control condition

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

| Aspect | Attention Matching paper | kv_compaction_experiment | kv_operator_matching |
|--------|--------------------------|--------------------------|----------------------|
| Core object | Attention output + mass per head | Attention distribution softmax(q K^T) | Response operator A_mu(q) = N_mu(q) / Z_mu(q) |
| Factorization | Implicit: β enters as log(w) bias | None explicit | Explicit N/Z: separate Z (partition) and N (numerator) |
| Composition across blocks | Proved in Appendix A.2; motivates mass matching | Not clean; softmax re-normalizes | First-class: A_{P||F} = (N_P + N_F) / (Z_P + Z_F) |
| Support restriction | Subset of original keys Ck ⊆ K | Always drawn from original cache | Can be synthetic/merged; β are fit coefficients |
| β parameterization | β_j = log(w_j), fit via NNLS on Z | Not present | β_j ≥ 0 directly; same math as paper's w_j |
| Cv / value fitting | Separate LS step after β | Not present | Unified in L_N; equivalent to paper's Cv fit |
| Reference queries | Separate prefill pass (context/repeat/self-study) | Ad hoc from live inference | Live query bank; online collection is the hypothesis |
| Surrogate losses | L_Z then L_N sequential (paper's two-step) | Ad hoc | L_Z + L_N jointly (L_lin); equivalent, potentially tighter |
| Theoretical grounding | Concatenation property proved; NNLS for β | Informal | N/Z extends paper's framing; formal in same sense |
| Scope | Single-context compaction; offline query gen | Online experiment scaffold | Online evidence → operator approximation |

The key conceptual shift from the old experiment repo: the old repo approximated attention matching informally, without the paper's decomposition. This repo adopts the paper's decomposition as a foundation and generalizes it toward online evidence collection, synthetic support, and progressive structure.

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
