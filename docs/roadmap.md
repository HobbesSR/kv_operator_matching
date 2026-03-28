# Roadmap: kv_operator_matching

This document describes the phased development plan. Phases are ordered by dependency and concreteness. Phase 1 is current work; Phase 4 is long-term motivation only and may evolve significantly before it is attempted.

---

## Phase 1 (Current): Foundations

**Goal**: Get a working, well-documented scaffold that correctly implements the N/Z objectives and enables the first real experiment.

Tasks:
- [x] Repo scaffolding and directory structure
- [x] Core theory documentation (`docs/theory_sketch.md`)
- [x] Roadmap and relationship docs
- [ ] Core objective implementations: `compute_z`, `compute_n`, `compute_response`, `loss_z`, `loss_n`, `loss_lin`, `loss_true_response`
- [ ] Empirical query bank with recency weighting and max-size trimming
- [ ] Fixed-support beta-fit: placeholder NNLS (scipy `nnls` or projected gradient)
- [ ] Baseline stubs: recency, attention-mass, uniform selection
- [ ] Verification gate: holdout response-error check
- [ ] Qwen 2 experiment scaffold: config, argument parsing, inference loop skeleton
- [ ] Basic unit tests for objectives (correctness on trivial cases)

Deliverable: a repo where the objectives are implemented, the query bank runs, and the experiment scaffold is in place — even if end-to-end inference is not yet wired.

---

## Phase 2: Baseline Comparison Pipeline

**Goal**: Run the first real comparison of beta-refit vs. baselines on Qwen 2 with live query evidence.

Tasks:
- [ ] Wire up live query collection hooks in the Qwen 2 inference loop (adapt patterns from `kv_compaction_experiment`)
- [ ] Implement full baseline comparison pipeline:
  - Recency selection (no refit)
  - Attention-mass selection (no refit)
  - Recency selection + beta-refit
  - Attention-mass selection + beta-refit
  - Uniform selection (sanity check)
- [ ] Held-out verification pipeline: split query bank into fit / holdout, evaluate L_true on holdout
- [ ] Results logging and basic plots (response error vs. budget fraction, by baseline)
- [ ] Better support proposal logic:
  - Improve attention-mass selection to use the live query bank rather than uniform-query mass
  - Implement N/Z-aware greedy selection: greedily add support points to minimize L_lin
- [ ] N/Z-aware greedy alternative to OMP: project residuals in N/Z space rather than attention space

Deliverable: a results table showing beta-refit improves over baseline selection for at least one budget fraction, with verification passing on held-out queries.

---

## Phase 3: Merged and Synthetic Support

**Goal**: Move beyond selecting support from the existing KV cache to generating support points that do not correspond to individual tokens.

Tasks:
- [ ] Implement K-means-style support proposal: cluster key vectors, use cluster centroids as support keys, fit values and betas jointly
- [ ] Implement exponential-family merge: given two KV pairs, compute a merged point that matches the first two moments of their contribution to Z and N
- [ ] Empirical tests of response sparsity (Open Question 1 from theory sketch)
- [ ] Empirical tests of spectral decay in the query-key kernel matrix (Open Question 2)
- [ ] Positional encoding interaction study: does RoPE structure in key vectors affect support quality? (Open Question 5)
- [ ] More principled merge proposal logic: conditions under which a good merge exists (toward Open Question 6)

Deliverable: at least one synthetic support method that improves over best Phase 2 baseline at tight budgets.

---

## Phase 4: Progressive Residual and Dynamic Policy (Long-Term)

**Note**: This phase is long-term motivation only. The scope will change significantly based on what is learned in Phases 1-3. Do not plan implementation details here.

High-level direction:
- **Progressive residual representation**: represent the KV state as mu = mu_0 + delta_1 + delta_2 + ... where each delta_t is a small correction added incrementally as new tokens arrive
- **Tree / hierarchy**: organize representations at multiple resolutions; coarse representation for distant tokens, fine for recent; concatenation property enables composability
- **Dynamic resolution policy**: decide at each checkpoint whether to refit, which resolution to use, and how to trade off memory vs. accuracy — based on live query evidence
- **Learned router**: a small model that predicts the best resolution given context signals

Why this is Phase 4 and not earlier: the progressive representation requires the concatenation property to be reliably exploited, which requires Phase 2-3 to establish that the N/Z surrogate fits work well enough to trust in a streaming setting. The dynamic policy requires empirical data on how often refitting is needed and how much it helps, which requires Phase 2 experiments.

---

## Notes on Priority

- Phases 1 and 2 are the research core. Everything else is dependent on them working.
- Phase 3 is the natural extension if Phase 2 shows beta-refit helps.
- Phase 4 is the long-term vision. It should inform design decisions in Phases 1-3 (keep the concatenation property central, keep Z and N as separate objects) but should not drive implementation now.
