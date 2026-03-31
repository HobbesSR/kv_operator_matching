# Roadmap: kv_operator_matching

This document describes the phased development plan. Phases are ordered by dependency and concreteness. Phase 1 is current work; Phase 4 is long-term motivation only and may evolve significantly before it is attempted.

---

## Phase 1 (Complete): Foundations

**Goal**: Get a working, well-documented scaffold that correctly implements the N/Z objectives and enables the first real experiment.

Tasks:
- [x] Repo scaffolding and directory structure
- [x] Core theory documentation (`docs/theory_sketch.md`)
- [x] Roadmap and relationship docs
- [x] Core objective implementations: `compute_z`, `compute_n`, `compute_response`, `loss_z`, `loss_n`, `loss_lin`, `loss_true_response`
- [x] Empirical query bank with recency weighting and max-size trimming
- [x] Fixed-support beta-fit scaffold and anchored value-fit variants
- [x] Baselines: recency, attention-mass, uniform selection
- [x] Verification gate: holdout response-error check
- [x] Qwen 2 experiment scaffold: config, argument parsing, inference loop skeleton
- [ ] Basic unit tests for objectives (correctness on trivial cases)

Deliverable: a repo where the objectives are implemented, the query bank runs, and the experiment scaffold is in place — even if end-to-end inference is not yet wired.

---

## Phase 2 (Mapped): Evidence-Regime and Support-Tradeoff Mapping

**Goal**: Map how support quality, repairability, and evidence regime interact on Qwen 2 under the N/Z framing.

Tasks:
- [x] Wire up live query collection hooks in the Qwen 2 inference loop (adapt patterns from `kv_compaction_experiment`)
- [x] Add `repeat-prefill` reference query strategy as a control: run "Context. Repeat it. Context." prefill and extract query vectors. This is the paper's cheapest strong baseline (~8s for 60k tokens) and establishes a reference ceiling for what a good offline query bank can achieve.
- [x] Add `teacher-forced` decode as the middle collection regime between prefill and free online generation.
- [x] Add collector parity and opportunity-accounting utilities for regime comparison.
- [x] Implement working baseline comparison pipeline with:
  - Recency selection (no refit)
  - Attention-mass selection (no refit)
  - Recency selection + beta-refit
  - Attention-mass selection + beta-refit
  - OMP-style support selection + anchored value refit
  - Uniform selection (sanity check)
- [x] Held-out verification pipeline: split query bank into fit / holdout, evaluate L_true on holdout
- [x] Results logging (JSON per-head metrics and regime-comparison artifacts)
- [ ] Basic plots (response error vs. budget fraction, by baseline)
- [x] Comparative support forensics:
  - repair-substrate geometry
  - weak-direction drift exposure
  - baseline-quality versus repairability tradeoff
- [ ] Downstream QA comparison against the motivating paper surface
- [ ] Richer query-bank controls closer to self-study + repeat-prefill
- [ ] N/Z-aware greedy alternative to OMP: project residuals in N/Z space rather than attention space

Current note: the Phase 2 findings and supporting artifacts live in [phase2_evidence_collection.md](/home/csmith/projects/kv_operator_matching/docs/phase2_evidence_collection.md). The main result is no longer just "which method wins", but:

- baseline support quality and repair substrate quality are distinct axes
- their tradeoff depends on continuous evidence properties, not just named regimes
- sparse online evidence currently favors recency-like coherent supports for local value repair
- richer decode-like evidence allows OMP-like baseline quality to matter more

Phase 2 deliverable is therefore not a single final selector. It is a mapped tradeoff surface and a set of predictive evidence/support proxies that justify a continuous hybrid selector in Phase 3.

---

## Phase 3 (Current): Hybrid Support Objective and Constructed Support

**Goal**: Build a single support-selection strategy whose behavior changes continuously with the observed evidence state, then extend it to merged or synthetic support.

Phase 2 suggests the right family is:

`J(S; E) = B(S) + alpha(E) * R(S; E) - beta(E) * C(S; E)`

where:
- `B(S)` is baseline support quality
- `R(S; E)` is repairability under evidence state `E`
- `C(S; E)` is instability / spread cost
- `alpha(E), beta(E)` are continuous weights driven by observed evidence properties

Interpretation update:

- baseline support quality should increasingly be read as "how small the
  initial output-scaled quotient residual already is" on the operational bank
- repairability should increasingly be read as "how much the allowed
  approximation class can move the residual toward the local null direction"
  rather than as geometry in the abstract
- current geometry proxies remain useful only insofar as they predict that
  quotient-aware behavior

The evidence state should be represented by continuous proxies rather than hard labels, for example:
- support-normalized stable rank
- weak-direction drift risk / low-singular update share
- evidence density / opportunity coverage
- query diversity / occupancy
- support span / locality / age dispersion

Interpretation note:

- the tree is the persistent support manifold
- the hybrid objective is the frontier policy over that manifold
- online evidence should update local split/collapse values and a rolling
  evidence-state vector, not trigger a global support search after every token

That means Phase 3 naturally splits into:
- basis construction: build a sensible hierarchical atom structure
- frontier policy: maintain local split/collapse preferences from online stats

Tasks:
- [x] Phase 3A handoff checkpoint: keep the first target narrow and explicit.
  The first selector should use only original-token candidates and answer one
  question: can one continuous hybrid selector interpolate between recency-like
  and OMP-like behavior as evidence-state changes, without hand-switching
  regimes?
- [x] Phase 3A: implement a first hybrid selector with score
  `J_add = ΔB + alpha(E) * ΔQ_coh - beta(E) * ΔQ_span`
  where:
  - `ΔB` is baseline-fidelity gain
  - `ΔQ_coh` is a repairability / coherence proxy with stable-rank improvement
    as the main term
  - `ΔQ_span` is a spread penalty, starting with temporal/support span
- [x] Phase 3A constraint: keep this as a single selector, not a hidden
  regime switch.
- [x] Phase 3A constraint: `alpha(E)` and `beta(E)` must be functions of
  continuous observables already available in the harness.
- [x] Phase 3A constraint: do not introduce merged or synthetic atoms in the
  first selector.
- [x] Phase 3A: evaluate whether the continuous selector moves between
  recency-like and OMP-like behavior as evidence-state changes.
- [x] Phase 3A: ablate the hybrid score terms (`ΔB`, `ΔQ_coh`, `ΔQ_span`,
  evidence-dependent weighting) to identify which ingredients are decisive.
- [x] Phase 3A: compare hybrid support geometry and post-vfit weak-direction
  drift directly against recency and OMP on the winning surfaces.
- [x] Phase 3A-prime: simplify the live selector around the current payload
  (`ΔB + ΔQ_coh`).
- [x] Phase 3A-prime: test one targeted repair-risk proxy in place of raw
  span, and record the failure of the direct low-singular-risk replacement.
- [x] Phase 3A-prime: test a different risk term family, and record the
  failure of the redundancy replacement on the broad surface.
- [x] Phase 3A-prime: treat `ΔB + ΔQ_coh` as the stable Phase 3A selector
  after the failed third-term tranches.
- [ ] Phase 3B: implement K-means-style support proposal: cluster key vectors, use cluster centroids as support keys, fit values and betas jointly
- [ ] Phase 3B: implement exponential-family merge: given two KV pairs, compute a merged point that matches the first two moments of their contribution to Z and N
- [x] Phase 3B negative tranche: test a first source-grounded constructed
  substrate (original tokens plus adjacent pair merges under the fixed
  `ΔB + ΔQ_coh` selector), and record that it is a live substrate shift but
  does not improve the original-token hybrid on either the broad surface or the
  denser richer-regime slice.
- [ ] Empirical tests of response sparsity (Open Question 1 from theory sketch)
- [ ] Empirical tests of spectral decay in the query-key kernel matrix (Open Question 2)
- [ ] Positional encoding interaction study: does RoPE structure in key vectors affect support quality? (Open Question 5)
- [ ] More principled merge proposal logic: conditions under which a good merge exists (toward Open Question 6)
- [ ] Quotient-residual forensic tranche: add explicit residual diagnostics to
  the Phase 2 / 3 forensic scripts and check whether they explain the known
  beta-only, `vfit`, hybrid, and merge results better than the current proxy
  geometry alone

Current note: the first Phase 3A selector and its supporting artifacts live in
[phase3a_hybrid_selector.md](/home/csmith/projects/kv_operator_matching/docs/phase3a_hybrid_selector.md).
The corrected Phase 3A result is:

- the first continuous hybrid selector really does adapt with evidence state
- its live core is currently `ΔB + ΔQ_coh`
- it robustly beats recency
- it can beat OMP on sparse online surfaces
- but raw span, evidence-dependent weighting, and the direct low-singular-risk
  and redundancy replacements do not yet clearly earn their place

Current Phase 3B note: the first constructed-support tranche lives in
[phase3b_pairmerge.md](/home/csmith/projects/kv_operator_matching/docs/phase3b_pairmerge.md).
Its conclusion is negative but useful:

- a conflict-aware selector over original tokens plus adjacent pair merges is a
  real substrate shift and uses merged atoms materially
- but this simple adjacent-pair mean construction does not beat the
  original-token hybrid selector
- the next Phase 3B construction family should therefore change the merge rule,
  not merely tweak the same adjacent-pair family

The next conservative follow-up also produced a useful negative result in
[phase3b_fitted_pairmerge.md](/home/csmith/projects/kv_operator_matching/docs/phase3b_fitted_pairmerge.md):

- compatibility-filtered, locally fitted adjacent-pair representatives beat the
  failed mean-merge family
- but mostly by over-pruning the candidate pool so hard that the selector
  almost reverted to the original-token hybrid
- so this is not yet a meaningful constructed-support win either

The next cheap precursor in
[phase3b_candidate_geometry.md](/home/csmith/projects/kv_operator_matching/docs/phase3b_candidate_geometry.md)
also narrowed the path:

- a locality-constrained operator-role compatibility score does surface a much
  richer non-adjacent local pool than strict adjacency
- but in its current form it does not beat adjacency broadly as the top local
  partner rule
- so the next step should improve the compatibility score itself before another
  full merge-construction tranche

The next cheap score iteration in
[phase3b_candidate_geometry_rerank.md](/home/csmith/projects/kv_operator_matching/docs/phase3b_candidate_geometry_rerank.md)
also failed cleanly:

- stronger locality regularization chose closer partners
- but it further reduced candidate richness and still failed to beat adjacency
  as a top partner rule
- so the cheap local-pair scoring path now looks exhausted enough that the next
  3B move should change the framing, not just retune the same score

The next framing shift in
[phase3b_anchor_region.md](/home/csmith/projects/kv_operator_matching/docs/phase3b_anchor_region.md)
also failed broadly:

- anchor-conditioned local construction around the good hybrid anchors is a
  heavy, real substrate shift rather than a no-op
- it can modestly improve baseline support under `repeat-prefill`
- but it loses after `vfit` in all three regimes, so it does not preserve the
  original-token hybrid's repairability advantage
- so simple local constructed-atom families now look weak enough that the next
  3B move should be a larger framing shift rather than another small local
  construction tweak

The centroid/operator-role assignment precursor in
[phase3b_centroid_region_precursor.md](/home/csmith/projects/kv_operator_matching/docs/phase3b_centroid_region_precursor.md)
also failed:

- centroid-conditioned assignment does improve internal assignment coherence
- and it slightly reduces low-singular update share
- but it still lowers stable-rank-like geometry and loses after `vfit` in all
  three regimes
- so simple centroid-conditioned regional construction is not yet the missing
  bridge either

Deliverable: a hybrid support strategy that outperforms the best fixed selector on the relevant evidence surfaces, plus at least one merged or synthetic support method that inherits that tradeoff instead of hard-coding a single regime.

---

## Phase 4: Progressive Residual and Dynamic Policy (Long-Term)

**Note**: This phase is long-term motivation only. The scope will change significantly based on what is learned in Phases 1-3. Do not plan implementation details here.

High-level direction:
- **Progressive residual representation**: represent the KV state as mu = mu_0 + delta_1 + delta_2 + ... where each delta_t is a small correction added incrementally as new tokens arrive
- **Tree / hierarchy**: organize representations at multiple resolutions; coarse representation for distant tokens, fine for recent; concatenation property enables composability
- **Dynamic resolution policy**: decide at each checkpoint whether to refit, which resolution to use, and how to trade off memory vs. accuracy — based on live query evidence
- **Learned router**: a small model that predicts the best resolution given context signals

Why this is Phase 4 and not earlier: the N/Z formalism is already *algebraically compatible* with progressive and tree-structured representations — the concatenation property and the admissible representation class both support it (see `docs/theory_sketch.md` Section 8). What Phase 4 requires that the current formalism does not yet supply is (a) a construction rule for merged or synthetic atoms, (b) a hybrid support objective worth applying recursively, and (c) empirical evidence that the operator is compressible enough to justify hierarchy. Phases 2-3 must answer those first.

---

## Notes on Priority

- Phases 1 and 2 establish the evidence and tradeoff surface. Everything else depends on that map being trustworthy.
- Phase 3 is now the natural extension because Phase 2 identified a structural tradeoff between baseline quality and repairability, not because a single selector already solved the problem.
- Phase 4 is the long-term vision. It should inform design decisions in Phases 1-3 (keep the concatenation property central, keep Z and N as separate objects) but should not drive implementation now.
