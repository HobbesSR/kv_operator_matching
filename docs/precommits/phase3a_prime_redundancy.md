# Phase 3A-Prime Precommit: Redundancy Term

- `Hypothesis`: Adding one redundancy penalty to the live `ΔB + ΔQ_coh`
  selector will improve `hybrid+vfit` over the current Phase 3A core on the
  broad surface, with the biggest gains expected in `teacher-forced` and
  `repeat-prefill`.
- `Surface`: prompts `near_capacity_dispatch.json`,
  `near_capacity_dispatch_safe.json`, `near_capacity_network_dispatch_safe.json`,
  `relational_binding_probe.json`; layers `4 12 20 28`; budgets `0.25 0.5`;
  regimes `online teacher-forced repeat-prefill`; query-bank density
  `max_queries=256`, `max_new_tokens=32`.
- `Primary metric`: mean holdout `L_true` of `redundancy+vfit` versus the live
  `ΔB + ΔQ_coh` core on the broad surface.
- `Mechanism expectation`: if the redundancy term helps, it should reduce
  support duplication pressure without collapsing into recency; relative to the
  live core, it should improve holdout `L_true` while keeping support span
  broad and lowering a key-space redundancy proxy.
- `Promotion rule`: promote only if the redundancy variant improves mean
  holdout `L_true` over the live core on the broad surface in at least two
  regimes, including one decode-like regime (`online` or `teacher-forced`).
  Slice-only wins count only as follow-up interest.
- `Kill rule`: if the redundancy variant is neutral-to-worse on the broad
  surface in all decode-like regimes, or clearly worse overall, demote this
  risk family for Phase 3A and do not rescue it with dense/stress follow-ups.
- `Artifact path`: `results/scratch/phase3a_prime_redundancy_q256_t32.json`
- `Pivot condition / follow-up trigger`: if the main hypothesis fails but the
  variant shows a broad baseline improvement without post-vfit benefit, treat
  that as a new selector-design lead, not a success for this tranche.
