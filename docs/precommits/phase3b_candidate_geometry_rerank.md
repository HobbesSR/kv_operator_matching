# Phase 3B Precommit: Locality-Regularized Candidate Rerank

- `Hypothesis`: A locality-regularized compatibility score that treats
  adjacency as the default prior will beat the current compatibility score on
  the top-neighbor metric and materially improve `top_score_beats_adjacent_frac`
  over the current ~`0.19-0.20` broad result.
- `Surface`:
  prompts `near_capacity_dispatch.json`, `near_capacity_dispatch_safe.json`,
  `near_capacity_network_dispatch_safe.json`, `relational_binding_probe.json`;
  layers `4 12 20 28`; budgets `0.25 0.5`; regimes
  `online teacher-forced repeat-prefill`; query-bank density
  `max_queries=256`, `max_new_tokens=32`.
- `Primary metric`: `top_score_beats_adjacent_frac`, compared directly between
  the old score and the locality-regularized rerank.
- `Mechanism expectation`: the rerank should preserve the strong “beats random”
  property and a materially richer local pool than the fitted-pair tranche,
  while choosing closer neighbors and improving top-neighbor quality.
- `Promotion rule`: justify another geometry-led merge tranche only if the
  rerank materially improves `top_score_beats_adjacent_frac` across regimes
  without collapsing candidate richness back toward adjacency-only behavior.
- `Kill rule`: if the rerank still cannot materially improve on the current
  top-neighbor metric, stop iterating cheap score tweaks and reconsider the
  local-pair framing before another merge-construction tranche.
- `Artifact path`: `results/scratch/phase3b_candidate_geometry_rerank_q256_t32.json`
