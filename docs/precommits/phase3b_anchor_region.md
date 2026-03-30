# Phase 3B Precommit: Anchor-Conditioned Local Construction

- `Hypothesis`: Replacing each selected hybrid anchor with one conservative
  local regional representative will improve over the original-token
  `hybrid+vfit` baseline on the broad surface by preserving the good anchor
  geometry while capturing nearby redundant structure.
- `Surface`:
  prompts `near_capacity_dispatch.json`, `near_capacity_dispatch_safe.json`,
  `near_capacity_network_dispatch_safe.json`, `relational_binding_probe.json`;
  layers `4 12 20 28`; budgets `0.25 0.5`; regimes
  `online teacher-forced repeat-prefill`; query-bank density
  `max_queries=256`, `max_new_tokens=32`.
- `Primary metric`: holdout `L_true` after anchored value refit, comparing
  `hybrid_anchor_region+vfit` against `hybrid+vfit`.
- `Mechanism expectation`: baseline `L_true` should improve modestly without
  materially worsening the post-`vfit` result, while the regional atoms remain
  a real substrate shift with nontrivial changed-atom and assigned-token
  fractions.
- `Promotion rule`: justify a denser follow-up only if the anchor-conditioned
  construction beats `hybrid+vfit` on the broad surface and the construction is
  materially used rather than collapsing back to the original-token support.
- `Kill rule`: if the construction loses broadly to `hybrid+vfit` on the broad
  surface, demote this simple local constructed-atom family and do not run the
  denser follow-up.
- `Artifact path`: `results/scratch/phase3b_anchor_region_q256_t32.json`
