# Phase 3B Precommit: Centroid-Conditioned Regional Assignment Precursor

- `Hypothesis`: Keeping the same hybrid anchors and the same conservative
  regional representative, centroid/operator-role assignment will produce
  better regional grouping than locality-only assignment. That should show up as
  better internal assignment coherence and a better or less harmful
  `vfit` outcome on the broad surface.
- `Surface`:
  prompts `near_capacity_dispatch.json`, `near_capacity_dispatch_safe.json`,
  `near_capacity_network_dispatch_safe.json`, `relational_binding_probe.json`;
  layers `4 12 20 28`; budgets `0.25 0.5`; regimes
  `online teacher-forced repeat-prefill`; query-bank density
  `max_queries=256`, `max_new_tokens=32`.
- `Primary metric`: holdout `L_true` after anchored value refit, comparing
  `anchor_region_centroid+vfit` against `anchor_region_local+vfit`.
- `Mechanism expectation`: centroid-conditioned assignment should improve
  assignment coherence and should not worsen post-`vfit` geometry
  (`stable_rank`, `low_sv_delta_share`) relative to locality-only assignment.
- `Promotion rule`: justify a centroid-conditioned constructed-atom tranche
  only if centroid assignment beats locality-only assignment broadly on the
  precommitted surface and the mechanism metrics move in the same direction.
- `Kill rule`: if centroid assignment does not beat locality-only assignment on
  the broad surface, or if the mechanism story conflicts with the outcome,
  stop treating simple centroid-conditioned regional construction as the next
  obvious 3B path.
- `Artifact path`: `results/scratch/phase3b_centroid_region_precursor_q256_t32.json`
