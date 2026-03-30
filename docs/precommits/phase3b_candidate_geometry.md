# Phase 3B Precommit: Candidate-Geometry Diagnostic

- `Hypothesis`: A locality-constrained operator-role compatibility score will
  identify better local pair candidates than plain adjacency on the broad
  surface, and a nontrivial share of the best candidates will occur beyond
  distance `1`.
- `Surface`:
  prompts `near_capacity_dispatch.json`, `near_capacity_dispatch_safe.json`,
  `near_capacity_network_dispatch_safe.json`, `relational_binding_probe.json`;
  layers `4 12 20 28`; budgets `0.25 0.5`; regimes
  `online teacher-forced repeat-prefill`; query-bank density
  `max_queries=256`, `max_new_tokens=32`.
- `Primary metric`: fraction of selected support anchors where the top
  compatibility-scored neighbor has lower conservative local pair-fit loss than
  the best adjacent neighbor.
- `Mechanism expectation`: if candidate geometry is the missing ingredient,
  then:
  - top compatibility neighbors should beat adjacency more often than chance
  - top compatibility distance should often exceed `1`
  - eligible candidate fractions should be materially higher than the
    over-pruned fitted-pair tranche
- `Promotion rule`: justify the next centroid-compatible local merge tranche
  only if the compatibility diagnostic beats adjacency broadly and yields a
  materially richer local candidate pool than the fitted-pair tranche.
- `Kill rule`: if compatibility scoring does not beat adjacency broadly or the
  eligible local pool remains tiny, do not move forward with centroid-compatible
  merge construction yet.
- `Artifact path`: `results/scratch/phase3b_candidate_geometry_q256_t32.json`
