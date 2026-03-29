# Results Policy

This directory is part of the research record for `kv_operator_matching`.

Policy:

- `results/checkpoints/` is for tracked milestone artifacts that support a doc,
  a conclusion, or a phase boundary.
- `results/scratch/` is for exploratory runs, reruns, and transient analysis.
  It is ignored by git.

Guidelines:

- If a result file is cited in a doc or needed to reconstruct an important
  experimental conclusion, put it under `results/checkpoints/`.
- If a run is mainly exploratory or likely to be superseded quickly, write it
  under `results/scratch/`.
- Prefer phase-oriented naming under `results/checkpoints/`, for example:
  - `results/checkpoints/phase1/...`
  - `results/checkpoints/phase2/...`
- Result artifacts should be self-describing. Include arguments, collection
  mode, and any other metadata needed to interpret the file without rerunning
  the experiment.
