# Experiment: Qwen 2 Online N/Z Operator Matching

**Status**: Runnable smoke harness. Supports `prefill`, `repeat-prefill`,
`teacher-forced-suffix`, `teacher-forced-full-prompt`, and `online` query
collection modes. The legacy `teacher-forced` CLI name remains as an explicit
alias for `teacher-forced-suffix`.

Project policy: for dense prompt-side evidence collection, prefer `prefill`.
Use `teacher-forced-full-prompt` as a prompt-side control when you specifically
want to measure incremental replay effects rather than as the default prompt
evidence path.

---

## Goal

Replicate the spirit of the Qwen 2 online experiment from `kv_compaction_experiment` under the new N/Z operator framing. The key difference: instead of matching attention distributions, we match the full response operator A_mu(q) = N_mu(q) / Z_mu(q), using live query evidence collected during inference.

---

## What this experiment will do

1. **Load Qwen 2** (7B Instruct, bfloat16, on CUDA)
2. **Run inference** on a set of evaluation prompts
3. **Collect query vectors**:
   - `online`: prefill the prompt into cache, then capture decode-step queries during greedy generation
   - `teacher-forced-suffix`: prefill the prompt into cache, then capture decode-step queries while feeding a fixed known continuation
   - `teacher-forced-full-prompt`: replay the prompt token by token from an empty cache to collect incremental prompt queries; optionally continue into real online decode at the boundary. This is a control path, not the preferred dense prompt-evidence regime.
   - `repeat-prefill`: use `prompt + "Repeat it." + prompt` as a stronger offline control
   - `prefill`: use prompt queries as an offline proxy bank. This is the preferred dense prompt-side evidence regime.
4. **Build the empirical query bank**: maintain a rolling bank of query vectors per head, weighted by recency (configurable)
5. **At each KV boundary** (configurable checkpoint interval):
   - For each head and layer, propose a compact support (recency / attention-mass / uniform)
   - Fit beta coefficients using the query bank and the L_lin surrogate
   - Run the verification gate on held-out queries
   - Record metric values for each baseline and beta-refit variant
6. **Compare baselines**: recency, attention-mass, uniform — both with and without beta-refit
7. **Log results**: per-head, per-layer, per-prompt response error at each budget fraction

---

## Configuration

See `config.yaml` for the experiment parameters. Key knobs:
- `model.name`: which Qwen 2 variant to use
- `query_bank.max_queries`: how many query vectors to retain
- `beta_fit.support_size`: compact representation size (budget)
- `experiment.budget_fractions`: list of budget fractions to evaluate (e.g., 0.25 = retain 25% of cache)
- `experiment.n_prompts`: how many prompts to evaluate

---

## Remaining Phase 2 work

- [x] Cache-aware online query collection loop
- [x] KV cache extraction for Qwen 2
- [x] Attention-mass support using the experiment query bank
- [x] `repeat-prefill` control path
- [x] `teacher-forced` decode control path
- [x] Collector parity harness
- [x] Prompt-side prefill vs full-replay control
- [x] Opportunity-aware evidence accounting in experiment artifacts
- [ ] Broader online evaluation over multiple prompts and boundaries
- [ ] Port prompt loading / batching utilities
- [ ] Port results logging (CSV / JSON per-head metrics)
- [x] Wire up `QueryBank`, `beta_fit.fit_beta`, and `verification.verify` from `src/kv_operator_matching/`

See [phase2_evidence_collection.md](/home/csmith/projects/kv_operator_matching/docs/phase2_evidence_collection.md) for the current task list and imported lessons from `kv_compaction_experiment`.
See [phase3c_quotient_selector.md](/home/csmith/projects/kv_operator_matching/docs/phase3c_quotient_selector.md) for the next selector-side experiment tranche.
See [phase3c_shortlist_sweep.md](/home/csmith/projects/kv_operator_matching/docs/phase3c_shortlist_sweep.md) for the shortlist-policy follow-up with fixed downstream OMP + `vfit`.
See [phase3c_quotient_refit.md](/home/csmith/projects/kv_operator_matching/docs/phase3c_quotient_refit.md) for the fixed-support quotient-aware refit tranche.
See [phase3c_qvfit_diagnostics.md](/home/csmith/projects/kv_operator_matching/docs/phase3c_qvfit_diagnostics.md) for the qvfit compatibility mechanism.
See [phase3c_qvfit_policy.md](/home/csmith/projects/kv_operator_matching/docs/phase3c_qvfit_policy.md) for the hard-gated and tempered qvfit policy comparison.
See [phase3c_qvfit_representativeness.md](/home/csmith/projects/kv_operator_matching/docs/phase3c_qvfit_representativeness.md) for the ESS- and divergence-controlled qvfit follow-up.
See [phase3c_qfit_controller.md](/home/csmith/projects/kv_operator_matching/docs/phase3c_qfit_controller.md) for the diagnostic-conditioned measured qfit controller.
See [phase3c_stack_readjudication.md](/home/csmith/projects/kv_operator_matching/docs/phase3c_stack_readjudication.md) for the canonical broad-surface stack re-adjudication.
See [qfit_metric_family.md](/home/csmith/projects/kv_operator_matching/docs/qfit_metric_family.md) for the unified row-metric view of `vfit`, `qvfit`, and the controlled refit family.

Current selector-side status:
- direct quotient-aware ranking is not a standalone replacement for
  `attn_mass`
- quotient-aware shortlist construction is now the preferred Phase 3C direction
  because it has produced real wins on `online` / `repeat-prefill` slices under
  tight shortlist pressure
- quotient-aware refit is also now a live experiment result, but it is support-
  conditioned rather than a default replacement for anchored `vfit`
- hard-gated `qvfit` is now the simple refit-side baseline to beat
- the best current refit policy is the diagnostic-conditioned `qfit` controller
- representativeness-controlled `qvfit` remains important because it supplies
  the middle branch of that controller
- the best tested overall broad stack is now `attn_mass+qfit_diag` on the
  canonical `q256/t32` Phase 3 surface

---

## Running

Example online smoke run:

```bash
python run_experiment.py --collection-mode online --layers 4 --budgets 0.25
```
