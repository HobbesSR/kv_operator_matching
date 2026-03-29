# Experiment: Qwen 2 Online N/Z Operator Matching

**Status**: Runnable smoke harness. Supports `prefill`, `repeat-prefill`,
`teacher-forced`, and `online` query collection modes.

---

## Goal

Replicate the spirit of the Qwen 2 online experiment from `kv_compaction_experiment` under the new N/Z operator framing. The key difference: instead of matching attention distributions, we match the full response operator A_mu(q) = N_mu(q) / Z_mu(q), using live query evidence collected during inference.

---

## What this experiment will do

1. **Load Qwen 2** (7B Instruct, bfloat16, on CUDA)
2. **Run inference** on a set of evaluation prompts
3. **Collect query vectors**:
   - `online`: prefill the prompt into cache, then capture decode-step queries during greedy generation
   - `teacher-forced`: prefill the prompt into cache, then capture decode-step queries while feeding a fixed known continuation
   - `repeat-prefill`: use `prompt + "Repeat it." + prompt` as a stronger offline control
   - `prefill`: use prompt queries as an offline proxy bank
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
- [x] Opportunity-aware evidence accounting in experiment artifacts
- [ ] Broader online evaluation over multiple prompts and boundaries
- [ ] Port prompt loading / batching utilities
- [ ] Port results logging (CSV / JSON per-head metrics)
- [x] Wire up `QueryBank`, `beta_fit.fit_beta`, and `verification.verify` from `src/kv_operator_matching/`

See [phase2_evidence_collection.md](/home/csmith/projects/kv_operator_matching/docs/phase2_evidence_collection.md) for the current task list and imported lessons from `kv_compaction_experiment`.

---

## Running

Example online smoke run:

```bash
python run_experiment.py --collection-mode online --layers 4 --budgets 0.25
```
