# Experiment: Qwen 2 Online N/Z Operator Matching

**Status**: Scaffold only. Not yet runnable end-to-end.

---

## Goal

Replicate the spirit of the Qwen 2 online experiment from `kv_compaction_experiment` under the new N/Z operator framing. The key difference: instead of matching attention distributions, we match the full response operator A_mu(q) = N_mu(q) / Z_mu(q), using live query evidence collected during inference.

---

## What this experiment will do

1. **Load Qwen 2** (7B Instruct, bfloat16, on CUDA)
2. **Run inference** on a set of evaluation prompts
3. **Collect live query vectors**: register hooks on each attention layer to capture the query tensors at each decoding step
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

## TODO: Adapt from kv_compaction_experiment

- [ ] Port inference loop hook structure for live query capture
- [ ] Port KV cache extraction (past_key_values handling for Qwen 2)
- [ ] Port attention mass computation logic
- [ ] Port prompt loading / batching utilities
- [ ] Port results logging (CSV / JSON per-head metrics)
- [ ] Wire up `QueryBank`, `beta_fit.fit_beta`, and `verification.verify` from `src/kv_operator_matching/`

---

## Running

Not yet runnable. Once the TODOs above are resolved, the intended entry point is:

```bash
python run_experiment.py --config config.yaml --output-dir outputs/run_001
```
