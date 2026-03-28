# kv_operator_matching

**KV cache operator approximation via N/Z factorization.**

Experimental and research-oriented. APIs and objectives are unstable and will change.

---

## What this repo is

This is a new direction, not a continuation of `kv_compaction_experiment`. The prior repo explored KV cache compression under an Attention Matching framing — matching compressed attention distributions to full-cache distributions. That work produced useful baselines and experiment patterns, but the framing has limitations: it is head-local in the wrong way, it does not compose cleanly across KV blocks, and it does not expose the natural operator structure of attention.

This repo reframes the problem from scratch around the **N/Z factorization**.

---

## The N/Z operator framing

For a single attention head with KV cache state mu = {(k_i, v_i)}, define:

- **Z_mu(q)** = sum_i exp(< q, k_i >)  — the partition function
- **N_mu(q)** = sum_i exp(< q, k_i >) * v_i  — the numerator (value-weighted)
- **A_mu(q)** = N_mu(q) / Z_mu(q)  — the attention response operator

The operator A_mu maps a query vector to an output vector. It is a rational function of the query, parameterized by the full KV cache.

The goal is: given the current KV state mu, find a **compact representation** mu_hat (with far fewer support points) such that A_mu_hat(q) approximately equals A_mu(q) for the queries q that will actually be issued during future inference.

The approximation is measured over an **empirical query bank** — live query vectors collected during the current inference session.

---

## Why the N/Z factorization matters

**It survives concatenation.** If you split the KV cache into a past block P and a future block F:

    A_{P || F}(q) = (N_P(q) + N_F(q)) / (Z_P(q) + Z_F(q))

This means compressed representations of P can be concatenated with uncompressed F and the arithmetic is exact. There is no need to refit when new tokens arrive — you just update the numerator and denominator separately. This is the key property that makes the operator framing practical for streaming inference.

**It exposes operator structure.** Z_mu and N_mu are linear in the support measure mu. The approximation problem can be studied as a linear operator approximation problem, which is more tractable than working with the ratio A_mu directly.

**It supports synthetic and merged support elements.** Because the framing is measure-theoretic, support points (k_hat_j, v_hat_j) do not need to be drawn from the original KV cache. They can be learned, merged, or synthetically generated, as long as the induced operator A_mu_hat is close to A_mu over the query bank.

---

## First-stage scope (what is in this repo now)

- **Live query evidence collection**: hooks to collect query vectors during inference
- **Empirical query bank**: rolling bank of weighted query vectors that defines the matching objective
- **Fixed-support beta-refit**: given a fixed support (recency-selected or attention-mass-selected), fit nonneg coefficients beta_j to minimize the N/Z surrogate loss
- **Baseline comparisons**: recency, attention-mass, uniform selection — all returning CompactRepresentation objects compatible with the N/Z objective
- **Verification gate**: check that a fitted representation meets a response-error threshold on held-out queries before deployment

---

## What is NOT yet in scope

- Progressive residual representation: mu = mu_0 + delta_1 + delta_2 + ...
- Learned or adaptive router over representations
- Full dynamic resolution policy (when to refit, when to switch levels)
- Principled synthetic/merged support generation
- Tree or hierarchy structure over the cache

These are Phase 3 and Phase 4 work. See `docs/roadmap.md`.

---

## Immediate milestones

**Phase 1** (current):
- Repo scaffolding and docs (this phase)
- Core objective implementations (L_Z, L_N, L_lin, L_true)
- Empirical query bank with recency weighting
- Fixed-support beta-fit (NNLS placeholder, proper solver next)
- Qwen 2 experiment scaffold

**Phase 2** (next):
- Baseline comparison pipeline (recency, attention-mass, fixed-support no-refit, beta-refit)
- Held-out verification pipeline
- Better support proposal logic
- N/Z-aware greedy selection alternatives

See `docs/roadmap.md` for full phased plan.

---

## Relationship to kv_compaction_experiment

See `docs/old_repo_relationship.md` for a detailed accounting of what carries forward and what is new.

---

## Disclaimer

This is an active research repo. Nothing here is production code. Objectives, APIs, and experiment designs will change as the theory develops. See `docs/theory_sketch.md` for the current theoretical framing and open questions.
