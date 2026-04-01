# Quotient Residual Note

This note promotes the project-relevant part of the quotient-residual
conjecture into the stable repo docs.

Status:

- the algebraic identity is exact
- the interpretation is strongly relevant to current experiments
- the resulting metrics are now worth testing empirically
- this does **not** replace the existing `L_Z` / `L_N` / `L_lin` fitting story

---

## Core Identity

Let

$$
O(q) = \frac{N(q)}{Z(q)}, \qquad \hat O(q) = \frac{\hat N(q)}{\hat Z(q)}
$$

and define

$$
\delta N(q) = \hat N(q) - N(q), \qquad \delta Z(q) = \hat Z(q) - Z(q).
$$

Then:

$$
\hat O(q) - O(q) = \frac{\delta N(q) - O(q)\delta Z(q)}{Z(q) + \delta Z(q)}.
$$

Define the **quotient residual**

$$
E(q) := \delta N(q) - O(q)\delta Z(q).
$$

So the exact response error is

$$
\hat O(q) - O(q) = \frac{E(q)}{Z(q) + \delta Z(q)}.
$$

The project consequence is simple:

- output error is governed by a coupled numerator/denominator object
- `δN` alone is not the observable error
- `δZ` alone is not the observable error
- a weighted sum of separate `N` and `Z` errors is only a surrogate

---

## Null Direction

If

$$
\delta N(q) = O(q)\delta Z(q),
$$

then `E(q) = 0`, and the output is preserved exactly at that query as long as
`Z(q) + δZ(q) != 0`.

So perturbations near the direction

$$
(\delta N, \delta Z) = (O(q)\epsilon, \epsilon)
$$

are cheap, while transverse perturbations are expensive.

This is the local geometry that later approximation stages should care about.

---

## Omission Formula

If a subset `R` of support atoms is removed, then

$$
\delta Z_R(q) = - \sum_{i \in R} w_i(q), \qquad
\delta N_R(q) = - \sum_{i \in R} w_i(q) v_i.
$$

So

$$
E_R(q) = - \sum_{i \in R} w_i(q)\left(v_i - O(q)\right).
$$

For a single atom `i`,

$$
E_i(q) = - w_i(q)\left(v_i - O(q)\right).
$$

This gives a quotient-aware importance story:

- large attention weight is not enough to make an atom important
- a high-mass atom close to the current output can be cheap to remove
- a moderate-mass atom with a distinctive value can be expensive to remove

The associated static proxy is:

$$
\|E_i(q)\| \propto w_i(q)\|v_i - O(q)\|.
$$

---

## Why This Matters For Current Results

### Beta-only weakness

If support keys and values are fixed and only `beta` is refit, then the fit can
move `δZ` directly but can only move `δN` inside the span of the retained value
vectors. That gives a structural reason beta-only repair can fail even when
mass matching improves.

This matches the current Phase 2 empirical pattern:

- beta-only repair is often weak or harmful
- anchored value refit is much more consistently useful

### Repairability versus baseline quality

The Phase 2 and Phase 3 docs already distinguish:

- better baseline support quality
- better repair substrate quality

The quotient-residual lens sharpens that split:

- baseline quality is about how small the initial output-scaled quotient
  residual already is
- repairability is about whether the allowed approximation class can move that
  residual toward the null direction on the operational query bank

### Constructed support and merging

A merge should not be judged only by mass preservation, locality, or averaged
value quality. The sharper criterion is:

$$
\delta N_C(q) \approx O(q)\delta Z_C(q)
$$

on the operational query bank.

If that fails, the merge is likely to look plausible under separate `N` and
`Z` checks while still harming the final output.

---

## Practical Metric Family

For a query bank `Q_b`, useful diagnostics include:

- mean quotient-residual energy:
  $$
  \mathbb E_{q \sim Q_b}\|E(q)\|_2^2
  $$
- output-scaled quotient-residual energy:
  $$
  \mathbb E_{q \sim Q_b}\frac{\|E(q)\|_2^2}{Z(q)^2}
  $$
- worst-case output-scaled residual:
  $$
  \sup_{q \in Q_b}\frac{\|E(q)\|_2}{Z(q)}
  $$

The output-scaled version is especially important because it is the exact
response-error quantity written in quotient form.

In this repo, these metrics are currently intended for:

- forensic evaluation
- mechanism checks
- support / merge proposal screening

They are not yet the fitting objective.

---

## Caveats

Two caveats matter immediately.

### Query-bank dependence

All quotient-residual diagnostics are conditional on the empirical query bank.
They are only as useful as the evidence regime that produced that bank.

### Temporal drift within a session

The query bank is slightly backward-looking. Queries observed now may not
perfectly match the queries issued later in the same generation.

So the quotient-residual framing sharpens the mechanism story, but it does not
remove the need for held-out verification and evidence-regime controls.

---

## Near-Term Experimental Use

The right next use is not a full objective rewrite. It is:

1. add quotient-residual diagnostics to the existing forensic scripts
2. check whether they explain known Phase 2 and Phase 3 wins and failures
3. only then consider quotient-aware support ranking or merge selection

That keeps the theory tied to the current empirical program instead of turning
into a parallel speculative branch.

The first selector-side experiment should be deliberately narrow:

- rank original tokens by a bank-aggregated quotient-aware omission score
- compare that against plain attention-mass support
- then check whether any gain survives anchored value refit
- then test the same score only as a shortlist for the existing OMP path

This is the smallest direct test of whether the quotient lens improves support
selection itself rather than only explaining results after the fact.

One caveat should stay explicit: exact local cancellation does not remove the
need for mass fidelity under concatenation. Even a selector built from local
`E(q)` still has to be judged on deployed-style held-out response error, not on
local cancellation alone.

Current repo status:

- direct quotient-aware omission ranking did not become a new standalone
  selector winner
- quotient-aware shortlist construction *did* produce real downstream wins once
  combined with fixed downstream OMP + `vfit`
- those wins currently concentrate in `online` and `repeat-prefill`, especially
  at tight shortlist multipliers (`1.5x-2.0x`)
- fixed-support quotient-aware refit also produced real gains, but only for
  some support families:
  `attn_mass+qvfit` beat `attn_mass+vfit` across all three tested regimes,
  while `OMP` / `hybrid` generally did not benefit
- the first compatibility diagnostics suggest why:
  `qvfit` success is tracked much more by quotient-specific row-scaling
  statistics like `zhat_over_zref_cv` and quotient-row concentration than by
  ordinary normalized-design conditioning
- so the quotient lens is now operationally relevant as a shortlist prior,
  and as a support-conditioned refit objective, while still not replacing the
  broader support-search stack
