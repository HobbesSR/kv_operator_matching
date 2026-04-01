# QFit Metric Family

This note formalizes the current relationship between `vfit`, `qvfit`, and
their controlled variants.

The main point is that these are best understood as one fixed-support
least-squares family with different **row metrics** over the query bank, not
as unrelated fitting methods.

---

## Setup

Fix:

- a retained support of size `t`
- fixed support keys
- fixed support betas
- a bank of `n` queries

For each bank query `q_i`, define the compact unnormalized support weights

$$
s_i \in \mathbb{R}^{1 \times t},
\qquad
\hat z_i := s_i \mathbf 1,
\qquad
x_i := \frac{s_i}{\hat z_i}.
$$

Stacking rows gives

$$
S \in \mathbb{R}^{n \times t},
\qquad
D := \operatorname{diag}(\hat z_1, \dots, \hat z_n),
\qquad
X := D^{-1} S.
$$

Let

$$
O \in \mathbb{R}^{n \times d}
$$

be the bank target outputs from the reference cache.

Then the compact values `C \in \mathbb{R}^{t \times d}` are fit against the
same bank in two closely related ways.

---

## Neutral Metric: VFit

Ordinary value refit solves the normalized bank fit:

$$
\min_C \|X C - O\|_F^2.
$$

This is the **neutral row-metric** member of the family.
Each bank row is treated evenly except for the base query-bank weights
themselves.

In the code path, this is implemented by using normalized compact attention
weights as the design matrix:

- [value_fit.py](/home/csmith/projects/kv_operator_matching/src/kv_operator_matching/value_fit.py)

---

## Quotient Metric: QVFit

The fixed-support quotient-aware refit solves the exact local quotient solve:

$$
\min_C \|S C - D O\|_F^2.
$$

Since `S = D X`, this is exactly

$$
\min_C \|D(X C - O)\|_F^2
=
\sum_{i=1}^n \hat z_i^2 \|x_i C - O_i\|_2^2.
$$

So `qvfit` is not a different species of fit. It is the same bank fit under a
different row metric:

$$
\min_C \|W (X C - O)\|_F^2
$$

with

$$
W = D.
$$

This is the formal reason the current diagnostics focus on quotient-induced row
scaling rather than on ordinary design conditioning alone.

---

## Special-Case Relation

`vfit` is the special case of the same family where the row metric is neutral:

$$
W = c I
$$

for any positive constant `c`.

Because multiplying the objective by a positive constant does not change the
minimizer, this reduces exactly to the `vfit` objective.

So the useful conceptual relation is:

- `vfit` = neutral row metric
- `qvfit` = quotient row metric

The empirical program has now validated that this unification is useful:

- when the quotient row metric is mild, `qvfit` can improve over `vfit`
- when the quotient row metric becomes too dispersed, raw `qvfit` can become
  unstable

---

## Controlled Variants

Once `vfit` and `qvfit` are viewed as one row-metric family, the current
control strategies become structurally clean.

### Hard Gate

Choose between the two row metrics using a compatibility statistic:

- use `qvfit` if the quotient row metric is sufficiently well behaved
- otherwise use `vfit`

The current winning policy uses:

- `zhat_over_zref_cv`

as the compatibility statistic and gates at a fixed threshold.

### Tempered Quotient Metric

Replace the full quotient row scale by a tempered one:

$$
W_\gamma = D^\gamma,
\qquad
0 \le \gamma \le 1.
$$

Then:

- `\gamma = 0` recovers the neutral metric
- `\gamma = 1` gives the full quotient metric
- `0 < \gamma < 1` gives a tempered quotient-aware fit

This is exactly the current `qvfit_temp` interpretation.

### Clipped Or Controlled Quotient Metric

More aggressive controls keep the same family but alter the row metric:

- quotient-ratio clipping
- row-scale clipping
- quotient-metric preconditioning

These are not arbitrary blends. They are controls on **how strongly the bank is
reweighted by the quotient structure**.

---

## Why This Matters

The recent Phase 3C results now have a cleaner interpretation:

1. `qvfit` is not a random second refit method.
   It is the quotient-weighted member of the same bank-fit family.

2. The key failure mode is not generic bad support geometry.
   It is that the quotient row metric becomes too heterogeneous across the
   bank, so the solve becomes hostage-like to a small subset of rows.

3. A gate or controlled metric is therefore not a hack.
   It is a principled way to decide when and how strongly to apply the
   quotient-induced row geometry.

That is why the refit-side story has progressed from:

- quotient mechanism

to:

- quotient compatibility statistic

to:

- quotient-aware control law
