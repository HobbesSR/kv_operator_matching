# Phase 3C Precommit: Quotient-Aware Selector

This tranche is a selector-side test, not a fitting-objective rewrite.

The quotient-residual note implies that attention-only support ranking is
incomplete because the local omission damage of token `i` depends on both:

- its attention weight
- its value deviation from the current output

So the first precommitted sequence is:

1. keep downstream `beta` / `C_v` fitting fixed
2. replace attention-only support ranking with a bank-aggregated
   quotient-aware omission score over original tokens
3. test that selector directly
4. test the same score as a shortlist for the existing OMP path
5. only if selector-side gains plateau, consider full E-aware OMP or
   alternating joint `(beta, C_v)` refinement

The immediate structural follow-up after the first selector tranche is:

- compare shortlist policies rather than raw selector scores
- use only parameter-light policies:
  - `attn_mass`
  - `quotient_omit`
  - equal-rank blend
  - two-stage gate
- hold downstream OMP + `vfit` fixed

Two guardrails:

- local exactness via `E(q)=0` is not enough for deployed behavior; mass
  fidelity under concatenation still matters
- the null hypothesis is that value-aware selection explains most of the gain,
  and more ambitious joint fitting adds little beyond the current pipeline

Primary evaluation targets:

- extreme compaction ratios
- concatenation-sensitive regimes
- topic / branch / session-switch prompts

Primary metrics:

- holdout `L_true`
- quotient-residual energy
- quotient cancellation gain
- worst-case output-scaled quotient residual

Observed update:

- the direct quotient-aware selector did not replace `attn_mass`
- quotient-aware shortlist construction with fixed downstream OMP + `vfit`
  *did* produce real wins on the first slice and the broader stability slice
- the wins are narrow rather than universal:
  strongest in `online` and `repeat-prefill`, especially at `1.5x-2.0x`
  shortlist pressure, while `teacher-forced-suffix` still prefers
  `attn_mass`

So the precommitted next question is now:

- treat quotient as shortlist architecture, not a monolithic selector
- inspect shortlist-size interaction and shortlist-content differences before
  any full E-aware OMP or alternating `(beta, C_v)` rewrite
- separately test fixed-support quotient-aware refit rather than assuming the
  identity's leverage is only on the selection side

Observed follow-up:

- fixed-support quotient-aware refit did produce real gains, but only for some
  support families
- `attn_mass+qvfit` beat `attn_mass+vfit` across all three tested regimes
- `OMP` / `hybrid` generally did not benefit, so the refit leverage is also
  support-conditioned rather than universal
