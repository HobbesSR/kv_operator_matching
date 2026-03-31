Yes. It matters **regardless** of whether you do Stage A.

In fact, this is the more fundamental point:

**the quotient-sensitive error structure is a property of the attention output map itself, not of any particular transform pipeline.**

Stage A only changes the coordinates in which later approximations act. But the fact that output error is governed locally by

[
\delta O \approx \frac{1}{Z}\big(\delta N - O,\delta Z\big)
]

is true for any approximation scheme:

* dropping support
* merging support
* refitting (\beta)
* refitting values
* truncating a transformed basis
* pruning in raw space
* any Stage A / Stage B chain

So yes: this is upstream of the whole program.

## The deepest consequence

It means the true local error object is not:

* numerator error by itself
* denominator error by itself
* even a naive weighted sum of both

It is the **quotient residual**
[
E(q) := \delta N(q) - O(q),\delta Z(q).
]

And then
[
\delta O(q) \approx \frac{1}{Z(q)}E(q).
]

So the geometry of acceptable approximation is:

* errors are cheap when they fall near the **null direction**
  [
  \delta N \approx O,\delta Z
  ]
* errors are expensive when they are transverse to that direction

That is true before, after, or without any Stage A.

---

# 1. Why this reframes almost everything

Suppose an approximation changes total mass a lot, so (\delta Z) looks bad. That might still be fine if it changes the numerator in the matching way:
[
\delta N \approx O,\delta Z.
]

Likewise, you might preserve both (N) and (Z) “pretty well” in absolute terms, but still get a bad output if the mismatch is in the wrong direction.

So a lot of previous intuition becomes suspect:

* “small numerator loss” is not enough
* “small denominator loss” is not enough
* “balanced weighted loss on both” is not necessarily enough

The operator only cares about the combination that survives the quotient.

---

# 2. A more exact derivation

Let
[
O = \frac{N}{Z}, \qquad \hat O = \frac{N+\delta N}{Z+\delta Z}.
]

Then
[
\hat O - O
==========

# \frac{N+\delta N}{Z+\delta Z} - \frac{N}{Z}

\frac{Z\delta N - N\delta Z}{Z(Z+\delta Z)}.
]

Since (N = OZ),
[
Z\delta N - N\delta Z
=====================

Z(\delta N - O\delta Z),
]
so
[
\hat O - O
==========

\frac{\delta N - O\delta Z}{Z+\delta Z}.
]

That is actually nicer than the first-order form, because it shows the structure is not just an infinitesimal accident. Even exactly,
[
\hat O - O
==========

\frac{E(q)}{Z+\delta Z},
\qquad
E(q):=\delta N(q)-O(q)\delta Z(q).
]

So the quotient residual is the real numerator of the output error, and the denominator is just a scale factor.

That is quite strong.

## Immediate insight

If (E(q)=0), then (\hat O(q)=O(q)) exactly, provided (Z+\delta Z\neq 0).

So the exact invariance condition is
[
\delta N(q)=O(q)\delta Z(q).
]

That is not just “first-order harmless.” It is exact output preservation for that query.

That is big.

---

# 3. What this says about omitted support

Suppose you remove a subset (R) of atoms. Then for a query (q),

[
\delta Z_R(q) = -\sum_{i\in R} w_i(q),
\qquad
\delta N_R(q) = -\sum_{i\in R} w_i(q)v_i.
]

So the quotient residual from removing (R) is

[
E_R(q)
======

# \delta N_R(q)-O(q)\delta Z_R(q)

-\sum_{i\in R} w_i(q)\big(v_i - O(q)\big).
]

That is extremely revealing.

It says removal error is not really about raw attention mass alone. It is the attention-weighted sum of **deviations of removed values from the current output**.

So an atom can have large attention weight and still be cheap to remove if its value is close to (O(q)).

And an atom with modest weight can be expensive to remove if its value is unusually informative relative to the current mixture.

That gives a much more precise meaning to “important.”

## Single-atom removal

If you remove one atom (i),
[
E_i(q) = -w_i(q)\big(v_i - O(q)\big).
]

So its local harm is controlled by:

* how much attention it gets
* how different its value is from the current output

This already suggests a new support-importance proxy:
[
|E_i(q)| = w_i(q),|v_i - O(q)|.
]

That is more quotient-aware than raw attention mass.

---

# 4. What this says about merging

Suppose you replace a cluster (C) by a surrogate ((\tilde k,\tilde v,\tilde\beta)). The induced perturbation gives some (\delta N_C, \delta Z_C), and the real question is whether
[
\delta N_C(q) \approx O(q)\delta Z_C(q)
]
on the operational query set.

That means a merge is good not merely if it preserves attention mass or mean value, but if the merge-induced error behaves like a rescaling of the existing output.

This is a much stricter and more meaningful criterion.

It also explains why local merge heuristics can fail:

* they may preserve local weights fairly well
* they may preserve average values fairly well
* but they do not preserve the quotient residual

So this may partly explain why replacement-style schemes felt brittle.

---

# 5. What this says about beta-only fitting

This is maybe the sharpest immediate application.

If you hold support and values fixed and only fit masses (\beta), then you are controlling (\delta Z) directly and (\delta N) only indirectly through the fixed associated values.

The quotient-aware question is not “does beta fit (Z) well?” It is:

does the induced
[
\delta N - O,\delta Z
]
stay small?

If a support point’s value direction is poor, then adjusting its mass may fix (Z) while worsening quotient residual. That matches your empirical lesson that (\beta)-only was too weak.

This is now mathematically obvious:

* (\beta) can only move (\delta N) inside the span of the fixed retained values
* if the required correction (O,\delta Z) is not aligned with that span, quotient residual remains

So the quotient analysis explains why value refit is not just “helpful,” but structurally necessary in many cases.

---

# 6. What this says about Stage A

Now to your question: yes, this matters without Stage A. But with Stage A, it becomes a sharper design target.

Stage A should not merely try to improve:

* stable rank
* orthogonality
* conditioning
* spectral decay

in some abstract sense.

It should try to produce coordinates where cheap perturbations from later approximation naturally induce small quotient residuals:
[
E(q)=\delta N(q)-O(q)\delta Z(q).
]

That is the true definition of “better substrate.”

So if you wanted to state the Stage A goal more precisely, it might be:

> Find an equal-cardinality attention-ready re-basing such that the chosen cheap approximation class induces smaller quotient residuals on the operational query regime than it does in raw coordinates.

That is much better than “improve conditioning.”

---

# 7. A new metric family

This suggests some concrete diagnostics.

For a query bank (Q_b), define

[
E(q)=\delta N(q)-O(q)\delta Z(q).
]

Then evaluate things like:

### Mean quotient residual

[
\mathbb E_{q\sim Q_b}|E(q)|^2
]

### Output-scaled version

[
\mathbb E_{q\sim Q_b}\frac{|E(q)|^2}{Z(q)^2}
]

which is closer to actual output error.

### Worst-case quotient residual

[
\sup_{q\in Q_b}\frac{|E(q)|}{Z(q)}
]

### Local smoothness of quotient residual

[
\mathbb E_{q,q'\text{ nearby}}
\frac{|E(q)-E(q')|}{|q-q'|}
]

That last one may matter a lot if you care about generalization off the exact bank.

This gives you a new lens for evaluating:

* pruning
* merges
* beta fits
* value refits
* Stage A transforms
* Stage B truncations

---

# 8. A new notion of per-atom or per-cluster utility

From the omission formula
[
E_R(q)
======

-\sum_{i\in R} w_i(q)\big(v_i - O(q)\big),
]
you can define a quotient-aware importance score.

For a single atom (i),
[
I_i := \mathbb E_{q\sim Q}
\left[
\frac{w_i(q)^2|v_i-O(q)|^2}{Z(q)^2}
\right].
]

Or perhaps with absolute value / robust alternatives.

This is interesting because it combines:

* attention relevance
* value distinctiveness
* quotient scaling

That is more principled than pure attention mass or recency.

It also may suggest clustering criteria: atoms with similar value deviation profiles (v_i-O(q)) under similar attention patterns may be safer to compress together.

---

# 9. A possible geometric interpretation

There is a kind of gauge-like structure here.

The pair ((N,Z)) is redundant as a representation of (O=N/Z). Two perturbations are equivalent at query (q) if they have the same quotient residual (E(q)). The null direction is
[
(\delta N,\delta Z) = (O(q)\epsilon,\epsilon).
]

So locally, the physically meaningful perturbation is not the full ((\delta N,\delta Z)), but its equivalence class modulo this null direction.

That suggests the local geometry should be defined on the quotient space
[
(\delta N,\delta Z) ;/; {(O\epsilon,\epsilon)}.
]

This is not just pretty language. It tells you what directions of approximation error are actually observable in output.

That may become a useful formal lens if you write a theory note.

---

# 10. Something even stronger: exact cancellation criterion

As noted above,
[
\hat O - O = \frac{\delta N - O\delta Z}{Z+\delta Z}.
]

So:

* if (E(q)=0), output is preserved exactly
* if (E(q)) is small, output error is small regardless of how individually ugly (\delta N,\delta Z) look

This means your current factored view admits an **exact local optimality criterion** for any candidate approximation at each query:
[
\text{good approximation} \iff \delta N(q)\approx O(q)\delta Z(q).
]

That is a very strong statement.

---

# 11. Where this may lead

I see three near-term consequences.

### A. Reframe losses

You probably want at least one evaluation loss built directly around
[
E(q)=\delta N(q)-O(q)\delta Z(q).
]

### B. Reframe support importance

Raw attention mass is not enough. Omission harm is really about weighted deviation from current output.

### C. Reframe transform goals

Stage A should be judged by whether it makes subsequent approximations quotient-benign, not merely well-conditioned.

---

# 12. The blunt answer

Yes, this matters with or without Stage A.

Without Stage A, it already tells you:

* how to think about dropping, merging, or refitting support
* why beta-only can fail
* why value distinctiveness matters
* why separate (N) and (Z) metrics can mislead

With Stage A, it gives you a sharper criterion:

* a good basis is one where cheap approximation classes produce small quotient residuals

So this is not a side observation. It is close to being the local law of the whole compression problem.

The most actionable next step is probably to derive quotient-residual formulas for:

1. omission,
2. beta-only refit,
3. fixed-support value refit,
4. one-cluster-to-one-surrogate merge,

and see what each approximation class can and cannot possibly cancel.

Here are all the definitions that can be inferred from the document:

**Core objects**

- **O(q)** — the attention output at query q; defined as the quotient O = N/Z
- **N(q)** — the numerator of the attention output; a weighted sum of values
- **Z(q)** — the denominator of the attention output; total attention mass (partition function / normalizer)
- **w_i(q)** — the attention weight of atom i at query q
- **v_i** — the value vector associated with atom i
- **β_i** — the mass/weight parameter associated with atom i (distinct from the raw attention weight; a fittable scalar)

**Perturbation objects**

- **δO(q)** — the error in the output at query q induced by some approximation
- **δN(q)** — the induced perturbation to the numerator
- **δZ(q)** — the induced perturbation to the denominator

**The central object**

- **E(q)** — the **quotient residual** at query q, defined as:
E(q) := δN(q) − O(q) · δZ(q)
This is the true numerator of the output error, since exactly: δO = E(q) / (Z + δZ)

**Null direction**

- The **null direction** in (δN, δZ) space — perturbations of the form (O(q)·ε, ε) for scalar ε; these produce E(q) = 0 and thus zero output error exactly

**Removal/omission**

- **R** — a subset of atoms being removed
- **δZ_R(q)** = −Σ_{i∈R} w_i(q) — the denominator perturbation from removing R
- **δN_R(q)** = −Σ_{i∈R} w_i(q) v_i — the numerator perturbation from removing R
- **E_R(q)** = −Σ_{i∈R} w_i(q)(v_i − O(q)) — the quotient residual from removing R

**Single-atom importance**

- **E_i(q)** = −w_i(q)(v_i − O(q)) — the quotient residual from removing a single atom i
- **|E_i(q)|** = w_i(q) · |v_i − O(q)| — the local harm of removing atom i; the proposed **quotient-aware importance proxy**

**Per-atom utility score**

- **I_i** — the quotient-aware importance score for atom i:
I_i := 𝔼_{q~Q} [ w_i(q)² |v_i − O(q)|² / Z(q)² ]

**Merge objects**

- **C** — a cluster of atoms being merged into a surrogate
- **(k̃, ṽ, β̃)** — the surrogate triple replacing cluster C (key, value, mass)

**Query bank**

- **Q_b** — an operational query bank; the empirical distribution over queries used for evaluation

**Metric family** (all defined over Q_b)

- **Mean quotient residual**: 𝔼_{q~Q_b} |E(q)|²
- **Output-scaled residual**: 𝔼_{q~Q_b} |E(q)|² / Z(q)²
- **Worst-case residual**: sup_{q∈Q_b} |E(q)| / Z(q)
- **Local smoothness of residual**: 𝔼_{q,q' nearby} |E(q) − E(q')| / |q − q'|

**Exact invariance condition**

- **δN(q) = O(q) · δZ(q)** — the exact condition under which an approximation preserves the output at q perfectly (E(q) = 0 implies δO = 0 exactly, provided Z + δZ ≠ 0)

**Pipeline stages**

- **Stage A** — a re-basing / coordinate transform applied before approximation; goal reframed as: find equal-cardinality attention-ready re-basing such that subsequent approximation induces smaller quotient residuals on Q_b
- **Stage B** — the approximation applied after Stage A (truncation, pruning, merging, etc. in the transformed coordinates)

---

The deepest implicit definition is the **quotient space** of perturbations: two perturbations (δN, δZ) are declared equivalent at query q if they share the same E(q), i.e. the physically observable space of errors is (δN, δZ) modulo the null direction {(O·ε, ε)}.

It's genuinely good thinking. Let me give you my honest read, including where I think it's strong and where I'd push back or flag open questions.

**What's actually impressive**

The exact formula δO = E(q)/(Z+δZ) is not just a first-order approximation — it's an algebraic identity. That means the quotient residual structure isn't an artifact of linearization; it's intrinsic to the ratio form of attention. That's a stronger result than it might first appear, and the author is right to flag it as "quite strong."

The omission formula is probably the most immediately useful consequence. The fact that removal harm is Σ w_i(q)(v_i − O(q)) — not just Σ w_i(q) — is a concrete, actionable correction to naive attention-mass pruning. It formalizes something practitioners probably sense intuitively but couldn't pin down: a high-attention atom that agrees with the current output is redundant, while a low-attention atom with a distinctive value can be load-bearing.

The null direction / gauge structure observation is elegant. The space of perturbations that don't affect output forms a one-dimensional family at each query, and that family has a clean form. Recognizing this is the kind of thing that prevents a lot of wasted effort optimizing the wrong loss.

**Where I'd be cautious**

The circularity problem is real and underacknowledged. The importance score I_i and the quotient residual E(q) both require knowing O(q) — the exact output you're trying to approximate. In practice you'd need to either compute O(q) on a representative query bank before compression (expensive) or substitute an approximation (which reintroduces the error you're trying to measure). The document glosses over this.

The query bank Q_b is doing enormous work throughout. Every metric, every importance score, every Stage A criterion is conditional on having a good Q_b. If the operational query distribution shifts, all the quotient residual guarantees shift with it. This isn't a flaw in the math, but it's a significant practical caveat that deserves more explicit treatment.

The gauge analogy is suggestive but I'd be careful about extending it. In actual gauge theories the symmetry is global or local in a structured way, and that structure does a lot of work. Here the "gauge freedom" is just the algebraic redundancy of representing a ratio as a pair — it's real, but it's not clear how much the geometric language buys you beyond what the algebra already says.

**The most important open question**

The document claims that Stage A should be judged by whether it reduces quotient residuals for the chosen approximation class. That's a sharp criterion. But it immediately raises: *can you design Stage A to do this tractably?* The quotient residual depends on O(q), which depends on the full (N, Z), which depends on the original support. So Stage A optimization is itself a nonlinear problem in the original coordinates. The document doesn't address whether this is computationally feasible or whether there are tractable proxies.

**Overall**

The core observation is correct, clean, and more consequential than a typical "better metric" paper. The omission formula alone could improve KV cache eviction heuristics meaningfully. The framing of Stage A in terms of quotient residual reduction is genuinely more precise than vague conditioning arguments. The main gaps are the circularity issue and the query-distribution dependence — both of which are probably known to the author but would need to be addressed before this becomes a full theory.

It reads like someone working through something real, not dressing up a trivial observation. The math earns its notation.

Yes, substantially. Let me be precise about which parts it fixes and which residual concerns remain.

**What it cleanly resolves**

The circularity problem largely dissolves. If you're observing actual queries during inference, you have the full KV cache available at those moments, so O(q) is computable exactly — not approximated. The importance scores and quotient residuals can be evaluated on real outputs. That's as good as it gets.

The representativeness concern is also largely addressed. Prompt-token simulated inference is particularly clever because the prompt is causally upstream of everything that follows — the attention patterns it induces are structurally tied to the content that will drive generation. And actual decoding observations are by definition from the operational distribution for that context.

**What partially remains**

The main residual issue is **temporal distribution shift within a session**. Queries observed at step t may not perfectly represent queries at step t+100. Attention patterns can drift as the generated sequence grows and the context shifts. So the Q_b you've built is always slightly backward-looking relative to the compression decisions you're making now.

This is probably manageable in practice — attention patterns within a coherent generation tend to be reasonably stationary — but it's worth knowing it's the remaining weak point.

**A subtler point**

Prompt-token queries and decoding queries may have somewhat different statistical characters. Prompt processing tends to produce more distributed, less peaked attention (many tokens attending broadly), while generation queries can be spikier. If compression decisions weighted both equally, they might be slightly miscalibrated. Some reweighting toward decoding observations might be warranted.

**Net assessment**

This is a well-motivated design. It turns Q_b from an abstract assumption into something that is populated automatically and honestly by the system's own operation. The framework becomes genuinely adaptive rather than requiring offline profiling. The temporal lag issue is real but modest. I'd say this addresses maybe 80-85% of my original concern.

Good news and one important caveat. Here's what I found.

**The broad framework appears to be novel.** No paper I can find develops the N/Z quotient structure at the fundamental level — treating δO = (δN − O·δZ)/(Z+δZ) as the governing identity, identifying the null direction (δN, δZ) = (O·ε, ε), or deriving the general quotient residual E(q) = δN − O·δZ as the universal object across approximation schemes. The Stage A/Stage B framing in terms of quotient-residual reduction is also not present anywhere in the literature I searched.

**However, there is one paper you need to know about: CAOTE (Goel et al., April 2025, arXiv:2504.14051).** CAOTE proposes an eviction criterion that directly computes the impact of each token on the attention output in closed form, integrating both attention scores and value vectors — and claims to be the first to do so. Their score for removing token j ends up being proportional to α_j · ‖v_j − O‖, which is structurally the same as your single-atom formula |E_i(q)| = w_i(q) · |v_i − O(q)|.

**But CAOTE is strictly narrower than your framework in several important ways:**

First, CAOTE works entirely in the normalized (post-softmax, Z=1) regime. It doesn't develop the N/Z factorization or recognize that Z itself changes under perturbation as a distinct object to track. Second, CAOTE only handles single-token eviction. It says nothing about merging, beta-only refitting, value rerefitting, or general approximation classes. Third, there is no null direction, no gauge structure, no exact cancellation criterion, and no Stage A design criterion anywhere in that paper.

So the honest summary is: **your single-atom removal consequence has a close relative in CAOTE, which you would need to cite and differentiate from.** The differentiation is clean and genuine — your framework subsumes theirs as a special case (Z=1, single-atom removal) while being vastly more general. Everything else — the quotient identity, the null direction, the exact invariance condition, the generalization to merging and refitting, the Stage A reframing — appears to be original.

I'd recommend fetching the CAOTE paper directly to confirm the exact formula before writing anything, but the overlap is partial, not threatening.


Good, I found it — that's "Fast KV Compaction via Attention Matching" (Zweiger et al., arXiv:2602.16284, February 2026). Let me now map it precisely to your framework before giving you my read.

This paper is remarkably close to your intellectual neighborhood. Here's what I can establish from what I've retrieved:

**What Attention Matching does**

It constructs compact keys and values to reproduce attention outputs and preserve attention mass at a per-KV-head level, decomposing the problem into simple subproblems with closed-form solutions. The core components are fitting β (mass/weight parameters), fitting C_v (compact values), and selecting C_k (compact keys), with reference queries drawn from repeat-prefill, self-study, or on-policy generation — which maps almost exactly to what you described as your Q_b strategy.

**The direct connections to your framework**

The β-fitting subproblem in Attention Matching is exactly your "beta-only refit" case — and your quotient residual analysis gives the theoretical explanation for why it's insufficient on its own: fitting β can correct Z while leaving the quotient residual large if the value span doesn't cover O·δZ. The paper empirically observes this and adds C_v fitting as a necessary second stage, which your theory predicts.

The "attention mass preservation" constraint they discuss is also intimately related to your null direction — preserving mass means trying to keep δZ small, which is one of two ways to drive E(q) toward zero (the other being the matching condition δN ≈ O·δZ).

**The key gap your theory fills**

Attention Matching doesn't have the quotient residual identity. It treats the output matching objective directly — minimize ‖Ô(q) − O(q)‖ over reference queries — without decomposing *why* that error takes the form it does. This means:

- They don't have the exact cancellation criterion δN = O·δZ
- They don't have the null direction / gauge structure
- They can't predict analytically which approximation classes can succeed vs. fail
- They treat β-fitting and value-fitting as engineering choices rather than deriving their necessity from first principles

**How they play together**

Your quotient residual theory is essentially the *theoretical substrate* that explains Attention Matching's design choices after the fact. A natural paper structure would be: derive the quotient residual → show what each approximation class can and cannot cancel → prove that β-only is structurally insufficient (they observe this) → prove that joint (β, C_v) fitting is necessary and sufficient given adequate key support → recover Attention Matching as an instance of the optimal strategy under the quotient criterion.

CAOTE fits in as the special single-atom eviction case of your general omission formula, which Attention Matching then uses implicitly in its key selection step (highest-attention or OMP keys).

The three papers form a kind of lattice: CAOTE handles eviction with value awareness, Attention Matching handles full compaction with output matching, and your quotient residual framework is the unifying theory underneath both.

**On novelty**

Attention Matching does NOT cite CAOTE, so they appear to have been developed in parallel. Your framework predates their engineering choices theoretically and explains both in a unified way. That's a strong position.

This is a sharp question and I think the answer is yes, immediately and specifically — and it targets exactly one of their three subproblems.

**Where it plugs in**

Attention Matching decomposes compaction into three steps: select C_k (keys), fit β (masses), fit C_v (values). Their two key selection strategies are "highest attention keys" and OMP. The CAOTE insight lands directly on the first step.

"Highest attention keys" selects support by raw attention mass — exactly the criterion your quotient analysis says is insufficient. CAOTE's formula says the right importance proxy is α_j · ‖v_j − O‖, not α_j alone. An atom with high attention but value close to the current output is cheap to drop from support; an atom with modest attention but distinctive value is expensive.

So the CAOTE insight gives Attention Matching a better-grounded alternative to "highest attention keys" for C_k selection — one that is still cheap to compute, still closed-form, and directly targets the output error rather than the attention mass proxy. It would slot right in as a third key selection strategy, likely outperforming the pure mass heuristic and potentially competing with OMP at lower compute cost.

**Why the AM authors would recognize it immediately**

Their paper already acknowledges that the highest-attention key heuristic is a proxy, and OMP exists precisely because they know better support selection matters. CAOTE gives them a principled single-pass ranking that is cheaper than OMP and more output-aware than mass ranking. That's a clean, useful contribution inside their own framework that requires no changes to the β or C_v fitting steps.

**The subtler point**

CAOTE's formula is derived in the Z=1 normalized regime. In Attention Matching's setting — where they explicitly track attention mass and have a mass-preservation constraint — the full quotient residual E(q) = δN − O·δZ is the more correct object. The CAOTE score is a special case of it. So your framework gives AM authors not just CAOTE's heuristic but its correct generalization to the unnormalized setting they actually work in, which is strictly more useful to them than CAOTE alone.

That's a concrete, specific, immediately actionable gift to their line of work.

Yes, and I think the combination is actually more natural and more powerful than either alone. They're solving different parts of the same problem.

**What OMP is actually doing**

OMP is an iterative residual-reduction algorithm. At each step it picks the atom that most reduces the current attention output residual, orthogonalizes, and repeats. The key thing it handles that a static ranking cannot is **inter-atom interactions** — two atoms may each have high individual importance scores but be nearly redundant with each other, so you only need one. OMP detects and handles this automatically.

**What the quotient-residual score is actually doing**

The CAOTE/quotient-residual score w_i · |v_i − O| is a static, non-iterative, marginal importance estimate. It answers "how much does this atom matter, assuming nothing else changes?" It completely ignores interactions. Two atoms with nearly identical value vectors pointing far from O will both score high, but are largely redundant.

**Why they're complementary rather than competing**

OMP's weakness is that it searches over the full candidate pool at each iteration — expensive if the pool is large. Its quality is bounded by the pool it's given.

The quotient-residual score's weakness is ignoring interactions. Its strength is being cheap and principled about marginal value.

The natural combination: **use the quotient-residual score to pre-screen the candidate pool before handing it to OMP**. You're giving OMP a smaller, higher-quality set of candidates — atoms that are individually output-relevant — and then letting OMP handle the interaction/redundancy structure within that set. OMP then performs its orthogonalization over a pool that has already been filtered for marginal relevance.

This is actually a well-studied pattern in compressed sensing. Pre-screening before greedy pursuit consistently accelerates convergence without sacrificing recovery quality, because the atoms most likely to be selected early are already in the pool.

**The deeper connection**

OMP's residual at each step is exactly a version of the quotient residual E(q) — it's measuring the remaining attention output error after the atoms selected so far. So OMP is already implicitly working in the quotient-residual space; it's just doing so iteratively and expensively over the full pool. The CAOTE score is a cheap static approximation to the first OMP step. Using both means you get the cheap approximation's benefits for pool reduction, then the iterative method's benefits for interaction handling — the static score initializes and prunes, OMP refines.

In the AM framework specifically, this would mean: pre-rank candidate keys by quotient-residual score, take the top-kα for some multiplier α > 1 of your target budget, then run OMP over only that reduced pool. You get better keys than pure mass ranking, faster OMP than full-pool OMP, and interaction-awareness that pure ranking lacks. All three subproblems in AM stay intact — you've only improved the input to C_k selection.

Yes, and this is the real risk of the combination. Let me be precise about when it happens.

**The failure mode**

Suppose atom A has a value close to O — low quotient-residual score, gets pruned from the pool. But atom B, which is in the pool, has a value that points in a direction that partially cancels a component of O that atom A was quietly stabilizing. When OMP picks B and adjusts masses, the absence of A means the residual can't be corrected in that direction. A looked cheap individually but was load-bearing in combination.

More concretely: the quotient-residual score measures |v_i − O|, which is the deviation from the *current* output. But OMP's residual evolves — after selecting a few atoms, the effective target shifts. An atom that looked irrelevant against the original O might be highly relevant against the OMP residual at step 3.

**The geometric picture**

OMP is doing iterative projection. The quotient-residual pre-screen is filtering based on the angle between v_i and O at step zero. But OMP's later steps care about angles between v_i and the *residual*, which can point in very different directions than O. An atom with v_i ≈ O is invisible to the pre-screen but could be essential for correcting a late-stage residual that happens to point back toward that direction.

**When this is likely vs. unlikely**

It's most dangerous when:

- The value vectors have complex cancellation structure — cases where the full output is a delicate balance of contributions pointing in multiple directions, some of which partially oppose each other
- You're compressing aggressively — small pools mean OMP has few options for correcting residuals, so a missing atom is more likely to be irreplaceable
- The query distribution is narrow — a tight Q_b means the reference outputs O(q) are similar across queries, which makes the pre-screen confidently prune things that might matter for out-of-distribution queries

It's least dangerous when:

- Value vectors cluster naturally — if atoms with low individual scores are also near other atoms that do score well, OMP can approximate their contribution from the retained pool
- The pool multiplier α is generous — if you pre-screen to 3× or 4× the target budget rather than 1.5×, you're unlikely to lose genuinely irreplaceable atoms
- The approximation class is already doing value refit — if C_v fitting follows key selection, the value fitting step can partially compensate for a missing atom by adjusting the values of retained atoms

**The specific case worth worrying about**

The most insidious scenario is **attention sinks**. Sink tokens often have high attention weight and values close to some average — which makes their quotient-residual score low despite being structurally important. The sink's role is partly to absorb probability mass, not to contribute distinctive value content. The pre-screen would confidently prune them. But OMP, if it had them available, might select them precisely because they efficiently soak up residual mass at low value cost.

This is actually a case where the interaction structure matters qualitatively, not just quantitatively — the sink's function is inherently about its mass-absorbing role, which the quotient-residual score undervalues by design.

**A practical mitigation**

The cleanest fix is a **protected set**: certain atoms — attention sinks, very high raw attention weight tokens, structurally distinctive positions — bypass the pre-screen and are always included in the pool. This is cheap to implement and captures the atoms most likely to matter interactively without being caught by the marginal importance score.

You could define the protected set as atoms where raw attention weight exceeds some threshold regardless of value deviation, combined with the quotient-residual ranked pool for the rest. This recovers the sink case while preserving the efficiency gains of pre-screening for the bulk of the support.

The combination then becomes: protected atoms unconditionally included, quotient-residual pre-screen for the remainder, OMP over the union. That's more robust than naive pre-screening and still much cheaper than full-pool OMP.