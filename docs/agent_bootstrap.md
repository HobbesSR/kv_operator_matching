Create a new GitHub repo and local project named `kv_operator_matching`.

This repo is a fresh experimental branch of work that grows out of an earlier repo named `kv_compaction_experiment`. That earlier repo focused on approximating / reproducing the **Attention Matching** paper direction on Qwen 2 with online evidence collection experiments. The new repo is not just a continuation under the same framing. It is a shift in framing:

* old framing: compact KV by matching attention behavior
* new framing: approximate the **head-local operator** induced by the cache, using a numerator/denominator factorization
* old repo name has become a misnomer
* new repo should be initialized cleanly, but with awareness of the old experiment as guidance and precedent

I want you to set up the new repo so it is ready for iterative experiments, with documentation that captures the theory sketch and a practical initial experimental plan.

## Immediate repo tasks

1. Create a new git repo locally named `kv_operator_matching`.
2. Initialize git.
3. Create a sensible Python project structure for experiments.
4. Create a GitHub repo named `kv_operator_matching` and push the initial commit.
5. Create a strong `README.md`, plus one or more docs files capturing:

   * project motivation
   * theory sketch
   * mathematical framing
   * experimental roadmap
   * relationship to the old repo / old experiments
6. Add a minimal but clean Python environment setup:

   * `pyproject.toml`
   * basic package layout
   * lint/format config if lightweight
   * `.gitignore`
7. Do **not** try to port the entire old repo immediately. Start with clean scaffolding and documentation first.
8. Leave obvious placeholders / TODOs for later code import or adaptation from the old repo.

## What this repo is about

The new program is about moving from “attention matching” to a broader and more natural framing:

For each head, the KV cache induces an operator over future queries. We want to approximate that operator under a memory budget, using online-collected evidence from real queries, and eventually with a progressive / multiresolution representation.

The current work is **not** yet the full progressive system. The first concrete goal is much narrower and more practical:

* collect live online query evidence
* use boundary-time or checkpoint-time compaction / refit
* approximate the head response through a factorization into numerator and denominator terms
* start with fixed support and β-refit / support reweighting
* compare against prior baselines similar to the Qwen 2 online experiment from the old repo

This repo should make that scope clear.

---

# Background / relationship to previous work

The earlier repo (`kv_compaction_experiment`) explored attention matching style compaction on Qwen 2, including online evidence collection and experiments that approximate the paper’s strategy.

This new repo should treat that previous work as:

* implementation guidance
* a source of useful experiment patterns
* a source of baseline methodology

But conceptually, this repo is a new direction.

The earlier repo’s framing was approximately:

* compact a block of KV cache
* preserve attention outputs / attention mass on a reference query set
* approximate paper-style Attention Matching

The new framing is:

* the cache induces a **head-local operator**
* preserve / approximate that operator on a query distribution induced by live use
* use a factorization into:

  * denominator / mass-like term
  * numerator / value-weighted term
* use that as the basis for selection, merging, refit, and later progressive representation

So this repo is not “Attention Matching 2”. It is a new operator-matching direction that starts from lessons learned from the old work.

---

# Core theoretical framing

For a single attention head with cache entries ((k_i, v_i)), define:

[
Z_\mu(q) = \sum_i e^{\langle q, k_i \rangle}
]

[
N_\mu(q) = \sum_i e^{\langle q, k_i \rangle} v_i
]

and the induced head response operator:

[
A_\mu(q) = \frac{N_\mu(q)}{Z_\mu(q)}
]

The goal is to replace the original cache-induced representation (\mu) with a compact approximation (\hat{\mu}) such that, for the relevant future queries (q), the induced response remains close:

[
A_{\hat{\mu}}(q) \approx A_\mu(q)
]

The factorization matters because:

* it exposes the structure of the operator
* it is more expressive than plain attention-mass matching
* it lends itself to synthetic / merged elements
* it is naturally compatible with future progressive residual representations

## Why this generalizes Attention Matching

Attention Matching can be seen as a restricted special case of the broader operator-preservation problem:

* match local attention output
* match attention mass
* on a finite reference query set
* under a compacted KV representation

This new repo generalizes that perspective:

* broader operator framing
* explicit (N/Z) factorization
* room for merged/synthetic elements
* room for progressive residual structure
* room for online evidence-driven update policies

## Important concatenation property

One reason the (N/Z) factorization is useful is that it survives concatenation with arbitrary future KV blocks.

For a prefix block (P) and arbitrary future block (F),

[
A_{P \Vert F}(q)
================

\frac{N_P(q) + N_F(q)}{Z_P(q) + Z_F(q)}
]

So if a compact approximation preserves (N_P) and (Z_P) well, its effect under arbitrary future concatenation also remains well-behaved. This is one of the main reasons the factorization is the right object.

---

# Practical first-stage objective

The first practical stage should stay conservative.

We are **not** yet trying to build:

* a full dynamic multiresolution service policy
* a fully online hierarchical progressive memory structure
* a learned router
* an end-to-end latent memory system

We **are** trying to build:

* live query evidence collection
* empirical query-bank based evaluation
* fixed-support operator matching using (N/Z)
* online or boundary-time β-refit / support reweighting
* clean comparisons to familiar baselines

## Empirical query-bank objective

For a weighted empirical query bank ({(q_r, w_r)}), define:

[
\widehat{\mathcal{L}}_Z
=======================

\frac{1}{W}
\sum_r w_r
\left(
Z_{\hat{\mu}}(q_r) - Z_\mu(q_r)
\right)^2
]

[
\widehat{\mathcal{L}}_N
=======================

\frac{1}{W}
\sum_r w_r
\left|
N_{\hat{\mu}}(q_r) - N_\mu(q_r)
\right|_2^2
]

with

[
W = \sum_r w_r
]

and the combined surrogate:

[
\widehat{\mathcal{L}}_{\mathrm{lin}}
====================================

\widehat{\mathcal{L}}_Z + \widehat{\mathcal{L}}_N
]

The true quantity we care about is response error:

[
\widehat{\mathcal{L}}_{\mathrm{true}}
=====================================

\frac{1}{W}
\sum_r w_r
\left|
A_{\hat{\mu}}(q_r) - A_\mu(q_r)
\right|_2^2
]

Operationally:

* optimize the tractable surrogate
* verify on the true response metric before accepting a candidate

This should be one of the central ideas documented in the repo.

---

# Representation for the first implementation

Start with the simplest practical family:

[
\hat{\mu} = \sum_j \beta_j \delta_{(\hat{k}_j, \hat{v}_j)}
]

where initially:

* ((\hat{k}_j, \hat{v}_j)) are fixed support elements
* (\beta_j \ge 0) are fit / refit coefficients
* support may initially come from baseline selection rules or imported candidate sets
* later this can be extended to merged/synthetic support elements

Important:

* fixed support + β-refit is the first theorem-friendly / experiment-friendly fast path
* do not jump straight to a complicated progressive tree

---

# Online / service-style architecture

The repo should explain the system architecture clearly.

## Online path

During inference:

* collect live query vectors
* collect attention evidence or enough statistics to build an empirical query bank
* update online evidence structures cheaply
* do not mutate the live cache representation directly in a risky way

## Boundary / checkpoint path

At prompt/response boundaries or similar checkpoints:

* form or refresh a candidate compact representation
* solve β-refit on a fixed support or candidate support
* evaluate on a held-out or reference query subset
* only activate / swap the new representation if it verifies well enough

This is important:

* expensive work happens at checkpoints
* the hot path stays cheap
* verification gates deployment

That service-safe discipline should be reflected in both docs and code structure.

---

# Progressive representation: motivation, not immediate scope

The progressive / residual story is an important motivation, but not the first implementation target.

The intended future direction is something like:

[
\mu = \mu_0 + \delta_1 + \delta_2 + \cdots
]

where:

* (\mu_0) is a coarse approximation
* (\delta_i) are residual refinements
* coarse-only should be cheap and useful
* deeper levels should add detail where needed
* future service policy might dynamically vary effective resolution

But this should be presented as:

* future architecture motivation
* not required for the first success criterion

The docs should make that distinction very explicit.

---

# Key open theoretical / experimental questions

Please document these explicitly as open questions / future theory hooks.

## 1. Response sparsity

The key abstraction is not token sparsity but **sparsity in operator importance under the observed query distribution**.

We want to test whether the cache-induced operator is compressible under the empirical query bank.

## 2. Spectral decay

One high-leverage future test:

* build empirical feature / response Gram structures
* test whether they exhibit strong spectral decay

If they do, this supports progressive / low-dimensional approximation.

## 3. Tree compatibility

Even if the response geometry is low-dimensional, it may not be naturally captured by a tree of nested merges. This is a distinct question.

## 4. Approximate submodularity / diminishing returns

Greedy ordering or hierarchy-building depends on whether response gains show approximate diminishing returns in practice.

## 5. Positional structure

Open issue:

* can positional behavior be handled implicitly by response preservation?
* or does it require explicit guardrails / descriptors / reindexing logic?
* likely the answer depends on regime

## 6. Merge / synthetic support generation

Attention mass and OMP may not be the right upstream criterion under the new framing.
Alternatives to explore later:

* (N/Z)-residual greedy selection
* leverage / Gram-aware scoring
* behavioral clustering proposal mechanisms
* merge + fit + verify strategies

These should go into the theory / roadmap docs.

---

# First experimental goal

The first concrete experiment in this repo should mirror the spirit of the old Qwen 2 online experiment, but under the new objective.

Goal:

* build a Qwen 2 online experiment scaffold
* collect live query vectors / live evidence
* construct an empirical query bank
* compare fixed-support (N/Z)-matching and β-refit against existing-style baselines

## Candidate baselines to include

Use similar targets to the old repo where practical:

* recency baseline
* highest-attention / raw attention mass baseline
* fixed-support no-refit control
* operator-matching β-refit path

Potentially later:

* OMP-like or greedy residual-based support selection under (N/Z)

But the first version should stay manageable.

---

# Expectations for initial repo contents

Please create at least the following:

## Top-level

* `README.md`
* `pyproject.toml`
* `.gitignore`
* `LICENSE` if you think a default is appropriate; otherwise leave TODO
* `src/kv_operator_matching/`
* `experiments/`
* `docs/`

## Suggested docs

* `docs/theory_sketch.md`
* `docs/roadmap.md`
* `docs/old_repo_relationship.md`

## Suggested source structure

* `src/kv_operator_matching/__init__.py`
* `src/kv_operator_matching/config.py`
* `src/kv_operator_matching/query_bank.py`
* `src/kv_operator_matching/objectives.py`
* `src/kv_operator_matching/beta_fit.py`
* `src/kv_operator_matching/verification.py`
* `src/kv_operator_matching/baselines.py`
* `src/kv_operator_matching/types.py`

## Suggested experiment structure

* `experiments/qwen2_online_nz_match/`
* placeholder script or notebook entrypoint for the first online experiment
* clear TODO comments for where old-repo code patterns may be adapted

---

# README expectations

The README should:

* explain the repo is a new direction after `kv_compaction_experiment`
* explain the (N/Z) operator framing succinctly
* explain the first-stage implementation scope
* explain what is already motivated versus what is future work
* list immediate milestones
* state that this is experimental and research-oriented

---

# Roadmap expectations

The roadmap doc should roughly separate:

## Phase 1

* repo scaffolding
* docs
* basic objective implementations
* empirical query bank
* fixed-support β-fit
* Qwen 2 online experiment scaffold

## Phase 2

* baseline comparison
* held-out verification
* better support proposal logic
* (N/Z)-aware greedy alternatives to attention mass / OMP

## Phase 3

* merged / synthetic support elements
* more principled merge proposal logic
* response sparsity / spectral tests

## Phase 4

* progressive residual representation
* tree / hierarchy work
* dynamic resolution policy experiments

---

# Important implementation / writing guidance

* Keep code and docs clean and restrained.
* Avoid pretending the full progressive system already exists.
* Be explicit about what is theorem-friendly now versus only a future idea.
* Preserve the connection to the old experiment lineage, but make the conceptual shift very clear.
* Do not overstate rigor. State what is a sketch, what is an algorithmic plan, and what is an open problem.
* Prioritize making the repo a good launchpad for iterative experiments rather than prematurely filling it with speculative code.

---

# Deliverable style

When done:

* ensure the repo is initialized and pushed
* make the initial commit message sensible
* summarize what was created
* highlight any TODOs or places where old repo code should likely be referenced next

If there are points where you need to make a sensible choice without asking me, do so conservatively and document the choice in the repo.
