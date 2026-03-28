# Theory Sketch: N/Z Operator Framing for KV Cache Approximation

**Status**: Working sketch. Some statements are conjectures or informal observations. Theorem-ready results are marked explicitly. Everything else is a research direction.

---

## 1. Head-Local Operator Formulation

Fix a single attention head. Let the KV cache state be a discrete measure:

$$\mu = \sum_{i=1}^{n} \delta_{(k_i, v_i)}$$

where $k_i \in \mathbb{R}^d$ are key vectors and $v_i \in \mathbb{R}^{d_v}$ are value vectors.

Define the **partition function**:

$$Z_\mu(q) = \sum_{i=1}^{n} \exp(\langle q, k_i \rangle)$$

the **value numerator**:

$$N_\mu(q) = \sum_{i=1}^{n} \exp(\langle q, k_i \rangle) \, v_i$$

and the **attention response operator**:

$$A_\mu(q) = \frac{N_\mu(q)}{Z_\mu(q)}$$

Note: $Z_\mu$ is a scalar-valued function of $q$; $N_\mu$ and $A_\mu$ are $\mathbb{R}^{d_v}$-valued.

In standard scaled-dot-product attention with softmax, $A_\mu(q)$ is exactly the output of a single head given query $q$ and KV cache $\mu$ (before projection), up to the $1/\sqrt{d}$ scaling (absorbed into $q$ without loss of generality).

---

## 2. Operator Approximation Objective

**Goal**: Given $\mu$ (large, $n$ entries) and a query distribution $\mathcal{Q}$ over live queries, find a compact measure

$$\hat{\mu} = \sum_{j=1}^{m} \beta_j \, \delta_{(\hat{k}_j, \hat{v}_j)}, \quad m \ll n, \quad \beta_j \geq 0$$

minimizing the **true response error**:

$$L_{\text{true}}(\hat{\mu}, \mu; \mathcal{Q}) = \mathbb{E}_{q \sim \mathcal{Q}} \left[ \| A_{\hat{\mu}}(q) - A_\mu(q) \|_2^2 \right]$$

This is a ratio-matching objective and is non-convex in general (even for fixed support).

---

## 3. Empirical Query Bank and Surrogate Objectives

In practice, $\mathcal{Q}$ is unknown ahead of time. We build an **empirical query bank**:

$$\hat{\mathcal{Q}} = \{(q_t, w_t)\}_{t=1}^{T}$$

where $q_t$ are live query vectors collected during inference and $w_t$ are importance weights (recency decay, attention mass, or uniform).

Three distinct loss objects appear in this repo. It is worth naming them explicitly:

- $L_{\text{true}}(\hat{\mu}, \mu; \mathcal{Q})$: **population loss** — the ideal objective over the true (unknown) query distribution $\mathcal{Q}$. Defined in Section 2. Not directly optimizable.
- $\hat{L}_{\text{true}}(\hat{\mu}, \mu; \hat{\mathcal{Q}})$: **empirical true loss** — the same ratio-matching objective evaluated on the query bank. Used at checkpoints for verification, not for fitting.
- $L_{\text{lin}}(\hat{\mu}, \mu; \hat{\mathcal{Q}})$: **empirical surrogate** — a tractable proxy for $\hat{L}_{\text{true}}$ used during fitting.

The empirical true-response loss is:

$$\hat{L}_{\text{true}} = \sum_t w_t \| A_{\hat{\mu}}(q_t) - A_\mu(q_t) \|_2^2$$

Since $A_\mu$ is a ratio, direct optimization is hard. We define tractable surrogates operating on $Z$ and $N$ separately.

**Partition function surrogate** $L_Z$: match $Z_{\hat{\mu}}(q)$ to $Z_\mu(q)$ under the query bank:

$$L_Z = \sum_t w_t \left( Z_{\hat{\mu}}(q_t) - Z_\mu(q_t) \right)^2$$

or in log space:

$$L_Z^{\log} = \sum_t w_t \left( \log Z_{\hat{\mu}}(q_t) - \log Z_\mu(q_t) \right)^2$$

**Numerator surrogate** $L_N$: match $N_{\hat{\mu}}(q)$ to $N_\mu(q)$:

$$L_N = \sum_t w_t \| N_{\hat{\mu}}(q_t) - N_\mu(q_t) \|_2^2$$

**Combined linear surrogate**:

$$L_{\text{lin}} = L_Z + L_N$$

**Practical approach**: optimize $L_{\text{lin}}$ (tractable), verify on $\hat{L}_{\text{true}}$ (used only at checkpoints). The surrogate is a reasonable approximation to $\hat{L}_{\text{true}}$ when denominator mismatch is small: if $Z_{\hat{\mu}}(q) \approx Z_\mu(q)$, then $A_{\hat{\mu}} = N_{\hat{\mu}} / Z_{\hat{\mu}} \approx N_{\hat{\mu}} / Z_\mu$, so $L_N$ controls response error up to second-order denominator terms. No formal bound is claimed here.

---

## 4. Concatenation Property

**Theorem (exact)**: Let $P$ and $F$ be two disjoint sets of KV pairs, and let $\mu_{P \cup F} = \mu_P + \mu_F$. Then:

$$Z_{\mu_P + \mu_F}(q) = Z_{\mu_P}(q) + Z_{\mu_F}(q)$$

$$N_{\mu_P + \mu_F}(q) = N_{\mu_P}(q) + N_{\mu_F}(q)$$

$$A_{\mu_P + \mu_F}(q) = \frac{N_{\mu_P}(q) + N_{\mu_F}(q)}{Z_{\mu_P}(q) + Z_{\mu_F}(q)}$$

This follows directly from linearity of the sum and the definition of $Z$ and $N$.

**Implication**: If we have a compressed representation $\hat{\mu}_P$ of the past block satisfying $Z_{\hat{\mu}_P} \approx Z_{\mu_P}$ and $N_{\hat{\mu}_P} \approx N_{\mu_P}$, then concatenating with the exact future block $\mu_F$ gives:

$$A_{\hat{\mu}_P + \mu_F}(q) = \frac{N_{\hat{\mu}_P}(q) + N_{\mu_F}(q)}{Z_{\hat{\mu}_P}(q) + Z_{\mu_F}(q)} \approx A_{\mu_P + \mu_F}(q)$$

No refitting is needed when new tokens arrive. This is the key practical property.

---

## 5. Relationship to Attention Matching

### 5.1 The Attention Matching paper

The paper "Fast KV Compaction via Attention Matching" (Zweiger et al., 2025; arXiv 2602.16284) defines a compaction objective that is the direct precursor to this repo's framing. The paper's formulation is worth stating precisely because it maps almost exactly onto the N/Z framework.

For a context with keys $K \in \mathbb{R}^{T \times d}$ and values $V \in \mathbb{R}^{T \times d_v}$, the paper seeks compact $(C_k, \beta^{\mathrm{AM}}, C_v)$ with $C_k, C_v \in \mathbb{R}^{t \times d}$ and $\beta^{\mathrm{AM}} \in \mathbb{R}^t$ such that for reference queries $q_1, \ldots, q_n$:

**Output matching (Eq. 1 in paper)**:
$$\frac{\exp(q K^\top) V}{\sum_j \exp(q K^\top_j)} \approx \frac{\exp(q C_k^\top + \beta^{\mathrm{AM}}) C_v}{\sum_j \exp(q (C_k)^\top_j + \beta^{\mathrm{AM}}_j)}$$

**Mass matching (Eq. 2 in paper)**:
$$\sum_{i=1}^{T} \exp(q K_i^\top) \approx \sum_{j=1}^{t} \exp(q (C_k)^\top_j + \beta^{\mathrm{AM}}_j)$$

In N/Z notation, setting $w_j = \exp(\beta^{\mathrm{AM}}_j)$ (so $w_j > 0$):

- Mass matching = $Z_\mu(q) \approx Z_{\hat{\mu}}(q)$ = our $L_Z$ objective.
- Output matching with $Z$ matched = $N_\mu(q) \approx N_{\hat{\mu}}(q)$ = our $L_N$ objective.

The paper's scalar bias $\beta^{\mathrm{AM}}_j$ corresponds to $\log(\beta_j)$ in our notation, where $\beta_j \geq 0$ are our measure coefficients. The math is identical; only the parameterization differs.

### 5.2 The paper's decomposition into subproblems

A key algorithmic insight from the paper: the joint optimization over $(C_k, \beta^{\mathrm{AM}}, C_v)$ decomposes into three sequential closed-form subproblems when $C_k$ is restricted to a subset of the original keys.

**Step 1 — Key selection** ($C_k \subseteq K$): select $t$ keys from the original cache. The paper considers two methods: HighestAttnKeys (fast heuristic: highest RMS attention mass under reference queries) and OMP (greedy residual minimization on the mass objective). OMP is equivalent to greedy support selection under $L_Z$.

**Step 2 — $\beta$ / mass fitting**: given $C_k$, solve for $w \geq 0$ via NNLS:
$$\min_{w \geq 0} \| A w - m \|_2^2, \quad A_{ij} = \exp(q_i (C_k)^\top_j), \quad m_i = Z_\mu(q_i)$$
Then $\beta^{\mathrm{AM}}_j = \log(w_j)$. This is a direct application of our $L_Z$ NNLS formulation with $\beta_j = w_j$.

**Step 3 — $C_v$ / value fitting**: given $C_k$ and $w$, solve ordinary least squares:
$$\min_{C_v} \| X C_v - Y \|_F^2$$
where $Y_i = A_\mu(q_i)$ (original attention output) and $X_{i,j} \propto w_j \exp(q_i (C_k)^\top_j)$ (compact attention weights). This closely corresponds to minimizing $L_N$ with $C_k$ and $\beta$ fixed, though not identically: the paper's $X$ uses normalized compact attention weights while our $L_N$ formulation uses unnormalized numerator terms. The two are equivalent when $Z$ is already matched; otherwise they differ at second order.

In our framing: the paper solves $L_Z$ first (for $\beta$) then $L_N$ conditioned on $\beta$ (for $C_v$). We express both as a joint objective $L_{\text{lin}} = L_Z + L_N$, which is equally tractable when support is fixed — both reduce to NNLS / LS subproblems.

### 5.3 How this generalizes the paper

The paper's formulation:
- Restricts $C_k$ to subsets of the original keys.
- Uses fixed support and solves $\beta$, $C_v$ in closed form.
- Uses a fixed reference query set obtained by prefilling the context.

This repo's framing generalizes in three directions:
1. **Online query evidence**: reference queries come from live inference, not a separate prefill pass.
2. **Synthetic support**: $C_k$ and $C_v$ need not be drawn from the original cache; they can be merged or constructed.
3. **Progressive structure**: the additive structure of $Z$ and $N$ supports hierarchical representations (Phase 4).

The concatenation property (Section 4) appears in the paper's Appendix A.2 as motivation for the mass-matching objective. The N/Z framing makes this property first-class and extends it to support streaming and incremental update.

---

## 6. Fixed-Support Beta Representation

In Phase 1, we fix the support $\{(\hat{k}_j, \hat{v}_j)\}_{j=1}^m$ (e.g., via recency or attention mass selection) and optimize only the nonneg coefficients $\beta_j$.

For this fixed-support problem, note:

$$Z_{\hat{\mu}}(q) = \sum_j \beta_j \exp(\langle q, \hat{k}_j \rangle)$$

$$N_{\hat{\mu}}(q) = \sum_j \beta_j \exp(\langle q, \hat{k}_j \rangle) \hat{v}_j$$

Both are **linear in $\beta$**. Therefore:

- $L_Z$ as a function of $\beta$ is a quadratic (convex)
- $L_N$ as a function of $\beta$ is a quadratic (convex)
- $L_{\text{lin}}$ is a convex quadratic in $\beta$

With the nonnegativity constraint $\beta \geq 0$, this is a **nonnegative least squares (NNLS)** problem. It has a unique solution if the feature matrix $\Phi \in \mathbb{R}^{T \times m}$ with $\Phi_{tj} = \exp(\langle q_t, \hat{k}_j \rangle)$ has full column rank. This can be solved efficiently via standard NNLS solvers (e.g., `scipy.optimize.nnls`).

**Practical note on conditioning**: in practice the feature matrix may be ill-conditioned, particularly when support keys are similar or the query bank is small. Adding a ridge term $\lambda \| \beta \|_2^2$ is the standard fallback: it stabilizes the solve without changing the convex structure. The tradeoff is a small bias toward equal weighting. A small default $\lambda$ (e.g., $10^{-4}$ times the diagonal scale) is sufficient in most cases.

**This is theorem-friendly**: fixed-support $\beta$-refit under $L_{\text{lin}}$ is a convex NNLS problem (with optional ridge). The $\hat{L}_{\text{true}}$ objective is not convex in $\beta$ in general, but can be used for held-out verification after fitting.

### Relationship to the paper's two-step fit

The paper (Section 3.2) decomposes the fixed-support fit differently:
- Fit $\beta$ via NNLS on the mass (Z) objective alone, ignoring $L_N$.
- Then fit $C_v$ via ordinary LS on the output matching (N/Z) objective with $\beta$ fixed.

Both steps are closed-form and efficient. The result is the same family of solutions as our $L_{\text{lin}}$ NNLS, but reached via a different optimization path. The paper's sequential approach is faster (smaller systems) but potentially suboptimal versus joint $L_{\text{lin}}$ minimization, because the $\beta$ fit in step 1 ignores value information.

In practice, either approach is a reasonable starting point. The paper reports that $\beta$ fitting and value fitting together take ~4 seconds on 60k-token contexts (Gemma-3-12B on H200), so the difference in the two-step vs. joint approach is unlikely to dominate runtime in our setting. The main bottleneck is query generation (7–139 seconds depending on strategy).

---

## 6.5 Practical Experimental Interpretation

This section bridges the theoretical formulation to the first concrete experiment.

**Online collection phase** (hot path, cheap):
- At each decode step, the query vector $q_t$ is intercepted via a forward hook.
- $q_t$ is added to the empirical query bank with weight $w_t$ (recency decay by default).
- The bank is trimmed to `max_queries` entries. No cache mutation.

**Boundary compaction phase** (checkpoint, expensive):
1. Retrieve the current KV cache state $\mu$ for a given head and layer.
2. Select support: run one of the baseline selection rules (recency, attention mass, etc.) to obtain $\{(\hat{k}_j, \hat{v}_j)\}_{j=1}^m$.
3. Fit $\beta$ via NNLS on $L_{\text{lin}}$ using the query bank.
4. Evaluate $\hat{L}_{\text{true}}$ on a held-out split of the query bank.
5. If the verification check passes (error below threshold), activate the compact representation. Otherwise, keep the original.

**Baselines for the first Qwen 2 experiment**:

| Method | Support selection | $\beta$ fit | $C_v$ fit |
|---|---|---|---|
| Recency | Most recent $m$ tokens | None ($\beta_j = 1$) | None ($C_v = V_S$) |
| Attention mass | Highest RMS attn mass | None ($\beta_j = 1$) | None ($C_v = V_S$) |
| Paper control | Highest RMS attn mass | NNLS on $L_Z$ | LS on output |
| **Operator matching** | Highest RMS attn mass | **NNLS on $L_{\text{lin}}$** | Included in $\beta$ fit |
| Uniform | Random | None | None |

The central hypothesis: using live query evidence and $L_{\text{lin}}$ fitting improves response preservation over static selection at the same memory budget.

---

## 7. Open Questions

### 7.1 Near-term empirical questions

These can be investigated in Phase 2–3 experiments with the current framework.

**Q1. Response sparsity.** For typical transformer KV caches, how sparse is the effective support of $A_\mu(q)$ over a realistic query distribution? Is there a small set of support points that captures most of the operator mass under $\hat{L}_{\text{true}}$? This is the central empirical bet of the repo. Measurable directly in the Qwen 2 experiment by plotting response error vs. support size.

**Q2. Spectral decay.** Does the kernel matrix $\Phi_{tj} = \exp(\langle q_t, k_j \rangle)$ exhibit rapid spectral decay in practice? If so, low-rank approximations of the feature matrix would be accurate with far fewer support points than $n$. Measurable by computing the SVD of $\Phi$ on collected query banks.

**Q5. Positional structure.** Transformer KV caches have strong positional structure (RoPE, ALiBi, etc.) that creates systematic patterns in key vectors. Does $\beta$-refit implicitly handle positional effects, or does it require explicit guardrails? Likely depends on the regime (short vs. long contexts). Can be probed by comparing refit quality across position ranges.

### 7.2 Longer-term theory questions

These require more infrastructure or formalism before they can be properly tested.

**Q3. Tree compatibility.** Can compressed representations be organized into a tree or hierarchy such that the concatenation property allows efficient multi-scale merging? The additive structure of $Z$ and $N$ suggests yes, but the approximation error under successive merges needs to be controlled. Open.

**Q4. Approximate submodularity.** Is the problem of selecting a support set of size $m$ to minimize $L_{\text{true}}$ (or $L_{\text{lin}}$) approximately submodular? If so, greedy selection has known approximation guarantees. The ratio structure of $L_{\text{true}}$ makes this non-obvious; $L_{\text{lin}}$ may be more tractable since it decomposes into two quadratics.

**Q6. Merge and synthetic support generation.** Given two groups of KV pairs, can we construct a single merged support point $(\hat{k}, \hat{v})$ with coefficient $\hat{\beta}$ that approximates their combined contribution to $Z$ and $N$? This is related to exponential family moment matching. Formal conditions under which a good merge exists are not yet worked out.

---

## 8. What This Framing Does and Does Not Give You

This section states the boundary honestly.

### What the current formalism gives you

**Algebraic compatibility with progressive and tree-structured representations.** The representation $\hat{\mu} = \sum_j \beta_j \delta_{(\hat{k}_j, \hat{v}_j)}$ places no restriction on where $(\hat{k}_j, \hat{v}_j)$ come from. They do not have to be original KV pairs. Merged atoms, synthetic atoms, and residual correction terms are all admissible within the same objective. The formalism does not need to change to accommodate them.

**Compositionality via the concatenation property.** Since $Z$ and $N$ add over disjoint blocks (Section 4), a compact approximation of a past block is transparently composable with any exact future block. This is the property that makes a progressive or hierarchical representation plausible in a streaming setting: approximation error in the past does not grow with future tokens, it persists at the level it was when the compaction was applied.

**A clear optimization target at each level.** $L_{\text{lin}}$ is defined in terms of $Z$ and $N$, not in terms of token counts, positions, or layer indices. A merged atom or residual tier can be evaluated against the same objective as a retained token. This means Phase 1 experiments directly measure what Phase 3–4 extensions will need to improve on.

**Headroom.** Phase 1 (fixed-support NNLS on live query evidence) is not a dead end. It is the conservative end of a range of representations that all live inside the same objective framework. The decision to move from retained-subset support to merged or synthetic support is a decision about the admissible representation class, not a change in the objective.

### What the current formalism does not yet give you

**A construction rule for merged or synthetic atoms.** Knowing that $(\hat{k}_j, \hat{v}_j)$ can be a merged object is not the same as knowing how to construct a good one. The merge question (Q6 above) is open: under what conditions does a single merged atom approximate two groups of KV pairs well under $L_{\text{lin}}$? This requires either a closed-form answer (related to exponential family moment matching) or an empirical search procedure.

**Evidence that the operator is actually compressible.** Algebraic admissibility of progressive representations is not the same as empirical viability. Whether $A_\mu$ under realistic query distributions is sparse enough for aggressive compaction is an empirical question (Q1). The progressive/tree direction is only justified if Phases 2–3 show meaningful compressibility at modest support sizes.

**A policy for dynamic resolution.** Even if good merged atoms exist and the operator is compressible, deciding *when* to recompact, *how many* tiers to maintain, and *which resolution to serve under inference pressure* requires a policy layer that is entirely outside the current scope.

**The approximation error bound under successive merges.** The concatenation property is exact. But approximation error introduced at one level of a merge hierarchy may compound across levels. Controlling this is a distinct open problem (Q3).

### Summary

The current formalism is the right mathematical envelope for the progressive direction. It gives algebraic headroom and a consistent objective at all levels of a potential hierarchy. What it does not give — and what Phases 2–4 must supply — is the construction rule for merged atoms, empirical evidence of compressibility, and a policy for dynamic resolution. Phase 1 is not a commitment to staying in the fixed-subset regime; it is a commitment to establishing the baseline before lifting the representation class.
