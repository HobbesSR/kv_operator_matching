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

**Practical approach**: optimize $L_{\text{lin}}$ (tractable), verify on $\hat{L}_{\text{true}}$ (expensive but used only at checkpoints). The surrogate is faithful when $Z_{\hat{\mu}}(q) \approx Z_\mu(q)$, since then $A_{\hat{\mu}} = N_{\hat{\mu}} / Z_{\hat{\mu}} \approx N_{\hat{\mu}} / Z_\mu$.

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
where $Y_i = A_\mu(q_i)$ (original attention output) and $X_{i,j} \propto w_j \exp(q_i (C_k)^\top_j)$ (compact attention weights). This is equivalent to minimizing the output-matching part of our $L_N$ with $C_k$ and $\beta$ fixed.

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

With the nonnegativity constraint $\beta \geq 0$, this is a **nonnegative least squares (NNLS)** problem. It has a unique solution (if the feature matrix has full column rank) and can be solved efficiently.

**This is theorem-friendly**: fixed-support $\beta$-refit under $L_{\text{lin}}$ is a convex NNLS problem. The $L_{\text{true}}$ objective is not convex in $\beta$ in general, but can be used for held-out verification.

### Relationship to the paper's two-step fit

The paper (Section 3.2) decomposes the fixed-support fit differently:
- Fit $\beta$ via NNLS on the mass (Z) objective alone, ignoring $L_N$.
- Then fit $C_v$ via ordinary LS on the output matching (N/Z) objective with $\beta$ fixed.

Both steps are closed-form and efficient. The result is the same family of solutions as our $L_{\text{lin}}$ NNLS, but reached via a different optimization path. The paper's sequential approach is faster (smaller systems) but potentially suboptimal versus joint $L_{\text{lin}}$ minimization, because the $\beta$ fit in step 1 ignores value information.

In practice, either approach is a reasonable starting point. The paper reports that $\beta$ fitting and value fitting together take ~4 seconds on 60k-token contexts (Gemma-3-12B on H200), so the difference in the two-step vs. joint approach is unlikely to dominate runtime in our setting. The main bottleneck is query generation (7–139 seconds depending on strategy).

---

## 7. Open Theoretical Questions

**Q1. Response sparsity.** For typical transformer KV caches, how sparse is the effective support of $A_\mu(q)$ over a realistic query distribution? Is there a small set of support points that captures most of the operator mass? Empirical answer pending; formal characterization is open.

**Q2. Spectral decay.** Does the kernel matrix $K_{ij} = \exp(\langle q_i, k_j \rangle)$ for query-key pairs exhibit rapid spectral decay in practice? If so, low-rank approximations of the feature matrix would be accurate with far fewer support points than $n$. Related to existing attention approximation literature but not directly answered for the N/Z framing.

**Q3. Tree compatibility.** Can compressed representations be organized into a tree or hierarchy such that the concatenation property allows efficient multi-scale merging? The additive structure of $Z$ and $N$ suggests yes, but the approximation error under successive merges needs to be controlled. Open.

**Q4. Approximate submodularity.** Is the problem of selecting a support set of size $m$ to minimize $L_{\text{true}}$ (or $L_{\text{lin}}$) approximately submodular? If so, greedy selection has known approximation guarantees. The ratio structure of $L_{\text{true}}$ makes this non-obvious; $L_{\text{lin}}$ may be more tractable since it decomposes into two quadratics.

**Q5. Positional structure.** Transformer KV caches have strong positional structure (RoPE, ALiBi, etc.) that creates systematic patterns in the key vectors. Can this structure be exploited in the support selection or $\beta$-fit step? How does positional encoding interact with the inner product $\langle q, k_i \rangle$ in the N/Z formulation? Open and likely practically important.

**Q6. Merge and synthetic support generation.** Given two groups of KV pairs, can we construct a single merged support point $(\hat{k}, \hat{v})$ with coefficient $\hat{\beta}$ that approximates their combined contribution to $Z$ and $N$? This is related to exponential family moment matching. Formal conditions under which a good merge exists are not yet worked out.
