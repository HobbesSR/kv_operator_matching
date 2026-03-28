"""
Configuration dataclasses for kv_operator_matching experiments.

All configs use dataclasses with sensible defaults. They can be
constructed manually or loaded from YAML (e.g., via dacite or omegaconf).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class QueryBankConfig:
    """Configuration for the empirical query bank.

    Attributes:
        max_queries: Maximum number of query vectors retained in the bank.
            Older queries are evicted when the bank is full (FIFO with
            optional recency reweighting before eviction).
        weighting_scheme: How to assign importance weights to queries.
            - "uniform": all queries get equal weight (default)
            - "recency": exponential decay by collection order
            - "attention_mass": weight by total attention mass received
              (requires live attention evidence; not yet implemented)
        decay_factor: Multiplicative decay applied to existing weights
            when new queries are added, used when weighting_scheme is
            "recency". Values close to 1.0 retain history longer.
    """

    max_queries: int = 512
    weighting_scheme: str = "uniform"  # options: uniform, recency, attention_mass
    decay_factor: float = 0.99


@dataclass
class BetaFitConfig:
    """Configuration for the fixed-support beta coefficient fitting step.

    Phase 1a (current default): beta-only refit. A single scalar beta_j
    scales both the mass contribution (Z) and the value contribution (N)
    of each support point. Simple, convex, and theorem-friendly, but more
    restricted than the paper's full (Ck, beta, Cv) decomposition.

    Phase 1b (planned): value refit. With fit_values=True, after fitting
    beta, compact values are refit via least squares on the output matching
    objective (paper-style sequential fit). This recovers the full
    expressivity of the paper's fixed-support regime.

    Attributes:
        support_size: Number of support points m in the compact representation.
        max_iter: Maximum iterations for the NNLS / projected gradient solver.
        tol: Convergence tolerance for the solver.
        nonneg: If True, enforce beta >= 0 (standard NNLS). If False,
            allow negative betas (unconstrained least squares — generally
            not recommended since it can produce non-physical results).
        surrogate: Which surrogate loss to minimize during fitting.
            - "lin": L_Z + (1/d_v)*L_N (convex quadratic in beta; default)
            - "true_response": L_true response error (non-convex; expensive)
        normalize_lin: If True, divide L_N by d_v in L_lin to equalize
            the per-dimension scale of Z and N terms. Should be True
            unless debugging. Ignored when surrogate != "lin".
        fit_values: If True, run a separate least-squares value refit step
            after beta fitting (Phase 1b / paper-style). Not yet implemented;
            present as a flag to make the design space explicit.
        ridge: Small ridge regularization added to the NNLS diagonal for
            numerical stability when the feature matrix is ill-conditioned.
            Set to 0.0 to disable.
    """

    support_size: int = 64
    max_iter: int = 200
    tol: float = 1e-6
    nonneg: bool = True
    surrogate: str = "lin"  # options: lin, true_response
    normalize_lin: bool = True
    fit_values: bool = False  # Phase 1b; not yet implemented
    ridge: float = 1e-4


@dataclass
class VerificationConfig:
    """Configuration for the held-out verification gate.

    Attributes:
        threshold: Maximum acceptable L_true response error for a
            compact representation to pass verification.
        metric: Which metric to use for verification.
            - "response_l2": mean squared L2 error in A_mu(q) output
        holdout_fraction: Fraction of the query bank to hold out for
            verification (not used in beta-fit).
    """

    threshold: float = 0.05
    metric: str = "response_l2"  # options: response_l2
    holdout_fraction: float = 0.2


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration.

    Attributes:
        query_bank: Query bank configuration.
        beta_fit: Beta fitting configuration.
        verification: Verification gate configuration.
        device: PyTorch device string for tensor operations.
    """

    query_bank: QueryBankConfig = field(default_factory=QueryBankConfig)
    beta_fit: BetaFitConfig = field(default_factory=BetaFitConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    device: str = "cpu"
