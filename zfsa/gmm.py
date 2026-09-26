"""PCA and Gaussian-mixture machinery, the permutation statistics, and plot style.

The GMM step is built so that **K = 1 ("no clusters") is a real outcome**, not
a formality to be beaten:

* BIC over K = 1..k_max x four covariance structures, and K = 1 wins ties
  (the best K >= 2 model must beat it by ``tie_dbic``).
* A **parametric single-Gaussian null**: datasets drawn from one Gaussian
  fitted to the observed scores go through the identical grid and selection
  rule, and ``p_null`` is the share whose ΔBIC reaches the observed one.
* No minimum cluster size by default (``min_cluster_n = 1``), so a single run
  may be a cluster. A component that sits on one or two runs can collapse its
  covariance to the regularisation floor, which makes its likelihood unbounded
  -- such components are flagged as degenerate rather than hidden.

All permutation tests permute labels **within date**, because condition is
confounded with day in this design.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

COV_TYPES = ["spherical", "diag", "tied", "full"]

# Plot palette (light mode).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#8a8983", "#e1e0d9", "#c3c2b7", "#fcfcfb"
DIV_NEG, DIV_MID, DIV_POS = "#2a78d6", "#f0efec", "#e34948"
SEQ = ["#f0efec", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
        "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False,
        "lines.linewidth": 2, "savefig.dpi": 160, "savefig.bbox": "tight",
    })


# ---------------------------------------------------------------------- PCA
def parallel_analysis(X: np.ndarray, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, int]:
    """Horn's parallel analysis: components whose eigenvalue beats the 95th
    percentile of ``n`` column-permuted matrices. Never sees condition labels."""
    eig = PCA().fit(X).explained_variance_
    null = np.empty((n, X.shape[1]))
    for b in range(n):
        Xp = np.column_stack([rng.permutation(X[:, j]) for j in range(X.shape[1])])
        null[b] = PCA().fit(Xp).explained_variance_
    q95 = np.quantile(null, 0.95, axis=0)
    above = eig > q95
    k = int(np.argmin(above)) if not above.all() else len(eig)
    return eig, q95, k


def dmso_pca(X_dmso: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Principal axes of the vehicle runs alone. Returns ``(mean, components, evr)``."""
    X = np.asarray(X_dmso, dtype=float)
    mean = X.mean(axis=0)
    _, S, Vt = np.linalg.svd(X - mean, full_matrices=False)
    var = S**2 / (len(X) - 1)
    return mean, Vt[:k], var[:k] / var.sum()


# ---------------------------------------------------------------------- GMM
def fit_gmm(S, K, cov, seed, n_init=20):
    return GaussianMixture(K, covariance_type=cov, n_init=n_init, random_state=seed,
                           max_iter=1000, reg_covar=1e-6).fit(S)


def fit_grid(S: np.ndarray, seed: int, *, k_max: int = 5, cov_types=COV_TYPES, n_init: int = 20,
             min_cluster_n: int = 1) -> pd.DataFrame:
    """BIC for every K x covariance, and whether each fit is admissible
    (every cluster holds at least ``min_cluster_n`` runs; K = 1 always is)."""
    rows = []
    for cov in cov_types:
        for K in range(1, k_max + 1):
            g = fit_gmm(S, K, cov, seed, n_init)
            lab = g.predict(S)
            rows.append({
                "K": K, "covariance": cov, "bic": g.bic(S), "aic": g.aic(S),
                "loglik": g.score(S) * len(S), "n_params": g._n_parameters(),
                "converged": g.converged_, "min_cluster_n": int(np.bincount(lab, minlength=K).min()),
            })
    grid = pd.DataFrame(rows)
    grid["admissible"] = (grid.K == 1) | (grid.min_cluster_n >= min_cluster_n)
    return grid


def select(grid: pd.DataFrame, tie_dbic: float = 2.0) -> tuple[pd.Series, float, pd.Series]:
    """Returns ``(chosen, ΔBIC of best admissible K>=2 vs best K=1, best K>=2 row)``.

    K = 1 is chosen unless the best admissible K >= 2 model beats it by
    ``tie_dbic``. The third element is the partition to *describe* when K = 1
    wins (the best admissible K >= 2 fit, or the best K = 2 if none is).
    """
    k1 = grid[grid.K == 1]
    k1best = k1.loc[k1.bic.idxmin()]
    adm = grid[grid.admissible & (grid.K >= 2)]
    if adm.empty:
        k2 = grid[grid.K == 2]
        return k1best, 0.0, k2.loc[k2.bic.idxmin()]
    best = adm.loc[adm.bic.idxmin()]
    dbic = float(max(k1best.bic - best.bic, 0.0))
    chosen = best if dbic >= tie_dbic else k1best
    return chosen, dbic, best


def null_rep(mean, cov, n, seed, grid_kw: dict, tie_dbic: float) -> dict:
    """One single-Gaussian null dataset through the identical grid and rule."""
    rng = np.random.default_rng(seed)
    S = rng.multivariate_normal(mean, cov, size=n)
    chosen, dbic, _ = select(fit_grid(S, seed, **grid_kw), tie_dbic)
    return {"seed": seed, "delta_bic": dbic, "selected_K": int(chosen.K), "selected_cov": chosen.covariance}


def boot_rep(S, K, cov, seed, ref_labels, n_init=20):
    """Refit on a bootstrap resample, label every run, ARI against the full fit."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(S), len(S))
    g = fit_gmm(S[idx], K, cov, seed, n_init)
    lab = g.predict(S)
    return adjusted_rand_score(ref_labels, lab), lab


# --------------------------------------------------------------- statistics
def cramers_v(a, b) -> float:
    # Contingency table by integer codes; called once per permutation, where
    # pd.crosstab would dominate the run time.
    _, ai = np.unique(np.asarray(a), return_inverse=True)
    _, bi = np.unique(np.asarray(b), return_inverse=True)
    ct = np.zeros((ai.max() + 1, bi.max() + 1), dtype=np.int64)
    np.add.at(ct, (ai, bi), 1)
    if min(ct.shape) < 2:
        return 0.0
    chi2 = stats.chi2_contingency(ct, correction=False)[0]
    return float(np.sqrt(chi2 / (ct.sum() * (min(ct.shape) - 1))))


def kw_h(values, labels) -> float:
    groups = [values[labels == k] for k in np.unique(labels)]
    if len(groups) < 2 or np.ptp(values) == 0:
        return 0.0
    return float(stats.kruskal(*groups).statistic)


def strata_groups(strata: np.ndarray) -> list[np.ndarray]:
    """Row indices of each stratum, for :func:`permute_within`."""
    return [np.flatnonzero(strata == s) for s in np.unique(strata)]


def permute_within(labels: np.ndarray, groups: list[np.ndarray], rng) -> np.ndarray:
    """Shuffle ``labels`` within each index group from :func:`strata_groups`."""
    out = labels.copy()
    for i in groups:
        out[i] = labels[rng.permutation(i)]
    return out


def perm_p(stat_fn, labels, strata, n, rng) -> tuple[float, float]:
    obs = stat_fn(labels)
    groups = strata_groups(strata)
    null = np.array([stat_fn(permute_within(labels, groups, rng)) for _ in range(n)])
    return obs, float((1 + (null >= obs - 1e-12).sum()) / (1 + n))


def co_cluster_rate(labels: np.ndarray, cond: np.ndarray, eligible: np.ndarray) -> float:
    """Share of same-condition pairs (among ``eligible`` runs) that share a cluster."""
    if not eligible.any():
        return np.nan
    _, ci = np.unique(cond[eligible], return_inverse=True)
    _, li = np.unique(labels[eligible], return_inverse=True)

    def n_pairs(sizes):
        return int((sizes * (sizes - 1) // 2).sum())

    tot = n_pairs(np.bincount(ci))
    same = n_pairs(np.bincount(ci * (li.max() + 1) + li))
    return same / tot if tot else np.nan


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3g}") -> str:
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else floatfmt.format(v)
        return str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


# ---------------------------------------------------------------- self-test
def self_test(seed: int, grid_kw: dict, tie_dbic: float, n_jobs: int = -1) -> dict:
    """Planted structure must be found; absent structure must not be."""
    from joblib import Parallel, delayed

    rng = np.random.default_rng(seed)
    n = 89
    lab = np.repeat([0, 1], [45, 44])
    S = rng.normal(size=(n, 3))
    S[lab == 1, 0] += 6.0
    chosen, _, _ = select(fit_grid(S, seed, **grid_kw), tie_dbic)
    g = fit_gmm(S, int(chosen.K), chosen.covariance, seed, grid_kw.get("n_init", 20))
    ari = adjusted_rand_score(lab, g.predict(S))
    assert chosen.K == 2 and ari > 0.9, f"planted 2-cluster test failed: K={chosen.K}, ARI={ari:.2f}"

    reps = Parallel(n_jobs=n_jobs)(
        delayed(null_rep)(np.zeros(3), np.eye(3), n, seed + 50_000 + i, grid_kw, tie_dbic) for i in range(50))
    k1 = np.mean([r["selected_K"] == 1 for r in reps])
    assert k1 >= 0.9, f"single-Gaussian test selected K=1 in only {k1:.0%} of draws"
    return {"planted_K": int(chosen.K), "planted_ARI": float(ari), "single_gaussian_K1_rate": float(k1)}
