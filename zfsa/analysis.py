"""Run-level PCA -> Gaussian mixture on the 14 pose, timing and trigger features.

The unit is the **run** (five fish shoal, so triggers within a run are not
independent). Everything here is exploratory and hypothesis-generating; it is
not a hit gate.

Pipeline, all reading the tables ``run_pipeline.py`` writes to ``out/``:

1. **Features.** 3 trigger rates (Tier A), 5 timing features and 6 pose
   features per run. Runs below the effort floor (< k in-window active bouts)
   have no pose features and are not scored; a run missing more than
   ``max(1, 10% of p)`` features is not scored; residual gaps are imputed to
   the vehicle median.
2. **Scaling.** Fixed transform (log1p for rates and counts, logit for
   proportions) -> EB day-centring on the vehicle runs -> **rank-normal
   scores** Φ⁻¹((rank - 0.5)/n) as the primary scaling. In vehicle-MAD units a
   feature with tight controls and widely spread treated runs can reach |z| of
   25-40 and single-handedly define PC1; normal scores give every feature equal
   weight. The vehicle-MAD version is refitted as a robustness check.
3. **PCA** with Horn's parallel analysis (capped at ``pc_cap`` PCs).
4. **GMM** grid + single-Gaussian null + bootstrap + DMSO-only-PCA refit +
   flagged-run refit + robustness scalings (see :mod:`zfsa.gmm`).
5. **Interpretation**: cluster profiles, compound lists, and within-date
   permutation tests of each cluster against nuisance variables (date, group,
   detection quality, when pose frames were sampled), design (vehicle vs
   treated, replicate co-clustering) and endpoint validators.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402

from . import config as config_mod  # noqa: E402
from . import gmm, metadata, normalize, pose_features  # noqa: E402

VEHICLE = "DMSO"

#: feature -> (block, construct, transform, definition)
CATALOG: dict[str, tuple[str, str, str, str]] = {
    "inactive_bout_rate_per_obs_min": ("trigger", "inactive responding", "log1p",
        "inactive bouts per observable minute, analysis window"),
    "active_bout_rate_per_obs_min": ("trigger", "active responding", "log1p",
        "active bouts per observable minute, analysis window"),
    "mean_triggers_per_bout": ("trigger", "doses per visit", "log1p", "mean active triggers per bout"),
    "nontrig_pref_index": ("pose", "shoal platform attention", "identity",
        "(d_inactive - d_active)/(d_inactive + d_active), non-trigger fish"),
    "mean_nn_distance": ("pose", "cohesion", "identity", "mean nearest-neighbour distance, all fish"),
    "polarization_nontrigger": ("pose", "alignment", "logit", "heading alignment, non-trigger fish"),
    "frac_in_rim": ("pose", "thigmotaxis", "logit", "share of non-trigger fish in the rim band"),
    "trig_radial_offset": ("pose", "trigger-fish visit", "identity",
        "trigger fish distance from platform centre / half-side"),
    "trig_heading_outward": ("pose", "trigger-fish visit", "identity",
        "cos(heading - bearing from platform centre): +1 leaving"),
    "pref_change": ("timing", "preference trajectory", "identity",
        "late (bins >= 3) minus early (bins <= 1) log-odds of active vs inactive bouts"),
    "switch_excess": ("timing", "platform switching", "identity",
        "log((observed + 0.5)/(chance + 0.5)) count of active bouts followed by an inactive bout"),
    "bin_slope": ("timing", "within-run decline", "identity",
        "slope of log active bout rate across time bins"),
    "first_inactive_latency_log": ("timing", "latency", "identity",
        "log(1 + s from window start to first inactive trigger); censored at window end if none"),
    "first_active_latency_log": ("timing", "latency", "identity",
        "log(1 + s from window start to first active trigger)"),
}

INPUT_TABLES = ["runs", "events", "bouts", "exposure", "effort_index", "effort_status",
                "calibration", "pose_detections", "pose_frame_qc", "features_tier_a",
                "features_timing", "features_tier_c"]


def construct(f: str) -> str:
    return CATALOG[f][1]


# --------------------------------------------------------------------- plots
def _label_column(ax, D, idx, xlim):
    """Stack labels for the runs in ``idx`` in a column beside the data, untangled."""
    right = D.loc[idx][D.loc[idx, "PC1"] >= 0]
    left = D.loc[idx][D.loc[idx, "PC1"] < 0]
    ylo, yhi = D.PC2.min(), D.PC2.max()
    for sub, xt in [(right, xlim[1] - 0.18 * (xlim[1] - xlim[0])),
                    (left, xlim[0] + 0.01 * (xlim[1] - xlim[0]))]:
        if sub.empty:
            continue
        ys = np.linspace(yhi, ylo, len(sub)) if len(sub) > 1 else np.array([sub.PC2.iloc[0]])
        sub = sub.assign(ang=np.arctan2(sub.PC2 - (yhi + ylo) / 2, abs(xt - sub.PC1))).sort_values("ang", ascending=False)
        P = sub[["PC1", "PC2"]].to_numpy()
        slot = list(range(len(sub)))

        def cross(a, b2, c, d):
            o = lambda p, q, r: np.sign((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))  # noqa: E731
            return o(a, b2, c) != o(a, b2, d) and o(c, d, a) != o(c, d, b2)
        for _ in range(200):
            changed = False
            for i in range(len(sub)):
                for k in range(i + 1, len(sub)):
                    if cross(P[i], (xt, ys[slot[i]]), P[k], (xt, ys[slot[k]])):
                        slot[i], slot[k] = slot[k], slot[i]
                        changed = True
            if not changed:
                break
        for i, r in enumerate(sub.itertuples()):
            ax.annotate(r.condition, (r.PC1, r.PC2), xytext=(xt, ys[slot[i]]), textcoords="data",
                        fontsize=7.5, color=gmm.INK2, va="center", ha="left",
                        arrowprops=dict(arrowstyle="-", color=gmm.AXIS, lw=0.7, shrinkA=2, shrinkB=4))


def cluster_plot(A, k_used, ev_ratio, Kd, covd, descriptive_only, path):
    D = A.copy()
    fig, ax = plt.subplots(figsize=(10.5, 7))
    for j in range(Kd):
        for isv, mk in [(True, "o"), (False, "^")]:
            s = D[(D.cluster == j) & (D.is_vehicle == isv)]
            if s.empty:
                continue
            ax.scatter(s.PC1, s.PC2, s=58, marker=mk, color=gmm.SERIES[j % len(gmm.SERIES)],
                       edgecolor=gmm.SURFACE, lw=1.2, zorder=3,
                       label=f"Cluster {j}{' — core (largest)' if j == 0 else ''} · "
                             f"{'vehicle' if isv else 'treated'} (n={len(s)})")
    pc = D[D.is_positive_control]
    if len(pc):
        ax.scatter(pc.PC1, pc.PC2, s=150, facecolor="none", edgecolor=gmm.INK, lw=1.0, zorder=4,
                   label="positive control")
    fl = D[D.is_flagged]
    if len(fl):
        ax.scatter(fl.PC1, fl.PC2, s=90, marker="x", color=gmm.INK, lw=1.4, zorder=5, label="flagged run")
    minority = D.index[D.cluster != 0]
    span = D.PC1.max() - D.PC1.min()
    few = len(minority) <= 16
    xlim = (D.PC1.min() - 0.08 * span - (0.25 * span if (D.loc[minority, "PC1"] < 0).any() and few else 0),
            D.PC1.max() + (0.45 * span if few else 0.08 * span))
    ax.set_xlim(*xlim)
    if few:
        _label_column(ax, D, minority, xlim)
    ax.axhline(0, color=gmm.AXIS, lw=0.8, zorder=1)
    ax.axvline(0, color=gmm.AXIS, lw=0.8, zorder=1)
    ax.set_xlabel(f"PC1 ({ev_ratio[0]:.0%} of variance)")
    ax.set_ylabel(f"PC2 ({ev_ratio[1]:.0%} of variance)")
    ax.set_title(f"{len(D)} runs on the top two PCs, coloured by GMM cluster (K={Kd}, {covd})", pad=48)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=8)
    note = (f"GMM fitted on the first {k_used} PC(s) retained by parallel analysis"
            + ("; PC2 shown for display only." if k_used < 2 else ".")
            + (" Clusters are a descriptive cut of a continuum (see REPORT.md)." if descriptive_only else "")
            + ("" if few else " Minority clusters too large to label; see cluster_compounds.md."))
    fig.text(0.01, -0.02, note, fontsize=7.5, color=gmm.MUTED)
    fig.savefig(path)
    plt.close(fig)


# ====================================================================== main
def run(out_dir: Path, cfg: dict, *, log=print) -> dict:
    """Run the analysis on the tables in ``out_dir``; write ``out_dir/pca_gmm/``."""
    t0 = time.time()
    a = cfg["analysis"]
    SEED = int(a["seed"])
    FEATURES = list(a["features"])
    unknown = [f for f in FEATURES if f not in CATALOG]
    if unknown:
        raise ValueError(f"analysis.features has unknown feature(s) {unknown}; known: {sorted(CATALOG)}")
    N_PA, PC_CAP, N_NULL, N_BOOT, N_PERM = (int(a["n_parallel_analysis"]), int(a["pc_cap"]),
                                            int(a["n_null"]), int(a["n_boot"]), int(a["n_perm"]))
    N_INIT, N_JOBS, TIE = int(a["n_init"]), int(a["n_jobs"]), float(a["tie_dbic"])
    grid_kw = {"k_max": int(a["k_max"]), "cov_types": list(a["covariance_types"]), "n_init": N_INIT,
               "min_cluster_n": int(a["min_cluster_n"])}
    DEGENERATE_EIG, MAD_SD_MIN = float(a["degenerate_eig"]), float(a["mad_sd_min"])
    window = config_mod.window(cfg)

    HERE = Path(out_dir) / "pca_gmm"
    FIG = HERE / "figures"
    FIG.mkdir(parents=True, exist_ok=True)
    gmm.style()

    log("self-test ...")
    selftest = gmm.self_test(SEED, grid_kw, TIE, N_JOBS)
    log(f"   {selftest}")

    T = {k: pd.read_parquet(Path(out_dir) / f"{k}.parquet") for k in INPUT_TABLES}
    R = metadata.build_conditions(T["runs"], VEHICLE)
    R["date"] = pd.to_datetime(R.date).dt.strftime("%Y-%m-%d")
    R = R.sort_values("run_id").reset_index(drop=True)

    # ---- features
    log("assembling features ...")
    cand = R[["run_id"]].copy()
    exclusion = []
    tim = T["features_timing"]
    n_censored = int(tim.first_inactive_censored.sum())
    cand = cand.merge(tim.drop(columns="first_inactive_censored"), on="run_id")
    trig_cols = [f for f in FEATURES if CATALOG[f][0] == "trigger"]
    cand = cand.merge(T["features_tier_a"][["run_id", *trig_cols]], on="run_id")
    pose_cols = [f for f in FEATURES if CATALOG[f][0] == "pose"]
    frames = pose_features.frame_table(T["pose_detections"], T["calibration"],
                                       rim_inner=cfg["features"]["rim_inner"],
                                       rim_outer=cfg["features"]["rim_outer"])
    agg = pose_features.aggregate_runs(frames, pose_features.POSE_FEATURES, T["bouts"], T["effort_index"],
                                       gap_s=cfg["events"]["bout_gap_s"])
    ok = set(T["effort_status"].run_id[T["effort_status"].effort_equalized])
    agg = agg[agg.run_id.isin(ok)]
    k_eff = int(cfg["effort"]["k"])
    for rid in sorted(set(cand.run_id) - set(agg.run_id)):
        exclusion.append({"run_id": rid, "reason": f"below effort floor (<{k_eff} in-window active bouts): no pose features"})
    cand = cand.merge(agg[["run_id", *pose_cols]], on="run_id")
    cand = cand.merge(R[["run_id", "date", "condition", "compound", "is_vehicle"]], on="run_id")
    cand.to_parquet(HERE / "candidate_features.parquet", index=False)
    agg.to_parquet(HERE / "pose_features_with_draw_sd.parquet", index=False)
    n_cand = len(cand)
    sel_log = pd.DataFrame([
        {"order": i + 1, "feature": f, "block": CATALOG[f][0], "construct": CATALOG[f][1],
         "transform": CATALOG[f][2], "definition": CATALOG[f][3],
         "missing_frac": float(cand[f].isna().mean())}
        for i, f in enumerate(FEATURES)])
    sel_log.to_csv(HERE / "features_used.csv", index=False)

    # per-run missingness: tolerate at most max(1, 10% of p) gaps, imputed to the vehicle median
    allow = max(1, int(0.1 * len(FEATURES)))
    n_miss = cand[FEATURES].isna().sum(axis=1)
    for rid in cand.run_id[n_miss > allow]:
        exclusion.append({"run_id": rid, "reason": f"more than {allow} feature(s) missing"})
    raw = cand[n_miss <= allow].reset_index(drop=True)
    raw.to_parquet(HERE / "run_features.parquet", index=False)
    excluded = pd.DataFrame(exclusion, columns=["run_id", "reason"]).merge(
        R[["run_id", "condition"]], on="run_id", how="left")

    n_scored = len(raw)
    n_veh_scored = int(raw.is_vehicle.sum())
    if n_scored < 20:
        raise RuntimeError(f"only {n_scored} runs have all features; a mixture model needs at least 20")
    if n_veh_scored < metadata.MIN_VEHICLE_RUNS:
        raise RuntimeError(f"only {n_veh_scored} vehicle runs have all features; need at least "
                           f"{metadata.MIN_VEHICLE_RUNS} for scaling and day-centring")
    rpf = int(a["runs_per_feature"])
    small_n = n_scored < rpf * len(FEATURES)

    # ---- normalise (transform -> EB day-centre -> vehicle-robust scale)
    Rs = R[R.run_id.isin(raw.run_id)].reset_index(drop=True)
    spec = {f: CATALOG[f][2] for f in FEATURES if CATALOG[f][2] != "identity"}
    Z, scale_rep, cent_rep = normalize.scale_guarded(raw[["run_id", *FEATURES]], Rs, spec, mad_sd_min=MAD_SD_MIN)
    log(f"  scale methods: {scale_rep.set_index('feature').method.to_dict()}")
    n_imputed = int(Z[FEATURES].isna().sum().sum())
    Z[FEATURES] = Z[FEATURES].fillna(0.0)
    Z = Z.merge(Rs[["run_id", "date", "condition", "compound", "is_vehicle", "is_positive_control", "flag",
                    "fish_group_resolved", "run_order_within_day"]], on="run_id")
    Z.to_parquet(HERE / "run_features_z_dmso_mad.parquet", index=False)
    scale_rep.merge(cent_rep, on="feature", how="left").to_csv(HERE / "normalization_report.csv", index=False)
    X_mad = Z[FEATURES].to_numpy(float)
    mad_max_abs = pd.Series(np.abs(X_mad).max(0), index=FEATURES)
    for f in FEATURES:
        Z[f] = stats.norm.ppf((stats.rankdata(Z[f]) - 0.5) / len(Z))
    Z.to_parquet(HERE / "run_features_scores.parquet", index=False)

    X = Z[FEATURES].to_numpy(float)
    assert np.isfinite(X).all()
    n = len(X)
    veh = Z.is_vehicle.to_numpy()
    flagged = (Z.flag.fillna("").astype(str).str.strip() != "").to_numpy()
    rng = np.random.default_rng(SEED)

    # ---- PCA
    log("PCA + parallel analysis ...")
    pca = PCA().fit(X)
    eig, pa_q95, k_pa = gmm.parallel_analysis(X, N_PA, rng)
    k = int(np.clip(k_pa, 1, PC_CAP))
    pcs = [f"PC{i + 1}" for i in range(len(FEATURES))]
    loadings = pd.DataFrame(pca.components_.T, index=FEATURES, columns=pcs)
    loadings.insert(0, "construct", [construct(f) for f in FEATURES])
    loadings.to_csv(HERE / "pca_loadings.csv", index_label="feature")
    var = pd.DataFrame({
        "component": pcs, "eigenvalue": eig, "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative": np.cumsum(pca.explained_variance_ratio_), "parallel_analysis_q95": pa_q95,
        "retained": [i < k for i in range(len(pcs))]})
    var.to_csv(HERE / "pca_variance.csv", index=False)
    scores_all = pca.transform(X)
    S = scores_all[:, :k]
    sc = Z[["run_id", "date", "condition", "compound", "is_vehicle"]].copy()
    for i, p in enumerate(pcs):
        sc[p] = scores_all[:, i]
    sc.to_parquet(HERE / "pca_scores.parquet", index=False)

    # ---- GMM
    log("GMM grid ...")
    grid = gmm.fit_grid(S, SEED, **grid_kw)
    assert len(grid) == grid_kw["k_max"] * len(grid_kw["cov_types"]) and grid.bic.notna().all()
    chosen, dbic_obs, multi = gmm.select(grid, TIE)
    grid["selected"] = (grid.K == chosen.K) & (grid.covariance == chosen.covariance)
    grid.to_csv(HERE / "gmm_model_selection.csv", index=False)
    log(f"  selected K={chosen.K} ({chosen.covariance}), dBIC vs K=1 = {dbic_obs:.2f}")

    log(f"parametric null ({N_NULL}) ...")
    mu, cov = S.mean(axis=0), np.atleast_2d(np.cov(S, rowvar=False))
    null = pd.DataFrame(Parallel(n_jobs=N_JOBS)(
        delayed(gmm.null_rep)(mu, cov, n, SEED + 1 + i, grid_kw, TIE) for i in range(N_NULL)))
    null.to_csv(HERE / "gmm_null.csv", index=False)
    p_null = float((1 + (null.delta_bic >= dbic_obs).sum()) / (1 + N_NULL))
    null_k_rate = null.selected_K.value_counts(normalize=True).sort_index().to_dict()
    log(f"  p_null = {p_null:.3f}")

    desc = chosen if chosen.K > 1 else multi
    Kd, covd = int(desc.K), desc.covariance
    descriptive_only = bool(chosen.K == 1 or p_null > 0.05)
    gm = gmm.fit_gmm(S, Kd, covd, SEED, N_INIT)
    post = gm.predict_proba(S)
    assert np.allclose(post.sum(axis=1), 1.0)
    order = np.argsort(-np.bincount(post.argmax(1), minlength=Kd), kind="stable")
    post = post[:, order]
    labels = post.argmax(1)
    Kd = int(len(np.unique(labels)))  # an empty component cannot be described
    post = post[:, :Kd] / post[:, :Kd].sum(1, keepdims=True)
    entropy = -(post * np.log(np.clip(post, 1e-300, None))).sum(1) / np.log(max(Kd, 2))

    # A component sitting on one or a few runs can shrink its covariance to the
    # regularisation floor; its likelihood is then unbounded, which inflates
    # ΔBIC. Flag those rather than hide them.
    def _cov_eigs(gm_, j):
        c = gm_.covariances_
        M = {"full": lambda: c[j], "tied": lambda: c, "diag": lambda: np.diag(c[j]),
             "spherical": lambda: np.eye(S.shape[1]) * c[j]}[gm_.covariance_type]()
        return np.linalg.eigvalsh(np.atleast_2d(M))
    cluster_min_eig = np.array([_cov_eigs(gm, order[j]).min() for j in range(Kd)])
    cluster_degenerate = cluster_min_eig < DEGENERATE_EIG
    cluster_sizes = np.bincount(labels, minlength=Kd)
    log(f"  cluster sizes: {cluster_sizes.tolist()} degenerate: {cluster_degenerate.tolist()}")

    log(f"bootstrap ({N_BOOT}) ...")
    boot = Parallel(n_jobs=N_JOBS)(
        delayed(gmm.boot_rep)(S, int(desc.K), covd, SEED + 100_000 + i, labels, N_INIT) for i in range(N_BOOT))
    aris = np.array([b_[0] for b_ in boot])
    co = np.zeros((n, n))
    for _, lab in boot:
        co += lab[:, None] == lab[None, :]
    co /= N_BOOT

    m_d, c_d, _ = gmm.dmso_pca(X[veh], k)
    S_d = (X - m_d) @ c_d.T
    ari_dmso = adjusted_rand_score(labels, gmm.fit_gmm(S_d, int(desc.K), covd, SEED, N_INIT).predict(S_d))
    cosines = np.linalg.svd(pca.components_[:k] @ c_d.T, compute_uv=False)
    if flagged.any():
        keep = ~flagged
        S_b = PCA(n_components=k).fit_transform(X[keep])
        ari_flag = adjusted_rand_score(labels[keep], gmm.fit_gmm(S_b, int(desc.K), covd, SEED, N_INIT).predict(S_b))
    else:
        ari_flag = np.nan
    stability = pd.DataFrame([
        {"check": "bootstrap ARI (median)", "value": float(np.median(aris))},
        {"check": "bootstrap ARI (2.5%)", "value": float(np.quantile(aris, 0.025))},
        {"check": "bootstrap ARI (97.5%)", "value": float(np.quantile(aris, 0.975))},
        {"check": "vehicle-only PCA refit ARI", "value": float(ari_dmso)},
        {"check": "vehicle-only vs all-run PC subspace min cosine", "value": float(cosines.min())},
        {"check": f"drop flagged runs refit ARI ({int(flagged.sum())} flagged)", "value": float(ari_flag)},
        {"check": "mean posterior entropy (0=certain, 1=uniform)", "value": float(entropy.mean())},
        {"check": "runs with max posterior < 0.8", "value": float((post.max(1) < 0.8).sum())},
    ])
    stability.to_csv(HERE / "gmm_stability.csv", index=False)

    # ---- robustness to scaling
    log("robustness ...")
    minority = labels != 0
    rob = []
    for rname, Xr in [
        ("vehicle-MAD z (no rank transform)", X_mad),
        ("vehicle-MAD z winsorized at ±5", np.clip(X_mad, -5, 5)),
    ]:
        _, _, kr_pa = gmm.parallel_analysis(Xr, N_PA, np.random.default_rng(SEED + 3))
        kr = int(np.clip(kr_pa, 1, PC_CAP))
        Sr = PCA().fit_transform(Xr)[:, :kr]
        if np.corrcoef(Sr[:, 0], S[:, 0])[0, 1] < 0:
            Sr[:, 0] *= -1
        gr = gmm.fit_grid(Sr, SEED, **grid_kw)
        chr_, dbr, mr = gmm.select(gr, TIE)
        dr = chr_ if chr_.K > 1 else mr
        labr = gmm.fit_gmm(Sr, int(dr.K), dr.covariance, SEED, N_INIT).predict(Sr)
        auc = (stats.mannwhitneyu(Sr[minority, 0], Sr[~minority, 0]).statistic
               / (minority.sum() * (~minority).sum())) if minority.any() else np.nan
        rob.append({"scaling": rname, "k_pcs": kr, "selected_K": int(chr_.K),
                    "selected_covariance": chr_.covariance, "delta_bic_vs_K1": dbr,
                    "ARI_vs_main_partition": adjusted_rand_score(labels, labr),
                    "PC1_AUROC_main_minority_clusters": float(auc),
                    "PC1_corr_with_main_PC1": float(np.corrcoef(Sr[:, 0], S[:, 0])[0, 1])})
    rob = pd.DataFrame(rob)
    rob.to_csv(HERE / "gmm_robustness.csv", index=False)
    # A split only counts as robust if the alternative scalings recover the *same*
    # runs (ARI >= 0.5), not merely the same number of clusters.
    rob_same_partition = bool((rob.selected_K >= 2).all() and (rob.ARI_vs_main_partition >= 0.5).all())
    tails_not_states = bool((rob.selected_K == 1).any() or cluster_degenerate.any()
                            or np.median(aris) < 0.75 or not rob_same_partition or ari_dmso < 0.5)
    descriptive_only = descriptive_only or tails_not_states

    # ---- assignments
    A = Z[["run_id", "date", "condition", "compound", "is_vehicle", "is_positive_control", "flag",
           "fish_group_resolved", "run_order_within_day"]].copy()
    A["is_flagged"] = flagged
    A["cluster"] = labels
    for j in range(Kd):
        A[f"p_cluster{j}"] = post[:, j]
    A["posterior_entropy"] = entropy
    A["bootstrap_stability"] = [co[i, labels == labels[i]].mean() for i in range(n)]
    A["PC1"], A["PC2"] = scores_all[:, 0], scores_all[:, 1]
    A["model_K"], A["model_covariance"], A["descriptive_only"] = Kd, covd, descriptive_only
    A["cluster_size"] = cluster_sizes[labels]
    A["cluster_degenerate"] = cluster_degenerate[labels]
    A.to_parquet(HERE / "gmm_assignments.parquet", index=False)
    A.drop(columns=[c for c in A.columns if c.startswith("p_cluster")]).to_csv(HERE / "gmm_assignments.csv", index=False)

    # ---- profiles, composition, compound lists
    prof = []
    brng = np.random.default_rng(SEED + 7)
    for j in range(Kd):
        Xi = X[labels == j]
        for fi, f in enumerate(FEATURES):
            bs = [brng.choice(Xi[:, fi], len(Xi)).mean() for _ in range(2000)]
            prof.append({"cluster": j, "n_runs": len(Xi), "feature": f, "construct": construct(f),
                         "mean_z": Xi[:, fi].mean(), "ci_low": np.quantile(bs, .025), "ci_high": np.quantile(bs, .975)})
    prof = pd.DataFrame(prof)
    prof.to_csv(HERE / "cluster_profiles.csv", index=False)

    n_rep = A.condition.value_counts()
    comp_rows, md_lines = [], ["# Compounds per cluster", "",
                                f"{n} runs, GMM K={Kd} ({covd})."
                                + (" **Descriptive partition of a continuum, not discrete states** (see REPORT.md)."
                                   if descriptive_only else ""), "",
                                "`a of b` = runs of that condition in this cluster, out of all its scored runs.", ""]
    for j in range(Kd):
        s = A[A.cluster == j]
        md_lines += [f"## Cluster {j} — {len(s)} runs ({int(s.is_vehicle.sum())} vehicle)", ""]
        vc = s.condition.value_counts()
        for c, cnt in sorted(vc.items(), key=lambda t: (t[0] != VEHICLE, -t[1], t[0])):
            tot = int(n_rep[c])
            all_in = " — all replicates" if cnt == tot and tot > 1 else ""
            md_lines.append(f"- {c}: {cnt} of {tot}{all_in}")
            comp_rows.append({"cluster": j, "condition": c, "runs_in_cluster": int(cnt),
                              "runs_total": tot, "all_replicates_here": bool(cnt == tot)})
        md_lines.append("")
    if len(excluded):
        md_lines += ["## Not scored", ""] + [f"- {r.condition} ({r.run_id}): {r.reason}" for r in excluded.itertuples()] + [""]
    (HERE / "cluster_compounds.md").write_text("\n".join(md_lines), encoding="utf-8")
    pd.DataFrame(comp_rows).to_csv(HERE / "cluster_compounds.csv", index=False)
    comp = A.groupby("cluster").agg(n_runs=("run_id", "size"), n_vehicle=("is_vehicle", "sum"),
                                    n_dates=("date", "nunique"))

    # ---- confounds / design / validators
    log("permutations ...")
    C = A.merge(T["features_tier_c"], on="run_id", how="left", suffixes=("", "_c"))
    C = C.merge(T["features_tier_a"], on="run_id", how="left", suffixes=("", "_a"))
    date = C.date.to_numpy()
    none = np.zeros(n, dtype=int)
    prng = np.random.default_rng(SEED + 11)
    rows = []

    def add(kind, name, stat_name, fn, strata, note):
        obs, p = gmm.perm_p(fn, labels, strata, N_PERM, prng)
        rows.append({"kind": kind, "variable": name, "statistic": stat_name, "value": obs,
                     "p_perm": p, "permutation": note})

    add("nuisance", "date", "Cramér's V", lambda L: gmm.cramers_v(L, date), none, "unrestricted")
    grp = C.fish_group_resolved.astype(str).to_numpy()
    add("nuisance", "fish_group_resolved", "Cramér's V", lambda L: gmm.cramers_v(L, grp), none, "unrestricted")
    for v_ in ["run_order_within_day", "mean_conf_5th", "frac_frames_under_5", "mean_max_iou",
               "subsample_mean_time_bin", "blind_fraction", "n_bouts_available", "trigger_fish_present_fraction"]:
        vals = C[v_].astype(float).to_numpy()
        okm = np.isfinite(vals)
        add("nuisance", v_, "Kruskal-Wallis H", lambda L, vals=vals, okm=okm: gmm.kw_h(vals[okm], L[okm]),
            date, "within date")
    isv = C.is_vehicle.to_numpy()
    add("design", "is_vehicle", "Cramér's V", lambda L: gmm.cramers_v(L, isv), date, "within date")
    val_vars = ["active_bout_rate_per_obs_min", "inactive_bout_rate_per_obs_min", "logodds_active",
                "n_active_bouts_in_window"]
    for v_ in val_vars:
        vals = C[v_].astype(float).to_numpy()
        okm = np.isfinite(vals)
        add("validator", v_, "Kruskal-Wallis H", lambda L, vals=vals, okm=okm: gmm.kw_h(vals[okm], L[okm]),
            date, "within date")
    tests = pd.DataFrame(rows)
    cond = C.condition.to_numpy()
    counts = pd.Series(cond).value_counts()
    elig = np.array([(c != VEHICLE) and counts[c] >= 2 for c in cond])
    obs_rep = gmm.co_cluster_rate(labels, cond, elig)
    null_rep = []
    for _ in range(N_PERM):
        cp = gmm.permute_within(cond, date, prng)
        cnt = pd.Series(cp).value_counts()
        el = np.array([(c != VEHICLE) and cnt[c] >= 2 for c in cp])
        null_rep.append(gmm.co_cluster_rate(labels, cp, el))
    null_rep = np.array(null_rep)
    p_rep = float((1 + (null_rep >= obs_rep).sum()) / (1 + N_PERM))
    dmso_rate = gmm.co_cluster_rate(labels, cond, cond == VEHICLE)
    base_rate = float((np.bincount(labels) / n) @ (np.bincount(labels) / n))
    tests = pd.concat([tests, pd.DataFrame([{
        "kind": "design", "variable": "same-condition replicate co-clustering (non-vehicle, n>=2)",
        "statistic": "same-cluster pair rate", "value": obs_rep, "p_perm": p_rep,
        "permutation": "condition labels within date"}])], ignore_index=True)
    inputs_or_derived = set(FEATURES)
    if {"active_bout_rate_per_obs_min", "inactive_bout_rate_per_obs_min"} <= set(FEATURES):
        inputs_or_derived |= {"logodds_active"}
    if "active_bout_rate_per_obs_min" in FEATURES:
        inputs_or_derived |= {"n_active_bouts_in_window"}
    tests["flag"] = np.where((tests.kind == "nuisance") & (tests.p_perm <= 0.05), "cluster tracks nuisance",
                    np.where((tests.kind == "validator") & tests.variable.isin(inputs_or_derived),
                             "not independent: input or derived from inputs", ""))
    tests.to_csv(HERE / "cluster_confounds.csv", index=False)
    vcols = ["active_bout_rate_per_obs_min", "inactive_bout_rate_per_obs_min", "logodds_active",
             "n_active_bouts_in_window", "mean_conf_5th", "subsample_mean_time_bin"]
    validators = C.groupby("cluster")[vcols].median().add_prefix("median_")
    validators.insert(0, "n_runs", C.groupby("cluster").size())
    validators.to_csv(HERE / "cluster_validators.csv")

    # ================================================================ figures
    log("figures ...")
    cc = gmm.SERIES
    fig, ax = plt.subplots(figsize=(6, 3.6))
    xs = np.arange(1, len(eig) + 1)
    ax.plot(xs, eig, color=cc[0], marker="o", ms=6, label="observed eigenvalue")
    ax.plot(xs, pa_q95, color=gmm.MUTED, ls="--", marker="o", ms=4, label="parallel analysis 95th pct")
    ax.axvspan(0.5, k + 0.5, color=cc[0], alpha=0.07, lw=0)
    ax.set_xticks(xs)
    ax.set_xlabel("component")
    ax.set_ylabel("variance (normal-score units²)")
    ax.set_title(f"PCA scree — {k_pa} component(s) above noise, {k} retained")
    ax.legend(loc="upper right")
    fig.savefig(FIG / "01_scree_parallel_analysis.png")
    plt.close(fig)

    div = LinearSegmentedColormap.from_list("div", [gmm.DIV_NEG, gmm.DIV_MID, gmm.DIV_POS])
    show = max(k, min(4, len(pcs)))
    L = loadings[pcs[:show]].to_numpy()
    fig, ax = plt.subplots(figsize=(1.3 * show + 3.8, 0.42 * len(FEATURES) + 1.6))
    im = ax.imshow(L, cmap=div, vmin=-1, vmax=1, aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(show))
    ax.set_xticklabels([f"{p}\n{r:.0%}{' ✓' if i < k else ''}" for i, (p, r) in
                        enumerate(zip(pcs[:show], pca.explained_variance_ratio_[:show]))])
    ax.set_yticks(range(len(FEATURES)))
    ax.set_yticklabels([f"{f}  [{CATALOG[f][0]}]" for f in FEATURES])
    for i in range(L.shape[0]):
        for j in range(show):
            ax.text(j, i, f"{L[i, j]:+.2f}", ha="center", va="center", fontsize=8,
                    color=gmm.INK if abs(L[i, j]) < 0.6 else "#ffffff")
    fig.colorbar(im, ax=ax, shrink=0.7, label="loading")
    ax.set_title("PCA loadings (✓ = retained for GMM)")
    fig.savefig(FIG / "02_pca_loadings.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for ci, cv in enumerate(grid_kw["cov_types"]):
        g = grid[grid.covariance == cv]
        ax.plot(g.K, g.bic, color=cc[ci], marker="o", ms=6, label=cv)
        ax.text(g.K.iloc[-1] + 0.08, g.bic.iloc[-1], cv, color=gmm.INK2, fontsize=8, va="center")
    ax.scatter([chosen.K], [chosen.bic], s=160, facecolor="none", edgecolor=gmm.INK, lw=1.5, zorder=4)
    ax.set_xticks(range(1, grid_kw["k_max"] + 1))
    ax.set_xlim(0.7, grid_kw["k_max"] + 1.0)
    ax.set_xlabel("K (components)")
    ax.set_ylabel("BIC (lower is better)")
    ax.set_title(f"GMM selection — K={chosen.K} ({chosen.covariance}), ΔBIC vs K=1 = {dbic_obs:.1f}")
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(FIG / "03_gmm_bic_grid.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.hist(null.delta_bic, bins=40, color=cc[0], edgecolor=gmm.SURFACE, linewidth=1)
    ax.axvline(dbic_obs, color=gmm.INK, lw=2)
    ax.text(dbic_obs, ax.get_ylim()[1] * 0.92, f"  observed ΔBIC = {dbic_obs:.1f}\n  p_null = {p_null:.3f}",
            color=gmm.INK, fontsize=8, va="top")
    ax.set_xlabel("ΔBIC (best model vs best K=1)")
    ax.set_ylabel("null replicates")
    ax.set_title(f"Parametric null: {N_NULL} single-Gaussian datasets")
    fig.savefig(FIG / "04_gmm_null_delta_bic.png")
    plt.close(fig)

    o = np.lexsort((-post.max(1), labels))
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(co[np.ix_(o, o)], cmap=LinearSegmentedColormap.from_list("seq", gmm.SEQ), vmin=0, vmax=1)
    ax.grid(False)
    for e in np.cumsum(np.bincount(labels))[:-1] - 0.5:
        ax.axhline(e, color=gmm.INK, lw=0.8)
        ax.axvline(e, color=gmm.INK, lw=0.8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("runs, ordered by cluster")
    fig.colorbar(im, ax=ax, shrink=0.8, label="bootstrap co-assignment rate")
    ax.set_title(f"Bootstrap co-assignment, median ARI {np.median(aris):.2f}")
    fig.savefig(FIG / "05_bootstrap_coassignment.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, Kd, figsize=(3.1 * Kd + 3.2, 0.42 * len(FEATURES) + 1.8), sharey=True, sharex=True)
    axes = np.atleast_1d(axes)
    for j, axj in enumerate(axes):
        p = prof[prof.cluster == j].set_index("feature").loc[FEATURES]
        yy = np.arange(len(FEATURES))
        col = cc[j % len(cc)]
        axj.axvline(0, color=gmm.AXIS, lw=1)
        axj.hlines(yy, p.ci_low, p.ci_high, color=col, lw=2)
        axj.scatter(p.mean_z, yy, s=40, color=col, edgecolor=gmm.SURFACE, lw=1.2, zorder=3)
        axj.set_title(f"cluster {j} (n={int(p.n_runs.iloc[0])}, vehicle {int(comp.n_vehicle.iloc[j])})")
        axj.set_xlabel("mean normal score (0 = median run)")
    axes[0].set_yticks(np.arange(len(FEATURES)))
    axes[0].set_yticklabels(FEATURES)
    axes[0].invert_yaxis()
    fig.suptitle("Cluster feature profiles, 95% bootstrap CI", fontweight="bold")
    fig.savefig(FIG / "06_cluster_profiles.png")
    plt.close(fig)

    vplot = ["active_bout_rate_per_obs_min", "inactive_bout_rate_per_obs_min", "logodds_active",
             "mean_conf_5th", "subsample_mean_time_bin"]
    fig, axes = plt.subplots(1, len(vplot), figsize=(3.0 * len(vplot), 3.8))
    jr = np.random.default_rng(1)
    for axv, v_ in zip(axes, vplot):
        for j in range(Kd):
            vals = C.loc[C.cluster == j, v_].to_numpy(float)
            axv.scatter(j + jr.uniform(-0.18, 0.18, len(vals)), vals, s=22, color=cc[j % len(cc)],
                        edgecolor=gmm.SURFACE, lw=0.8, zorder=3)
            if np.isfinite(vals).any():
                axv.hlines(np.nanmedian(vals), j - 0.3, j + 0.3, color=gmm.INK, lw=2, zorder=4)
        pv = tests.loc[tests.variable == v_, "p_perm"]
        tag = " (input)" if v_ in inputs_or_derived else ""
        axv.set_title(f"{v_}{tag}\n" + (f"p_perm={pv.iloc[0]:.3f}" if len(pv) else ""), fontsize=8)
        axv.set_xticks(range(Kd))
        axv.set_xticklabels([f"c{j}" for j in range(Kd)])
    fig.suptitle("Endpoints and QC by cluster", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "07_cluster_validators.png")
    plt.close(fig)

    cluster_plot(A, k, pca.explained_variance_ratio_, Kd, covd, descriptive_only,
                 FIG / "08_clusters_PC1_PC2.png")

    # ================================================================= report
    md = gmm.md_table
    sizes = np.bincount(labels)
    nuis_flag = tests[(tests.kind == "nuisance") & (tests.p_perm <= 0.05)]
    verdict = (
        f"**BIC selects K={int(chosen.K)} ({chosen.covariance}).** " +
        ("With the tie rule (ΔBIC < 2 → K=1) the data do not support more than one component. "
         if chosen.K == 1 else "") +
        f"ΔBIC vs the best K=1 model = {dbic_obs:.1f}, p_null = {p_null:.3f} ({N_NULL} single-Gaussian datasets, "
        "identical grid and selection rule). " +
        (f"**{int(cluster_degenerate.sum())} cluster(s) are degenerate** (sizes "
         f"{', '.join(str(s) for s, d in zip(cluster_sizes, cluster_degenerate) if d)}): their covariance collapsed "
         "to the regularisation floor, so their likelihood is unbounded and ΔBIC is inflated by them — these are "
         "outlier runs, not subpopulations. "
         if cluster_degenerate.any() else "") +
        (f"Cluster sizes {', '.join(map(str, sizes))}; bootstrap median ARI {np.median(aris):.2f} "
         f"(95% {np.quantile(aris, .025):.2f}–{np.quantile(aris, .975):.2f}); vehicle-only PCA axes recover it at "
         f"ARI {ari_dmso:.2f}; under the alternative scalings BIC selects "
         + "; ".join(f"K={r.selected_K} with ARI {r.ARI_vs_main_partition:.2f} against this split ({r.scaling})"
                     for r in rob.itertuples()) + ". "
         if tails_not_states else "") +
        ("**Treat the clusters as a descriptive cut of a continuum, not discrete behavioural states.**"
         if descriptive_only else
         "**The split exceeds a single Gaussian, is bootstrap-stable, and is recovered under the alternative "
         "scalings and on vehicle-only PCA axes.**")
    )
    dates_min = A.loc[A.cluster != 0, "date"].unique()
    sameday = (A[A.date.isin(dates_min)].sort_values(["date", "run_order_within_day"])
               [["date", "run_order_within_day", "condition", "cluster", "PC1", "bootstrap_stability"]])
    sameday.to_csv(HERE / "cluster_sameday.csv", index=False)
    size_tbl = pd.DataFrame({"cluster": range(Kd), "n_runs": cluster_sizes,
                             "n_vehicle": [int(A.is_vehicle[A.cluster == j].sum()) for j in range(Kd)],
                             "min covariance eigenvalue": cluster_min_eig,
                             "degenerate": np.where(cluster_degenerate, "yes", "")})
    size_tbl.to_csv(HERE / "cluster_sizes.csv", index=False)
    small_n_note = (
        f"\n> **Warning: {n_scored} scored runs for {len(FEATURES)} features** (fewer than {rpf} runs per "
        "feature). The PCA and mixture are fragile at this size; read the null, bootstrap and robustness "
        "results before anything else.\n" if small_n else "")
    rep = f"""# Run-level PCA → GMM on pose, timing and trigger features

Generated by `run_pipeline.py` in {time.time() - t0:.0f} s (analysis step). Exploratory and hypothesis-generating; **not a hit gate**.
{small_n_note}
## Verdict

{verdict}

- Stability: bootstrap median ARI {np.median(aris):.2f} (95% {np.quantile(aris, .025):.2f}–{np.quantile(aris, .975):.2f}); vehicle-only PCA refit ARI {ari_dmso:.2f}; drop-flagged-runs refit ARI {'' if np.isnan(ari_flag) else f'{ari_flag:.2f}'}{' (no runs flagged)' if np.isnan(ari_flag) else f' ({int(flagged.sum())} flagged)'}.
- Nuisance variables the clusters track (p_perm ≤ 0.05): {', '.join(nuis_flag.variable) if len(nuis_flag) else 'none'}.
- Vehicle vs treated composition within date: p_perm = {tests.loc[tests.variable == 'is_vehicle', 'p_perm'].iloc[0]:.3f}. Same-condition replicates share a cluster at rate {obs_rep:.2f} (within-date permutation null median {np.nanmedian(null_rep):.2f}, p = {p_rep:.3f}); vehicle–vehicle pair rate {dmso_rate:.2f}; chance rate Σp² {base_rate:.2f}.

## Data and features

- {n} runs scored ({int(veh.sum())} vehicle) of {n_cand} with pose features and {len(R)} in the metadata after exclusions. {len(excluded)} not scored{': ' + '; '.join(f'{r.run_id} ({r.reason})' for r in excluded.itertuples()) if len(excluded) else ''}.
- **{len(FEATURES)} fixed features**: 3 trigger rates from the event TSVs, 5 timing features, 6 pose features from the YOLO keypoints (listed below).
- **Primary scaling is rank-normal scores.** Features were transformed (logit for proportions, log1p for rates), EB day-centred on the vehicle runs, then converted to normal scores Φ⁻¹((rank − 0.5)/n). In vehicle-MAD units the largest |z| was {', '.join(f'{f} {v:.0f}' for f, v in mad_max_abs.sort_values(ascending=False).head(3).items())}; features with tight controls and widely spread treated runs would alone define PC1. Normal scores give every feature equal weight; the vehicle-MAD version is refitted as a robustness check below.
- Missing values: {n_imputed} residual missing cell(s) imputed to the vehicle median (at most {allow} per run).
- Scaling guard (vehicle-MAD robustness version): where the vehicle MAD is below {MAD_SD_MIN:g} × the vehicle SD, the vehicle SD is used instead: {', '.join(f'{r.feature} ({r.method})' for r in scale_rep.itertuples() if r.method != 'mad') or 'none needed'}.
- `first_inactive_latency_log` is censored at the window end for {n_censored} run(s) with no in-window inactive trigger.

### Features

{md(sel_log[['order', 'feature', 'block', 'construct', 'transform', 'definition', 'missing_frac']])}

## PCA

Parallel analysis: {k_pa} component(s) exceed the 95th percentile of {N_PA} column-permuted matrices; **{k} retained** (cap {PC_CAP}).

{md(var.head(max(k + 2, 4))[['component', 'eigenvalue', 'parallel_analysis_q95', 'explained_variance_ratio', 'cumulative', 'retained']])}

Loadings (first {max(k, 2)} PCs):

{md(loadings.reset_index().rename(columns={'index': 'feature'})[['feature', 'construct', *pcs[:max(k, 2)]]], '{:+.2f}')}

## GMM

Grid K = 1…{grid_kw['k_max']} × {grid_kw['cov_types']}, n_init {N_INIT}, BIC, K=1 wins ties (ΔBIC < {TIE:g}). Minimum cluster size: {grid_kw['min_cluster_n']} run(s){' — a cluster may hold a single run' if grid_kw['min_cluster_n'] <= 1 else ''}. The null replicates follow the same rule, so p_null compares like with like.

A one- or two-run component can shrink its covariance to the regularisation floor (min eigenvalue < {DEGENERATE_EIG:g}), which makes its likelihood unbounded. Such components are flagged as degenerate rather than removed.

{md(grid.sort_values('bic').head(8)[['K', 'covariance', 'bic', 'n_params', 'min_cluster_n', 'converged']], '{:.1f}')}

### Cluster sizes

{md(size_tbl, '{:.3g}')}

Null selected K: {', '.join(f'K={kk}: {vv:.1%}' for kk, vv in null_k_rate.items())}; null ΔBIC 95th percentile {null.delta_bic.quantile(.95):.1f}.
{'' if chosen.K > 1 else f'K=1 was selected; the partition below is the best K ≥ 2 model ({covd}), shown descriptively.'}

Compound lists per cluster: **`cluster_compounds.md`**. Cluster plot: **`figures/08_clusters_PC1_PC2.png`**. Per-run assignments: `gmm_assignments.csv`.

### Cluster profiles (mean normal score; 0 = median run, ±1 ≈ 1 SD)

{md(prof.pivot(index='feature', columns='cluster', values='mean_z').loc[FEATURES].reset_index(), '{:+.2f}')}

### Confounds, design association, validators

{md(tests[['kind', 'variable', 'statistic', 'value', 'p_perm', 'permutation', 'flag']])}

{md(validators.reset_index())}

### Same-day comparison (dates carrying a minority-cluster run)

{md(sameday)}

## Robustness to feature scaling

{md(rob)}

## Stability

{md(stability)}

## Caveats

- n = {n} runs over {A.condition.nunique()} conditions: a mixture on this many points is fragile; the null, tie rule, bootstrap and robust-scaling refits guard against false structure but cannot add power.
- Condition is confounded with day when a compound is run on few days; the within-date tests are the relevant ones, and date/group associations are expected when compounds share days.
- The trigger rates are the screen's endpoint. Because they are inputs, the validator tests on them are not independent (flagged above).
- Pose features exist only at trigger times (frames are saved when a fish triggers), so they are conditioned on the behaviour being measured, and they cannot measure locomotion away from the platforms. A compound that slows fish without changing drug-seeking can still move the trigger rates; a food self-administration control is the specificity check.
- With no minimum cluster size, BIC can reward components that fit single outlier runs. Check `cluster_sizes.csv` before reading any small cluster as a group.

## Self-test

Planted 2-cluster case → K={selftest['planted_K']}, ARI {selftest['planted_ARI']:.2f}; single Gaussian → K=1 in {selftest['single_gaussian_K1_rate']:.0%} of 50 draws.
"""
    (HERE / "REPORT.md").write_text(rep, encoding="utf-8")

    import scipy
    import sklearn
    results = {"n_runs": n, "n_vehicle": int(veh.sum()), "k_parallel_analysis": k_pa, "k_retained": k,
               "selected_K": int(chosen.K), "selected_covariance": chosen.covariance, "delta_bic": dbic_obs,
               "p_null": p_null, "described_K": Kd, "described_covariance": covd,
               "descriptive_only": descriptive_only, "tails_not_states": tails_not_states,
               "cluster_sizes": cluster_sizes.tolist(), "cluster_degenerate": cluster_degenerate.tolist(),
               "bootstrap_ari_median": float(np.median(aris)), "ari_vehicle_pca": float(ari_dmso),
               "robustness": rob.to_dict("records"), "self_test": selftest}
    prov = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"), "runtime_s": round(time.time() - t0, 1),
        "config_hash": config_mod.hash_config(cfg), "config": cfg,
        "features": FEATURES, "results": results,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "scikit-learn": sklearn.__version__, "scipy": scipy.__version__},
    }
    (HERE / "provenance.json").write_text(json.dumps(prov, indent=2, default=str), encoding="utf-8")
    return results
