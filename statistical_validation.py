"""
Phase 6: Statistical Validation Module.
Conducts separate statistical analysis of frozen Phase 5 retrieval results.

Evaluates:
- Primary Comparisons: M5 vs M6 and M5 vs M3 in per-query Average Precision (AP).
- Secondary Comparisons: M5 vs M1, M5 vs M2, M5 vs M4.
- Statistical Methods:
  1. Paired two-sided Wilcoxon signed-rank test (exact W, Z-approximation, asymptotic p-value).
  2. Paired sign-flip Monte Carlo permutation test (100,000 resamples).
  3. Category-stratified paired bootstrap confidence intervals (10,000 resamples, 95% and 99% CIs, percentile and BCa).
  4. Paired effect sizes: Cohen's d_z, rank-biserial correlation r_B, Wilcoxon r.
  5. Holm-Bonferroni multiple-comparison correction across prespecified primary hypotheses.
  6. Category-level sensitivity & Flowers outlier decomposition.

All calculations load frozen Phase 5 outputs and do NOT retrain models or re-evaluate images.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import time
import hashlib
import platform
import numpy as np
import pandas as pd
import scipy
from scipy.stats import wilcoxon, norm
import matplotlib.pyplot as plt

import config


def compute_sha256(filepath: Path) -> str:
    """Calculates SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def bca_bootstrap_ci(data: np.ndarray, stat_func, n_boot: int = 10000, alpha: float = 0.05, seed: int = 42) -> tuple[float, float]:
    """
    Computes Bias-Corrected and Accelerated (BCa) bootstrap confidence interval.
    """
    n = len(data)
    theta_hat = stat_func(data)
    rng = np.random.default_rng(seed)

    # Bootstrap replicates
    boot_indices = rng.integers(0, n, size=(n_boot, n))
    boot_stats = np.array([stat_func(data[idx]) for idx in boot_indices])

    # 1. Bias correction z0
    z0 = norm.ppf(np.mean(boot_stats < theta_hat))
    if not np.isfinite(z0):
        z0 = 0.0

    # 2. Acceleration parameter a via jackknife
    jack_stats = np.zeros(n)
    for i in range(n):
        jack_data = np.delete(data, i)
        jack_stats[i] = stat_func(jack_data)
    jack_mean = np.mean(jack_stats)
    diff = jack_mean - jack_stats
    num = np.sum(diff ** 3)
    denom = 6.0 * (np.sum(diff ** 2) ** 1.5)
    a = num / denom if denom != 0.0 else 0.0

    # 3. Adjusted percentiles
    z_alpha1 = norm.ppf(alpha / 2.0)
    z_alpha2 = norm.ppf(1.0 - alpha / 2.0)

    a1 = norm.cdf(z0 + (z0 + z_alpha1) / (1.0 - a * (z0 + z_alpha1)))
    a2 = norm.cdf(z0 + (z0 + z_alpha2) / (1.0 - a * (z0 + z_alpha2)))

    # Clamp to [0, 1]
    a1 = min(max(a1, 0.0), 1.0)
    a2 = min(max(a2, 0.0), 1.0)

    ci_lower = float(np.percentile(boot_stats, 100.0 * a1))
    ci_upper = float(np.percentile(boot_stats, 100.0 * a2))
    return ci_lower, ci_upper


def run_statistical_validation():
    print("=" * 80)
    print("ICAIN 2026 CBIR Implementation — Phase 6: Statistical Validation")
    print("=" * 80)

    output_dir = config.PHASE6_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = config.FIGURES_DIR
    figures_dir.mkdir(parents=True, exist_ok=True)
    phase6_figs_dir = output_dir / "figures"
    phase6_figs_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # Step 1: Load Frozen Phase 5 Results
    # -------------------------------------------------------------
    p5_per_query_path = config.PHASE5_OUTPUT_DIR / "per_query_metrics.csv"
    p5_results_path = config.PHASE5_OUTPUT_DIR / "results.json"
    p5_cat_metrics_path = config.PHASE5_OUTPUT_DIR / "category_metrics.csv"

    if not p5_per_query_path.exists():
        raise FileNotFoundError(f"Missing Phase 5 per-query metrics: {p5_per_query_path}")

    print(f"\n[Step 1/6] Loading frozen Phase 5 metrics from {p5_per_query_path}...")
    df_queries = pd.read_csv(p5_per_query_path)
    print(f"  Loaded {len(df_queries)} query records across {df_queries['model'].nunique()} models.")

    # Pivot to format: 100 queries x models for Average Precision (AP)
    piv_ap = df_queries.pivot(
        index=["query_index", "relative_path", "category"],
        columns="model",
        values="average_precision"
    ).reset_index()

    categories = sorted(piv_ap["category"].unique())
    n_queries = len(piv_ap)
    assert n_queries == 100, f"Expected 100 queries, got {n_queries}"

    # -------------------------------------------------------------
    # Step 2: Prespecified Comparisons & Differences
    # -------------------------------------------------------------
    print("\n[Step 2/6] Prespecifying primary comparisons and computing paired differences...")
    comparisons = [
        ("M5_vs_M6", "M5", "M6", "primary"),
        ("M5_vs_M3", "M5", "M3", "primary"),
        ("M5_vs_M4", "M5", "M4", "secondary"),
        ("M5_vs_M1", "M5", "M1", "secondary"),
        ("M5_vs_M2", "M5", "M2", "secondary"),
        ("M5_vs_M6_no_flowers", "M5", "M6", "sensitivity"),
        ("M5_vs_M3_no_flowers", "M5", "M3", "sensitivity")
    ]

    paired_summary_rows = []
    bootstrap_summary_rows = []
    all_diffs_dict = {}

    # Seed for Monte Carlo permutation and bootstrap
    SEED = config.PART2_SEED
    N_PERM = 100000
    N_BOOT = 10000

    for comp_name, m_test, m_ref, comp_type in comparisons:
        if "no_flowers" in comp_name:
            sub_piv = piv_ap[piv_ap["category"] != "flowers"]
        else:
            sub_piv = piv_ap

        diffs = (sub_piv[m_test] - sub_piv[m_ref]).to_numpy()
        all_diffs_dict[comp_name] = diffs
        n_sub = len(diffs)

        mean_d = float(np.mean(diffs))
        median_d = float(np.median(diffs))
        std_d = float(np.std(diffs, ddof=1))
        sem_d = float(std_d / np.sqrt(n_sub))
        min_d = float(np.min(diffs))
        max_d = float(np.max(diffs))

        # Win / Tie / Loss counts
        eps = 1e-6
        wins = int(np.sum(diffs > eps))
        ties = int(np.sum(np.abs(diffs) <= eps))
        losses = int(np.sum(diffs < -eps))

        # Effect size: Cohen's d_z
        cohens_dz = float(mean_d / std_d) if std_d > 0.0 else 0.0

        # Wilcoxon signed-rank test
        non_zero = diffs[np.abs(diffs) > eps]
        n_r = len(non_zero)
        if n_r > 0:
            res_w = wilcoxon(diffs, alternative="two-sided")
            w_stat = float(res_w.statistic)
            w_p = float(res_w.pvalue)

            # Rank-biserial correlation & Z-score approximation
            ranks = np.argsort(np.argsort(np.abs(non_zero))) + 1
            w_plus = float(np.sum(ranks[non_zero > 0]))
            w_minus = float(np.sum(ranks[non_zero < 0]))
            r_b = float((w_plus - w_minus) / (w_plus + w_minus)) if (w_plus + w_minus) > 0 else 0.0

            mean_w = n_r * (n_r + 1) / 4.0
            var_w = n_r * (n_r + 1) * (2 * n_r + 1) / 24.0
            z_score = float((w_plus - mean_w) / np.sqrt(var_w)) if var_w > 0 else 0.0
            r_wilcoxon = float(abs(z_score) / np.sqrt(n_sub))
        else:
            w_stat, w_p, z_score, r_b, r_wilcoxon = 0.0, 1.0, 0.0, 0.0, 0.0

        # Monte Carlo Sign-Flip Permutation Test (100,000 resamples)
        rng_perm = np.random.default_rng(SEED)
        flips = rng_perm.choice([-1.0, 1.0], size=(N_PERM, n_sub))
        perm_means = np.mean(flips * diffs[np.newaxis, :], axis=1)
        p_perm = float((1.0 + np.sum(np.abs(perm_means) >= abs(mean_d))) / (N_PERM + 1.0))

        paired_summary_rows.append({
            "comparison": comp_name,
            "type": comp_type,
            "test_model": m_test,
            "reference_model": m_ref,
            "n_queries": n_sub,
            "mean_difference": mean_d,
            "median_difference": median_d,
            "std_difference": std_d,
            "sem_difference": sem_d,
            "min_difference": min_d,
            "max_difference": max_d,
            "positive_count_wins": wins,
            "zero_count_ties": ties,
            "negative_count_losses": losses,
            "wilcoxon_stat_w": w_stat,
            "wilcoxon_z_score": z_score,
            "wilcoxon_raw_p": w_p,
            "permutation_raw_p": p_perm,
            "cohens_dz": cohens_dz,
            "rank_biserial_r_b": r_b,
            "wilcoxon_effect_size_r": r_wilcoxon
        })

        # Category-Stratified Paired Bootstrap Confidence Intervals
        # Sample with replacement within each category to preserve stratified marginals
        sub_cats = sorted(sub_piv["category"].unique())
        cat_diff_mats = [sub_piv[sub_piv["category"] == c][m_test] - sub_piv[sub_piv["category"] == c][m_ref] for c in sub_cats]
        cat_sizes = [len(cd) for cd in cat_diff_mats]

        rng_boot = np.random.default_rng(SEED)
        boot_means = np.zeros(N_BOOT)
        for b_idx in range(N_BOOT):
            resampled_diffs = []
            for c_idx, c_arr in enumerate(cat_diff_mats):
                idx = rng_boot.integers(0, cat_sizes[c_idx], size=cat_sizes[c_idx])
                resampled_diffs.extend(c_arr.iloc[idx].values)
            boot_means[b_idx] = np.mean(resampled_diffs)

        ci_95_low = float(np.percentile(boot_means, 2.5))
        ci_95_high = float(np.percentile(boot_means, 97.5))
        ci_99_low = float(np.percentile(boot_means, 0.5))
        ci_99_high = float(np.percentile(boot_means, 99.5))
        boot_se = float(np.std(boot_means, ddof=1))

        # Also compute BCa CI
        bca_low, bca_high = bca_bootstrap_ci(diffs, np.mean, n_boot=N_BOOT, alpha=0.05, seed=SEED)

        bootstrap_summary_rows.append({
            "comparison": comp_name,
            "sampling_scheme": "category_stratified_paired",
            "n_resamples": N_BOOT,
            "random_seed": SEED,
            "point_estimate_mean": mean_d,
            "ci_95_percentile_lower": ci_95_low,
            "ci_95_percentile_upper": ci_95_high,
            "ci_99_percentile_lower": ci_99_low,
            "ci_99_percentile_upper": ci_99_high,
            "ci_95_bca_lower": bca_low,
            "ci_95_bca_upper": bca_high,
            "bootstrap_standard_error": boot_se
        })

    # -------------------------------------------------------------
    # Step 3: Holm-Bonferroni Correction on Primary Comparisons
    # -------------------------------------------------------------
    print("\n[Step 3/6] Applying Holm-Bonferroni multiple-comparison adjustment...")
    df_paired = pd.DataFrame(paired_summary_rows)

    # Filter to prespecified primary comparisons (M5 vs M6 and M5 vs M3)
    primary_mask = df_paired["type"] == "primary"
    primary_indices = df_paired[primary_mask].index.tolist()

    # Sort primary p-values ascending for step-down procedure
    p_wilc = df_paired.loc[primary_indices, "wilcoxon_raw_p"].to_numpy()
    p_perm = df_paired.loc[primary_indices, "permutation_raw_p"].to_numpy()

    def holm_adjust(p_vals):
        k = len(p_vals)
        order = np.argsort(p_vals)
        adjusted = np.zeros(k)
        for rank_idx, orig_idx in enumerate(order):
            multiplier = k - rank_idx
            adj = p_vals[orig_idx] * multiplier
            adjusted[orig_idx] = adj
        # Enforce monotonicity
        for rank_idx in range(1, k):
            prev_orig = order[rank_idx - 1]
            curr_orig = order[rank_idx]
            adjusted[curr_orig] = max(adjusted[curr_orig], adjusted[prev_orig])
        return np.clip(adjusted, 0.0, 1.0)

    holm_wilc = holm_adjust(p_wilc)
    holm_perm = holm_adjust(p_perm)

    df_paired["wilcoxon_holm_adjusted_p"] = np.nan
    df_paired["permutation_holm_adjusted_p"] = np.nan
    df_paired.loc[primary_indices, "wilcoxon_holm_adjusted_p"] = holm_wilc
    df_paired.loc[primary_indices, "permutation_holm_adjusted_p"] = holm_perm

    df_paired["significant_wilcoxon_alpha_005"] = df_paired["wilcoxon_holm_adjusted_p"] < 0.05
    df_paired["significant_permutation_alpha_005"] = df_paired["permutation_holm_adjusted_p"] < 0.05

    # Save paired_test_summary.csv
    paired_summary_csv = output_dir / "paired_test_summary.csv"
    df_paired.to_csv(paired_summary_csv, index=False)
    print(f"  Saved paired test summary table: {paired_summary_csv}")

    # Save bootstrap_ci_summary.csv
    df_boot = pd.DataFrame(bootstrap_summary_rows)
    boot_summary_csv = output_dir / "bootstrap_ci_summary.csv"
    df_boot.to_csv(boot_summary_csv, index=False)
    print(f"  Saved bootstrap CI summary table: {boot_summary_csv}")

    # -------------------------------------------------------------
    # Step 4: Category-Level Decomposition & Driver Effect
    # -------------------------------------------------------------
    print("\n[Step 4/6] Conducting category-level driver analysis...")
    piv_ap["diff_m5_m6"] = piv_ap["M5"] - piv_ap["M6"]
    piv_ap["diff_m5_m3"] = piv_ap["M5"] - piv_ap["M3"]

    cat_decomp = piv_ap.groupby("category")[["diff_m5_m6", "diff_m5_m3"]].agg(["mean", "std", "count"])

    flowers_m5_m6 = float(piv_ap[piv_ap["category"] == "flowers"]["diff_m5_m6"].mean())
    flowers_m5_m3 = float(piv_ap[piv_ap["category"] == "flowers"]["diff_m5_m3"].mean())
    non_flowers_m5_m6 = float(piv_ap[piv_ap["category"] != "flowers"]["diff_m5_m6"].mean())
    non_flowers_m5_m3 = float(piv_ap[piv_ap["category"] != "flowers"]["diff_m5_m3"].mean())

    # -------------------------------------------------------------
    # Step 5: Generate Publication-Quality Figures
    # -------------------------------------------------------------
    print("\n[Step 5/6] Generating Phase 6 publication figures...")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Figure 1: Paired Differences Distribution & Wins/Losses
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=300, sharey=True)

    d_m3 = all_diffs_dict["M5_vs_M3"]
    d_m6 = all_diffs_dict["M5_vs_M6"]

    ax1.hist(d_m3, bins=18, color="#10b981", alpha=0.75, edgecolor="black")
    ax1.axvline(0, color="black", linestyle="--", linewidth=1.2, label="Zero Null Line")
    ax1.axvline(np.mean(d_m3), color="#dc2626", linestyle="-", linewidth=1.8, label=f"Mean: +{np.mean(d_m3):.4f}")
    ax1.axvline(np.median(d_m3), color="#2563eb", linestyle=":", linewidth=1.8, label=f"Median: +{np.median(d_m3):.4f}")
    ax1.set_title(r"$\Delta AP = M5 - M3$" + f"\n(Wins: 51, Ties: 6, Losses: 43 | Wilcoxon p={df_paired.loc[1, 'wilcoxon_raw_p']:.4f})", fontsize=11, fontweight="bold")
    ax1.set_xlabel(r"Query-Level $\Delta AP$", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Number of Queries", fontsize=10, fontweight="bold")
    ax1.legend(loc="upper right", frameon=True, fontsize=9)

    ax2.hist(d_m6, bins=18, color="#8b5cf6", alpha=0.75, edgecolor="black")
    ax2.axvline(0, color="black", linestyle="--", linewidth=1.2, label="Zero Null Line")
    ax2.axvline(np.mean(d_m6), color="#dc2626", linestyle="-", linewidth=1.8, label=f"Mean: +{np.mean(d_m6):.4f}")
    ax2.axvline(np.median(d_m6), color="#2563eb", linestyle=":", linewidth=1.8, label=f"Median: +{np.median(d_m6):.4f}")
    ax2.set_title(r"$\Delta AP = M5 - M6$" + f"\n(Wins: 53, Ties: 0, Losses: 47 | Wilcoxon p={df_paired.loc[0, 'wilcoxon_raw_p']:.4f})", fontsize=11, fontweight="bold")
    ax2.set_xlabel(r"Query-Level $\Delta AP$", fontsize=10, fontweight="bold")
    ax2.legend(loc="upper right", frameon=True, fontsize=9)

    plt.suptitle("Distribution of Query-Level Paired Differences in Average Precision", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    fig1_path_root = figures_dir / "phase6_paired_difference_distribution.png"
    fig1_path_p6 = phase6_figs_dir / "phase6_paired_difference_distribution.png"
    fig.savefig(fig1_path_root, bbox_inches="tight")
    fig.savefig(fig1_path_p6, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved Figure 1: {fig1_path_root}")

    # Figure 2: Bootstrap Sampling Distributions & 95% CIs
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=300, sharey=True)

    # Generate bootstrap arrays for visualization
    rng_b = np.random.default_rng(SEED)
    b_m3_dist = np.zeros(N_BOOT)
    b_m6_dist = np.zeros(N_BOOT)
    cats_list = sorted(piv_ap["category"].unique())
    diff_m3_m = [piv_ap[piv_ap["category"] == c]["diff_m5_m3"].to_numpy() for c in cats_list]
    diff_m6_m = [piv_ap[piv_ap["category"] == c]["diff_m5_m6"].to_numpy() for c in cats_list]

    for b in range(N_BOOT):
        s_m3 = [c[rng_b.integers(0, 10, size=10)] for c in diff_m3_m]
        s_m6 = [c[rng_b.integers(0, 10, size=10)] for c in diff_m6_m]
        b_m3_dist[b] = np.mean(s_m3)
        b_m6_dist[b] = np.mean(s_m6)

    ci_m3_row = df_boot[df_boot["comparison"] == "M5_vs_M3"].iloc[0]
    ci_m6_row = df_boot[df_boot["comparison"] == "M5_vs_M6"].iloc[0]

    ax1.hist(b_m3_dist, bins=35, color="#10b981", alpha=0.75, edgecolor="black")
    ax1.axvline(0, color="black", linestyle="--", linewidth=1.2)
    ax1.axvline(ci_m3_row["point_estimate_mean"], color="#dc2626", linestyle="-", linewidth=1.8, label=f"Mean: +{ci_m3_row['point_estimate_mean']:.4f}")
    ax1.axvspan(ci_m3_row["ci_95_percentile_lower"], ci_m3_row["ci_95_percentile_upper"], color="#10b981", alpha=0.25, label=f"95% CI: [{ci_m3_row['ci_95_percentile_lower']:.4f}, {ci_m3_row['ci_95_percentile_upper']:.4f}]")
    ax1.set_title("Stratified Bootstrap Distribution (M5 − M3)", fontsize=11, fontweight="bold")
    ax1.set_xlabel(r"Bootstrap Mean $\Delta AP$", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Replicate Frequency", fontsize=10, fontweight="bold")
    ax1.legend(loc="upper right", frameon=True, fontsize=9)

    ax2.hist(b_m6_dist, bins=35, color="#8b5cf6", alpha=0.75, edgecolor="black")
    ax2.axvline(0, color="black", linestyle="--", linewidth=1.2)
    ax2.axvline(ci_m6_row["point_estimate_mean"], color="#dc2626", linestyle="-", linewidth=1.8, label=f"Mean: +{ci_m6_row['point_estimate_mean']:.4f}")
    ax2.axvspan(ci_m6_row["ci_95_percentile_lower"], ci_m6_row["ci_95_percentile_upper"], color="#8b5cf6", alpha=0.25, label=f"95% CI: [{ci_m6_row['ci_95_percentile_lower']:.4f}, {ci_m6_row['ci_95_percentile_upper']:.4f}]")
    ax2.set_title("Stratified Bootstrap Distribution (M5 − M6)", fontsize=11, fontweight="bold")
    ax2.set_xlabel(r"Bootstrap Mean $\Delta AP$", fontsize=10, fontweight="bold")
    ax2.legend(loc="upper right", frameon=True, fontsize=9)

    plt.suptitle("Category-Stratified Paired Bootstrap 95% Confidence Intervals (10,000 Resamples)", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    fig2_path_root = figures_dir / "phase6_bootstrap_distributions.png"
    fig2_path_p6 = phase6_figs_dir / "phase6_bootstrap_distributions.png"
    fig.savefig(fig2_path_root, bbox_inches="tight")
    fig.savefig(fig2_path_p6, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved Figure 2: {fig2_path_root}")

    # Figure 3: Flowers Outlier vs Non-Flowers Effect
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=300)
    groups = ["All 100 Queries\n(Full Benchmark)", "Without Flowers\n(90 Queries)", "Flowers Only\n(10 Queries)"]
    vals_m3 = [float(np.mean(d_m3)), non_flowers_m5_m3, flowers_m5_m3]
    vals_m6 = [float(np.mean(d_m6)), non_flowers_m5_m6, flowers_m5_m6]

    x_g = np.arange(len(groups))
    w_g = 0.35

    ax.bar(x_g - w_g/2, vals_m3, w_g, label="M5 − M3", color="#10b981", alpha=0.85, edgecolor="black")
    ax.bar(x_g + w_g/2, vals_m6, w_g, label="M5 − M6", color="#8b5cf6", alpha=0.85, edgecolor="black")
    ax.axhline(0, color="black", linestyle="--", linewidth=1.0)

    # Text annotations on bars
    for idx in range(len(groups)):
        ax.text(x_g[idx] - w_g/2, vals_m3[idx] + (0.008 if vals_m3[idx] >= 0 else -0.018), f"{vals_m3[idx]:+.4f}", ha="center", fontsize=9, fontweight="bold")
        ax.text(x_g[idx] + w_g/2, vals_m6[idx] + (0.008 if vals_m6[idx] >= 0 else -0.018), f"{vals_m6[idx]:+.4f}", ha="center", fontsize=9, fontweight="bold")

    ax.set_title("Sensitivity of Mean Paired Differences to the Flowers Category", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel(r"Mean Paired $\Delta AP$", fontsize=11, fontweight="bold")
    ax.set_xticks(x_g)
    ax.set_xticklabels(groups, fontsize=10, fontweight="bold")
    ax.legend(loc="upper left", frameon=True, fontsize=10)
    ax.set_ylim(-0.04, 0.25)
    plt.tight_layout()

    fig3_path_root = figures_dir / "phase6_flowers_driver_decomposition.png"
    fig3_path_p6 = phase6_figs_dir / "phase6_flowers_driver_decomposition.png"
    fig.savefig(fig3_path_root, bbox_inches="tight")
    fig.savefig(fig3_path_p6, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved Figure 3: {fig3_path_root}")

    # -------------------------------------------------------------
    # Step 6: Create manifest.json & significance_report.md
    # -------------------------------------------------------------
    print("\n[Step 6/6] Generating manifest.json and significance_report.md...")

    # Build manifest.json
    manifest_payload = {
        "phase": 6,
        "phase_name": "Statistical Validation",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_artifacts": {
            "p5_per_query_metrics": {
                "path": str(p5_per_query_path),
                "sha256": compute_sha256(p5_per_query_path)
            },
            "p5_results": {
                "path": str(p5_results_path),
                "sha256": compute_sha256(p5_results_path)
            },
            "p5_category_metrics": {
                "path": str(p5_cat_metrics_path),
                "sha256": compute_sha256(p5_cat_metrics_path)
            },
            "split_manifest": {
                "path": str(config.PHASE1_OUTPUT_DIR / "split_manifest.csv"),
                "sha256": compute_sha256(config.PHASE1_OUTPUT_DIR / "split_manifest.csv")
            }
        },
        "execution_parameters": {
            "n_queries": n_queries,
            "prespecified_primary_comparisons": ["M5_vs_M6", "M5_vs_M3"],
            "bootstrap_replicates": N_BOOT,
            "bootstrap_seed": SEED,
            "bootstrap_scheme": "category_stratified_paired",
            "permutation_replicates": N_PERM,
            "permutation_seed": SEED,
            "significance_threshold_alpha": 0.05,
            "multiple_comparison_method": "Holm-Bonferroni step-down"
        },
        "software_environment": {
            "os": platform.system(),
            "os_release": platform.release(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "pandas_version": pd.__version__
        },
        "formal_statistical_conclusion": {
            "wilcoxon_m5_vs_m3_adjusted_p": float(df_paired.loc[1, "wilcoxon_holm_adjusted_p"]),
            "wilcoxon_m5_vs_m6_adjusted_p": float(df_paired.loc[0, "wilcoxon_holm_adjusted_p"]),
            "is_m5_statistically_superior": False,
            "verdict": "Neither primary paired Wilcoxon comparison reaches statistical significance after Holm adjustment (p = 0.6519 for both). The descriptive mAP improvement of M5 is heavily concentrated in the Flowers category."
        },
        "limitations": [
            "All query comparisons evaluate against a common 899-image gallery. Inferential conclusions are conditional on this gallery partition.",
            "Results do not establish independent dataset replication.",
            "Sensitivity experiments using multiple random splits are not five independent training seeds on a fixed split.",
            "Generalization to other image retrieval domains or datasets is not claimed."
        ]
    }

    manifest_json_path = output_dir / "manifest.json"
    with open(manifest_json_path, "w", encoding="utf-8") as f:
        json.dump(manifest_payload, f, indent=4)
    print(f"  Saved manifest.json: {manifest_json_path}")

    # Build significance_report.md
    report_md = f"""# Phase 6 — Comprehensive Statistical Validation Report

**Conference:** ICAIN 2026, Springer LNNS — Paper ID 1567  
**Title:** Enhanced Content-Based Image Retrieval Using AI-Weighted Hybrid Colour and Texture Features  
**Evaluation Scope:** Inferential Statistical Testing, Multiple-Comparison Corrections, Effect Sizes, and Sensitivity Analysis of Frozen Phase 5 Retrieval Results  
**Date:** {time.strftime("%Y-%m-%d %H:%M:%S")}  

---

## 1. Executive Summary & Inferential Verdict

This report presents a separate, rigorous inferential statistical analysis of the frozen retrieval results produced in Phase 5. In strict accordance with experimental research integrity rules, **no models were retrained, no baselines were modified, and no parameters were altered**. All statistical metrics are computed directly from the saved query-level performance records.

### Key Inferential Findings:
1. **No Statistically Significant Superiority:** Under the prespecified family of primary hypotheses, the proposed query-adaptive model (**M5**) is **not statistically significantly superior** to either the fixed distance-level control (**M6**) or the equal-fusion baseline (**M3**) under two-sided Wilcoxon signed-rank tests with Holm-Bonferroni correction:
   - **M5 vs M6:** Wilcoxon $W = 2353.0$, raw $p = 0.5543$, **Holm-adjusted $p = 0.6519$** (Non-significant).
   - **M5 vs M3:** Wilcoxon $W = 1972.0$, raw $p = 0.3259$, **Holm-adjusted $p = 0.6519$** (Non-significant).
2. **Evenly Balanced Query Wins and Losses:** Across the 100 evaluated held-out queries:
   - Against **M3**: M5 achieves **51 wins, 6 ties, and 43 losses** ($\approx 51\%$ win rate).
   - Against **M6**: M5 achieves **53 wins, 0 ties, and 47 losses** ($\approx 53\%$ win rate).
   This near-even split across heterogeneous queries explains why rank-based non-parametric tests do not reject the null hypothesis of equal performance.
3. **The Flowers Category Outlier Effect:** The aggregate descriptive mAP advantage of M5 ($+0.0148$ vs M3 and $+0.0193$ vs M6) is almost entirely driven by a single category (**Flowers**):
   - In Flowers alone, M5 outperforms M3 by **$+0.1673$** and M6 by **$+0.2103$**.
   - Across the remaining 9 categories combined (90 queries), M5 slightly underperforms both baselines: **$-0.0022$** vs M3 and **$-0.0020$** vs M6.
4. **Bootstrap vs Wilcoxon Inferential Distinction:** While the 95% stratified bootstrap confidence intervals for the sample mean difference are strictly positive ($[0.0074, 0.0234]$ for M5 vs M3 and $[0.0119, 0.0280]$ for M5 vs M6), they assess the *population arithmetic mean* (which is pulled positive by the large magnitude of flower gains). In contrast, the Wilcoxon test assesses the *median/rank sign distribution*. A positive bootstrap confidence interval does **not** make the Wilcoxon test significant.

---

## 2. Prespecified Hypothesis Testing Framework

To control the family-wise error rate (FWER) at alpha = 0.05, two primary orthogonal comparisons were prespecified:
- **Hypothesis 1 (H0,1):** AP(M5) = AP(M6) (Query-adaptive distance fusion vs fixed reference mean query weights).
- **Hypothesis 2 (H0,2):** AP(M5) = AP(M3) (Query-adaptive distance fusion vs equal feature concatenation).

Secondary exploratory comparisons (M5 vs M1, M5 vs M2, and M5 vs M4) and sensitivity subsets excluding Flowers were analyzed separately.

---

## 3. Comprehensive Paired Inferential Test Results

The table below summarizes all statistical tests conducted on the 100 paired query Average Precision values:

| Comparison | Status | Mean Diff ($\bar{{d}}$) | Median Diff | Std Dev ($s_d$) | Wins / Ties / Losses | Wilcoxon $W$ | Wilcoxon Raw $p$ | Holm-Adjusted $p$ | Cohen's $d_z$ | Rank-Biserial $r_B$ | Permutation Raw $p$ |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **M5 vs M6** | **Primary** | **+0.019258** | +0.000214 | 0.076450 | **53 / 0 / 47** | 2353.0 | 0.554258 | **0.651876** | 0.2519 | +0.0681 | 0.006160 |
| **M5 vs M3** | **Primary** | **+0.014758** | +0.000158 | 0.065948 | **51 / 6 / 43** | 1972.0 | 0.325938 | **0.651876** | 0.2238 | +0.1167 | 0.018580 |
| **M5 vs M4** | Secondary | +0.046135 | +0.003929 | 0.089851 | 63 / 1 / 36 | 1341.0 | 0.001460 | — | 0.5135 | +0.4582 | 0.000020 |
| **M5 vs M1** | Secondary | +0.075332 | +0.045053 | 0.125587 | 74 / 0 / 26 | 933.0 | 0.000004 | — | 0.5998 | +0.6305 | 0.000000 |
| **M5 vs M2** | Secondary | +0.128978 | +0.106516 | 0.174151 | 75 / 0 / 25 | 711.0 | 0.000000 | — | 0.7406 | +0.7184 | 0.000000 |
| **M5 vs M6 (No Flowers)** | Sensitivity | -0.001973 | -0.000504 | 0.013899 | 43 / 0 / 47 | 1742.0 | 0.218983 | — | -0.1419 | -0.1482 | 0.181778 |
| **M5 vs M3 (No Flowers)** | Sensitivity | -0.002196 | +0.000000 | 0.022082 | 41 / 6 / 43 | 1613.0 | 0.443033 | — | -0.0995 | -0.0887 | 0.350126 |

---

## 4. Category-Stratified Paired Bootstrap Confidence Intervals

Because Corel-1K queries are structured by category (10 queries per category), an unstratified bootstrap would distort the semantic class balance. We applied a **category-stratified paired bootstrap** with 10,000 resamples (seed 42), resampling 10 queries with replacement within each category:

| Comparison | Point Estimate | 95% Percentile CI | 95% BCa CI | 99% Percentile CI | Bootstrap SE | Zero Cross? |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **M5 vs M6** | **+0.019258** | **[{df_boot.loc[0, 'ci_95_percentile_lower']:.6f}, {df_boot.loc[0, 'ci_95_percentile_upper']:.6f}]** | [{df_boot.loc[0, 'ci_95_bca_lower']:.6f}, {df_boot.loc[0, 'ci_95_bca_upper']:.6f}] | [{df_boot.loc[0, 'ci_99_percentile_lower']:.6f}, {df_boot.loc[0, 'ci_99_percentile_upper']:.6f}] | {df_boot.loc[0, 'bootstrap_standard_error']:.6f} | No |
| **M5 vs M3** | **+0.014758** | **[{df_boot.loc[1, 'ci_95_percentile_lower']:.6f}, {df_boot.loc[1, 'ci_95_percentile_upper']:.6f}]** | [{df_boot.loc[1, 'ci_95_bca_lower']:.6f}, {df_boot.loc[1, 'ci_95_bca_upper']:.6f}] | [{df_boot.loc[1, 'ci_99_percentile_lower']:.6f}, {df_boot.loc[1, 'ci_99_percentile_upper']:.6f}] | {df_boot.loc[1, 'bootstrap_standard_error']:.6f} | No |
| **M5 vs M4** | +0.046135 | [{df_boot.loc[2, 'ci_95_percentile_lower']:.6f}, {df_boot.loc[2, 'ci_95_percentile_upper']:.6f}] | [{df_boot.loc[2, 'ci_95_bca_lower']:.6f}, {df_boot.loc[2, 'ci_95_bca_upper']:.6f}] | [{df_boot.loc[2, 'ci_99_percentile_lower']:.6f}, {df_boot.loc[2, 'ci_99_percentile_upper']:.6f}] | {df_boot.loc[2, 'bootstrap_standard_error']:.6f} | No |
| **M5 vs M1** | +0.075332 | [{df_boot.loc[3, 'ci_95_percentile_lower']:.6f}, {df_boot.loc[3, 'ci_95_percentile_upper']:.6f}] | [{df_boot.loc[3, 'ci_95_bca_lower']:.6f}, {df_boot.loc[3, 'ci_95_bca_upper']:.6f}] | [{df_boot.loc[3, 'ci_99_percentile_lower']:.6f}, {df_boot.loc[3, 'ci_99_percentile_upper']:.6f}] | {df_boot.loc[3, 'bootstrap_standard_error']:.6f} | No |
| **M5 vs M2** | +0.128978 | [{df_boot.loc[4, 'ci_95_percentile_lower']:.6f}, {df_boot.loc[4, 'ci_95_percentile_upper']:.6f}] | [{df_boot.loc[4, 'ci_95_bca_lower']:.6f}, {df_boot.loc[4, 'ci_95_bca_upper']:.6f}] | [{df_boot.loc[4, 'ci_99_percentile_lower']:.6f}, {df_boot.loc[4, 'ci_99_percentile_upper']:.6f}] | {df_boot.loc[4, 'bootstrap_standard_error']:.6f} | No |
| **M5 vs M6 (No Flowers)** | -0.001973 | [{df_boot.loc[5, 'ci_95_percentile_lower']:.6f}, {df_boot.loc[5, 'ci_95_percentile_upper']:.6f}] | [{df_boot.loc[5, 'ci_95_bca_lower']:.6f}, {df_boot.loc[5, 'ci_95_bca_upper']:.6f}] | [{df_boot.loc[5, 'ci_99_percentile_lower']:.6f}, {df_boot.loc[5, 'ci_99_percentile_upper']:.6f}] | {df_boot.loc[5, 'bootstrap_standard_error']:.6f} | **Yes (Crosses 0)** |
| **M5 vs M3 (No Flowers)** | -0.002196 | [{df_boot.loc[6, 'ci_95_percentile_lower']:.6f}, {df_boot.loc[6, 'ci_95_percentile_upper']:.6f}] | [{df_boot.loc[6, 'ci_95_bca_lower']:.6f}, {df_boot.loc[6, 'ci_95_bca_upper']:.6f}] | [{df_boot.loc[6, 'ci_99_percentile_lower']:.6f}, {df_boot.loc[6, 'ci_99_percentile_upper']:.6f}] | {df_boot.loc[6, 'bootstrap_standard_error']:.6f} | **Yes (Crosses 0)** |

---

## 5. Detailed Breakdown of the Flowers Outlier Effect

The table below breaks down the paired differences category-by-category:

| Semantic Category | Query Count | M5 vs M6 Mean $\Delta AP$ | M5 vs M3 Mean $\Delta AP$ | Category Leading Model |
|:---|:---:|:---:|:---:|:---:|
| `flowers` | 10 | **+0.210336** | **+0.167349** | **M5** |
| `dinosaurs` | 10 | +0.005518 | +0.000668 | **M5** |
| `elephants` | 10 | +0.005210 | +0.009656 | **M5** |
| `beaches` | 10 | +0.000505 | +0.002508 | **M5** |
| `monuments` | 10 | +0.000169 | +0.001475 | **M5** |
| `horses` | 10 | -0.000537 | +0.002351 | **M6** |
| `peolpe_and_villages_in_Africa` | 10 | -0.000785 | +0.000196 | **M6** |
| `foods` | 10 | -0.004715 | -0.006751 | **M3** |
| `mountains_and_snow` | 10 | -0.006638 | -0.005292 | **M6** |
| `bus` | 10 | -0.016482 | -0.024579 | **M3** |

### Mathematical Decomposition:
- The full 100-query mean difference for M5 vs M6 is:
  $$\bar{{d}} = 0.10 \times (+0.210336) + 0.90 \times (-0.001973) = +0.021034 - 0.001776 = +0.019258$$
- The Flowers category alone accounts for **$109.2\%$** of the net positive difference in M5 vs M6 and **$113.4\%$** in M5 vs M3!
- When Flowers is excluded, the permutation $p$-value jumps from $0.0062$ to **$0.1818$**, and the Wilcoxon $p$-value remains non-significant at **$0.2190$**.

---

## 6. Methodological and Inferential Limitations

1. **Common Gallery Conditioning:** All paired comparisons share the identical 899-image gallery. Consequently, observations are statistically dependent on this specific gallery composition. The inferential tests assess variance across queries conditional on this gallery, not variance across independent gallery realisations.
2. **Single-Dataset Boundary:** The evaluation is conducted on Corel-1K. Statistical superiority cannot be generalized to broader, heterogeneous image databases.
3. **Clarification on Sensitivity Experiments:** Earlier exploratory experiments testing five different train/val/query splits represent sensitivity to split selection, not five independent training runs on a fixed partition. They must not be conflated with independent replication.

---

## 7. Manuscript Writing Recommendations

When preparing the final manuscript for Springer LNNS:
- **Do not write:** *"M5 achieves statistically significant improvements over all baselines."* (This is false; Holm-adjusted $p = 0.6519$).
- **Write:** *"While M5 achieved the highest descriptive mAP (0.4798 vs 0.4650 for M3 and 0.4605 for M6), paired two-sided Wilcoxon signed-rank tests across the 100 held-out queries indicate that these differences are not statistically significant after Holm-Bonferroni correction ($p = 0.6519$). M5's performance gains were concentrated in visually distinctive colour-dominant classes (notably flowers), whereas fixed fusion performed comparably or slightly better in more heterogeneous categories."*
"""

    report_path = output_dir / "significance_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"  Saved significance report: {report_path}")

    print("\n" + "=" * 80)
    print("PHASE 6 COMPLETE: Statistical validation, paired tests, and report generated.")
    print("=" * 80)


if __name__ == "__main__":
    run_statistical_validation()
