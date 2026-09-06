"""
Figures for the federated PD speech-biomarker paper.

Plotting only. All numbers come from experiment.py; this module never
re-implements any part of the experiment.

    python make_figures.py

Uses cached results (results.npz) if present, otherwise runs the benchmark
first. Figures are written to figures/.
"""

from __future__ import annotations

import os

import numpy as np
from scipy import stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

import experiment as exp

OUTPUT_DIR = "figures"

# Palette. Teal reads as "healthy", red as "PD", consistently across figures.
C_HEALTHY = "#2A9D8F"
C_PD = "#E63946"
C_FED = "#264653"
C_LOCAL = "#9AA0A6"
C_CENTRAL = "#E9C46A"

DISPLAY_NAME = {
    "VSA": "VSA",
    "jitter_pct": "jitter",
    "shimmer_db": "shimmer",
    "hnr_db": "HNR",
    "f0_std_hz": "F0 SD",
}


def _save(fig, name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


# ---------------------------------------------------------------------
# Figure 1: vowel triangles, healthy vs PD
# ---------------------------------------------------------------------
def fig_vowel_triangle(s):
    fig, ax = plt.subplots(figsize=(6.5, 5))

    for label, colour, legend in (
        ("healthy", C_HEALTHY, "Healthy"),
        ("pd", C_PD, "Parkinson's"),
    ):
        tri = s["mean_vowels"][label]
        pts = np.array([tri["a"], tri["i"], tri["u"]])
        area = s["vsa"][label]
        ax.add_patch(
            Polygon(pts, closed=True, facecolor=colour, alpha=0.22,
                    edgecolor=colour, lw=2.5,
                    label=f"{legend} (VSA {area / 1000:.0f}k Hz$^2$)")
        )
        ax.scatter(pts[:, 0], pts[:, 1], color=colour, s=70, zorder=5)

    for v in exp.VOWEL_TARGETS:
        ax.annotate(f"/{v}/", s["mean_vowels"]["healthy"][v], fontsize=13,
                    fontweight="bold", xytext=(8, 8), textcoords="offset points")

    ax.set_xlabel("F1 (Hz)", fontsize=11)
    ax.set_ylabel("F2 (Hz)", fontsize=11)
    ax.set_title("Articulatory impairment: vowel space shrinkage",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9.5)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _save(fig, "fig1_vowel_triangle.pdf")


# ---------------------------------------------------------------------
# Figure 2: method comparison, absolute and paired
# ---------------------------------------------------------------------
def fig_method_comparison(s):
    fig, (ax_abs, ax_paired) = plt.subplots(
        1, 2, figsize=(9.4, 4.1), gridspec_kw={"width_ratios": [1, 1.05]}
    )

    # (a) absolute AUC with 95% CIs
    labels = ["Local-only\n(no sharing)", "Federated\n(data-minimizing)",
              "Centralized\n(pooled bound)"]
    values = [s["local_auc"].mean(), s["fed_auc"].mean(), s["central_auc"].mean()]
    errors = [exp.ci95(s["local_auc"]), exp.ci95(s["fed_auc"]),
              exp.ci95(s["central_auc"])]

    bars = ax_abs.bar(labels, values, yerr=errors, capsize=6,
                      color=[C_LOCAL, C_FED, C_CENTRAL],
                      edgecolor="black", lw=0.6)
    for bar, v, e in zip(bars, values, errors):
        ax_abs.text(bar.get_x() + bar.get_width() / 2, v + e + 0.006,
                    f"{v:.3f}", ha="center", fontsize=9.5, fontweight="bold")
    ax_abs.set_ylabel(f"Pooled-test AUC (mean $\\pm$ 95% CI, {len(s['fed_auc'])} seeds)",
                      fontsize=9.5)
    ax_abs.set_ylim(0.5, 1.0)
    ax_abs.set_title("(a) Absolute performance", fontsize=11, fontweight="bold")
    ax_abs.grid(alpha=0.25, axis="y")
    ax_abs.tick_params(labelsize=8.5)

    rng = np.random.default_rng(0)
    bound = exp.EQUIVALENCE_BOUND
    ax_paired.set_ylim(-0.030, 0.034)
    ax_paired.add_patch(
        Rectangle((0.55, -bound), 0.9, 2 * bound, facecolor="grey",
                  alpha=0.13, zorder=0, lw=0)
    )
    ax_paired.text(1.44, bound - 0.001, f"$\\pm${bound}\nequivalence",
                   fontsize=7.0, color="grey", ha="right", va="top")
    ax_paired.axhline(0, color="black", lw=0.9, zorder=1)

    for x, diffs, colour in ((0, s["d_local"], C_FED), (1, s["d_central"], C_CENTRAL)):
        jitter = rng.normal(0, 0.05, size=len(diffs))
        ax_paired.scatter(np.full_like(diffs, x) + jitter, diffs, s=22,
                          color=colour, alpha=0.55, edgecolor="white",
                          lw=0.4, zorder=2)
        ax_paired.errorbar(x, diffs.mean(), yerr=exp.ci95(diffs), fmt="o",
                           color="black", ms=7, capsize=6, lw=1.6, zorder=3)

    ax_paired.set_xticks([0, 1])
    ax_paired.set_xticklabels(["Fed $-$ Local", "Fed $-$ Central"], fontsize=9.5)
    ax_paired.set_xlim(-0.55, 1.62)
    ax_paired.set_ylabel("Paired $\\Delta$ AUC (per seed)", fontsize=9.5)
    ax_paired.set_title("(b) Paired differences across seeds",
                        fontsize=11, fontweight="bold")
    ax_paired.grid(alpha=0.25, axis="y")
    ax_paired.tick_params(labelsize=8.5)

    d = s["d_local"]
    ax_paired.annotate(
        f"$\\Delta$=+{d.mean():.3f}\n$p$={s['p_vs_local']:.0e} (paired $t$)\n"
        f"{(d > 0).sum()}/{len(d)} seeds $>$0",
        xy=(0, d.mean()), xytext=(-0.48, 0.0225), fontsize=7.6,
        color=C_FED, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C_FED, lw=1),
    )
    ax_paired.annotate(
        f"$\\Delta$={s['d_central'].mean():+.3f}\nTOST $p$={s['p_tost']:.0e}",
        xy=(1, s["d_central"].mean()), xytext=(1.16, -0.024), fontsize=7.6,
        color="#8A6D1A", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#8A6D1A", lw=1),
    )

    fig.tight_layout()
    _save(fig, "fig2_method_comparison.pdf")


# ---------------------------------------------------------------------
# Figure 3: coefficient recovery and sign stability
# ---------------------------------------------------------------------
def fig_coefficients(s):
    weights = s["weights"]
    mean_w = weights.mean(axis=0)
    ci_w = 1.96 * stats.sem(weights, axis=0)
    order = np.argsort(np.abs(mean_w))          # least to most important

    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    ax.set_xlim(-1.82, 0.98)

    for y, idx in enumerate(order):
        name = exp.FEATURE_NAMES[idx]
        m, e = mean_w[idx], ci_w[idx]
        colour = C_PD if m > 0 else C_HEALTHY
        ax.barh(y, m, xerr=e, capsize=4, color=colour, edgecolor="black",
                lw=0.5, height=0.62, zorder=2)
        # per-seed values, so the reader sees the spread, not just the mean
        ax.scatter(weights[:, idx],
                   np.full(len(weights), y)
                   + np.random.default_rng(idx).normal(0, 0.06, len(weights)),
                   s=10, color="black", alpha=0.28, zorder=3)
        mark = "\u2713" if np.sign(m) == exp.EXPECTED_SIGN[name] else "\u2717"
        ax.text(m + (e + 0.05) * np.sign(m), y, mark, va="center",
                ha="left" if m > 0 else "right", fontsize=12)

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([DISPLAY_NAME[exp.FEATURE_NAMES[i]] for i in order],
                       fontsize=10)
    ax.axvline(0, color="black", lw=0.9)
    ax.set_xlabel(
        f"Standardized coefficient (mean $\\pm$ 95% CI, {len(weights)} seeds)",
        fontsize=10,
    )
    ax.set_title("Coefficient recovery & sign stability",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.25, axis="x")
    ax.text(0.02, 0.965, "teal $\\rightarrow$ healthy", transform=ax.transAxes,
            color=C_HEALTHY, fontsize=9, fontweight="bold", va="top", ha="left")
    ax.text(0.985, 0.045, "red $\\rightarrow$ PD", transform=ax.transAxes,
            color=C_PD, fontsize=9, fontweight="bold", va="bottom", ha="right")

    # Report stability honestly: what is robust is the top feature and the
    # signs, NOT the full ranking of the phonatory descriptors.
    top = max(s["top_counts"], key=s["top_counts"].get)
    ax.text(
        0.985, 0.93,
        f"{DISPLAY_NAME[top]} ranked #1 in {s['top_counts'][top]}/{len(weights)} seeds\n"
        f"All {s['signs_correct']} signs match physiology ($\\checkmark$)",
        transform=ax.transAxes, fontsize=9, va="top", ha="right",
        fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.92,
                  edgecolor="gray"),
    )
    fig.tight_layout()
    _save(fig, "fig3_coefficient_recovery.pdf")


# ---------------------------------------------------------------------
# Figure 4 (supplementary): convergence over federated rounds
# ---------------------------------------------------------------------
def fig_convergence(s):
    hist = s["histories"]
    rounds = np.arange(1, hist.shape[1] + 1)
    mean = hist.mean(axis=0)
    band = 1.96 * stats.sem(hist, axis=0)

    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    ax.plot(rounds, mean, "-o", color=C_FED, lw=2.5, ms=4, label="FedAvg mean")
    ax.fill_between(rounds, mean - band, mean + band, color=C_FED, alpha=0.2,
                    label="95% CI")
    ax.set_xlabel("Federated round", fontsize=11)
    ax.set_ylabel("Pooled-test AUC", fontsize=11)
    ax.set_title(f"Convergence ({len(hist)} seeds)", fontsize=13,
                 fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _save(fig, "fig4_convergence.pdf")


def main():
    if os.path.exists(exp.RESULTS_FILE):
        print(f"Loading cached results from {exp.RESULTS_FILE}")
        s = exp.load_results()
    else:
        print("No cached results found, running benchmark first.")
        s = exp.run_benchmark()
        exp.save_results(s)

    print("\nGenerating figures:")
    fig_vowel_triangle(s)
    fig_method_comparison(s)
    fig_coefficients(s)
    fig_convergence(s)
    print("\nDone.")


if __name__ == "__main__":
    main()
