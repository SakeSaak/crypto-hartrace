"""Stylized facts (1-3) for BTC-EUR: distributions, clustering, long memory.

Inputs
------
  data/processed/realized_measures_btc_eur_15m.parquet

Outputs
-------
  outputs/figures/    PNG figures (distributions, ACFs, tails, long memory)
  outputs/tables/     CSV summary tables
  outputs/02_stylized_facts_summary.md  Markdown report
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from hartrace.stylized import (
    moments, fit_student_t, fit_hansen_skewt, qq_points,
    hill_estimator, gpd_tail,
    acf_with_ci, pacf_with_ci, ljung_box, arch_lm,
    hurst_rs, gph,
)


PROJECT = Path(__file__).parent.parent
INPUT = PROJECT / "data" / "processed" / "realized_measures_btc_eur_15m.parquet"
FIG_DIR = PROJECT / "outputs" / "figures"
TBL_DIR = PROJECT / "outputs" / "tables"
SUMMARY = PROJECT / "outputs" / "02_stylized_facts_summary.md"

plt.rcParams.update({
    'figure.dpi': 110, 'savefig.dpi': 130,
    'axes.spines.top': False, 'axes.spines.right': False,
    'font.size': 10,
})


# =====================================================================
# Helpers
# =====================================================================
def _save(fig, name):
    out = FIG_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"  → {out.relative_to(PROJECT)}")


def _ci_bands(ci_arr):
    """Convert statsmodels (n, 2) CI bands into ±-half-width around 0."""
    return float(ci_arr[1, 1] - ci_arr[1, 0]) / 2.0


# =====================================================================
# Main analysis
# =====================================================================
def main():
    print("=" * 64)
    print("02_stylized_facts.py — BTC-EUR realized measures (1-3)")
    print("=" * 64)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TBL_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)

    print("\n[1/5] Loading data")
    df = pd.read_parquet(INPUT)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df['log_RV'] = np.log(df['RV'])
    df['abs_r'] = np.abs(df['r_d'])
    df['r2'] = df['r_d'] ** 2
    df['neg_share'] = df['RS_neg'] / (df['RS_neg'] + df['RS_pos'])
    print(f"  N = {len(df)} days from {df['date'].min().date()} to {df['date'].max().date()}")

    # =================================================================
    # FACT 1 — Distribution properties
    # =================================================================
    print("\n[2/5] FACT 1: Distribution properties")
    r = df['r_d'].values
    lrv = df['log_RV'].values

    # Tables of moments
    mom_r = moments(r)
    mom_lrv = moments(lrv)

    # ML-fit Student-t and skewed-t
    t_r = fit_student_t(r)
    st_r = fit_hansen_skewt(r)
    t_lrv = fit_student_t(lrv)
    st_lrv = fit_hansen_skewt(lrv)

    print(f"  Returns r_d: skew={mom_r['skew']:.3f}, excess kurt={mom_r['excess_kurt']:.3f}, "
          f"JB-p={mom_r['jb_pvalue']:.3e}")
    print(f"  Returns r_d: Student-t ν̂={t_r['nu']:.2f};  skew-t ν̂={st_r['nu']:.2f}, λ̂={st_r['lam']:.3f}")
    print(f"  log RV    : skew={mom_lrv['skew']:.3f}, excess kurt={mom_lrv['excess_kurt']:.3f}, "
          f"JB-p={mom_lrv['jb_pvalue']:.3e}")
    print(f"  log RV    : Student-t ν̂={t_lrv['nu']:.2f};  skew-t ν̂={st_lrv['nu']:.2f}, λ̂={st_lrv['lam']:.3f}")

    # --- Figure 1: distributional properties of r_d ---
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    ax[0, 0].plot(df['date'], r, lw=0.7, color='#2b6cb0')
    ax[0, 0].set_title('Daily log-returns over time')
    ax[0, 0].set_ylabel(r'$r_t$')
    ax[0, 0].axhline(0, color='k', lw=0.5)

    # Histogram + densities
    bins = 50
    ax[0, 1].hist(r, bins=bins, density=True, alpha=0.45,
                   color='#a0aec0', edgecolor='white')
    xs = np.linspace(r.min(), r.max(), 400)
    from scipy.stats import norm
    ax[0, 1].plot(xs, norm.pdf(xs, np.mean(r), np.std(r, ddof=1)),
                   lw=1.5, color='#2b6cb0', label='Normal')
    from scipy.stats import t as student_t
    ax[0, 1].plot(xs, student_t.pdf(xs, df=t_r['nu'], loc=t_r['mu'], scale=t_r['sigma']),
                   lw=1.5, color='#d97706', label=f"Student-t (ν̂={t_r['nu']:.1f})")
    ax[0, 1].set_title('Histogram + fitted densities')
    ax[0, 1].set_xlabel(r'$r_t$')
    ax[0, 1].legend()

    # QQ vs Gaussian
    th, sa = qq_points(r, dist='norm')
    ax[1, 0].scatter(th, sa, s=10, color='#2b6cb0', alpha=0.6)
    lim = [min(th.min(), sa.min()), max(th.max(), sa.max())]
    ax[1, 0].plot(lim, lim, 'k--', lw=0.8)
    ax[1, 0].set_title('QQ-plot vs Gaussian')
    ax[1, 0].set_xlabel('Theoretical quantile')
    ax[1, 0].set_ylabel('Sample quantile')

    # QQ vs skewed-t
    th2, sa2 = qq_points(r, dist='skewt', params=st_r)
    ax[1, 1].scatter(th2, sa2, s=10, color='#d97706', alpha=0.6)
    lim2 = [min(th2.min(), sa2.min()), max(th2.max(), sa2.max())]
    ax[1, 1].plot(lim2, lim2, 'k--', lw=0.8)
    ax[1, 1].set_title(f"QQ-plot vs Hansen skew-t (ν̂={st_r['nu']:.1f}, λ̂={st_r['lam']:.2f})")
    ax[1, 1].set_xlabel('Theoretical quantile')
    ax[1, 1].set_ylabel('Sample quantile')

    fig.suptitle('Fact 1a — Daily log-returns $r_t$', y=1.00, fontsize=12, fontweight='bold')
    _save(fig, '01a_distribution_returns')

    # --- Figure 1b: log RV ---
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    ax[0, 0].plot(df['date'], lrv, lw=0.7, color='#7c3aed')
    ax[0, 0].set_title(r'$\log RV_t$ over time')
    ax[0, 0].set_ylabel(r'$\log RV_t$')

    ax[0, 1].hist(lrv, bins=bins, density=True, alpha=0.45,
                   color='#a0aec0', edgecolor='white')
    xs = np.linspace(lrv.min(), lrv.max(), 400)
    ax[0, 1].plot(xs, norm.pdf(xs, np.mean(lrv), np.std(lrv, ddof=1)),
                   lw=1.5, color='#7c3aed', label='Normal')
    ax[0, 1].set_title('Histogram + Normal density')
    ax[0, 1].set_xlabel(r'$\log RV_t$')
    ax[0, 1].legend()

    th, sa = qq_points(lrv, dist='norm')
    ax[1, 0].scatter(th, sa, s=10, color='#7c3aed', alpha=0.6)
    lim = [min(th.min(), sa.min()), max(th.max(), sa.max())]
    ax[1, 0].plot(lim, lim, 'k--', lw=0.8)
    ax[1, 0].set_title('QQ-plot vs Gaussian')

    th2, sa2 = qq_points(lrv, dist='skewt', params=st_lrv)
    ax[1, 1].scatter(th2, sa2, s=10, color='#d97706', alpha=0.6)
    lim2 = [min(th2.min(), sa2.min()), max(th2.max(), sa2.max())]
    ax[1, 1].plot(lim2, lim2, 'k--', lw=0.8)
    ax[1, 1].set_title(f"QQ-plot vs skew-t (ν̂={st_lrv['nu']:.1f}, λ̂={st_lrv['lam']:.2f})")

    fig.suptitle(r'Fact 1b — Log realized variance $\log RV_t$', y=1.00, fontsize=12, fontweight='bold')
    _save(fig, '01b_distribution_logRV')

    # --- Figure 1c: Heavy tails ---
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    k_r, xi_r = hill_estimator(r, side='two-sided')
    ax[0].plot(k_r, xi_r, lw=1.2, color='#2b6cb0', label='returns |$r_t$|')
    ax[0].axhline(0.5, color='r', lw=0.8, linestyle='--', label='ξ=1/2 (var = ∞ above)')
    ax[0].set_title('Hill estimator of tail index ξ = 1/α')
    ax[0].set_xlabel('k (largest order statistics)')
    ax[0].set_ylabel('ξ̂(k)')
    ax[0].legend()

    gpd_r = gpd_tail(r, quantile=0.95)
    gpd_lrv = gpd_tail(lrv - lrv.mean(), quantile=0.95)
    ax[1].text(0.05, 0.95, 'GPD fits above 95th percentile of |x|:\n', transform=ax[1].transAxes,
                fontsize=10, fontweight='bold', va='top')
    txt = (
        f"|r_t|:\n  ξ̂ = {gpd_r['xi']:+.3f}   (ξ > 0 ⇒ heavy tail)\n"
        f"  β̂ = {gpd_r['beta']:.6f}\n  n_exc = {gpd_r['n_exc']}\n\n"
        f"|log RV - mean|:\n  ξ̂ = {gpd_lrv['xi']:+.3f}\n"
        f"  β̂ = {gpd_lrv['beta']:.3f}\n  n_exc = {gpd_lrv['n_exc']}"
    )
    ax[1].text(0.05, 0.85, txt, transform=ax[1].transAxes, fontsize=10, family='monospace', va='top')
    ax[1].set_axis_off()
    fig.suptitle('Fact 1c — Tail behavior', y=1.02, fontsize=12, fontweight='bold')
    _save(fig, '01c_tails')

    # Save Fact 1 table
    pd.DataFrame({
        'r_d': mom_r, 'log_RV': mom_lrv,
    }).to_csv(TBL_DIR / "01_moments.csv")

    # =================================================================
    # FACT 2 — Volatility clustering / autocorrelation
    # =================================================================
    print("\n[3/5] FACT 2: Volatility clustering")
    nlags = 30

    series = {
        r'$r_t$': r,
        r'$|r_t|$': np.abs(r),
        r'$r_t^2$': r ** 2,
        r'$\log RV_t$': lrv,
    }
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax_, (label, x) in zip(axes.ravel(), series.items()):
        ac, ci = acf_with_ci(x, nlags=nlags)
        lags = np.arange(len(ac))
        cw = _ci_bands(ci)
        ax_.bar(lags, ac, width=0.6,
                 color=['#2b6cb0' if i > 0 else '#9ca3af' for i in lags])
        ax_.fill_between([0, nlags], -cw, cw, color='red', alpha=0.10)
        ax_.axhline(0, color='k', lw=0.5)
        ax_.set_title(f"ACF of {label}")
        ax_.set_xlim(-0.5, nlags + 0.5)
        ax_.set_xlabel('lag (days)')
    fig.suptitle('Fact 2 — Autocorrelation structure', y=1.00, fontsize=12, fontweight='bold')
    _save(fig, '02_acf_panel')

    # Ljung-Box and ARCH-LM tables
    rows = []
    for lab, x in [('r_d', r), ('|r_d|', np.abs(r)), ('r_d^2', r ** 2),
                    ('log_RV', lrv)]:
        for lg in (5, 10, 22):
            lb = ljung_box(x, lags=[lg])
            rows.append({
                'series': lab, 'test': 'Ljung-Box', 'lag': lg,
                'stat': float(lb['lb_stat'].iloc[0]),
                'pvalue': float(lb['lb_pvalue'].iloc[0]),
            })
    for lg in (5, 10, 22):
        lm = arch_lm(r, nlags=lg)
        rows.append({
            'series': 'r_d', 'test': 'ARCH-LM', 'lag': lg,
            'stat': lm['lm_stat'], 'pvalue': lm['lm_pvalue'],
        })
    fact2_tbl = pd.DataFrame(rows)
    fact2_tbl.to_csv(TBL_DIR / "02_clustering_tests.csv", index=False)
    print(fact2_tbl.to_string(index=False, float_format=lambda x: f'{x:.4g}'))

    # =================================================================
    # FACT 3 — Long memory
    # =================================================================
    print("\n[4/5] FACT 3: Long memory")
    rs_r = hurst_rs(np.abs(r))
    rs_lrv = hurst_rs(lrv)
    gph_r = gph(np.abs(r))
    gph_lrv = gph(lrv)

    print(f"  R/S Hurst on |r_d|   : H = {rs_r['H']:.3f} (SE {rs_r['se']:.3f})")
    print(f"  R/S Hurst on log RV  : H = {rs_lrv['H']:.3f} (SE {rs_lrv['se']:.3f})")
    print(f"  GPH d  on |r_d|      : d = {gph_r['d']:.3f} (SE {gph_r['se']:.3f}, "
          f"t = {gph_r['t_stat']:.2f}, p = {gph_r['pvalue_two_sided']:.3e})")
    print(f"  GPH d  on log RV     : d = {gph_lrv['d']:.3f} (SE {gph_lrv['se']:.3f}, "
          f"t = {gph_lrv['t_stat']:.2f}, p = {gph_lrv['pvalue_two_sided']:.3e})")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].scatter(rs_lrv['log_n'], rs_lrv['log_rs'], s=22, color='#7c3aed',
                   alpha=0.8, label='log RV')
    ax[0].plot(rs_lrv['log_n'],
                rs_lrv['fit_intercept'] + rs_lrv['H'] * rs_lrv['log_n'],
                color='#7c3aed', lw=1.2,
                label=f"H = {rs_lrv['H']:.3f}")
    ax[0].scatter(rs_r['log_n'], rs_r['log_rs'], s=22, color='#2b6cb0',
                   alpha=0.8, label='|r_d|')
    ax[0].plot(rs_r['log_n'],
                rs_r['fit_intercept'] + rs_r['H'] * rs_r['log_n'],
                color='#2b6cb0', lw=1.2,
                label=f"H = {rs_r['H']:.3f}")
    ax[0].plot(rs_lrv['log_n'], 0.5 * rs_lrv['log_n']
                + (rs_lrv['log_rs'].mean() - 0.5 * rs_lrv['log_n'].mean()),
                'k--', lw=0.8, label='H = 1/2 (no memory)')
    ax[0].set_xlabel(r'$\log n$ (chunk size)')
    ax[0].set_ylabel(r'$\log E[R/S]$')
    ax[0].set_title('R/S analysis — log-log scaling')
    ax[0].legend(fontsize=9)

    ax[1].text(0.05, 0.95,
                "Long-memory diagnostics\n" + "─" * 30 + "\n",
                transform=ax[1].transAxes, va='top',
                fontsize=10, fontweight='bold', family='monospace')
    txt = (
        f"R/S Hurst    H            SE     H>0.5?\n"
        f"  |r_d|     {rs_r['H']:+.3f}   {rs_r['se']:.3f}    {'YES' if rs_r['H']>0.5 else ' no'}\n"
        f"  log RV    {rs_lrv['H']:+.3f}   {rs_lrv['se']:.3f}    {'YES' if rs_lrv['H']>0.5 else ' no'}\n\n"
        f"GPH d        d̂           SE      p-value\n"
        f"  |r_d|     {gph_r['d']:+.3f}   {gph_r['se']:.3f}   {gph_r['pvalue_two_sided']:.3g}\n"
        f"  log RV    {gph_lrv['d']:+.3f}   {gph_lrv['se']:.3f}   {gph_lrv['pvalue_two_sided']:.3g}\n\n"
        f"Equity-vol benchmarks (S&P 500):\n"
        f"  H ≈ 0.75-0.85,  d ≈ 0.40-0.45"
    )
    ax[1].text(0.05, 0.85, txt, transform=ax[1].transAxes,
                family='monospace', fontsize=9, va='top')
    ax[1].set_axis_off()
    fig.suptitle('Fact 3 — Long memory', y=1.02, fontsize=12, fontweight='bold')
    _save(fig, '03_long_memory')

    # Tables
    pd.DataFrame({
        'series': ['|r_d|', 'log_RV'],
        'H_RS':   [rs_r['H'], rs_lrv['H']],
        'H_RS_se': [rs_r['se'], rs_lrv['se']],
        'd_GPH':  [gph_r['d'], gph_lrv['d']],
        'd_GPH_se': [gph_r['se'], gph_lrv['se']],
        'd_GPH_p': [gph_r['pvalue_two_sided'], gph_lrv['pvalue_two_sided']],
    }).to_csv(TBL_DIR / "03_long_memory.csv", index=False)

    # =================================================================
    # Summary markdown
    # =================================================================
    print("\n[5/5] Writing summary report")
    md = []
    md.append("# Stylized facts (1–3) — BTC-EUR\n")
    md.append(f"Sample: {df['date'].min().date()} → {df['date'].max().date()} "
               f"({len(df)} days, 15-min RV measures)\n")
    md.append("## Fact 1 — Distribution\n")
    md.append(f"- Daily returns: mean={mom_r['mean']:+.5f}, std={mom_r['std']:.4f}, "
               f"skew={mom_r['skew']:+.3f}, **excess kurt = {mom_r['excess_kurt']:+.3f}**, "
               f"JB p={mom_r['jb_pvalue']:.2e}")
    md.append(f"- log RV    : mean={mom_lrv['mean']:+.3f}, std={mom_lrv['std']:.3f}, "
               f"skew={mom_lrv['skew']:+.3f}, **excess kurt = {mom_lrv['excess_kurt']:+.3f}**, "
               f"JB p={mom_lrv['jb_pvalue']:.2e}")
    md.append(f"- Hansen skew-t fit on returns: **ν̂ = {st_r['nu']:.2f}**, "
               f"**λ̂ = {st_r['lam']:+.3f}**, log-lik = {st_r['loglik']:.2f}")
    md.append(f"- Hansen skew-t fit on log RV : ν̂ = {st_lrv['nu']:.2f}, λ̂ = {st_lrv['lam']:+.3f}")
    md.append(f"- GPD shape ξ̂ (95-pct excess of |r_d|): **{gpd_r['xi']:+.3f}**")
    md.append("")
    md.append("## Fact 2 — Volatility clustering\n")
    md.append("Ljung-Box and ARCH-LM (selected lags):\n")
    md.append(fact2_tbl.to_markdown(index=False, floatfmt=".3g"))
    md.append("")
    md.append("## Fact 3 — Long memory\n")
    md.append(f"- R/S Hurst on |r_d| : H = {rs_r['H']:+.3f} (SE {rs_r['se']:.3f})")
    md.append(f"- R/S Hurst on log RV: H = {rs_lrv['H']:+.3f} (SE {rs_lrv['se']:.3f})")
    md.append(f"- GPH d on |r_d|     : d = {gph_r['d']:+.3f} (SE {gph_r['se']:.3f}, p={gph_r['pvalue_two_sided']:.3g})")
    md.append(f"- GPH d on log RV    : d = {gph_lrv['d']:+.3f} (SE {gph_lrv['se']:.3f}, p={gph_lrv['pvalue_two_sided']:.3g})")
    md.append("")
    md.append("Figures saved in `outputs/figures/`, tables in `outputs/tables/`.")
    SUMMARY.write_text("\n".join(md))
    print(f"  → {SUMMARY.relative_to(PROJECT)}")
    print("\nDone.\n")


if __name__ == '__main__':
    main()
