"""Extended stylized facts on v2 realized measures (n≈2 300 days, 5-min RK).

Builds on the v1 analysis (scripts/02) with three additions:

  - Stylized fact 4: roughness diagnostics — Hurst on log-vol increments
                     (Takaishi 2020); compare with Hurst on levels.
  - Stylized fact 5: inverse leverage test — OLS of log_RV_{t+1} on lagged
                     return + signed/absolute interactions. Bouri et al.
                     report positive coefficient larger than negative for BTC.
  - Stylized fact 6: periodicity — F-test for day-of-week effect in log RV
                     plus weekend dummy (HAC SE).

Also redoes facts 1-3 quickly on the larger sample so we can see how the
key results change with n=2 300 vs n=316.
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
    moments, fit_student_t, fit_hansen_skewt,
    acf_with_ci, ljung_box, arch_lm,
    hurst_rs, gph,
    hurst_of_increments, inverse_leverage_test,
    periodicity_tests, gpd_tail,
)


PROJECT = Path(__file__).parent.parent
INPUT = PROJECT / 'data' / 'processed' / 'realized_measures_btc_eur_v2.parquet'
FIG_DIR = PROJECT / 'outputs' / 'figures'
TBL_DIR = PROJECT / 'outputs' / 'tables'
SUMMARY = PROJECT / 'outputs' / 'C1_stylized_facts_v2_summary.md'

plt.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 130,
                      'axes.spines.top': False, 'axes.spines.right': False,
                      'font.size': 10})


def main():
    print("=" * 70)
    print("C1_stylized_facts_v2.py — extended stylized facts on n≈2300")
    print("=" * 70)

    print("\n[1/5] Loading v2 measures")
    df = pd.read_parquet(INPUT)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df['log_RV'] = np.log(df['RV'])
    df['log_RK'] = np.log(df['RK'])
    print(f"  N = {len(df)}, {df['date'].min().date()} → {df['date'].max().date()}")

    r = df['r_d'].values
    lrv = df['log_RV'].values
    lrk = df['log_RK'].values
    md = []
    md.append("# Stylized facts v2 — BTC-EUR, 5-min RK, n≈2300\n")
    md.append(f"Sample: {df['date'].min().date()} → {df['date'].max().date()} "
               f"({len(df)} days)\n")

    # ==================================================================
    # FACT 1 - Distribution (redo on v2)
    # ==================================================================
    print("\n[2/5] Fact 1 (redone): distributions on n=2,334")
    mom_r = moments(r)
    mom_lrv = moments(lrv)
    mom_lrk = moments(lrk)
    st_r = fit_hansen_skewt(r)
    st_lrk = fit_hansen_skewt(lrk)
    print(f"  r_d : skew={mom_r['skew']:+.3f}, exkurt={mom_r['excess_kurt']:+.3f}, "
          f"JB-p={mom_r['jb_pvalue']:.2e}; skew-t ν̂={st_r['nu']:.2f}, λ̂={st_r['lam']:+.3f}")
    print(f"  log RK: skew={mom_lrk['skew']:+.3f}, exkurt={mom_lrk['excess_kurt']:+.3f}, "
          f"JB-p={mom_lrk['jb_pvalue']:.2e}; skew-t ν̂={st_lrk['nu']:.2f}, λ̂={st_lrk['lam']:+.3f}")
    md.append("## Fact 1 — Distribution (v2)\n")
    md.append(f"- daily returns r_d: skew={mom_r['skew']:+.3f}, "
               f"excess kurt={mom_r['excess_kurt']:+.3f}, "
               f"skew-t ν̂={st_r['nu']:.2f}, λ̂={st_r['lam']:+.3f}")
    md.append(f"- log RK         : skew={mom_lrk['skew']:+.3f}, "
               f"excess kurt={mom_lrk['excess_kurt']:+.3f}, "
               f"skew-t ν̂={st_lrk['nu']:.2f}, λ̂={st_lrk['lam']:+.3f}")
    md.append(f"- log RV (simple): skew={mom_lrv['skew']:+.3f}, "
               f"excess kurt={mom_lrv['excess_kurt']:+.3f}")

    gpd_r = gpd_tail(r, quantile=0.95)
    md.append(f"- GPD ξ̂ (|r| > q95): {gpd_r['xi']:+.3f} "
               f"(positive ⇒ heavy-tailed regime)")

    # ==================================================================
    # FACT 2 - Clustering (redo on v2)
    # ==================================================================
    print("\n[3/5] Fact 2: clustering (LB / ARCH-LM)")
    md.append("\n## Fact 2 — Clustering (v2)\n")
    rows = []
    for lab, x in [('r_d', r), ('|r_d|', np.abs(r)),
                     ('r_d^2', r ** 2), ('log_RK', lrk)]:
        for lg in (5, 10, 22):
            lb = ljung_box(x, lags=[lg])
            rows.append({'series': lab, 'test': 'LB', 'lag': lg,
                         'p': float(lb['lb_pvalue'].iloc[0])})
    for lg in (5, 10, 22):
        lm = arch_lm(r, nlags=lg)
        rows.append({'series': 'r_d', 'test': 'ARCH-LM', 'lag': lg,
                     'p': lm['lm_pvalue']})
    tbl2 = pd.DataFrame(rows)
    pivot2 = tbl2.pivot_table(index=['series', 'test'], columns='lag', values='p')
    print(pivot2.round(4).to_string())
    md.append(pivot2.round(4).to_markdown(floatfmt='.4g'))

    # ==================================================================
    # FACT 3 - Long memory (redo + new roughness)
    # ==================================================================
    print("\n[4/5] Fact 3: long memory + roughness")
    rs_r = hurst_rs(np.abs(r))
    rs_lrk = hurst_rs(lrk)
    gph_r = gph(np.abs(r))
    gph_lrk = gph(lrk)
    rough = hurst_of_increments(lrk, q_range=[0.5, 1.0, 2.0])
    print(f"  R/S H |r_d|       : {rs_r['H']:+.3f}  (SE {rs_r['se']:.3f})")
    print(f"  R/S H log RK      : {rs_lrk['H']:+.3f}  (SE {rs_lrk['se']:.3f})")
    print(f"  GPH d  log RK     : {gph_lrk['d']:+.3f}  (p={gph_lrk['pvalue_two_sided']:.3e})")
    print(f"  Hurst log-RK INCR : H(q=0.5)={rough['H_q0.5']:.3f}, "
          f"H(q=1.0)={rough['H_q1.0']:.3f}, H(q=2.0)={rough['H_q2.0']:.3f}")
    md.append("\n## Fact 3 — Long memory + roughness\n")
    md.append(f"- R/S Hurst H(|r_d|)  = {rs_r['H']:+.3f} (SE {rs_r['se']:.3f})")
    md.append(f"- R/S Hurst H(log RK) = {rs_lrk['H']:+.3f} (SE {rs_lrk['se']:.3f})")
    md.append(f"- GPH d (log RK)      = {gph_lrk['d']:+.3f} (SE {gph_lrk['se']:.3f}, "
               f"p={gph_lrk['pvalue_two_sided']:.3e})")
    md.append(f"- Hurst log-RK *increments* (Takaishi): "
               f"H(q=0.5)={rough['H_q0.5']:.3f}, "
               f"H(q=1.0)={rough['H_q1.0']:.3f}, "
               f"H(q=2.0)={rough['H_q2.0']:.3f}")
    md.append(f"  (H_incr < 0.5 ⇒ rough volatility, à la Gatheral-Jaisson-Rosenbaum 2018)")

    # ==================================================================
    # FACT 4 - Jump dynamics
    # ==================================================================
    print("\n[5a/5] Fact 4: jump dynamics over time")
    j_share = df['J'] / df['RV']
    md.append("\n## Fact 4 — Jump dynamics\n")
    md.append(f"- jump rate (BNS α=0.001): {df['J_flag'].mean()*100:.2f}%")
    md.append(f"- on jump-days, J/RV: mean = {j_share[df['J_flag']==1].mean():.3f}")
    # Jump clustering: AR(1) on J_flag indicator
    flags = df['J_flag'].values
    if len(flags) > 22:
        rho1 = float(pd.Series(flags).autocorr(lag=1))
        md.append(f"- AR(1) on jump indicator: ρ̂_1 = {rho1:+.3f}  "
                   f"(weak clustering if non-zero)")

    # ==================================================================
    # FACT 5 - Inverse leverage
    # ==================================================================
    print("[5b/5] Fact 5: inverse leverage test")
    lev = inverse_leverage_test(r, lrk)
    md.append("\n## Fact 5 — Inverse-leverage test (Bouri-style)\n")
    md.append("Regression: log RK_{t+1} on log RK_t, r_t, 1{r_t<0}·|r_t|, 1{r_t>0}·|r_t|. HAC SE.\n")
    rows = []
    for nm, b, s, p in zip(lev['names'], lev['beta'], lev['se'], lev['p_value']):
        rows.append({'param': nm, 'beta': b, 'se_hac': s, 'p': p})
    md.append(pd.DataFrame(rows).to_markdown(index=False, floatfmt='.4f'))
    print("  inverse-leverage results:")
    for nm, b, p in zip(lev['names'], lev['beta'], lev['p_value']):
        flag = '***' if p < 0.01 else '**' if p < 0.05 else ''
        print(f"    {nm:<14} = {b:+8.4f}  p={p:.4f} {flag}")
    md.append(f"\n  R² = {lev['r_squared']:.4f}, n = {lev['n']}")
    # Inverse-leverage interpretation
    pos = lev['beta'][4]
    neg = lev['beta'][3]
    if pos > neg:
        md.append(f"\n  **Inverse leverage detected**: β_pos ({pos:+.3f}) > β_neg ({neg:+.3f}). "
                   f"Positive returns predict next-day vol *more strongly* than negative — "
                   f"consistent with crypto-specific inverse-leverage (Bouri et al.).")
    else:
        md.append(f"\n  Standard leverage pattern: β_neg ({neg:+.3f}) > β_pos ({pos:+.3f}).")

    # ==================================================================
    # FACT 6 - Periodicity
    # ==================================================================
    print("[5c/5] Fact 6: periodicity (day-of-week, weekend)")
    perio = periodicity_tests(df['log_RK'], df['dow'])
    md.append("\n## Fact 6 — Periodicity\n")
    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    md.append("Day-of-week mean log-RK:")
    for i in range(7):
        if i in perio['dow_means']:
            md.append(f"  - {dow_names[i]}: {perio['dow_means'][i]:+.3f}")
    md.append(f"\n- F-test (all DOW means equal): F = {perio['F_dow']:.3f}, "
               f"p = {perio['p_F_dow']:.4f}")
    md.append(f"- Weekend dummy effect (HAC): β = {perio['weekend_diff']:+.4f}, "
               f"SE = {perio['weekend_se_hac']:.4f}, p = {perio['weekend_p_hac']:.4f}")
    print(f"  F-test DOW: F={perio['F_dow']:.2f}, p={perio['p_F_dow']:.4f}")
    print(f"  Weekend effect: β={perio['weekend_diff']:+.4f}, p={perio['weekend_p_hac']:.4f}")

    # ==================================================================
    # Figures: composite v2 panel
    # ==================================================================
    print("\nFigures …")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    axes[0,0].plot(df['date'], lrk, lw=0.5, color='#7c3aed')
    axes[0,0].set_title(r'$\log RK_t$ over time (v2)')
    axes[0,0].set_ylabel(r'$\log RK_t$')

    axes[0,1].plot(df['date'], np.sqrt(df['RK']*365)*100, lw=0.6, color='#dc2626')
    axes[0,1].set_title('Annualised vol (RK)')
    axes[0,1].set_ylabel('% annualised')

    axes[0,2].plot(df['date'], df['J_flag'].rolling(50, min_periods=10).mean(),
                    lw=0.8, color='#f59e0b')
    axes[0,2].set_title('Rolling 50-day jump frequency')
    axes[0,2].set_ylabel('fraction')
    axes[0,2].axhline(df['J_flag'].mean(), color='k', linestyle='--', lw=0.6,
                       label=f"avg {df['J_flag'].mean():.3f}")
    axes[0,2].legend()

    ac, ci = acf_with_ci(lrk, nlags=40)
    lags = np.arange(len(ac))
    cw = (ci[1,1] - ci[1,0]) / 2
    axes[1,0].bar(lags, ac, color='#2b6cb0', width=0.6)
    axes[1,0].fill_between([0, 40], -cw, cw, color='red', alpha=0.10)
    axes[1,0].axhline(0, color='k', lw=0.4)
    axes[1,0].set_title(r'ACF of $\log RK_t$ (40 lags)')

    # Day-of-week boxplot
    dow_labels = [dow_names[d] for d in sorted(df['dow'].unique())]
    df.boxplot(column='log_RK', by='dow', ax=axes[1,1], grid=False)
    axes[1,1].set_title('log RK by day-of-week')
    axes[1,1].set_xlabel('day-of-week (0=Mon)')
    axes[1,1].set_xticklabels(dow_labels)
    fig.suptitle('')

    # Inverse leverage: scatter r_t vs log RK_{t+1}
    axes[1,2].scatter(r[:-1], lrk[1:], s=4, alpha=0.25, color='#0891b2')
    axes[1,2].set_xlabel(r'$r_t$')
    axes[1,2].set_ylabel(r'$\log RK_{t+1}$')
    axes[1,2].set_title('Return → next-day log RK')

    fig.suptitle('Stylized facts v2 — BTC-EUR, 5-min RK, n=2 334',
                  y=1.00, fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'C1_stylized_facts_v2.png', bbox_inches='tight')
    plt.close(fig)
    print(f"  saved → {(FIG_DIR / 'C1_stylized_facts_v2.png').relative_to(PROJECT)}")

    # Markdown
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text("\n".join(md))
    print(f"  saved → {SUMMARY.relative_to(PROJECT)}")
    print("\nDone.\n")


if __name__ == '__main__':
    main()
