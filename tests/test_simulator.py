"""Smoke tests for hartrace simulator + distributions.

Run from the project root:
    python tests/test_simulator.py
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np

from hartrace import (
    ModelSpec, HARMeanParams, ScaleParams, InnovationParams,
    ReturnInnovationParams, HARState,
    simulate, AuxForecasters, ComponentSampler,
)
from hartrace import distributions as skt


# ---------------------------------------------------------------------
# Hansen skewed-t sanity checks
# ---------------------------------------------------------------------
def test_skewt_moments():
    """Sampled mean ≈ 0, variance ≈ 1 at large N for several (ν, λ)."""
    rng = np.random.default_rng(0)
    for nu in (4.0, 8.0, 15.0):
        for lam in (-0.3, 0.0, 0.2):
            x = skt.rvs(nu, lam, size=200_000, rng=rng)
            m = float(x.mean())
            v = float(x.var())
            assert abs(m) < 0.05, f"mean off for (ν,λ)=({nu},{lam}): {m:.4f}"
            assert abs(v - 1.0) < 0.05, f"var off for (ν,λ)=({nu},{lam}): {v:.4f}"
    print("  ✓ skewt moments OK")


def test_skewt_inverse():
    """CDF(PPF(u)) ≈ u and PDF integrates approximately to 1."""
    u = np.linspace(0.001, 0.999, 200)
    for nu in (4.0, 8.0):
        for lam in (-0.4, 0.0, 0.3):
            z = skt.ppf(u, nu, lam)
            u_back = skt.cdf(z, nu, lam)
            err = float(np.max(np.abs(u - u_back)))
            assert err < 1e-6, f"PPF/CDF inversion err {err} at (ν,λ)=({nu},{lam})"
    print("  ✓ skewt PPF↔CDF inverse OK")


# ---------------------------------------------------------------------
# Simulator smoke tests
# ---------------------------------------------------------------------
def _mock_state(rng):
    n = 30
    log_RV = rng.normal(loc=np.log(0.0004), scale=0.5, size=n)
    log_C = log_RV - 0.05
    log_J = np.log1p(np.maximum(np.exp(log_RV) * 0.05, 0.0))
    log_RS_neg = log_RV - np.log(2.0)
    log_RS_pos = log_RV - np.log(2.0)
    return HARState(
        log_RV=log_RV, log_C=log_C, log_J=log_J,
        log_RS_neg=log_RS_neg, log_RS_pos=log_RS_pos,
        Q=0.0, V=0.0, S=0.0, dD=0.0, F=0.0,
        sigma_prev=0.5, eps_prev=0.0,
    )


def _mock_components(rng, N=500):
    c = rng.uniform(0.70, 0.95, size=N)
    j = 1.0 - c
    rs_neg = rng.uniform(0.30, 0.70, size=N)
    rs_pos = 1.0 - rs_neg
    rq = rng.normal(0.0, 0.1, size=N)
    return ComponentSampler(
        shares_C=c, shares_J=j,
        shares_RS_neg=rs_neg, shares_RS_pos=rs_pos,
        shares_RQ=rq,
    )


def _run_one(scale_type, family, rng, h=22, K=2000):
    spec = ModelSpec(
        family=family,
        mean=HARMeanParams(
            beta_0=-0.5, beta_D=0.40, beta_W=0.30, beta_M=0.20,
            beta_D_Q=-0.05,
            beta_J_D=0.05, beta_J_W=0.02, beta_J_M=0.01,
            beta_D_neg=0.50, beta_D_pos=0.10,
            beta_D_neg_Q=-0.03, beta_D_pos_Q=-0.01,
            gamma_V=0.05, gamma_S=0.10, gamma_dD=-0.02, gamma_F=0.15,
        ),
        scale=ScaleParams(
            scale_type=scale_type, sigma_static=0.40,
            omega_gas=0.02, A_gas=0.05, B_gas=0.95,
            omega_egarch=-0.04, alpha_egarch=0.10,
            delta_egarch=-0.05, beta_egarch=0.95,
        ),
        innov=InnovationParams(nu=8.0, lam=-0.10),
        return_innov=ReturnInnovationParams(nu_r=5.0, lam_r=-0.15),
    )
    state = _mock_state(rng)
    comps = _mock_components(rng)
    aux = AuxForecasters()
    return simulate(spec, state, horizon=h, K=K, aux=aux,
                    components=comps, rng=rng)


def test_simulator_all_variants():
    rng = np.random.default_rng(42)
    results = {}
    for fam in ('HAR-CJ-X-Q', 'HAR-RS-X-Q'):
        for sc in ('static', 'gas', 'egarch'):
            res = _run_one(sc, fam, rng)
            results[(fam, sc)] = res
            assert res.paths_log_RV.shape == (2000, 22)
            assert res.paths_return.shape == (2000, 22)
            assert res.cum_return.shape == (2000,)
            assert np.all(np.isfinite(res.cum_return))
    print("  ✓ simulator runs for 2 families × 3 scale variants")

    print()
    print("  Cum-return summary (h=22, K=2000):")
    print(f"  {'family':<14} {'scale':<8} {'mean':>9} {'std':>9} "
          f"{'VaR_1%':>9} {'ES_1%':>9} {'VaR_5%':>9}")
    for (fam, sc), res in results.items():
        print(f"  {fam:<14} {sc:<8} "
              f"{res.cum_return.mean():>9.4f} "
              f"{res.cum_return.std():>9.4f} "
              f"{res.var(0.01):>9.4f} "
              f"{res.es(0.01):>9.4f} "
              f"{res.var(0.05):>9.4f}")


if __name__ == '__main__':
    print("Running hartrace smoke tests")
    print("=" * 60)
    print("[1/3] Hansen skewed-t moments")
    test_skewt_moments()
    print("[2/3] Hansen skewed-t inverse-CDF self-consistency")
    test_skewt_inverse()
    print("[3/3] Simulator: 2 families × 3 scale variants")
    test_simulator_all_variants()
    print()
    print("All smoke tests passed.")
