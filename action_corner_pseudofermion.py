"""
M_act_4: Pseudofermion corner-state extraction.

Stochastic estimator of G^bdy = (M^{-1})[(τ=0); (τ=N_E−1)] via Gaussian
pseudofermion sources η:
    χ = M^{-1} η,   ⟨χ η^T⟩_η = M^{-1}
For each pseudofermion sample: pick a single source on the τ=N_E−1 boundary
slice; solve M χ = η; extract χ at τ=0 → one stochastic estimate of G^bdy
column.

Cross-validate against direct M_act_3 at toy scale.
"""
from __future__ import annotations
import numpy as np
import time
from dataclasses import dataclass

from action_z2_staggered import (
    LatticeGeometry, Z2GaugeConfig, build_dirac_matrix, compute_G_bdy,
)


def pseudofermion_G_bdy_estimate(
    geom: LatticeGeometry, U: Z2GaugeConfig,
    n_samples: int, rng: np.random.Generator,
) -> np.ndarray:
    """Stochastic G^bdy via N_pf pseudofermion samples.

    For each sample:
      - η: Gaussian random vector with support ONLY on τ=N_E−1 slice
      - Solve M χ = η
      - Extract χ at τ=0
      - Single-sample G_bdy[a, b] = χ_a · η_b
    Average over samples.

    Returns: V_3 × V_3 G^bdy estimate.
    """
    V3, V4, N_E = geom.V_3, geom.V_4, geom.N_E
    M = build_dirac_matrix(geom, U)
    # Indices of the two boundary slices
    bdy_top_idx = np.array([
        geom.site_idx(0, k // geom.Ly, k % geom.Ly) for k in range(V3)
    ])
    bdy_bot_idx = np.array([
        geom.site_idx(N_E - 1, k // geom.Ly, k % geom.Ly) for k in range(V3)
    ])

    G_acc = np.zeros((V3, V3), dtype=np.float64)
    for _ in range(n_samples):
        # Real Gaussian source at the bottom boundary
        eta = np.zeros(V4, dtype=np.float64)
        eta_bdy = rng.standard_normal(V3)
        eta[bdy_bot_idx] = eta_bdy
        # Solve M χ = η
        chi = np.linalg.solve(M, eta)
        chi_bdy = chi[bdy_top_idx]
        # Single-sample estimate: G_bdy[a, b] ≈ χ_a · η_b
        G_acc += np.outer(chi_bdy, eta_bdy)
    return G_acc / n_samples


def cross_validate_against_direct(
    geom: LatticeGeometry, U: Z2GaugeConfig,
    N_samples_list: list, seed: int = 2026,
):
    """Compare PF stochastic G^bdy estimator against direct."""
    G_direct = np.real_if_close(compute_G_bdy(geom, U))
    if np.iscomplexobj(G_direct):
        G_direct = G_direct.real

    print(f"|G^bdy direct| range: [{np.abs(G_direct).min():.4f}, "
          f"{np.abs(G_direct).max():.4f}]")
    print(f"||G^bdy direct||_F = {np.linalg.norm(G_direct):.4f}")
    print()
    print(f"  {'N_pf':>8} | {'||G_pf - G_direct||_F':>22} | {'rel error':>12} | {'time (s)':>9}")
    print("  " + "-" * 65)
    rng = np.random.default_rng(seed)
    for N_pf in N_samples_list:
        t0 = time.time()
        G_pf = pseudofermion_G_bdy_estimate(geom, U, N_pf, rng)
        elapsed = time.time() - t0
        err = np.linalg.norm(G_pf - G_direct)
        rel = err / np.linalg.norm(G_direct)
        print(f"  {N_pf:>8} | {err:>22.4e} | {rel:>12.4e} | {elapsed:>9.2f}")


def _self_test():
    geom = LatticeGeometry(Lx=2, Ly=2, N_E=4, m=0.5)
    rng = np.random.default_rng(42)
    U = Z2GaugeConfig.random(geom, rng)

    print(f"Pseudofermion G^bdy estimator on random Z₂ config")
    print(f"Geometry: 2×2 spatial × {geom.N_E} temporal, m = {geom.m}")
    print()
    cross_validate_against_direct(geom, U, [100, 1000, 10000, 100000])


if __name__ == '__main__':
    _self_test()
