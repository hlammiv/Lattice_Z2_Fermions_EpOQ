"""
M_act_6 v2: end-to-end P2 pipeline (replaces action_endtoend_pipeline.py).

Identical to v1 EXCEPT Phase 2 uses the exact P2 W-matrix sampler from
`action_corner_direct_v2` (which combines Agent A's per-step T̂_F^single
with the Cauchy-Binet 70-term K_BC) instead of the G^bdy-based weights.

For sign in the ratio estimator: the v2 sampler returns float-valued
"signs" = Re(W) / |W| ∈ [-1, 1] (the real projection of the phase factor
when W has complex eigvals from the CB decomposition). The estimator
formulas are the same as v1; integer ↔ float substitution is transparent.

Reference targets (from `physical_sector_reference` at β=N_E=4, K=1):
  t = 0.00 → C = +0.045
  t = 0.50 → C = -0.003
  t = 1.00 → C = -0.039
  t = 2.00 → C = +0.039

Previous v1 result (with G^bdy):
  t = 0.00 → C = -0.014 ± 0.027  (sign-wrong, projector observable!)
  t = 0.50 → C = -0.010 ± 0.025
  t = 1.00 → C = -0.003 ± 0.020
  t = 2.00 → C = +0.005 ± 0.017

Expected v2 result: match reference within bootstrap error.
"""
from __future__ import annotations
import sys
import time
import numpy as np
from scipy.linalg import expm

sys.path.insert(0, '/home/hlamm/Desktop/QC/logdet/m1_toy')

from action_z2_staggered import LatticeGeometry, Z2GaugeConfig, build_dirac_matrix, gauge_action
from action_z2_metropolis import run_metropolis
from action_corner_direct_v2 import sample_corner_pairs_stratified
from action_minkowski_stitch import (
    matrix_element_minkowski, qc_basis_index, static_observable_diagonal,
)
from action_endtoend_pipeline import physical_sector_reference


def run_action_p2_pipeline(
    geom: LatticeGeometry, K: float, times: list[float],
    n_gauge: int = 100, n_per_sector: int = 5,
    n_warmup_gauge: int = 50, seed: int = 2026,
    n_trotter_real: int = 200,
):
    """Run the P2 (exact-W, stratified-sampled) end-to-end pipeline.

    Stratified sampling: n_per_sector pairs from each particle-number sector
    per gauge config (V_3+1 sectors).  Total samples per gauge config =
    n_per_sector × (V_3 + 1).  Sample weight is W-proportional within each
    sector, with sector_w = sector_total / n_per_sector for unbiased
    combination across sectors.
    """
    rng = np.random.default_rng(seed)
    n_per_config = n_per_sector * (geom.V_3 + 1)

    # Phase 1: classical action MC for gauge configs
    print(f"Phase 1: classical action MC (n_gauge={n_gauge}, "
          f"n_warmup={n_warmup_gauge}, K={K})...")
    t0 = time.time()
    mc_result = run_metropolis(
        geom, K=K, n_sweeps=n_gauge, n_warmup=n_warmup_gauge, seed=seed,
    )
    t_mc = time.time() - t0
    print(f"  wall-clock: {t_mc:.1f}s, accept rate: {mc_result.accept_rate:.3f}")

    # Phase 2: corner-state extraction via P2 W-matrix (EXACT), stratified
    print(f"Phase 2: stratified corner-state extraction "
          f"(n_per_sector={n_per_sector}, n_per_config={n_per_config})...")
    t0 = time.time()
    all_samples = []   # list of (U, psi_i, psi_j, sign_slater, sector_w)
    for U_idx, U in enumerate(mc_result.configs):
        if U_idx % 25 == 0:
            print(f"  gauge config {U_idx}/{len(mc_result.configs)}...", flush=True)
        pairs, signs, sector_w, _ = sample_corner_pairs_stratified(
            geom, U, n_per_sector=n_per_sector, rng=rng)
        for k in range(len(pairs)):
            all_samples.append((U, int(pairs[k, 0]), int(pairs[k, 1]),
                                float(signs[k]), float(sector_w[k])))
    t_corner = time.time() - t0
    print(f"  wall-clock: {t_corner:.1f}s, n_samples_total: {len(all_samples)}")

    # Phase 3: Minkowski matrix elements
    print(f"Phase 3: QC matrix elements (n_trotter={n_trotter_real})...")
    t0 = time.time()
    contribs = {t: [] for t in times}
    is_diag = np.zeros(len(all_samples), dtype=bool)
    sign_vals = np.zeros(len(all_samples))
    sector_w_vals = np.zeros(len(all_samples))
    for k, (U, psi_i, psi_j, sign_slater, sec_w) in enumerate(all_samples):
        if k % 200 == 0:
            print(f"  sample {k}/{len(all_samples)}...", flush=True)
        bits_top = qc_basis_index(geom, U, 0, psi_i)
        bits_bot = qc_basis_index(geom, U, geom.N_E - 1, psi_j)
        is_diag[k] = (bits_top == bits_bot)
        sign_vals[k] = sign_slater
        sector_w_vals[k] = sec_w
        n0_j = static_observable_diagonal(geom, U, psi_j)
        for t in times:
            mat_el = matrix_element_minkowski(
                geom, U, psi_i, psi_j, t=t, n_trotter=n_trotter_real)
            # contribution: W_eff · n_0 · mat_el  where W_eff = sign · sec_w
            contribs[t].append(sign_slater * sec_w * n0_j * mat_el.real)
    t_qc = time.time() - t0
    print(f"  wall-clock: {t_qc:.1f}s")
    print(f"  diagonal sample fraction: {is_diag.mean():.3f}")
    print(f"  negative-sign fraction:   {np.mean(sign_vals < 0):.3f}")

    # Phase 4: ratio estimator + bootstrap.  For stratified sampling, the
    # estimator is:  ⟨n_0(t)⟩ = (Σ samples sign·sec_w·n_0·mat_el) / (Σ samples sign·sec_w·δ_diag)
    print("Phase 4: ratio estimator + bootstrap...")
    out_C = {}
    out_err = {}
    denom_full = float(np.sum(sign_vals * sector_w_vals * is_diag))
    print(f"  denominator (Σ sign·sec_w·δ_diag) = {denom_full:+.6f}")
    for t in times:
        contribs_arr = np.array(contribs[t])
        if abs(denom_full) < 1e-12:
            print(f"  WARNING: zero denominator at t={t}")
            out_C[t], out_err[t] = float('nan'), float('nan')
            continue
        N = len(contribs_arr)
        rng_b = np.random.default_rng(31337)
        means = []
        for _ in range(200):
            idx = rng_b.choice(N, N, replace=True)
            num = contribs_arr[idx].sum()
            den = float(np.sum(sign_vals[idx] * sector_w_vals[idx] * is_diag[idx]))
            if abs(den) < 1e-12:
                continue
            means.append(num / den)
        out_C[t] = float(np.mean(means)) if means else float('nan')
        out_err[t] = float(np.std(means)) if means else float('nan')
    return {
        'C': out_C, 'err': out_err,
        'n_samples': len(all_samples), 'is_diag_fraction': float(is_diag.mean()),
        'walltime': {'mc': t_mc, 'corner': t_corner, 'qc': t_qc},
    }


def _self_test():
    geom = LatticeGeometry(Lx=2, Ly=2, N_E=4, m=0.5)
    K = 1.0
    times = [0.0, 0.5, 1.0, 2.0]

    print("Computing physical-sector reference (β = N_E = 4)...")
    ref = physical_sector_reference(beta=float(geom.N_E), times=times)
    print(f"Reference C(t):")
    for t in times:
        print(f"  t={t}: {ref[t]:+.6f}")
    print()

    print("=" * 60)
    print("P2 pipeline (exact W-matrix, no PF)")
    print("=" * 60)
    # stratified: 20 gauge × 5 per_sector × 5 sectors = 500 samples; ~5-7 min
    result = run_action_p2_pipeline(
        geom, K=K, times=times,
        n_gauge=20, n_per_sector=5,
        n_warmup_gauge=50, seed=2026,
        n_trotter_real=200,
    )
    print(f"\nResults:")
    print(f"  {'t':>5} | {'C (P2)':>22} | {'reference':>12} | z-score")
    for t in times:
        rf = ref[t]
        m, e = result['C'][t], result['err'][t]
        z = (m - rf) / e if (e and e > 0) else float('nan')
        print(f"  {t:>5.2f} | {m:>9.6f} ± {e:.4f}    | {rf:>12.6f} | {z:+.2f}")


if __name__ == '__main__':
    _self_test()
