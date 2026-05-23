"""
continuum_convergence_mk.py — proper continuum convergence test using
M-K T̂_F (gives thermal observables, not OBC boundary partition function).

Per gauge config U: build T̂_F^{N_E-1} = ∏ T_F_step_t via M-K, project to
phi=0 chi-Fock, compute ⟨n̂_0⟩ at t=0 (diagonal entries weighted by n_0).
Sign-corrected gauge average.

For variable N_E: matrix dimension stays 256x256 (V_3=4), but number of
T̂_F multiplications increases.

Usage: python continuum_convergence_mk.py <N_E_list> <n_gauge> <n_warmup>
"""
from __future__ import annotations
import sys
import time
import numpy as np

sys.path.insert(0, '/home/hlamm/epoq_continuum_test/m1_toy')

from action_z2_staggered import LatticeGeometry, Z2GaugeConfig
from action_z2_metropolis import run_metropolis
from transfer_matrix_mk import (
    build_fermion_ops, mk_T_F_single_step, project_to_phi_vacuum,
)


def n0_from_MK(geom, U, chi_ops, phi_ops):
    """⟨n̂_0⟩ for given gauge config via M-K T̂_F^{N_E-1}."""
    V3 = geom.V_3
    T = np.eye(1 << (2 * V3), dtype=complex)
    for t in range(geom.N_E - 1):
        T_step = mk_T_F_single_step((geom.Lx, geom.Ly), chi_ops, phi_ops,
                                    U.U_x, U.U_y, t, geom.m)
        T = T_step @ T
    T_phi0 = project_to_phi_vacuum(T, V3)
    dim_chi = 1 << V3
    Z, num = 0.0, 0.0
    for st in range(dim_chi):
        w = T_phi0[st, st].real
        Z += w
        num += w * ((st >> 0) & 1)
    return num, Z


def run_at_N_E(N_E, n_gauge=100, n_warmup=500, seed=2026,
               m=0.5, K_E=1.0, K_M=0.5):
    print(f"\n--- N_E = {N_E} ---", flush=True)
    print(f"  params: m={m}, K_E={K_E}, K_M={K_M}, β_eff={N_E-1}", flush=True)
    geom = LatticeGeometry(Lx=2, Ly=2, N_E=N_E, m=m)
    chi_ops, phi_ops = build_fermion_ops(geom.V_3)
    cold = K_M >= 1.0
    t0 = time.time()
    mc = run_metropolis(
        geom, K_E=K_E, K_M=K_M,
        n_sweeps=n_gauge, n_warmup=n_warmup, seed=seed, cold_start=cold)
    print(f"  gauge MC: accept={mc.accept_rate:.3f}, ⟨sign⟩={mc.avg_sign:+.3f}, "
          f"configs={len(mc.configs)}, time={time.time()-t0:.1f}s", flush=True)

    sign_num_sum = 0.0
    sign_den_sum = 0.0
    t1 = time.time()
    for U_idx, U in enumerate(mc.configs):
        num_U, den_U = n0_from_MK(geom, U, chi_ops, phi_ops)
        sign_U = mc.sign_history[U_idx]
        sign_num_sum += sign_U * num_U
        sign_den_sum += sign_U * den_U
        if U_idx % 20 == 0:
            elapsed = time.time() - t1
            print(f"    config {U_idx}: cum ⟨n̂_0⟩ = "
                  f"{sign_num_sum/sign_den_sum if abs(sign_den_sum)>1e-12 else 0:+.5f}, "
                  f"t={elapsed:.0f}s", flush=True)

    n0 = sign_num_sum / sign_den_sum if abs(sign_den_sum) > 1e-12 else float('nan')
    print(f"  ⟨n̂_0⟩ = {n0:+.6f}, total M-K time={time.time()-t1:.1f}s", flush=True)
    return {'N_E': N_E, 'beta_eff': N_E - 1, 'accept': mc.accept_rate,
            'avg_sign': mc.avg_sign, 'n0_estimate': n0, 'n_configs': len(mc.configs)}


def main():
    N_E_list = ([int(x) for x in sys.argv[1].split(',')]
                if len(sys.argv) > 1 else [5, 7, 9])
    n_gauge = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    n_warmup = int(sys.argv[3]) if len(sys.argv) > 3 else 500

    print("=" * 78)
    print("M-K Continuum convergence: ⟨n̂_0⟩ vs N_E at fixed couplings")
    print("=" * 78)
    print(f"params: m=0.5, K_E=1.0, K_M=0.5 (matching z2_setup g_E=1, g_M=0.5)")
    print()
    from action_endtoend_pipeline import physical_sector_reference
    results = []
    for N_E in N_E_list:
        r = run_at_N_E(N_E, n_gauge=n_gauge, n_warmup=n_warmup)
        ref = physical_sector_reference(beta=float(N_E - 1), times=[0.0])
        r['ref'] = ref[0.0]
        r['err'] = abs(r['n0_estimate'] - ref[0.0]) if not np.isnan(r['n0_estimate']) else float('nan')
        print(f"  physical_sector_reference(β={N_E-1}) = {ref[0.0]:+.6f}")
        print(f"  |err| = {r['err']:.5f}", flush=True)
        results.append(r)

    print()
    print("=" * 78)
    print("Summary")
    print("=" * 78)
    print(f"{'N_E':>5} {'β':>5} {'accept':>8} {'⟨sign⟩':>8} {'⟨n̂_0⟩':>10} "
          f"{'reference':>12} {'|err|':>10}")
    for r in results:
        print(f"{r['N_E']:>5d} {r['beta_eff']:>5d} {r['accept']:>8.3f} "
              f"{r['avg_sign']:>+8.3f} {r['n0_estimate']:>+10.5f} "
              f"{r['ref']:>+12.5f} {r['err']:>10.5f}")


if __name__ == "__main__":
    main()
