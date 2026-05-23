"""
continuum_convergence_test_v2.py — simpler: fix all action parameters and vary N_E.

We're at Phase 7 conventions (m=0.5, K_E=1.0, K_M=0.5) which gave 0.074 at N_E=5.
Just vary N_E in {5, 7, 9, 11, 13} (odd, so T̂_F^{N-1} = even power = positive-def
per M-K).  See if pipeline result changes with N_E (= changes β as well).

Different a_τ scaling than fully-continuum, but useful to see whether lattice
corrections at this regime are large or small.

Reference: physical_sector_reference(β=N_E-1).
"""
from __future__ import annotations
import sys
import time
import numpy as np

sys.path.insert(0, '/home/hlamm/epoq_continuum_test/m1_toy')

from action_z2_staggered import LatticeGeometry, Z2GaugeConfig, build_dirac_matrix
from action_z2_metropolis import run_metropolis
from transfer_matrix_kbc import compute_combined_weight


def gauge_bits_at_slice(U, t_slice):
    def bit(s): return 0 if s == 1 else 1
    b0 = bit(U.U_y[t_slice, 0, 0])
    b1 = bit(U.U_y[t_slice, 1, 0])
    b2 = bit(U.U_x[t_slice, 0, 0])
    b3 = bit(U.U_x[t_slice, 0, 1])
    return b0 | (b1 << 1) | (b2 << 2) | (b3 << 3)


def compute_n0_at_t0(geom, U):
    g_top = gauge_bits_at_slice(U, 0)
    g_bot = gauge_bits_at_slice(U, geom.N_E - 1)
    if g_top != g_bot:
        return 0.0, 0.0
    W, info = compute_combined_weight(geom, U)
    V3 = geom.V_3
    from itertools import product
    num, den = 0.0, 0.0
    for i, n in enumerate(product((0, 1), repeat=V3)):
        psi = sum(int(nb) << k for k, nb in enumerate(n))
        w = W[i, i].real
        n_0 = 1.0 if (psi & 1) else 0.0
        num += w * n_0
        den += w
    return num, den


def run_at_N_E(N_E, n_gauge=500, n_warmup=2000, seed=2026,
               m=0.5, K_E=1.0, K_M=0.5):
    print(f"  N_E={N_E}, m={m}, K_E={K_E}, K_M={K_M}, β_eff={N_E-1}")
    geom = LatticeGeometry(Lx=2, Ly=2, N_E=N_E, m=m)
    cold = K_M >= 1.0
    t0 = time.time()
    mc = run_metropolis(
        geom, K_E=K_E, K_M=K_M,
        n_sweeps=n_gauge, n_warmup=n_warmup, seed=seed, cold_start=cold)
    print(f"  gauge MC: accept={mc.accept_rate:.3f}, ⟨sign⟩={mc.avg_sign:+.3f}, "
          f"configs={len(mc.configs)}, time={time.time()-t0:.1f}s")
    sum_sign_num = 0.0
    sum_sign_den = 0.0
    n_match = 0
    t1 = time.time()
    for U_idx, U in enumerate(mc.configs):
        sign_U = mc.sign_history[U_idx]
        num_U, den_U = compute_n0_at_t0(geom, U)
        if den_U != 0:
            n_match += 1
        sum_sign_num += sign_U * num_U
        sum_sign_den += sign_U * den_U
    if abs(sum_sign_den) > 1e-12:
        n0 = sum_sign_num / sum_sign_den
    else:
        n0 = float('nan')
    print(f"  n_diag={n_match}/{len(mc.configs)}, ⟨n̂_0⟩={n0:+.6f}, "
          f"W loop time={time.time()-t1:.1f}s")
    return {'N_E': N_E, 'n_configs': len(mc.configs), 'n_diag': n_match,
            'n0_estimate': n0, 'accept': mc.accept_rate, 'avg_sign': mc.avg_sign}


def main():
    N_E_list = ([int(x) for x in sys.argv[1].split(',')]
                if len(sys.argv) > 1 else [5, 7, 9, 11, 13])
    n_gauge = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    n_warmup = int(sys.argv[3]) if len(sys.argv) > 3 else 2000

    print("=" * 78)
    print("⟨n̂_0(t=0)⟩ vs N_E at fixed action couplings (m=0.5, K_E=1, K_M=0.5)")
    print("=" * 78)
    print(f"physical_sector_reference at β=N_E-1 used for comparison.")
    print()
    from action_endtoend_pipeline import physical_sector_reference
    results = []
    for N_E in N_E_list:
        print(f"\n--- N_E = {N_E} ---")
        r = run_at_N_E(N_E, n_gauge=n_gauge, n_warmup=n_warmup)
        ref = physical_sector_reference(beta=float(N_E - 1), times=[0.0])
        r['ref'] = ref[0.0]
        r['err'] = abs(r['n0_estimate'] - ref[0.0]) if not np.isnan(r['n0_estimate']) else float('nan')
        print(f"  physical_sector_reference(β={N_E-1}) = {ref[0.0]:+.6f}")
        results.append(r)

    print()
    print("=" * 78)
    print(f"{'N_E':>5} {'β':>5} {'accept':>8} {'⟨sign⟩':>8} {'n_diag':>8} "
          f"{'⟨n̂_0⟩':>10} {'reference':>12} {'|err|':>10}")
    for r in results:
        print(f"{r['N_E']:>5d} {r['N_E']-1:>5d} {r['accept']:>8.3f} {r['avg_sign']:>+8.3f} "
              f"{r['n_diag']:>8d} {r['n0_estimate']:>+10.5f} {r['ref']:>+12.5f} {r['err']:>10.5f}")


if __name__ == "__main__":
    main()
