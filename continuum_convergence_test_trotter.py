"""
continuum_convergence_test_trotter.py — Trotter T̂_F observable + choice of
action (central-diff legacy or forward-time block-structured).

Per-config observable: T̂_F = expm(-a_τ · H_KS[U_spatial(t)]) per slice
(Agent 1's transfer_matrix_kbc_trotterized — exact H_KS match per gauge
eigenstate).

MC weight: action's e^{-S_g} · |det M|.  action_type='forward' uses the
new block-upper-triangular M_forward (O(N_E · V_3³) det, QCD-scalable
linear-in-N_E scaling).  action_type='central' uses the legacy
action_z2_staggered.compute_det_M (O(V_4³)).

Optional plaq_flip_every: composite Z_2 plaquette-flip MC moves to reduce
autocorrelation in the gauge sector.

Speedups:
  - Vectorized W diagonal sum (numpy bit-mask precomputed once per V_3).
  - Block-structured det M (when action_type='forward').
  - Plaquette-flip composite moves.

Usage:
  python continuum_convergence_test_trotter.py <N_E_list> <n_gauge> <n_warmup>
                                                [action_type] [plaq_every]
  action_type defaults to 'forward'; plaq_every defaults to 5.
"""
from __future__ import annotations
import sys
import time
import numpy as np

sys.path.insert(0, '/home/hlamm/Desktop/QC/logdet/m1_toy')

from action_z2_staggered import LatticeGeometry, Z2GaugeConfig
from action_z2_metropolis import run_metropolis, run_metropolis_parallel
from transfer_matrix_kbc_trotterized import compute_combined_weight_trotter


def gauge_bits_at_slice(U, t_slice):
    """Encode spatial gauge config at slice t as 4-bit int (z2_setup ordering)."""
    def bit(s): return 0 if s == 1 else 1
    b0 = bit(U.U_y[t_slice, 0, 0])
    b1 = bit(U.U_y[t_slice, 1, 0])
    b2 = bit(U.U_x[t_slice, 0, 0])
    b3 = bit(U.U_x[t_slice, 0, 1])
    return b0 | (b1 << 1) | (b2 << 2) | (b3 << 3)


# Cache the n_0 bit-mask per V_3 (precomputed once, vectorizes the W-diag sum).
_N0_MASK_CACHE: dict[int, np.ndarray] = {}

def _n0_mask(V3: int) -> np.ndarray:
    if V3 not in _N0_MASK_CACHE:
        # bit at position (V_3 - 1) of the integer state index = site 0 occ
        dim = 1 << V3
        bit_pos = V3 - 1
        states = np.arange(dim)
        _N0_MASK_CACHE[V3] = ((states >> bit_pos) & 1).astype(np.float64)
    return _N0_MASK_CACHE[V3]


def compute_n0_at_t0_trotter(geom, U, a_tau=1.0):
    """Per-config ⟨n̂_0⟩ contributions via the Trotter T̂_F.

    EρOQ corner-state closure requires spatial gauge config to match at top
    and bottom OBC slices (gauge_top = gauge_bot).  When they differ, the
    corner-state contribution is zero.

    Trotter W diagonal:  W[Ψ, Ψ] = ⟨Ψ | T̂_F^{N_E-1} | Ψ⟩.
    Plaquette factors set to ZERO (K_E=K_M=0) — MC measure already
    weights configs by e^{-S_g}.  We only want the fermion-sector partition
    function from T̂_F here.

    Vectorized: precomputed n_0 bit-mask × np.real(np.diag(W)).

    Returns (num, den).
    """
    g_top = gauge_bits_at_slice(U, 0)
    g_bot = gauge_bits_at_slice(U, geom.N_E - 1)
    if g_top != g_bot:
        return 0.0, 0.0
    W, info = compute_combined_weight_trotter(
        geom, U, a_tau=a_tau, K_E=0.0, K_M=0.0, g_hop=0.5,
    )
    V3 = geom.V_3
    diag = np.real(np.diag(W))
    mask = _n0_mask(V3)
    num = float(np.sum(diag * mask))
    den = float(np.sum(diag))
    return num, den


def run_at_N_E(N_E, n_gauge=500, n_warmup=2000, seed=2026,
               m=0.5, K_E=1.0, K_M=0.5, a_tau=1.0,
               action_type: str = 'forward', plaq_flip_every: int = 5,
               n_chains: int = 1):
    print(f"  N_E={N_E}, m={m}, K_E={K_E}, K_M={K_M}, a_τ={a_tau}, β_eff={(N_E-1)*a_tau}")
    print(f"  action_type={action_type}, plaq_flip_every={plaq_flip_every}, n_chains={n_chains}")
    geom = LatticeGeometry(Lx=2, Ly=2, N_E=N_E, m=m)
    cold = K_M >= 1.0
    t0 = time.time()
    if n_chains > 1:
        mc = run_metropolis_parallel(
            geom, n_chains=n_chains, K_E=K_E, K_M=K_M,
            n_sweeps_per_chain=max(1, n_gauge // n_chains),
            n_warmup=n_warmup, seed=seed, cold_start=cold,
            action_type=action_type, plaq_flip_every=plaq_flip_every)
    else:
        mc = run_metropolis(
            geom, K_E=K_E, K_M=K_M,
            n_sweeps=n_gauge, n_warmup=n_warmup, seed=seed, cold_start=cold,
            action_type=action_type, plaq_flip_every=plaq_flip_every)
    print(f"  gauge MC: accept={mc.accept_rate:.3f}, ⟨sign⟩={mc.avg_sign:+.3f}, "
          f"configs={len(mc.configs)}, time={time.time()-t0:.1f}s")
    sum_sign_num = 0.0
    sum_sign_den = 0.0
    n_match = 0
    t1 = time.time()
    for U_idx, U in enumerate(mc.configs):
        sign_U = mc.sign_history[U_idx]
        num_U, den_U = compute_n0_at_t0_trotter(geom, U, a_tau=a_tau)
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
    action_type = sys.argv[4] if len(sys.argv) > 4 else 'forward'
    plaq_flip_every = int(sys.argv[5]) if len(sys.argv) > 5 else 5
    n_chains = int(sys.argv[6]) if len(sys.argv) > 6 else 1

    print("=" * 78)
    print("TROTTER observable + {} action MC".format(action_type.upper()))
    print("=" * 78)
    print(f"action couplings: m=0.5, K_E=1, K_M=0.5 (Phase 7 conventions)")
    print(f"Trotter T̂_F per slice = expm(-a_τ · H_KS[U_spatial(t)])")
    print(f"MC action_type={action_type}, plaq_flip_every={plaq_flip_every}")
    print(f"Reference: physical_sector_reference at β = (N_E-1)·a_τ")
    print()
    from action_endtoend_pipeline import physical_sector_reference
    results = []
    for N_E in N_E_list:
        print(f"\n--- N_E = {N_E} ---")
        r = run_at_N_E(N_E, n_gauge=n_gauge, n_warmup=n_warmup,
                       action_type=action_type, plaq_flip_every=plaq_flip_every,
                       n_chains=n_chains)
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
