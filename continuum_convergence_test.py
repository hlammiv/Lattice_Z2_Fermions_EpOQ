"""
continuum_convergence_test.py — test if EρOQ pipeline ⟨n̂_0⟩ converges to
physical_sector_reference(β=4) as lattice spacing a_τ → 0.

Physical parameters (QC Hamiltonian, z2_setup convention):
    g_E = 1.0, g_M = 0.5, m_phys = 0.5, β = 4.0

Lattice scaling for varying a_τ:
    a_τ = β / (N_E - 1)        # so β = const = 4
    m_lat = m_phys · a_τ        # bare mass scales with a_τ
    K_E = 1 / (a_τ · g_E²) = 1/a_τ            # electric (temporal plaquette)
    K_M = a_τ / g_M² = 4 · a_τ                # magnetic (spatial plaquette)

Observable at t=0 (no QC needed — just diagonal classical Euclidean):
    ⟨n̂_0⟩_t=0 = Σ_configs sign(det M) · n̂_0(Ψ_diag) · w(Ψ_diag, U)
              / Σ_configs sign(det M) · w(Ψ_diag, U)
where w(Ψ_diag, U) = W[Ψ, Ψ] for Ψ with bit pattern matching gauge_top(U) =
gauge_bot(U).

Usage:  python continuum_convergence_test.py <N_E> <n_gauge> <n_warmup>
"""
from __future__ import annotations
import sys
import time
import numpy as np

sys.path.insert(0, '/home/hlamm/epoq_continuum_test/m1_toy')

from action_z2_staggered import LatticeGeometry, Z2GaugeConfig, build_dirac_matrix
from action_z2_metropolis import run_metropolis
from transfer_matrix_kbc import compute_combined_weight


def gauge_bits_at_slice(U: Z2GaugeConfig, t_slice: int) -> int:
    """Encode spatial gauge config at slice t as int (matching z2_setup convention).

    For 2x2 OBC: 4 spatial links per slice (2 x + 2 y).
    Bit ordering (z2_setup convention):
        q0 = U_y[t, 0, 0]
        q1 = U_y[t, 1, 0]
        q2 = U_x[t, 0, 0]
        q3 = U_x[t, 0, 1]
    """
    def bit(sigma):
        return 0 if sigma == 1 else 1
    b0 = bit(U.U_y[t_slice, 0, 0])
    b1 = bit(U.U_y[t_slice, 1, 0])
    b2 = bit(U.U_x[t_slice, 0, 0])
    b3 = bit(U.U_x[t_slice, 0, 1])
    return b0 | (b1 << 1) | (b2 << 2) | (b3 << 3)


def fock_basis_index(psi_tuple) -> int:
    """Convert tuple of occupations to bit-int (site k -> bit k)."""
    return sum(int(n) << k for k, n in enumerate(psi_tuple))


def compute_n0_at_t0(geom: LatticeGeometry, U: Z2GaugeConfig):
    """For one gauge config U, compute:
        num_U = Σ_{Ψ: gauge_top(U)=gauge_bot(U) AND Ψ_top=Ψ_bot} W[Ψ, Ψ] · n_0(Ψ)
        den_U = Σ_{Ψ: same diag condition} W[Ψ, Ψ]
    Returns (num_U, den_U).  When gauge_top != gauge_bot: both = 0.
    """
    g_top = gauge_bits_at_slice(U, 0)
    g_bot = gauge_bits_at_slice(U, geom.N_E - 1)
    if g_top != g_bot:
        return 0.0, 0.0
    W, info = compute_combined_weight(geom, U)
    V3 = geom.V_3
    dim = 1 << V3
    # Diagonal entries of W are W[i, i].  Each corresponds to a Fock state
    # in original Fock basis (lex enum from itertools.product).
    from itertools import product
    num = 0.0
    den = 0.0
    for i, n in enumerate(product((0, 1), repeat=V3)):
        psi = fock_basis_index(n)  # bit k = site k occupation
        w = W[i, i].real
        n_0 = 1.0 if (psi & 1) else 0.0
        num += w * n_0
        den += w
    return num, den


def run_at_aτ(a_tau: float, n_gauge: int = 500, n_warmup: int = 1000,
              seed: int = 2026, beta_phys: float = 4.0):
    """Run pipeline at given temporal lattice spacing."""
    N_E = int(round(beta_phys / a_tau)) + 1   # so β_eff = (N_E-1)·a_τ = β_phys
    m_lat = 0.5 * a_tau
    K_E = 1.0 / a_tau          # g_E = 1 in QC
    K_M = 4.0 * a_tau          # g_M = 0.5 in QC → K_M = a_τ / g_M² = 4·a_τ
    print(f"  N_E={N_E}, a_τ={a_tau:.4f}, m_lat={m_lat:.4f}, "
          f"K_E={K_E:.3f}, K_M={K_M:.3f}, β_eff={(N_E-1)*a_tau:.4f}")

    geom = LatticeGeometry(Lx=2, Ly=2, N_E=N_E, m=m_lat)
    t0 = time.time()
    # Cold start when K_M ≥ 1 (ordered phase) so MC starts near equilibrium.
    cold = K_M >= 1.0
    mc = run_metropolis(
        geom, K_E=K_E, K_M=K_M,
        n_sweeps=n_gauge, n_warmup=n_warmup, seed=seed, cold_start=cold)
    print(f"  gauge MC: accept={mc.accept_rate:.3f}, "
          f"⟨sign⟩={mc.avg_sign:+.3f}, "
          f"configs={len(mc.configs)}, time={time.time()-t0:.1f}s")

    # Aggregate corner-state contributions
    sum_sign_num = 0.0
    sum_sign_den = 0.0
    n_match = 0
    for U_idx, U in enumerate(mc.configs):
        sign_U = mc.sign_history[U_idx]
        num_U, den_U = compute_n0_at_t0(geom, U)
        if den_U != 0:
            n_match += 1
        sum_sign_num += sign_U * num_U
        sum_sign_den += sign_U * den_U

    if abs(sum_sign_den) > 1e-12:
        n0_estimate = sum_sign_num / sum_sign_den
    else:
        n0_estimate = float('nan')
    print(f"  n_diag configs={n_match}/{len(mc.configs)}, ⟨n̂_0⟩={n0_estimate:+.6f}")
    return {
        'a_tau': a_tau, 'N_E': N_E, 'm_lat': m_lat,
        'K_E': K_E, 'K_M': K_M,
        'beta_eff': (N_E - 1) * a_tau,
        'accept_rate': mc.accept_rate, 'avg_sign': mc.avg_sign,
        'n_configs': len(mc.configs), 'n_diag': n_match,
        'n0_estimate': n0_estimate,
    }


def main():
    if len(sys.argv) > 1:
        a_tau_list = [float(x) for x in sys.argv[1].split(',')]
    else:
        a_tau_list = [1.0, 0.5, 0.25, 0.125]
    n_gauge = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    n_warmup = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 2026

    print("=" * 78)
    print("Continuum convergence test: ⟨n̂_0(t=0)⟩ vs a_τ at β=4 fixed")
    print("=" * 78)
    print(f"physical_sector_reference(β=4) = 0.044597")
    print(f"n_gauge={n_gauge}, n_warmup={n_warmup}, seed={seed}")
    print()
    results = []
    for a_tau in a_tau_list:
        print(f"\n--- a_τ = {a_tau} ---")
        r = run_at_aτ(a_tau, n_gauge=n_gauge, n_warmup=n_warmup, seed=seed)
        results.append(r)

    print()
    print("=" * 78)
    print("Summary")
    print("=" * 78)
    print(f"{'a_τ':>8} {'N_E':>5} {'m_lat':>8} {'K_E':>8} {'K_M':>8} "
          f"{'accept':>8} {'⟨sign⟩':>8} {'n_diag':>8} {'⟨n̂_0⟩':>10}  "
          f"{'|err vs 0.045|':>15}")
    for r in results:
        err = abs(r['n0_estimate'] - 0.044597) if not np.isnan(r['n0_estimate']) else float('nan')
        print(f"{r['a_tau']:>8.4f} {r['N_E']:>5d} {r['m_lat']:>8.4f} "
              f"{r['K_E']:>8.3f} {r['K_M']:>8.3f} "
              f"{r['accept_rate']:>8.3f} {r['avg_sign']:>+8.3f} {r['n_diag']:>8d} "
              f"{r['n0_estimate']:>+10.5f} {err:>15.4f}")


if __name__ == "__main__":
    main()
