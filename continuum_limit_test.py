"""
continuum_limit_test.py — pipeline ⟨n̂_0⟩ vs lattice spacing a_τ, β fixed.

Goal: validate that the corner-state EρOQ pipeline reproduces the
Hamiltonian observable physical_sector_reference(β=4) ≈ 0.0446 in the
a_τ → 0 limit, even though it doesn't at a_τ=1.

Reference Hamiltonian couplings (z2_setup):
  g_E_Ham = 1.0   (electric)
  g_M_Ham = 0.5   (magnetic)
  m_Ham   = 0.5   (fermion mass)
  g_hop   = 0.5   (spatial fermion hopping)

Continuum-limit action coupling scaling (a_τ → 0, β fixed):
  K_M_action(a_τ) = a_τ · g_M_Ham                          (diagonal Trotter)
  K_E_action(a_τ) = -(1/2) log tanh(a_τ · g_E_Ham)         (Suzuki transverse)
  m_action(a_τ)   = a_τ · m_Ham                            (Trotter mass)
  hop coeff       = 1/2 fixed (lattice-derivative form)

Observable T̂_F = expm(-a_τ · H_KS[U_spatial(t)]) per slice uses Hamiltonian
masses (m_Ham, g_hop) directly — the a_τ scaling is absorbed in the matrix
exponential.

Sweep a_τ ∈ {1, 0.5, 0.25, 0.125} → N_E ∈ {5, 9, 17, 33} at β=4.

Usage: python continuum_limit_test.py [n_configs] [n_chains]
"""
from __future__ import annotations
import sys
import time
import numpy as np

sys.path.insert(0, '/home/hlamm/Desktop/QC/logdet/m1_toy')

from action_z2_staggered import LatticeGeometry, Z2GaugeConfig
from action_z2_metropolis import run_metropolis_parallel
from transfer_matrix_kbc_trotterized import compute_combined_weight_trotter
from mc_error_analysis import blocked_ratio_jackknife, format_estimate


# ---------------------------------------------------------------------------
# Hamiltonian-side reference parameters (z2_setup defaults).
# ---------------------------------------------------------------------------
G_E_HAM = 1.0
G_M_HAM = 0.5
M_HAM = 0.5
G_HOP = 0.5
BETA = 4.0


def couplings_at_a_tau(a_tau: float) -> dict:
    """Return action couplings + observable couplings for the chosen a_τ."""
    K_M = a_tau * G_M_HAM
    K_E = -0.5 * np.log(np.tanh(a_tau * G_E_HAM))
    m_action = a_tau * M_HAM
    return {
        'a_tau': a_tau,
        'K_M_action': K_M,
        'K_E_action': K_E,
        'm_action': m_action,
        'm_obs': M_HAM,         # Hamiltonian mass for T̂_F observable
        'g_hop_obs': G_HOP,
    }


def gauge_bits_at_slice(U, t_slice):
    def bit(s): return 0 if s == 1 else 1
    b0 = bit(U.U_y[t_slice, 0, 0])
    b1 = bit(U.U_y[t_slice, 1, 0])
    b2 = bit(U.U_x[t_slice, 0, 0])
    b3 = bit(U.U_x[t_slice, 0, 1])
    return b0 | (b1 << 1) | (b2 << 2) | (b3 << 3)


# Precompute the n_0 bit-mask once per V_3 (site 0 occupation).
_N0_MASK_CACHE: dict[int, np.ndarray] = {}

def _n0_mask(V3: int) -> np.ndarray:
    if V3 not in _N0_MASK_CACHE:
        dim = 1 << V3
        bit_pos = V3 - 1
        states = np.arange(dim)
        _N0_MASK_CACHE[V3] = ((states >> bit_pos) & 1).astype(np.float64)
    return _N0_MASK_CACHE[V3]


def compute_n0_at_t0(geom, U, params):
    """Per-config corner-state contribution (num, den).  Uses m_obs in the
    Trotter observable, not the action mass."""
    g_top = gauge_bits_at_slice(U, 0)
    g_bot = gauge_bits_at_slice(U, geom.N_E - 1)
    if g_top != g_bot:
        return 0.0, 0.0
    W, info = compute_combined_weight_trotter(
        geom, U, a_tau=params['a_tau'],
        K_E=0.0, K_M=0.0,                    # plaq factors already in MC weight
        g_hop=params['g_hop_obs'],
        m_obs=params['m_obs'],
    )
    diag = np.real(np.diag(W))
    mask = _n0_mask(geom.V_3)
    num = float(np.sum(diag * mask))
    den = float(np.sum(diag))
    return num, den


def run_at_a_tau(a_tau: float, n_gauge: int, n_warmup: int = 2000,
                 seed: int = 2026, n_chains: int = 8):
    params = couplings_at_a_tau(a_tau)
    N_E = int(round(BETA / a_tau)) + 1
    beta_eff = (N_E - 1) * a_tau

    print(f"--- a_τ = {a_tau:.4f}, N_E = {N_E} (β_eff = {beta_eff:.4f}) ---")
    print(f"  Action: m={params['m_action']:.4f}, K_E={params['K_E_action']:.4f}, "
          f"K_M={params['K_M_action']:.4f}")
    print(f"  Observable: m_obs={params['m_obs']:.4f}, g_hop={params['g_hop_obs']:.4f}")

    geom = LatticeGeometry(Lx=2, Ly=2, N_E=N_E, m=params['m_action'])

    t0 = time.time()
    mc = run_metropolis_parallel(
        geom, n_chains=n_chains,
        K_E=params['K_E_action'], K_M=params['K_M_action'],
        n_sweeps_per_chain=max(1, n_gauge // n_chains),
        n_warmup=n_warmup, seed=seed, cold_start=False,
        action_type='forward', plaq_flip_every=5,
    )
    t_mc = time.time() - t0
    print(f"  gauge MC: accept={mc.accept_rate:.3f}, ⟨sign⟩={mc.avg_sign:+.3f}, "
          f"configs={len(mc.configs)}, t={t_mc:.1f}s")

    # Collect per-config sign·num and sign·den arrays for proper jackknife
    sign_arr = mc.sign_history
    num_arr = np.zeros(len(mc.configs))
    den_arr = np.zeros(len(mc.configs))
    n_match = 0
    t1 = time.time()
    for U_idx, U in enumerate(mc.configs):
        num_U, den_U = compute_n0_at_t0(geom, U, params)
        if den_U != 0:
            n_match += 1
        num_arr[U_idx] = sign_arr[U_idx] * num_U
        den_arr[U_idx] = sign_arr[U_idx] * den_U
    t_obs = time.time() - t1

    # Autocorrelation-aware jackknife error bar
    n0, sigma_jk, tau_int, jk_info = blocked_ratio_jackknife(num_arr, den_arr)

    print(f"  n_diag={n_match}/{len(mc.configs)}, obs loop time={t_obs:.1f}s")
    print(f"  {format_estimate(n0, sigma_jk, tau_int, label='⟨n̂_0⟩')}")
    print(f"  (σ_naive at b=1: {jk_info.get('sigma_naive_b1', float('nan')):.5f}, "
          f"σ_plateau: {sigma_jk:.5f}, blow-up factor: "
          f"{sigma_jk/max(jk_info.get('sigma_naive_b1', 1e-12), 1e-12):.1f}×)")
    return {'a_tau': a_tau, 'N_E': N_E, 'beta_eff': beta_eff,
            'n0': n0, 'sigma': sigma_jk, 'tau_int': tau_int,
            'sigma_naive': jk_info.get('sigma_naive_b1', float('nan')),
            'avg_sign': mc.avg_sign, 'n_diag': n_match,
            'n_configs': len(mc.configs), 't_mc': t_mc, 't_obs': t_obs,
            **params}


def main():
    n_configs = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
    n_chains = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    a_tau_list = [1.0, 0.5, 0.25, 0.125]

    print("=" * 78)
    print(f"Continuum-limit study: ⟨n̂_0⟩ vs a_τ at fixed β = {BETA}")
    print("=" * 78)
    print(f"  Hamiltonian: g_E={G_E_HAM}, g_M={G_M_HAM}, m={M_HAM}, g_hop={G_HOP}")
    print(f"  Action couplings scaled per a_τ (K_M=a_τ·g_M, K_E=-½log tanh(a_τ g_E))")
    print(f"  n_configs={n_configs}, n_chains={n_chains}, β={BETA}")
    print()

    # Compute BOTH references: Gauss-law-projected (what z2_setup quantum theory
    # gives) AND classical-gauge-sum (what action MC + Trotter actually targets).
    from action_endtoend_pipeline import physical_sector_reference
    ref_phys = physical_sector_reference(beta=BETA, times=[0.0])[0.0]
    # Classical-gauge-sum reference computed at the staggered Hamiltonian
    from staggered_ks_reference import build_pauli_terms_stagg, pauli_term_dense, matter_q, DIM
    from scipy.linalg import expm as _expm
    _H = sum(c * pauli_term_dense(fac) for c, fac
             in build_pauli_terms_stagg(g_E=G_E_HAM, g_M=G_M_HAM, g_hop=G_HOP, m=M_HAM))
    _H = (_H + _H.conj().T) / 2
    _rho = _expm(-BETA * _H)
    _n0 = 0.5 * np.eye(DIM, dtype=complex) - 0.5 * pauli_term_dense([(matter_q(0), 'Z')])
    ref_classical = float((np.trace(_rho @ _n0) / np.trace(_rho)).real)
    # Compare to BOTH (pipeline should approach ref_classical, NOT ref_phys)
    ref = ref_classical
    print(f"References:")
    print(f"  Gauss-law-physical (z2_setup quantum):       {ref_phys:+.5f}")
    print(f"  Classical-gauge-sum (pipeline target):        {ref_classical:+.5f}")
    print(f"  → Comparing pipeline to {ref:+.5f} (classical-gauge-sum)")
    print()

    results = []
    for a_tau in a_tau_list:
        r = run_at_a_tau(a_tau, n_gauge=n_configs, n_chains=n_chains)
        r['err'] = abs(r['n0'] - ref) if not np.isnan(r['n0']) else float('nan')
        results.append(r)
        print()

    print("=" * 78)
    print(f"Summary (Gauss-law-physical reference = {ref:+.5f}):")
    print(f"{'a_τ':>8} {'N_E':>5} {'⟨sign⟩':>8} {'n_diag':>8} "
          f"{'⟨n̂_0⟩':>11} {'σ_jack':>10} {'τ_int':>7} {'|err|':>9} {'err/σ':>7}")
    for r in results:
        err = r['err']
        ratio = err / r['sigma'] if r['sigma'] > 0 else float('inf')
        print(f"{r['a_tau']:>8.4f} {r['N_E']:>5d} {r['avg_sign']:>+8.3f} "
              f"{r['n_diag']:>8d} {r['n0']:>+11.5f} ± {r['sigma']:.5f} "
              f"{r['tau_int']:>7.1f} {err:>9.5f} {ratio:>7.1f}")


if __name__ == "__main__":
    main()
