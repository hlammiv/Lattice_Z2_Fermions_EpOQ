"""
verify_pipeline_pf_vs_exact_m2.py — full pipeline C(t) cross-check at m=2.

Generate gauge configs via two MC methods at the same parameters:
  1. Exact-det heat-bath (`heatbath_sweep` with action_type='forward').
  2. Hasenbusch PF Metropolis (`metropolis_sweep_hasenbusch`, Δm=2).

For each ensemble, run Phase 2+3 of the v3 pipeline (corner-state sampling
from Trotter W + cached U†OU lookup) and compute C(t) = ⟨n_0(t)·n_0(0)⟩.

Both MC methods target the same Boltzmann weight at m=2, K_E=0.386, K_M=0.25,
so they should give the SAME C(t) within statistical noise.  This is the
methodology-paper validation that PF-Hasenbusch is a drop-in replacement
for exact-det MC.

Monkey-patches M_MASS=2.0 in both epoq_quantum_simulator and epoq_classical_sampler
to keep the QC Hamiltonian consistent with the Euclidean side.
"""
from __future__ import annotations
import sys
import time
import numpy as np

sys.path.insert(0, '/home/hlamm/Desktop/QC/logdet/m1_toy')

# IMPORTANT: monkey-patch M_MASS to 2.0 BEFORE importing modules that build
# H using these constants.
import epoq_quantum_simulator
import epoq_classical_sampler
epoq_quantum_simulator.M_MASS = 2.0
epoq_classical_sampler.M_MASS = 2.0

from action_z2_staggered import LatticeGeometry, Z2GaugeConfig
from action_z2_metropolis import heatbath_sweep
from action_z2_pf_hasenbusch import metropolis_sweep_hasenbusch
from action_corner_direct_v3 import sample_corner_pairs_stratified
from action_minkowski_stitch import qc_basis_index, static_observable_diagonal
from transfer_matrix_kbc_trotterized import compute_combined_weight_trotter
from epoq_quantum_simulator import build_trottered_UOU_cache


def run_chain(geom, K_E, K_M, n_sweeps, n_warmup, seed, mode, delta_m=2.0):
    """Single-chain MC. mode = 'exact' or 'hasenbusch'."""
    rng = np.random.default_rng(seed)
    U = Z2GaugeConfig.random(geom, rng)
    configs, sign_hist = [], []
    for sw in range(n_sweeps + n_warmup):
        if mode == 'exact':
            U, dm, sg, _ = heatbath_sweep(
                geom, U, K_E=K_E, K_M=K_M, rng=rng, action_type='forward')
        else:
            U, dm, sg, _ = metropolis_sweep_hasenbusch(
                geom, U, K_E=K_E, K_M=K_M, rng=rng, delta_m=delta_m,
                refresh='per_link')
        if sw >= n_warmup:
            configs.append(Z2GaugeConfig(
                geom=geom, U_x=U.U_x.copy(), U_y=U.U_y.copy(), U_t=U.U_t.copy()))
            sign_hist.append(1 if dm > 0 else (-1 if dm < 0 else 0))
    return configs, np.array(sign_hist)


def compute_C_t(geom, configs, sign_hist, m_obs, times, n_per_sector,
                cached_UOU, seed):
    """For an ensemble of gauge configs, run Phase 2+3 to estimate C(t)."""
    rng = np.random.default_rng(seed)
    sum_num_t = {t: 0.0 for t in times}
    sum_denom = 0.0
    n_diag_total = 0
    for U_idx, U in enumerate(configs):
        try:
            pairs, signs, sector_w, _ = sample_corner_pairs_stratified(
                geom, U, n_per_sector=n_per_sector, rng=rng,
                a_tau=1.0, m_obs=m_obs, g_hop=0.5,
            )
        except RuntimeError:
            continue
        sign_g = float(sign_hist[U_idx])
        for k in range(len(pairs)):
            psi_i = int(pairs[k, 0]); psi_j = int(pairs[k, 1])
            sl_sign = float(signs[k]); sw = float(sector_w[k])
            bits_top = qc_basis_index(geom, U, geom.N_E - 1, psi_i)
            bits_bot = qc_basis_index(geom, U, 0, psi_j)
            is_diag_k = (bits_top == bits_bot)
            n0_at_0 = static_observable_diagonal(geom, U, psi_j)
            w = sign_g * sl_sign * sw
            if is_diag_k:
                sum_denom += w
                n_diag_total += 1
            for t in times:
                mat_el = cached_UOU[t][bits_bot, bits_top]
                sum_num_t[t] += w * n0_at_0 * float(mat_el.real)
    Cs = {}
    if abs(sum_denom) > 1e-12:
        for t in times:
            Cs[t] = sum_num_t[t] / sum_denom
    else:
        for t in times:
            Cs[t] = float('nan')
    return Cs, n_diag_total


def main():
    m_Ham = 2.0
    a_tau = 1.0
    K_E = -0.5 * np.log(np.tanh(a_tau * 1.0))     # 0.1362 (Suzuki for g_E=1)
    K_M = a_tau * 0.5                              # 0.5
    print(f"=== PF-Hasenbusch vs exact-det pipeline cross-check at m={m_Ham} ===")
    print(f"a_τ={a_tau}, K_E={K_E:.4f}, K_M={K_M}")

    geom = LatticeGeometry(Lx=2, Ly=2, N_E=5, m=m_Ham)   # action m at a_τ=1 = m_Ham

    # Cached U†OU at m_Ham=2.0 (after monkey-patch)
    times = [0.0, 0.5, 1.0, 2.0]
    print("Building U†OU cache with m=2.0...")
    t0 = time.time()
    cached_UOU = build_trottered_UOU_cache(times, n_trotter=200, order=2)
    print(f"  cache build: {time.time()-t0:.1f}s")

    # Reference: exact-Hamiltonian thermal correlator at m=2
    from scipy.linalg import expm
    from epoq_classical_sampler import build_pauli_terms
    DIM = 256
    P2 = {'I': np.eye(2, dtype=complex),
          'X': np.array([[0,1],[1,0]], dtype=complex),
          'Y': np.array([[0,-1j],[1j,0]], dtype=complex),
          'Z': np.diag([1,-1]).astype(complex)}
    def pauli_dense(factors, nq=8):
        by_q = {q: P2['I'] for q in range(nq)}
        for q, ax in factors: by_q[q] = P2[ax]
        r = by_q[nq-1]
        for q in range(nq-2, -1, -1): r = np.kron(r, by_q[q])
        return r
    H = sum(c * pauli_dense(fac) for c, fac in build_pauli_terms())
    H = (H + H.conj().T) / 2
    n0 = 0.5 * np.eye(DIM, dtype=complex) - 0.5 * pauli_dense([(4, 'Z')])
    rho = expm(-4.0 * H); Z = np.trace(rho).real
    refs = {}
    for t in times:
        Ut = expm(-1j * H * t)
        n0_t = Ut.conj().T @ n0 @ Ut
        refs[t] = (np.trace(rho @ n0_t @ n0) / Z).real
    print(f"Unprojected ED reference at β=4, m=2: " +
          ", ".join(f"C({t})={refs[t]:+.4f}" for t in times))
    print()

    # Run 4 chains each, modest stats
    n_chains = 4
    n_sweeps, n_warmup = 800, 200
    n_per_sector = 5

    for label, mode in [('exact', 'exact'), ('hasenbusch', 'hasenbusch')]:
        print(f"\n--- {label} MC: {n_chains} chains × {n_sweeps} sweeps + {n_warmup} warmup ---")
        chain_Cs = {t: [] for t in times}
        chain_t0 = time.time()
        for ch in range(n_chains):
            t0c = time.time()
            configs, sign_hist = run_chain(
                geom, K_E, K_M, n_sweeps, n_warmup, seed=10*ch+1, mode=mode)
            t_mc = time.time() - t0c
            Cs, n_diag = compute_C_t(
                geom, configs, sign_hist, m_obs=m_Ham, times=times,
                n_per_sector=n_per_sector, cached_UOU=cached_UOU,
                seed=10*ch+1000)
            for t in times:
                chain_Cs[t].append(Cs[t])
            print(f"  chain {ch}: MC {t_mc:.0f}s, n_diag={n_diag}, "
                  f"C={ {t: f'{Cs[t]:+.4f}' for t in times} }")
        print(f"  total walltime: {time.time()-chain_t0:.0f}s")
        print(f"  {label} chain-averaged C(t):")
        for t in times:
            arr = np.array(chain_Cs[t])
            arr = arr[np.isfinite(arr)]
            if len(arr) >= 2:
                m = arr.mean(); s = arr.std(ddof=1)/np.sqrt(len(arr))
                print(f"    C({t}) = {m:+.4f} ± {s:.4f}   ref = {refs[t]:+.4f}")


if __name__ == "__main__":
    main()
