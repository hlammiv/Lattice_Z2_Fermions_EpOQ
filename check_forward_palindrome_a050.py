"""Same as check_forward_a050.py but uses ORDER=2 (palindrome) for the W
matrix in the corner sampler. If this fixes C(0) (no more negative values
from non-Hermitian W diagonals), the diagnosis is confirmed."""
from __future__ import annotations
import sys, time
import numpy as np

sys.path.insert(0, '/home/hlamm/Desktop/QC/logdet/m1_toy')
import epoq_quantum_simulator; epoq_quantum_simulator.M_MASS = 2.0
import epoq_classical_sampler; epoq_classical_sampler.M_MASS = 2.0

from action_z2_staggered import LatticeGeometry
from action_z2_metropolis_pt import run_metropolis_pt, electric_plaquette_sum
from action_corner_direct_v3 import sample_corner_pairs_stratified
from action_minkowski_stitch import qc_basis_index, static_observable_diagonal
from epoq_quantum_simulator import build_trottered_UOU_cache

M_HAM, G_E_HAM, G_M_HAM, BETA = 2.0, 1.0, 0.5, 4.0
A_TAU = 0.5
N_WARMUP, N_SWEEPS, SWAP_EVERY = 5000, 10000, 5
N_PER_SECTOR = 50
N_INSTANCES = 4
W_ORDER = 2   # palindrome (Hermitian PD)


def build_K_E_ladder(K_E_target, n_replicas=8):
    K_E_low = max(0.10, K_E_target - 0.7)
    if K_E_low >= K_E_target - 0.05:
        K_E_low = K_E_target - 0.05
    return list(np.linspace(K_E_low, K_E_target, n_replicas))


def measure_C_t(geom, configs, sign_hist, times, n_per_sector, cached_UOU, seed):
    rng = np.random.default_rng(seed)
    sum_num = {t: 0.0 for t in times}
    sum_denom = 0.0
    n_diag = 0
    for U_idx in range(len(configs)):
        U = configs[U_idx]
        sign_g = float(sign_hist[U_idx])
        try:
            pairs, p_signs, sector_w, _ = sample_corner_pairs_stratified(
                geom, U, n_per_sector=n_per_sector, rng=rng,
                a_tau=A_TAU, m_obs=M_HAM, g_hop=0.5, order=W_ORDER)
        except RuntimeError:
            continue
        for k in range(len(pairs)):
            psi_i = int(pairs[k, 0]); psi_j = int(pairs[k, 1])
            sl_sign = float(p_signs[k]); sw = float(sector_w[k])
            bits_top = qc_basis_index(geom, U, geom.N_E - 1, psi_i)
            bits_bot = qc_basis_index(geom, U, 0, psi_j)
            is_diag = (bits_top == bits_bot)
            n0_at_0 = static_observable_diagonal(geom, U, psi_j)
            w = sign_g * sl_sign * sw
            if is_diag:
                sum_denom += w; n_diag += 1
            for t in times:
                mat_el = float(cached_UOU[t][bits_bot, bits_top].real)
                sum_num[t] += w * n0_at_0 * mat_el
    if abs(sum_denom) < 1e-12:
        return {t: float('nan') for t in times}, n_diag, sum_denom
    return {t: sum_num[t]/sum_denom for t in times}, n_diag, sum_denom


def run_one_PT(seed, cached_UOU, n_E_plaq, times):
    K_E_target = -0.5 * np.log(np.tanh(A_TAU * G_E_HAM))
    K_M = A_TAU * G_M_HAM
    N_E = int(round(BETA / A_TAU)) + 1
    m_action = A_TAU * M_HAM
    geom = LatticeGeometry(Lx=2, Ly=2, N_E=N_E, m=m_action)
    K_E_ladder = build_K_E_ladder(K_E_target)
    t0 = time.time()
    pt_results, sw_stats = run_metropolis_pt(
        geom, K_E_ladder=K_E_ladder, K_M=K_M,
        n_sweeps=N_SWEEPS, n_warmup=N_WARMUP, swap_every=SWAP_EVERY,
        seed=seed, action_type='forward')
    t_mc = time.time() - t0
    top = pt_results[-1]
    PE = np.array([electric_plaquette_sum(geom, U)/n_E_plaq for U in top.configs])
    sgn = np.array(top.sign_history, dtype=int)
    Cs, n_diag, sum_denom = measure_C_t(
        geom, top.configs, top.sign_history, times=times,
        n_per_sector=N_PER_SECTOR, cached_UOU=cached_UOU, seed=seed+7777)
    return {'C': Cs, 'n_diag': n_diag, 'sum_denom': sum_denom,
            'mc_walltime': t_mc, 'swap_rates': sw_stats.rates.tolist(),
            'PE_mean': float(PE.mean()), 'sign_mean': float(sgn.mean())}


def main():
    times = [0.0, 0.5, 1.0, 2.0]
    print("=" * 80, flush=True)
    print(f"FORWARD-time PALINDROME W (order={W_ORDER}) check @ a_τ={A_TAU}")
    print(f"  m_Ham={M_HAM}, β={BETA}, {N_INSTANCES} indep PT-forward")
    print(f"  warm={N_WARMUP} swp={N_SWEEPS}, n_per_sector={N_PER_SECTOR}")
    print("=" * 80, flush=True)

    cached_UOU = build_trottered_UOU_cache(times, n_trotter=200, order=2)

    from scipy.linalg import expm
    P2 = {'I': np.eye(2, dtype=complex), 'X':np.array([[0,1],[1,0]], dtype=complex),
          'Y':np.array([[0,-1j],[1j,0]], dtype=complex),
          'Z':np.diag([1,-1]).astype(complex)}
    def pauli(factors, nq=8):
        by_q = {q: P2['I'] for q in range(nq)}
        for q, ax in factors: by_q[q] = P2[ax]
        r = by_q[nq-1]
        for q in range(nq-2, -1, -1): r = np.kron(r, by_q[q])
        return r
    H = sum(c * pauli(f) for c, f in epoq_classical_sampler.build_pauli_terms())
    H = (H + H.conj().T) / 2
    n0 = 0.5 * np.eye(256, dtype=complex) - 0.5 * pauli([(4, 'Z')])
    rho = expm(-BETA * H); Z = np.trace(rho).real
    refs = {}
    for t in times:
        Ut = expm(-1j * H * t)
        refs[t] = (np.trace(rho @ (Ut.conj().T @ n0 @ Ut) @ n0) / Z).real
    print("ED ref: " + "  ".join(f"C({t})={refs[t]:+.5f}" for t in times), flush=True)

    N_E = int(round(BETA / A_TAU)) + 1
    n_E_plaq = (N_E - 1) * 1 * 2 + (N_E - 1) * 2 * 1

    instances = []
    for i in range(N_INSTANCES):
        seed_i = 2026 + i * 11111
        print(f"\n--- instance {i+1}/{N_INSTANCES} (seed={seed_i}) ---", flush=True)
        r = run_one_PT(seed_i, cached_UOU, n_E_plaq, times)
        print(f"    MC walltime: {r['mc_walltime']:.0f}s, n_diag={r['n_diag']}",
              flush=True)
        print(f"    ⟨P_E⟩={r['PE_mean']:.4f}  ⟨sgn⟩={r['sign_mean']:+.3f}  "
              f"sum_denom={r['sum_denom']:+.3e}", flush=True)
        print("    " + "  ".join(f"C({t})={r['C'][t]:+.5f}" for t in times),
              flush=True)
        instances.append(r)

    print()
    print("=" * 70, flush=True)
    print(f"SUMMARY  @ a_τ={A_TAU}, FORWARD-time PALINDROME W (order=2)")
    print("=" * 70)
    print(f"{'t':>4} | {'C(t) ± SEM':>24} | {'ED ref':>10} | {'sigmas':>7}")
    for t in times:
        vals = np.array([inst['C'][t] for inst in instances])
        vals = vals[np.isfinite(vals)]
        if len(vals) < 2:
            print(f"  C({t}) = n/a")
            continue
        m = vals.mean(); s = vals.std(ddof=1)/np.sqrt(len(vals))
        gap = m - refs[t]
        sigmas = abs(gap)/s if s > 0 else float('nan')
        print(f"  {t:.1f} | {m:>+10.5f} ± {s:>.5f}   | {refs[t]:>+10.5f} | "
              f"{sigmas:>6.1f}σ")
    print()
    print("Previous result with order=1 (Lie-Trotter) for comparison:")
    print("  C(0.0) = -0.00977 ± 0.00583  (4.0σ low — NEGATIVE)")
    print("  C(0.5) = -0.03765 ± 0.00580  (4.3σ off)")
    print("  C(1.0) = -0.03432 ± 0.00515  (9.2σ off)")
    print("  C(2.0) = -0.02365 ± 0.00737  (5.0σ off)")


if __name__ == "__main__":
    main()
