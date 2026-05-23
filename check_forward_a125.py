"""Verify forward-time det M (transfer-matrix exact) fixes the C(0) mismatch.
Same setup as check_psi_index_fix.py but action_type='forward'.

If forward-time gives C(0) ≈ ED ⟨n_0⟩ = +0.01352, the diagnosis is confirmed
and we proceed to remove central-diff implementations. If not, more debugging."""
from __future__ import annotations
import sys, time
import numpy as np

sys.path.insert(0, '/home/hlamm/Desktop/QC/logdet/m1_toy')
import epoq_quantum_simulator; epoq_quantum_simulator.M_MASS = 2.0
import epoq_classical_sampler; epoq_classical_sampler.M_MASS = 2.0

from action_z2_staggered import LatticeGeometry
from action_z2_metropolis_pt import run_metropolis_pt
from action_corner_direct_v3 import sample_corner_pairs_stratified
from action_minkowski_stitch import qc_basis_index, static_observable_diagonal
from epoq_quantum_simulator import build_trottered_UOU_cache

M_HAM, G_E_HAM, G_M_HAM, BETA = 2.0, 1.0, 0.5, 4.0
N_WARMUP, N_SWEEPS, SWAP_EVERY = 10000, 10000, 5
N_PER_SECTOR = 50
A_TAU = 0.125
SEED = 2026


def build_K_E_ladder(K_E_target, n_replicas=8):
    K_E_low = max(0.30, K_E_target - 0.7)
    if K_E_low >= K_E_target - 0.10:
        K_E_low = K_E_target - 0.10
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
                a_tau=A_TAU, m_obs=M_HAM, g_hop=0.5)
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
        return {t: float('nan') for t in times}, n_diag
    return {t: sum_num[t]/sum_denom for t in times}, n_diag


def main():
    times = [0.0, 0.5, 1.0, 2.0]
    print("=" * 80, flush=True)
    print(f"FORWARD-TIME check @ a_τ={A_TAU}, m={M_HAM}, β={BETA}")
    print(f"  action_type='forward'  (vs the previous central-diff)")
    print(f"  Single PT instance, warm={N_WARMUP} swp={N_SWEEPS}, "
          f"n_per_sector={N_PER_SECTOR}, seed={SEED}")
    print("=" * 80, flush=True)

    t0 = time.time()
    cached_UOU = build_trottered_UOU_cache(times, n_trotter=200, order=2)
    print(f"cache: {time.time()-t0:.1f}s", flush=True)

    # ED ref
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
    print("\nED ref: " + "  ".join(f"C({t})={refs[t]:+.5f}" for t in times),
          flush=True)

    K_E_target = -0.5 * np.log(np.tanh(A_TAU * G_E_HAM))
    K_M = A_TAU * G_M_HAM
    N_E = int(round(BETA / A_TAU)) + 1
    m_action = A_TAU * M_HAM
    geom = LatticeGeometry(Lx=2, Ly=2, N_E=N_E, m=m_action)
    K_E_ladder = build_K_E_ladder(K_E_target)
    print(f"\nPT MC: N_E={N_E}, ladder={[f'{k:.3f}' for k in K_E_ladder]}", flush=True)

    t0 = time.time()
    pt_results, sw_stats = run_metropolis_pt(
        geom, K_E_ladder=K_E_ladder, K_M=K_M,
        n_sweeps=N_SWEEPS, n_warmup=N_WARMUP, swap_every=SWAP_EVERY,
        seed=SEED, action_type='forward')   # <-- FORWARD!
    print(f"PT MC walltime: {time.time()-t0:.0f}s", flush=True)

    top = pt_results[-1]
    sgn = np.array(top.sign_history, dtype=int)
    print(f"top replica: {len(top.configs)} configs, ⟨sgn⟩={sgn.mean():+.3f}",
          flush=True)

    t0 = time.time()
    Cs, n_diag = measure_C_t(
        geom, top.configs, top.sign_history, times=times,
        n_per_sector=N_PER_SECTOR, cached_UOU=cached_UOU, seed=SEED+7777)
    print(f"Phase 2+3 walltime: {time.time()-t0:.0f}s, n_diag={n_diag}",
          flush=True)

    print()
    print("=" * 60, flush=True)
    print(f"{'t':>4} | {'ED ref':>10} | {'C(t) FORWARD':>14} | {'gap':>10}")
    print("-" * 60)
    for t in times:
        print(f"{t:>4.1f} | {refs[t]:>+10.5f} | {Cs[t]:>+14.5f} | "
              f"{Cs[t]-refs[t]:>+10.5f}")
    print()
    print("Previous CENTRAL-diff result was:")
    print("  C(0.0)=+0.00074  C(0.5)=-0.01204  C(1.0)=-0.00639  C(2.0)=+0.00140")


if __name__ == "__main__":
    main()
