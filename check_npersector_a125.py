"""Focused check: a_τ=0.125 only, n_per_sector=100 (vs the long run's 5).
Same warmup/sweeps/seeds as continuum_limit_pt_m2_long.py — only the corner
sampler density changes.

If C(0), C(1), C(2) snap toward ED (from the 13σ-off, 22σ-off, 13σ-off they
were stuck at), the ratio-estimator bias diagnosis is confirmed and we'll
re-run the full continuum study with the bigger n_per_sector."""
from __future__ import annotations
import sys, time, pickle, os
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
N_INSTANCES = 4
N_WARMUP, N_SWEEPS, SWAP_EVERY = 15000, 15000, 5
N_PER_SECTOR = 100   # ← 20x the long run's 5
A_TAU = 0.125
CKPT_DIR = "/tmp/continuum_pt_m2_npersector100_ckpt"
os.makedirs(CKPT_DIR, exist_ok=True)


def build_K_E_ladder(K_E_target, n_replicas=8):
    K_E_low = max(0.30, K_E_target - 0.7)
    if K_E_low >= K_E_target - 0.10:
        K_E_low = K_E_target - 0.10
    return list(np.linspace(K_E_low, K_E_target, n_replicas))


def measure_C_t(geom, configs, sign_hist, a_tau, m_obs, times,
                  n_per_sector, cached_UOU, seed):
    rng = np.random.default_rng(seed)
    sum_num_t = {t: 0.0 for t in times}
    sum_denom = 0.0
    n_diag_total = 0
    for U_idx in range(len(configs)):
        U = configs[U_idx]
        sign_g = float(sign_hist[U_idx])
        try:
            pairs, p_signs, sector_w, _ = sample_corner_pairs_stratified(
                geom, U, n_per_sector=n_per_sector, rng=rng,
                a_tau=a_tau, m_obs=m_obs, g_hop=0.5)
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
                sum_denom += w; n_diag_total += 1
            for t in times:
                mat_el = cached_UOU[t][bits_bot, bits_top]
                sum_num_t[t] += w * n0_at_0 * float(mat_el.real)
    if abs(sum_denom) < 1e-12:
        return {t: float('nan') for t in times}, n_diag_total
    return {t: sum_num_t[t]/sum_denom for t in times}, n_diag_total


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
        n_sweeps=N_SWEEPS, n_warmup=N_WARMUP,
        swap_every=SWAP_EVERY, seed=seed, action_type='forward')
    t_mc = time.time() - t0
    top = pt_results[-1]
    PE = np.array([electric_plaquette_sum(geom, U)/n_E_plaq for U in top.configs])
    sgn = np.array(top.sign_history, dtype=int)
    t1 = time.time()
    Cs, n_diag = measure_C_t(
        geom, top.configs, top.sign_history, a_tau=A_TAU, m_obs=M_HAM,
        times=times, n_per_sector=N_PER_SECTOR, cached_UOU=cached_UOU,
        seed=seed + 7777)
    t_p23 = time.time() - t1
    return {'C': Cs, 'n_diag': n_diag, 'mc_walltime': t_mc, 'p23_walltime': t_p23,
            'swap_rates': sw_stats.rates.tolist(),
            'K_E_target': K_E_target, 'K_M': K_M,
            'PE_mean': float(PE.mean()), 'sign_mean': float(sgn.mean())}


def main():
    times = [0.0, 0.5, 1.0, 2.0]
    print("=" * 80, flush=True)
    print(f"FOCUSED CHECK @ a_τ=0.125, m={M_HAM}, β={BETA}")
    print(f"  n_per_sector = {N_PER_SECTOR} (vs long-run 5)")
    print(f"  PT-exact × {N_INSTANCES} INDEP instances, warm={N_WARMUP} swp={N_SWEEPS}")
    print("=" * 80, flush=True)

    print("Building U†OU cache at m=2...", flush=True)
    t0 = time.time()
    cached_UOU = build_trottered_UOU_cache(times, n_trotter=200, order=2)
    print(f"  cache: {time.time()-t0:.1f}s", flush=True)

    # ED reference
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
        n0_t = Ut.conj().T @ n0 @ Ut
        refs[t] = (np.trace(rho @ n0_t @ n0) / Z).real
    print(f"\nED ref: " + "  ".join(f"C({t})={refs[t]:+.5f}" for t in times),
          flush=True)

    N_E = int(round(BETA / A_TAU)) + 1
    n_E_plaq = (N_E - 1) * 1 * 2 + (N_E - 1) * 2 * 1

    instances = []
    for i in range(N_INSTANCES):
        seed_i = 2026 + i * 11111
        ckpt = os.path.join(CKPT_DIR, f"inst_{i}_seed_{seed_i}.pkl")
        if os.path.exists(ckpt):
            with open(ckpt, 'rb') as f:
                r = pickle.load(f)
            print(f"\n[ckpt] inst {i+1} loaded", flush=True)
        else:
            print(f"\n--- instance {i+1}/{N_INSTANCES} (seed={seed_i}) ---", flush=True)
            r = run_one_PT(seed_i, cached_UOU, n_E_plaq, times)
            with open(ckpt, 'wb') as f:
                pickle.dump(r, f)
            print(f"    MC walltime: {r['mc_walltime']:.0f}s, "
                  f"Phase 2+3: {r['p23_walltime']:.0f}s, n_diag={r['n_diag']}",
                  flush=True)
            print(f"    ⟨P_E⟩={r['PE_mean']:.4f}  ⟨sgn⟩={r['sign_mean']:+.3f}",
                  flush=True)
            Cs = r['C']
            print(f"    " + "  ".join(f"C({t})={Cs[t]:+.5f}" for t in times),
                  flush=True)
        instances.append(r)

    print()
    print("=" * 80, flush=True)
    print(f"SUMMARY  @ a_τ=0.125, n_per_sector={N_PER_SECTOR}, "
          f"{N_INSTANCES} indep PT instances")
    print("=" * 80, flush=True)
    row = f"{'observable':>14} | {'PT (n_per_sec='+str(N_PER_SECTOR)+')':>22} | "
    row += f"{'ED ref':>10} | sigmas"
    print(row)
    for t in times:
        vals = np.array([inst['C'][t] for inst in instances])
        vals = vals[np.isfinite(vals)]
        if len(vals) < 2:
            print(f"  C({t}) = n/a")
            continue
        m = vals.mean(); s = vals.std(ddof=1)/np.sqrt(len(vals))
        gap = m - refs[t]
        sigmas = abs(gap)/s if s > 0 else float('nan')
        print(f"  C({t}) = {m:+.5f} ± {s:.5f}      {refs[t]:+.5f}      "
              f"{sigmas:.1f}σ")
    print()
    print("For comparison — long run @ a_τ=0.125 with n_per_sector=5:")
    print("  C(0.0) = +0.0006±0.0003  (45σ off ED)")
    print("  C(0.5) = -0.0126±0.0007  (0σ — matches)")
    print("  C(1.0) = -0.0062±0.0009  (22σ off, wrong sign)")
    print("  C(2.0) = +0.0001±0.0010  (13σ off)")


if __name__ == "__main__":
    main()
