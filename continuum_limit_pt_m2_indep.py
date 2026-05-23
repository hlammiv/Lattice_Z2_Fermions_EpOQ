"""
continuum_limit_pt_m2_indep.py — continuum-limit study at m=2 with
N_INSTANCES independent PT-exact chains (different seeds). Inter-PT-instance
SEM = true statistical error (no within-chain autocorrelation bias).

For each a_τ ∈ {0.5, 0.25, 0.125} at β=4 fixed:
  Run N_INSTANCES separate PT runs (seed = base_seed + i * 11111).
  Each PT: 8-replica K_E ladder, n_sweeps + n_warmup at swap_every.
  Take top-replica configs from each, run Phase 2+3 → one C(t) per instance.
  Mean ± SEM across N_INSTANCES at each t.

Sequential across PT instances; each PT uses Pool(8) internally, well below
20-CPU local box. Monkey-patches M_MASS=2.0 in epoq_quantum_simulator and
epoq_classical_sampler before importing dependents.
"""
from __future__ import annotations
import sys
import time
import pickle
import os
import numpy as np

sys.path.insert(0, '/home/hlamm/Desktop/QC/logdet/m1_toy')

import epoq_quantum_simulator
import epoq_classical_sampler
epoq_quantum_simulator.M_MASS = 2.0
epoq_classical_sampler.M_MASS = 2.0

from action_z2_staggered import LatticeGeometry
from action_z2_metropolis_pt import run_metropolis_pt
from action_corner_direct_v3 import sample_corner_pairs_stratified
from action_minkowski_stitch import qc_basis_index, static_observable_diagonal
from epoq_quantum_simulator import build_trottered_UOU_cache


M_HAM = 2.0
G_E_HAM = 1.0
G_M_HAM = 0.5
BETA = 4.0
N_INSTANCES = 4
CKPT_DIR = "/tmp/continuum_pt_m2_indep_ckpt"
os.makedirs(CKPT_DIR, exist_ok=True)


def build_K_E_ladder(K_E_target, n_replicas=8, K_E_low_min=0.30,
                      K_E_target_min_span=0.10):
    K_E_low = max(K_E_low_min, K_E_target - 0.7)
    if K_E_low >= K_E_target - K_E_target_min_span:
        K_E_low = K_E_target - K_E_target_min_span
    return list(np.linspace(K_E_low, K_E_target, n_replicas))


def measure_C_t(geom, configs, sign_hist, a_tau, m_obs, times,
                  n_per_sector, cached_UOU, seed):
    """Phase 2+3 over a list of configs → one C(t) dict."""
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
                a_tau=a_tau, m_obs=m_obs, g_hop=0.5,
            )
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
    return {t: sum_num_t[t] / sum_denom for t in times}, n_diag_total


def run_one_PT(a_tau, n_sweeps, n_warmup, swap_every, n_per_sector,
                 times, cached_UOU, seed):
    K_E_target = -0.5 * np.log(np.tanh(a_tau * G_E_HAM))
    K_M = a_tau * G_M_HAM
    N_E = int(round(BETA / a_tau)) + 1
    m_action = a_tau * M_HAM
    geom = LatticeGeometry(Lx=2, Ly=2, N_E=N_E, m=m_action)
    K_E_ladder = build_K_E_ladder(K_E_target)
    t0 = time.time()
    pt_results, swap_stats = run_metropolis_pt(
        geom, K_E_ladder=K_E_ladder, K_M=K_M,
        n_sweeps=n_sweeps, n_warmup=n_warmup,
        swap_every=swap_every, seed=seed,
        action_type='forward',
    )
    t_mc = time.time() - t0
    top = pt_results[-1]
    Cs, n_diag = measure_C_t(
        geom, top.configs, top.sign_history,
        a_tau=a_tau, m_obs=M_HAM, times=times,
        n_per_sector=n_per_sector, cached_UOU=cached_UOU,
        seed=seed + 7777,
    )
    return {'C': Cs, 'n_diag': n_diag, 'mc_walltime': t_mc,
            'swap_rates': swap_stats.rates.tolist(),
            'K_E_target': K_E_target, 'K_M': K_M,
            'K_E_ladder': K_E_ladder, 'n_top_configs': len(top.configs)}


def run_at_a_tau(a_tau, n_instances, n_sweeps, n_warmup, swap_every,
                  n_per_sector, times, cached_UOU, base_seed=2026):
    print(f"\n--- a_τ = {a_tau} (N_INSTANCES={n_instances}) ---")
    instance_results = []
    for i in range(n_instances):
        seed_i = base_seed + i * 11111
        print(f"  instance {i+1}/{n_instances} (seed={seed_i})...", flush=True)
        r = run_one_PT(
            a_tau, n_sweeps, n_warmup, swap_every, n_per_sector,
            times, cached_UOU, seed=seed_i,
        )
        Cs = r['C']
        print(f"    walltime={r['mc_walltime']:.0f}s  n_diag={r['n_diag']}  "
              f"C(0)={Cs[times[0]]:+.4f}  C({times[1]})={Cs[times[1]]:+.4f}  "
              f"C({times[2]})={Cs[times[2]]:+.4f}  C({times[3]})={Cs[times[3]]:+.4f}")
        print(f"    swap rates: [{', '.join(f'{x:.2f}' for x in r['swap_rates'])}]",
              flush=True)
        instance_results.append(r)
    return instance_results


def main():
    a_tau_list = [0.5, 0.25, 0.125]
    times = [0.0, 0.5, 1.0, 2.0]
    n_sweeps = 2000
    n_warmup = 500
    swap_every = 5
    n_per_sector = 5

    print("=" * 78)
    print(f"Continuum-limit @ m={M_HAM}, β={BETA}, "
          f"PT-exact × {N_INSTANCES} INDEPENDENT instances per a_τ")
    print(f"  n_sweeps={n_sweeps}, n_warmup={n_warmup}, "
          f"swap_every={swap_every}, n_per_sector={n_per_sector}")
    print("=" * 78)

    print("Building U†OU cache at m=2...")
    t0 = time.time()
    cached_UOU = build_trottered_UOU_cache(times, n_trotter=200, order=2)
    print(f"  cache: {time.time()-t0:.1f}s")

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
    rho = expm(-BETA * H); Z = np.trace(rho).real
    refs = {}
    for t in times:
        Ut = expm(-1j * H * t)
        n0_t = Ut.conj().T @ n0 @ Ut
        refs[t] = (np.trace(rho @ n0_t @ n0) / Z).real
    print(f"\nED reference (unprojected, m={M_HAM}, β={BETA}):")
    for t in times:
        print(f"  C({t}) = {refs[t]:+.5f}")

    all_results = {}
    for a_tau in a_tau_list:
        ckpt = os.path.join(CKPT_DIR, f"a_tau_{a_tau:.4f}.pkl")
        if os.path.exists(ckpt):
            with open(ckpt, 'rb') as f:
                all_results[a_tau] = pickle.load(f)
            print(f"\n[ckpt] loaded a_τ={a_tau}")
            continue
        instances = run_at_a_tau(
            a_tau, N_INSTANCES, n_sweeps, n_warmup, swap_every,
            n_per_sector, times, cached_UOU, base_seed=2026,
        )
        all_results[a_tau] = instances
        with open(ckpt, 'wb') as f:
            pickle.dump(instances, f)

    print()
    print("=" * 92)
    print(f"SUMMARY  (PT-exact m={M_HAM} β={BETA}, "
          f"{N_INSTANCES} independent PT instances per a_τ)")
    print("=" * 92)
    hdr = f"{'a_τ':>8} | {'K_E_top':>8} |"
    hdr += "".join(f"  {'C('+f'{t:.1f}'+')':>16}" for t in times)
    print(hdr)
    for a_tau in a_tau_list:
        instances = all_results.get(a_tau)
        if not instances:
            continue
        K_E_target = instances[0]['K_E_target']
        row = f"{a_tau:>8.4f} | {K_E_target:>8.4f} |"
        for t in times:
            vals = np.array([inst['C'][t] for inst in instances])
            vals = vals[np.isfinite(vals)]
            if len(vals) >= 2:
                m = vals.mean()
                s = vals.std(ddof=1) / np.sqrt(len(vals))
                row += f"   {m:+.4f}±{s:.4f}"
            else:
                row += f"        n/a       "
        print(row)
    row = f"{'':>8} | {'ED ref':>8} |"
    row += "".join(f"   {refs[t]:>+15.4f}" for t in times)
    print(row)
    print()
    for a_tau in a_tau_list:
        instances = all_results.get(a_tau)
        if not instances:
            continue
        for i, inst in enumerate(instances):
            print(f"a_τ={a_tau} instance {i+1}: swap rates "
                  f"[{', '.join(f'{x:.2f}' for x in inst['swap_rates'])}], "
                  f"n_diag={inst['n_diag']}")


if __name__ == "__main__":
    main()
