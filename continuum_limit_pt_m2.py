"""
continuum_limit_pt_m2.py — methodology-paper continuum-limit study at m=2,
using PARALLEL TEMPERING (exact-det) across K_E to overcome the mixing crisis
seen at small a_τ (high K_E_target).

For each a_τ ∈ {0.5, 0.25, 0.125} at β=4 fixed:
  1. Action couplings scaled per a_τ:
       K_M = a_τ · g_M_Ham = a_τ · 0.5
       K_E_target = -(1/2) log tanh(a_τ · g_E_Ham)
       m_action = a_τ · m_Ham = a_τ · 2.0
  2. PT-exact MC: 8-replica K_E ladder, target = K_E for this a_τ.
  3. Phase 2+3: corner sampling + cached U†OU lookups → C(t).
  4. Block top-replica configs into n_blocks for SEM.
  5. Compare to ED reference (unprojected, full staggered KS) at m=2, β=4.

Monkey-patches M_MASS=2.0 in epoq_quantum_simulator (QC H) and
epoq_classical_sampler (reference) before importing dependents.
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
CKPT_DIR = "/tmp/continuum_pt_m2_ckpt"
os.makedirs(CKPT_DIR, exist_ok=True)


def build_K_E_ladder(K_E_target, n_replicas=8, K_E_low_min=0.30,
                      K_E_target_min_span=0.10):
    """8-replica linear ladder from K_E_low to K_E_target."""
    K_E_low = max(K_E_low_min, K_E_target - 0.7)
    if K_E_low >= K_E_target - K_E_target_min_span:
        K_E_low = K_E_target - K_E_target_min_span
    return list(np.linspace(K_E_low, K_E_target, n_replicas))


def measure_C_t_blocked(geom, configs, sign_hist, a_tau, m_obs, times,
                          n_per_sector, cached_UOU, n_blocks, seed):
    """Phase 2+3 on a list of gauge configs, blocked into n_blocks for SEM."""
    n_cfg = len(configs)
    if n_cfg < n_blocks * 2:
        n_blocks = max(2, n_cfg // 2)
    block_size = n_cfg // n_blocks
    rng = np.random.default_rng(seed)
    block_Cs = {t: [] for t in times}
    block_denoms = []
    n_diag_total = 0
    for b in range(n_blocks):
        lo = b * block_size
        hi = (b + 1) * block_size if b < n_blocks - 1 else n_cfg
        sum_num_t = {t: 0.0 for t in times}
        sum_denom = 0.0
        for U_idx in range(lo, hi):
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
        block_denoms.append(sum_denom)
        if abs(sum_denom) > 1e-12:
            for t in times:
                block_Cs[t].append(sum_num_t[t] / sum_denom)
    return block_Cs, block_denoms, n_diag_total


def run_at_a_tau(a_tau, n_sweeps, n_warmup, swap_every, n_per_sector,
                  times, cached_UOU, n_blocks, base_seed=2026):
    """PT-exact at one a_τ + Phase 2+3. Returns dict."""
    K_E_target = -0.5 * np.log(np.tanh(a_tau * G_E_HAM))
    K_M = a_tau * G_M_HAM
    N_E = int(round(BETA / a_tau)) + 1
    m_action = a_tau * M_HAM
    geom = LatticeGeometry(Lx=2, Ly=2, N_E=N_E, m=m_action)
    K_E_ladder = build_K_E_ladder(K_E_target)

    print(f"\n  a_τ={a_tau:.4f}, N_E={N_E}, K_E_target={K_E_target:.4f}, "
          f"K_M={K_M:.4f}, m_action={m_action:.4f}")
    print(f"  ladder ({len(K_E_ladder)}): "
          f"[{', '.join(f'{k:.3f}' for k in K_E_ladder)}]")

    t0 = time.time()
    pt_results, swap_stats = run_metropolis_pt(
        geom, K_E_ladder=K_E_ladder, K_M=K_M,
        n_sweeps=n_sweeps, n_warmup=n_warmup,
        swap_every=swap_every, seed=base_seed,
        action_type='forward',
    )
    t_mc = time.time() - t0
    top = pt_results[-1]
    print(f"  PT MC walltime: {t_mc:.0f}s, top-replica configs: {len(top.configs)}")
    print(f"  swap rates: [{', '.join(f'{r:.3f}' for r in swap_stats.rates)}]")

    block_Cs, block_denoms, n_diag = measure_C_t_blocked(
        geom, top.configs, top.sign_history, a_tau=a_tau, m_obs=M_HAM,
        times=times, n_per_sector=n_per_sector, cached_UOU=cached_UOU,
        n_blocks=n_blocks, seed=base_seed + 7777,
    )
    return {
        'K_E_target': K_E_target, 'K_M': K_M, 'K_E_ladder': K_E_ladder,
        'swap_rates': swap_stats.rates.tolist(),
        'mc_walltime': t_mc, 'n_top_configs': len(top.configs),
        'block_Cs': block_Cs, 'block_denoms': block_denoms,
        'n_diag': n_diag,
    }


def main():
    a_tau_list = [0.5, 0.25, 0.125]
    times = [0.0, 0.5, 1.0, 2.0]
    n_sweeps = 2000
    n_warmup = 500
    swap_every = 5
    n_per_sector = 5
    n_blocks = 8

    print("=" * 78)
    print(f"Continuum-limit study at m={M_HAM}, β={BETA}")
    print(f"  PT-exact (8-replica K_E ladder), n_sweeps={n_sweeps}, "
          f"warmup={n_warmup}, swap_every={swap_every}")
    print(f"  Phase 2+3 blocked into {n_blocks} blocks for SEM")
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

    results = {}
    for a_tau in a_tau_list:
        ckpt = os.path.join(CKPT_DIR, f"a_tau_{a_tau:.4f}.pkl")
        if os.path.exists(ckpt):
            with open(ckpt, 'rb') as f:
                results[a_tau] = pickle.load(f)
            print(f"\n[ckpt] loaded a_τ={a_tau}")
            continue
        print(f"\n--- a_τ = {a_tau} ---")
        r = run_at_a_tau(
            a_tau, n_sweeps, n_warmup, swap_every, n_per_sector,
            times, cached_UOU, n_blocks, base_seed=2026,
        )
        results[a_tau] = r
        with open(ckpt, 'wb') as f:
            pickle.dump(r, f)

    print()
    print("=" * 90)
    print(f"SUMMARY  (PT-exact at m={M_HAM}, β={BETA}, "
          f"n_sweeps={n_sweeps}, swap_every={swap_every})")
    print("=" * 90)
    hdr = f"{'a_τ':>8} | {'K_E_top':>8} | {'n_diag':>7} |"
    hdr += "".join(f" {'C('+f'{t:.1f}'+')':>17}" for t in times)
    print(hdr)
    for a_tau in a_tau_list:
        if a_tau not in results:
            continue
        r = results[a_tau]
        row = f"{a_tau:>8.4f} | {r['K_E_target']:>8.4f} | {r['n_diag']:>7d} |"
        for t in times:
            arr = np.array(r['block_Cs'][t])
            arr = arr[np.isfinite(arr)]
            if len(arr) >= 2:
                m = arr.mean()
                s = arr.std(ddof=1) / np.sqrt(len(arr))
                row += f"  {m:+.4f}±{s:.4f}"
            else:
                row += f"       n/a       "
        print(row)
    row = f"{'':>8} | {'':>8} | {'ED ref':>7} |"
    row += "".join(f"  {refs[t]:>+15.4f}" for t in times)
    print(row)
    print()
    for a_tau in a_tau_list:
        if a_tau not in results:
            continue
        r = results[a_tau]
        print(f"a_τ={a_tau}: swap rates = "
              f"[{', '.join(f'{x:.2f}' for x in r['swap_rates'])}]")


if __name__ == "__main__":
    main()
