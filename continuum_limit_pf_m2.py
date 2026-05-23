"""
continuum_limit_pf_m2.py — methodology-paper continuum-limit study at m=2,
using PF-Hasenbusch MC as the QCD-scalable production MC.

For each a_τ ∈ {0.5, 0.25, 0.125} at β=4 fixed:
  1. Action couplings scaled per a_τ:
       K_M = a_τ · g_M_Ham = a_τ · 0.5
       K_E = -(1/2) log tanh(a_τ · g_E_Ham)
       m_action = a_τ · m_Ham = a_τ · 2.0
  2. Gauge MC: Hasenbusch (Δm=2) MULTI-CHAIN.
  3. Phase 2+3: corner sampling + cached U†OU lookups → C(t).
  4. Report inter-chain mean ± SEM at each t.
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
from multiprocessing import Pool

sys.path.insert(0, '/home/hlamm/Desktop/QC/logdet/m1_toy')

# Set M_MASS BEFORE any module imports it
import epoq_quantum_simulator
import epoq_classical_sampler
epoq_quantum_simulator.M_MASS = 2.0
epoq_classical_sampler.M_MASS = 2.0

from action_z2_staggered import LatticeGeometry, Z2GaugeConfig
from action_z2_metropolis import heatbath_sweep
from action_z2_pf_hasenbusch import metropolis_sweep_hasenbusch
from action_corner_direct_v3 import sample_corner_pairs_stratified
from action_minkowski_stitch import qc_basis_index, static_observable_diagonal
from epoq_quantum_simulator import build_trottered_UOU_cache


M_HAM = 2.0
G_E_HAM = 1.0
G_M_HAM = 0.5
BETA = 4.0
CKPT_DIR = "/tmp/continuum_pf_m2_ckpt"
os.makedirs(CKPT_DIR, exist_ok=True)


def _run_chain_worker(args):
    """Worker: one MC chain at (a_τ, K_E, K_M, mode, seed) → contribs to C(t)."""
    try:
        from threadpoolctl import threadpool_limits
        _ctx = threadpool_limits(limits=1)
    except Exception:
        _ctx = None
    try:
        from action_z2_kernels import warm_up
        warm_up()
    except Exception:
        pass

    (geom_args, a_tau, K_E, K_M, mode, n_sweeps, n_warmup, n_per_sector,
     times, seed, cached_UOU) = args

    import numpy as _np
    from action_z2_staggered import LatticeGeometry as LG, Z2GaugeConfig as ZC
    from action_z2_metropolis import heatbath_sweep as hb
    from action_z2_pf_hasenbusch import metropolis_sweep_hasenbusch as hbh
    from action_corner_direct_v3 import sample_corner_pairs_stratified as sample_corner
    from action_minkowski_stitch import qc_basis_index, static_observable_diagonal

    geom = LG(**geom_args)
    rng = _np.random.default_rng(seed)
    U = ZC.random(geom, rng)

    # Run MC, then for each measured config do Phase 2+3
    sum_num_t = {t: 0.0 for t in times}
    sum_denom = 0.0
    n_diag_total = 0
    rng_obs = _np.random.default_rng(seed + 1000)

    for sw in range(n_sweeps + n_warmup):
        if mode == 'exact':
            U, dm, sg, _ = hb(geom, U, K_E=K_E, K_M=K_M, rng=rng,
                              action_type='forward')
        else:  # 'hasenbusch'
            U, dm, sg, _ = hbh(geom, U, K_E=K_E, K_M=K_M, rng=rng,
                                delta_m=2.0, refresh='per_link')
        if sw < n_warmup:
            continue
        sign_g = 1.0 if dm > 0 else (-1.0 if dm < 0 else 0.0)
        # Per-config corner sampling
        try:
            pairs, signs, sector_w, _ = sample_corner(
                geom, U, n_per_sector=n_per_sector, rng=rng_obs,
                a_tau=a_tau, m_obs=M_HAM, g_hop=0.5)
        except RuntimeError:
            continue
        for k in range(len(pairs)):
            psi_i = int(pairs[k, 0]); psi_j = int(pairs[k, 1])
            sl_sign = float(signs[k]); sw_ = float(sector_w[k])
            bits_top = qc_basis_index(geom, U, geom.N_E - 1, psi_i)
            bits_bot = qc_basis_index(geom, U, 0, psi_j)
            is_diag = (bits_top == bits_bot)
            n0_at_0 = static_observable_diagonal(geom, U, psi_j)
            w = sign_g * sl_sign * sw_
            if is_diag:
                sum_denom += w; n_diag_total += 1
            for t in times:
                mat_el = cached_UOU[t][bits_bot, bits_top]
                sum_num_t[t] += w * n0_at_0 * float(mat_el.real)
    return {'sum_num_t': sum_num_t, 'sum_denom': sum_denom,
            'n_diag': n_diag_total}


def run_at_a_tau(a_tau, mode, n_chains, n_sweeps, n_warmup, n_per_sector,
                  times, cached_UOU, base_seed=2026):
    """Multi-chain MC + Phase 2+3 at one a_τ. Returns (chain_C_dict, chain_denom_list)."""
    K_E = -0.5 * np.log(np.tanh(a_tau * G_E_HAM))
    K_M = a_tau * G_M_HAM
    N_E = int(round(BETA / a_tau)) + 1
    m_action = a_tau * M_HAM
    geom_args = dict(Lx=2, Ly=2, N_E=N_E, m=m_action)
    print(f"\n  a_τ={a_tau:.4f}, N_E={N_E}, K_E={K_E:.4f}, K_M={K_M:.4f}, "
          f"m_action={m_action:.4f}, mode={mode}, n_chains={n_chains}")

    args_list = [
        (geom_args, a_tau, K_E, K_M, mode, n_sweeps, n_warmup, n_per_sector,
         list(times), base_seed + ch, cached_UOU)
        for ch in range(n_chains)
    ]
    t0 = time.time()
    with Pool(processes=n_chains) as pool:
        results = pool.map(_run_chain_worker, args_list)
    t_wall = time.time() - t0
    # Per-chain C(t) for inter-chain SEM
    chain_Cs = {t: [] for t in times}
    for r in results:
        if abs(r['sum_denom']) > 1e-12:
            for t in times:
                chain_Cs[t].append(r['sum_num_t'][t] / r['sum_denom'])
    print(f"  walltime: {t_wall:.0f}s")
    return chain_Cs, [r['sum_denom'] for r in results]


def main():
    a_tau_list = [0.5, 0.25, 0.125]
    times = [0.0, 0.5, 1.0, 2.0]
    n_chains = 8
    n_sweeps = 500
    n_warmup = 300
    n_per_sector = 5

    print("=" * 78)
    print(f"Continuum-limit study at m={M_HAM}, β={BETA}")
    print(f"  PF-Hasenbusch + exact-det cross-check pipeline")
    print(f"  {n_chains} chains × {n_sweeps} sweeps + {n_warmup} warmup each")
    print("=" * 78)

    # Cached U†OU at m=2 (after monkey-patch)
    print("Building U†OU cache at m=2...")
    t0 = time.time()
    cached_UOU = build_trottered_UOU_cache(times, n_trotter=200, order=2)
    print(f"  cache: {time.time()-t0:.1f}s")

    # ED reference
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

    # Run at each a_τ for both MC methods
    results = {}
    for a_tau in a_tau_list:
        ckpt = os.path.join(CKPT_DIR, f"a_tau_{a_tau:.4f}.pkl")
        if os.path.exists(ckpt):
            with open(ckpt, 'rb') as f:
                results[a_tau] = pickle.load(f)
            print(f"\n[ckpt] loaded a_τ={a_tau}")
            continue
        print(f"\n--- a_τ = {a_tau} ---")
        a_tau_results = {}
        for mode in ['exact', 'hasenbusch']:
            chain_Cs, chain_denoms = run_at_a_tau(
                a_tau, mode, n_chains, n_sweeps, n_warmup, n_per_sector,
                times, cached_UOU, base_seed=2026)
            a_tau_results[mode] = {'chain_Cs': chain_Cs, 'denoms': chain_denoms}
        results[a_tau] = a_tau_results
        with open(ckpt, 'wb') as f:
            pickle.dump(a_tau_results, f)

    # Summary table
    print()
    print("=" * 78)
    print(f"SUMMARY  (ED ref unprojected at m={M_HAM}, β={BETA})")
    print("=" * 78)
    print(f"{'a_τ':>8} | {'mode':>10} |" +
          "".join(f" {'C('+f'{t:.1f}'+')':>15}" for t in times))
    for a_tau in a_tau_list:
        if a_tau not in results:
            continue
        for mode in ['exact', 'hasenbusch']:
            row_str = f"{a_tau:>8.4f} | {mode:>10} |"
            for t in times:
                arr = np.array(results[a_tau][mode]['chain_Cs'][t])
                arr = arr[np.isfinite(arr)]
                if len(arr) >= 2:
                    m = arr.mean()
                    s = arr.std(ddof=1) / np.sqrt(len(arr))
                    row_str += f" {m:+.4f}±{s:.4f}"
                else:
                    row_str += f"     n/a       "
            print(row_str)
    print(f"{'':>8} | {'ED ref':>10} |" +
          "".join(f" {refs[t]:>+15.4f}" for t in times))


if __name__ == "__main__":
    main()
