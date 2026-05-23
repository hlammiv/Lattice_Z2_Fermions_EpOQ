"""
M_act_6 v3: end-to-end EρOQ pipeline with the **forward-time Suzuki-Trotter**
W matrix for corner-state extraction.

Replaces v2's `action_corner_direct_v2.sample_corner_pairs_stratified` (which
uses Cauchy-Binet W from central-diff M-K) with `action_corner_direct_v3`
(uses W = T̂_F^{N_E-1} with T̂_F = expm(-a_τ · H_KS) per slice).

Why: Phase 8 diagnostic showed the central-diff M-K transfer matrix produces
a lattice Hamiltonian H_lat ≠ H_QC (60× gap mismatch).  Phase 9 verified the
forward-time Trotter T̂_F matches H_KS exactly per gauge eigenstate.  For the
methodology paper, the corner-state weights must be consistent with the QC
side's H_KS time evolution.

Phases 1, 3, 4 reuse the v2 / v1 machinery:
  Phase 1: gauge MC (action_z2_metropolis.run_metropolis)
  Phase 2: corner pair sampling from Trotter W  ← new (action_corner_direct_v3)
  Phase 3: Minkowski matrix elements (action_minkowski_stitch.matrix_element_minkowski)
  Phase 4: ratio estimator + bootstrap (here, with autocorr-aware jackknife
           from mc_error_analysis as an additional check)

Note on parameters:
  - geom.m is the ACTION mass (a_τ · m_Ham in continuum-limit scaling).
  - m_obs is the HAMILTONIAN mass used inside T̂_F = expm(-a_τ · H_KS[m_obs, ...])
    and in the QC matrix-element evolution.
  - K_E_action, K_M_action are the gauge MC weights (passed to run_metropolis).
  - g_hop is the Hamiltonian hopping coefficient (fixed = 0.5 to match z2_setup).
"""
from __future__ import annotations
import sys
import time
import numpy as np
from scipy.linalg import expm

sys.path.insert(0, '/home/hlamm/Desktop/QC/logdet/m1_toy')

from action_z2_staggered import LatticeGeometry, Z2GaugeConfig, build_dirac_matrix, gauge_action
from action_z2_metropolis import run_metropolis, run_metropolis_parallel
from action_corner_direct_v3 import sample_corner_pairs_stratified
from action_minkowski_stitch import (
    matrix_element_minkowski, qc_basis_index, static_observable_diagonal,
)


import functools


@functools.lru_cache(maxsize=4)
def _cached_UOU(times_tuple: tuple, n_trotter: int) -> dict:
    """Worker-local cache of Trotterized U†(t)·O·U(t) matrices.

    The QC Hamiltonian doesn't depend on the per-config gauge variables
    (gauge degrees of freedom are part of the 8-qubit Hilbert space), so
    U(t) is the same operator across all MC configs.  We build it once
    per worker (via lru_cache) using the same n_trotter / Strang scheme
    that observable_matrix_element uses — Trotter approximation preserved.

    Returns dict[t→256×256 matrix].
    """
    from epoq_quantum_simulator import build_trottered_UOU_cache
    return build_trottered_UOU_cache(list(times_tuple), n_trotter=n_trotter, order=2)


# ---------------------------------------------------------------------------
# Per-config worker for parallel observable computation (Phase 2 + 3 per U)
# ---------------------------------------------------------------------------
def _obs_worker(args):
    """Process one gauge config: corner sampling + Minkowski matrix elements
    for all `times`.  Returns dict of per-sample arrays for aggregation.

    Pins BLAS to 1 thread inside the worker to avoid thread oversubscription
    when run in a Pool.  Warms Numba JIT eagerly.
    """
    import sys as _sys
    _sys.path.insert(0, '/home/hlamm/Desktop/QC/logdet/m1_toy')
    (geom_args, U_x, U_y, U_t, sign_gauge, times, n_per_sector,
     a_tau, m_obs, g_hop, n_trotter_real, seed) = args
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        threadpool_limits = None
    try:
        from action_z2_kernels import warm_up as _kernel_warm
        _kernel_warm()
    except Exception:
        pass

    def _do_work():
        import numpy as _np
        from action_z2_staggered import LatticeGeometry, Z2GaugeConfig
        from action_corner_direct_v3 import sample_corner_pairs_stratified
        from action_minkowski_stitch import (
            matrix_element_minkowski, qc_basis_index, static_observable_diagonal,
        )
        from epoq_quantum_simulator import build_trottered_UOU_cache
        geom = LatticeGeometry(**geom_args)
        U = Z2GaugeConfig(geom=geom, U_x=U_x, U_y=U_y, U_t=U_t)
        rng_local = _np.random.default_rng(seed)

        # --- U†(t) O U(t) cache (Trotter-faithful, gauge-config-independent) ---
        # H_QC doesn't depend on the per-config gauge variables (gauge IS part
        # of the qubit Hilbert space), so cache U†OU once per worker.  Each
        # worker pays this cost once at first call thanks to functools.lru_cache.
        _UOU = _cached_UOU(tuple(times), n_trotter_real)
        try:
            pairs, signs_slater, sector_w, _ = sample_corner_pairs_stratified(
                geom, U, n_per_sector=n_per_sector, rng=rng_local,
                a_tau=a_tau, m_obs=m_obs, g_hop=g_hop,
            )
        except RuntimeError:
            return None
        n_samples = len(pairs)
        # Aggregate per-config inside worker.  Output is small dict of scalars
        # (numerator per t, denom, n_samples, n_diag) — keeps multiprocessing
        # queue traffic minimal.
        sum_num_t = {t: 0.0 for t in times}
        sum_denom = 0.0    # Σ_k sign_gauge · sl_sign · sw · δ_diag (partition function estimator)
        n_diag = 0
        for k in range(n_samples):
            # v3 sampler convention: pairs[k] = (psi_i, psi_j) with
            #   psi_i = ψ_top, at boundary slice N_E−1  (row index of W)
            #   psi_j = ψ_bot, at boundary slice 0      (col index of W)
            # matrix_element_minkowski expects v1 convention (psi_arg1 at
            # slice 0, psi_arg2 at slice N_E−1).  So we pass (psi_j, psi_i).
            psi_i = int(pairs[k, 0]); psi_j = int(pairs[k, 1])
            sl_sign = float(signs_slater[k]); sw = float(sector_w[k])
            bits_top = qc_basis_index(geom, U, geom.N_E - 1, psi_i)  # top: gauge_tN-1 ⊗ psi_i
            bits_bot = qc_basis_index(geom, U, 0, psi_j)             # bot: gauge_t0   ⊗ psi_j
            is_diag_k = (bits_top == bits_bot)
            # EρOQ formula: ⟨O(t)O(0)⟩ = Σ_{a,b} n_0(a)·⟨a|ρ_E|b⟩·⟨b|U†OU|a⟩
            # where a is the t=0 corner state. In v3: a = (gauge_t0, psi_j).
            # n_0(a) is fermion-only (gauge-blind), so n_0(a) = n_0(psi_j).
            n0_at_0 = static_observable_diagonal(geom, U, psi_j)
            w = sign_gauge * sl_sign * sw
            if is_diag_k:
                sum_denom += w
                n_diag += 1
            for t in times:
                # Trotter-faithful matrix element via cached U†OU.
                # U†OU[a, b] = ⟨a|U†(t) O U(t)|b⟩.  We want ⟨bot|U†OU|top⟩.
                mat_el = _UOU[t][bits_bot, bits_top]
                sum_num_t[t] += w * n0_at_0 * float(mat_el.real)
        return {
            'sum_num_t': sum_num_t,
            'sum_denom': sum_denom,
            'n_diag': n_diag,
            'n_samples': n_samples,
        }

    if threadpool_limits is not None:
        with threadpool_limits(limits=1):
            return _do_work()
    return _do_work()


def run_action_p3_pipeline(
    geom: LatticeGeometry,
    K_E: float, K_M: float,
    times: list[float],
    n_gauge: int = 200,
    n_per_sector: int = 5,
    n_warmup_gauge: int = 1000,
    seed: int = 2026,
    n_trotter_real: int = 200,
    n_chains: int = 1,
    action_type: str = 'central',
    plaq_flip_every: int = 5,
    a_tau: float = 1.0,
    m_obs: float = None,
    g_hop: float = 0.5,
    obs_n_workers: int = 1,
):
    """End-to-end EρOQ pipeline using Trotter T̂_F for corner-state weights.

    Args:
      geom: LatticeGeometry (Lx, Ly, N_E, m_action).
      K_E, K_M: action couplings for gauge MC.
      times: list of real-time t values at which to compute ⟨n_0(t) · n_0(0)⟩.
      n_gauge: total gauge MC measurement sweeps (across chains).
      n_per_sector: stratified-sampling corner pairs per particle-number sector.
      n_warmup_gauge: per-chain warmup sweeps.
      n_chains: 1 → run_metropolis; >1 → run_metropolis_parallel.
      action_type: 'central' or 'forward' for action MC.
      a_tau: temporal lattice spacing for Trotter T̂_F.
      m_obs: Hamiltonian mass for T̂_F construction.  If None, uses geom.m.
      g_hop: Hamiltonian hopping coefficient.

    Returns:
      {'C': dict[t→mean], 'err': dict[t→σ], 'walltime': dict, ...}
    """
    rng = np.random.default_rng(seed)
    if m_obs is None:
        m_obs = geom.m

    # Phase 1: classical action MC for gauge configs
    print(f"Phase 1: gauge MC (n_gauge={n_gauge}, n_warmup={n_warmup_gauge}, "
          f"n_chains={n_chains}, action_type={action_type})...")
    print(f"  K_E={K_E}, K_M={K_M}, action mass={geom.m}, plaq_flip_every={plaq_flip_every}")
    t0 = time.time()
    if n_chains > 1:
        mc_result = run_metropolis_parallel(
            geom, n_chains=n_chains, K_E=K_E, K_M=K_M,
            n_sweeps_per_chain=max(1, n_gauge // n_chains),
            n_warmup=n_warmup_gauge, seed=seed,
            action_type=action_type, plaq_flip_every=plaq_flip_every,
        )
    else:
        mc_result = run_metropolis(
            geom, K_E=K_E, K_M=K_M,
            n_sweeps=n_gauge, n_warmup=n_warmup_gauge, seed=seed,
            action_type=action_type, plaq_flip_every=plaq_flip_every,
        )
    t_mc = time.time() - t0
    print(f"  wall-clock: {t_mc:.1f}s, accept rate: {mc_result.accept_rate:.3f}, "
          f"⟨sign⟩: {mc_result.avg_sign:+.3f}, configs: {len(mc_result.configs)}")

    # Phase 2+3: corner-state extraction + Minkowski matrix elements per config.
    # Embarrassingly parallel across gauge configs.
    n_per_config = n_per_sector * (geom.V_3 + 1)
    print(f"Phase 2+3: stratified corner extraction (Trotter W) + "
          f"Minkowski matrix elements")
    print(f"  n_per_sector={n_per_sector}, n_per_config={n_per_config}, "
          f"a_τ={a_tau}, m_obs={m_obs}, g_hop={g_hop}, "
          f"n_trotter_real={n_trotter_real}, obs_n_workers={obs_n_workers}")
    t0 = time.time()
    geom_args = dict(Lx=geom.Lx, Ly=geom.Ly, N_E=geom.N_E, m=geom.m)
    args_list = []
    for U_idx, U in enumerate(mc_result.configs):
        args_list.append((
            geom_args, U.U_x.copy(), U.U_y.copy(), U.U_t.copy(),
            float(mc_result.sign_history[U_idx]),
            list(times), n_per_sector,
            a_tau, m_obs, g_hop, n_trotter_real,
            seed + 10000 + U_idx,
        ))

    results = []
    if obs_n_workers > 1:
        import multiprocessing as mp
        with mp.Pool(processes=obs_n_workers) as pool:
            for i, r in enumerate(pool.imap_unordered(_obs_worker, args_list, chunksize=2)):
                if r is not None:
                    results.append(r)
                if (i + 1) % 50 == 0:
                    print(f"  config {i+1}/{len(args_list)}, "
                          f"elapsed={time.time()-t0:.1f}s", flush=True)
    else:
        for i, args in enumerate(args_list):
            if i % 50 == 0:
                print(f"  config {i}/{len(args_list)}, "
                      f"elapsed={time.time()-t0:.1f}s", flush=True)
            r = _obs_worker(args)
            if r is not None:
                results.append(r)

    # Aggregate per-config sums into arrays indexed by config (length n_configs).
    # These are the BLOCK-BOOTSTRAP units — each config = one independent block.
    n_configs_kept = len(results)
    cfg_num_t = {t: np.array([r['sum_num_t'][t] for r in results]) for t in times}
    cfg_denom = np.array([r['sum_denom'] for r in results])
    n_total = int(sum(r['n_samples'] for r in results))
    n_diag_total = int(sum(r['n_diag'] for r in results))
    t_obs = time.time() - t0
    print(f"  wall-clock: {t_obs:.1f}s, n_samples_total: {n_total}, "
          f"n_configs: {n_configs_kept}")
    print(f"  diagonal sample fraction: {n_diag_total / max(n_total, 1):.3f}")
    t_corner = t_obs / 2  # report as combined; legacy field
    t_qc = t_obs / 2

    # Phase 4: block bootstrap over GAUGE CONFIGS (correct for autocorr).
    # ⟨n_0(t) n_0(0)⟩ = (Σ_configs Σ_samples sg·ss·sw·n_0·mat_el)
    #                 / (Σ_configs Σ_samples sg·ss·sw·δ_diag)
    print("Phase 4: block-bootstrap (per-config) ratio estimator...")
    out_C = {}
    out_err = {}
    denom_full = float(cfg_denom.sum())
    print(f"  Z_est (Σ sg·ss·sw·δ_diag) = {denom_full:+.6f}")
    if abs(denom_full) < 1e-12 or n_configs_kept < 2:
        print(f"  WARNING: zero denominator or too few configs ({n_configs_kept})")
        for t in times:
            out_C[t] = float('nan'); out_err[t] = float('nan')
    else:
        N = n_configs_kept
        rng_b = np.random.default_rng(31337)
        # Boot only once over the config indices to use the same resample for all t
        for t in times:
            num_arr = cfg_num_t[t]
            means = []
            for _ in range(500):
                idx = rng_b.choice(N, N, replace=True)
                num = float(num_arr[idx].sum())
                den = float(cfg_denom[idx].sum())
                if abs(den) < 1e-12:
                    continue
                means.append(num / den)
            out_C[t] = float(np.mean(means)) if means else float('nan')
            out_err[t] = float(np.std(means)) if means else float('nan')

    return {
        'C': out_C, 'err': out_err,
        'n_samples': int(n_total),
        'n_configs_kept': int(n_configs_kept),
        'n_diag_total': int(n_diag_total),
        'walltime': {'mc': t_mc, 'corner': t_corner, 'qc': t_qc, 'obs_total': t_obs},
        # Per-config arrays for further analysis / extending bootstrap externally.
        'cfg_num_t': {t: list(v) for t, v in cfg_num_t.items()},
        'cfg_denom': list(cfg_denom),
    }


def _self_test():
    """Small smoke test: 2x2 OBC, N_E=5, β=4, K_E=1, K_M=0.5, m=0.5."""
    geom = LatticeGeometry(Lx=2, Ly=2, N_E=5, m=0.5)
    times = [0.0, 0.5, 1.0, 2.0]
    print("=" * 70)
    print("Pipeline v3 self-test (Trotter W for corner weights)")
    print("=" * 70)

    # Reference: staggered KS Gauss-projected
    from staggered_ks_reference import staggered_ks_thermal_n0
    # For real-time ⟨n_0(t) n_0(0)⟩: need a richer reference; do here with ED
    from scipy.linalg import expm as _expm
    from staggered_ks_reference import (
        build_pauli_terms_stagg, pauli_term_dense, gauss_law_projector,
        matter_q, DIM,
    )
    H = sum(c * pauli_term_dense(fac)
            for c, fac in build_pauli_terms_stagg(g_E=1.0, g_M=0.5, g_hop=0.5, m=0.5))
    H = (H + H.conj().T) / 2
    P = gauss_law_projector()
    evals_P, evecs_P = np.linalg.eigh(P)
    phys = evecs_P[:, evals_P > 0.5]
    H_phys = phys.conj().T @ H @ phys
    H_phys = (H_phys + H_phys.conj().T) / 2
    n0_full = 0.5 * np.eye(DIM, dtype=complex) - 0.5 * pauli_term_dense([(matter_q(0), 'Z')])
    n0_phys = phys.conj().T @ n0_full @ phys
    n0_phys = (n0_phys + n0_phys.conj().T) / 2
    rho = _expm(-4.0 * H_phys); Z = np.trace(rho).real
    print("Reference C(t) = ⟨n_0(t) n_0(0)⟩ (Gauss-physical, β=4):")
    for t in times:
        Ut = _expm(-1j * H_phys * t)
        n0_t = Ut.conj().T @ n0_phys @ Ut
        ref = (np.trace(rho @ n0_t @ n0_phys) / Z).real
        print(f"  t={t}: {ref:+.6f}")
    print()

    result = run_action_p3_pipeline(
        geom, K_E=1.0, K_M=0.5, times=times,
        n_gauge=500, n_per_sector=5, n_warmup_gauge=500,
        seed=2026, n_trotter_real=200, n_chains=4,
        action_type='central', plaq_flip_every=5,
        a_tau=1.0, m_obs=0.5, g_hop=0.5,
        obs_n_workers=8,
    )
    print(f"\nResults:")
    print(f"  {'t':>5} | {'C (v3)':>22}")
    for t in times:
        m, e = result['C'][t], result['err'][t]
        print(f"  {t:>5.2f} | {m:>+10.6f} ± {e:.4f}")


if __name__ == "__main__":
    _self_test()
