"""
M_act_2: Z₂ link Metropolis sampling of action e^(−S_g) det M_OBC.

Single-link proposals; full-det recomputation on each (fine at 2x2 toy scale).
Returns chains of gauge configurations, measured observables, acceptance, etc.
"""

from __future__ import annotations
import time
import numpy as np
from dataclasses import dataclass, field
from typing import List

from action_z2_staggered import (
    LatticeGeometry, Z2GaugeConfig,
    gauge_action, count_links_and_plaquettes,
)

# Forward-time action det (block-structured, QCD-scalable, O(N_E · V_3³)).
# This is the ONLY supported fermion action — see
# memory/feedback_no_central_diff_use_forward.md for why central-diff was removed.
from action_z2_staggered_forwardtime import compute_det_M_forward, build_dirac_matrix_forward


@dataclass
class MCMCResult:
    configs: List[Z2GaugeConfig]
    det_M_history: np.ndarray
    S_g_history: np.ndarray
    sign_history: np.ndarray         # sign(det M) per recorded config
    accept_rate: float
    walltime: float
    avg_sign: float = 0.0            # ⟨sign⟩ over recorded configs


def link_list(geom: LatticeGeometry) -> list[tuple]:
    """List all (link_type, t, x, y) tuples for sweep iteration. link_type ∈ {'x','y','t'}."""
    links = []
    for t in range(geom.N_E):
        for x in range(geom.Lx - 1):
            for y in range(geom.Ly):
                links.append(('x', t, x, y))
    for t in range(geom.N_E):
        for x in range(geom.Lx):
            for y in range(geom.Ly - 1):
                links.append(('y', t, x, y))
    for t in range(geom.N_E - 1):
        for x in range(geom.Lx):
            for y in range(geom.Ly):
                links.append(('t', t, x, y))
    return links


def flip_link(U: Z2GaugeConfig, link_type: str, t: int, x: int, y: int) -> None:
    """In-place flip of σ ↔ -σ at the specified link."""
    if link_type == 'x':
        U.U_x[t, x, y] *= -1
    elif link_type == 'y':
        U.U_y[t, x, y] *= -1
    elif link_type == 't':
        U.U_t[t, x, y] *= -1
    else:
        raise ValueError(link_type)


def _resolve_det_fn(action_type: str):
    """Pick the det-M function for the MC importance weight.

    - 'gauge_only' (CORRECT for EρOQ corner-state pipeline): no det M.
      The path-integral identity for ⟨g_a,ψ_a|e^{-βH}|g_b,ψ_b⟩ with explicit
      boundary fermion states uses W = T_F^{N-1}[U] (open-boundary fermion
      matrix element); det M would double-count (closed-boundary determinant).
      See memory/feedback_no_det_M_with_corner_states.md.  Sign-problem-free.

    - 'forward' (LEGACY / standard dynamical-fermion lattice QCD): includes
      |det M_forward| in importance, reweights observables by sgn(det M).
      Wrong for the corner-state pipeline; kept for cross-checks.  Has the
      ⟨sgn⟩→0 sign problem at a_τ ≲ 0.3.

    - 'central' (REMOVED): central-diff det M ≠ Tr T̂_F^{N-1} at finite a_τ.
      See memory/feedback_no_central_diff_use_forward.md.
    """
    if action_type == 'gauge_only':
        # Dummy det function returning +1 → log|det|=0 in the Metropolis ratio,
        # reducing the importance weight to pure e^{-S_g}.  sign_history
        # is all +1, so no reweighting occurs.
        return lambda geom, U: 1.0
    if action_type == 'forward':
        return compute_det_M_forward
    if action_type == 'central':
        raise ValueError(
            "action_type='central' is removed. Use 'gauge_only' (correct) "
            "or 'forward' (legacy). See feedback_no_central_diff_use_forward "
            "memory note.")
    raise ValueError(
        f"unknown action_type {action_type!r}; "
        f"valid options are 'gauge_only' (correct) and 'forward' (legacy).")


def heatbath_sweep(
    geom: LatticeGeometry,
    U: Z2GaugeConfig,
    K: float = 1.0, K_E: float = None, K_M: float = None,
    rng: np.random.Generator = None,
    action_type: str = 'forward',
    strang_M: bool = False,
) -> tuple[Z2GaugeConfig, float, float, int]:
    """Per-link Z_2 heat-bath sweep.  100% acceptance: each link is sampled
    from its exact conditional P(U_l | others) ∝ exp(-S_g(U_l)) · |det M(U_l)|.

    Sign of det M is tracked separately and can flip per update.
    action_type ∈ {'central', 'forward'} selects the Dirac matrix:
      'central' = central-difference time hop (action_z2_staggered)
      'forward' = forward-time only            (action_z2_staggered_forwardtime)
    Returns updated U, final det M, final S_g, and n_flipped (out of n_links).
    """
    if K_E is None:
        K_E = K
    if K_M is None:
        K_M = K
    det_fn = _resolve_det_fn(action_type)
    links = link_list(geom)
    rng.shuffle(links)
    n_flipped = 0
    cur_det = det_fn(geom, U)
    cur_Sg = gauge_action(geom, U, K=K, K_E=K_E, K_M=K_M, strang_M=strang_M)
    for link_type, t, x, y in links:
        det_cur = cur_det
        Sg_cur = cur_Sg
        flip_link(U, link_type, t, x, y)
        det_flip = det_fn(geom, U)
        Sg_flip = gauge_action(geom, U, K=K, K_E=K_E, K_M=K_M, strang_M=strang_M)
        # Compute conditional probability of FLIPPED state
        # ratio = (e^{-Sg_flip} |det_flip|) / (e^{-Sg_cur} |det_cur|)
        log_w_flip = -Sg_flip + (np.log(abs(det_flip)) if abs(det_flip) > 0 else -np.inf)
        log_w_cur = -Sg_cur + (np.log(abs(det_cur)) if abs(det_cur) > 0 else -np.inf)
        if log_w_flip > log_w_cur:
            p_flip = 1.0 / (1.0 + np.exp(log_w_cur - log_w_flip))
        else:
            p_flip = np.exp(log_w_flip - log_w_cur) / (1.0 + np.exp(log_w_flip - log_w_cur))
        if rng.random() < p_flip:
            cur_det, cur_Sg = det_flip, Sg_flip
            n_flipped += 1
        else:
            flip_link(U, link_type, t, x, y)  # revert (so unchanged)
    return U, cur_det, cur_Sg, n_flipped


def plaquette_flip_sweep(
    geom: LatticeGeometry,
    U: Z2GaugeConfig,
    K: float = 1.0, K_E: float = None, K_M: float = None,
    rng: np.random.Generator = None,
    action_type: str = 'forward',
    strang_M: bool = False,
) -> tuple[Z2GaugeConfig, float, float, int]:
    """Composite Metropolis move: flip all 4 links of a randomly-chosen
    plaquette simultaneously.  The plaquette itself is unchanged (sign(P)
    flips 4 times = same), but neighboring plaquettes change.  Reduces
    autocorrelation in the gauge sector compared to single-link updates.

    Proposes one xy (magnetic) and one xτ and one yτ plaquette flip per
    sweep iteration.  Each is its own inverse → detailed balance trivial.
    """
    if K_E is None:
        K_E = K
    if K_M is None:
        K_M = K
    det_fn = _resolve_det_fn(action_type)

    Lx, Ly, N_E = geom.Lx, geom.Ly, geom.N_E
    n_xy_plaqs = N_E * max(0, Lx - 1) * max(0, Ly - 1)
    n_xt_plaqs = max(0, N_E - 1) * max(0, Lx - 1) * Ly
    n_yt_plaqs = max(0, N_E - 1) * Lx * max(0, Ly - 1)
    n_plaqs_total = n_xy_plaqs + n_xt_plaqs + n_yt_plaqs
    if n_plaqs_total == 0:
        return U, det_fn(geom, U), gauge_action(geom, U, K=K, K_E=K_E, K_M=K_M, strang_M=strang_M), 0

    def flip_xy(t, x, y):
        U.U_x[t, x, y] *= -1
        U.U_y[t, x + 1, y] *= -1
        U.U_x[t, x, y + 1] *= -1
        U.U_y[t, x, y] *= -1

    def flip_xt(t, x, y):
        U.U_x[t, x, y] *= -1
        U.U_t[t, x + 1, y] *= -1
        U.U_x[t + 1, x, y] *= -1
        U.U_t[t, x, y] *= -1

    def flip_yt(t, x, y):
        U.U_y[t, x, y] *= -1
        U.U_t[t, x, y + 1] *= -1
        U.U_y[t + 1, x, y] *= -1
        U.U_t[t, x, y] *= -1

    n_accept = 0
    cur_det = det_fn(geom, U)
    cur_Sg = gauge_action(geom, U, K=K, K_E=K_E, K_M=K_M, strang_M=strang_M)
    # one pass: attempt all plaquettes once each
    plaq_list = []
    for t in range(N_E):
        for x in range(Lx - 1):
            for y in range(Ly - 1):
                plaq_list.append(('xy', t, x, y))
    for t in range(N_E - 1):
        for x in range(Lx - 1):
            for y in range(Ly):
                plaq_list.append(('xt', t, x, y))
    for t in range(N_E - 1):
        for x in range(Lx):
            for y in range(Ly - 1):
                plaq_list.append(('yt', t, x, y))
    rng.shuffle(plaq_list)

    for ptype, t, x, y in plaq_list:
        flipper = {'xy': flip_xy, 'xt': flip_xt, 'yt': flip_yt}[ptype]
        flipper(t, x, y)
        new_det = det_fn(geom, U)
        new_Sg = gauge_action(geom, U, K=K, K_E=K_E, K_M=K_M, strang_M=strang_M)
        log_w_new = -new_Sg + (np.log(abs(new_det)) if abs(new_det) > 0 else -np.inf)
        log_w_cur = -cur_Sg + (np.log(abs(cur_det)) if abs(cur_det) > 0 else -np.inf)
        log_r = log_w_new - log_w_cur
        if log_r >= 0 or rng.random() < np.exp(log_r):
            cur_det, cur_Sg = new_det, new_Sg
            n_accept += 1
        else:
            flipper(t, x, y)  # revert
    return U, cur_det, cur_Sg, n_accept


def metropolis_sweep(
    geom: LatticeGeometry,
    U: Z2GaugeConfig,
    K: float = 1.0, K_E: float = None, K_M: float = None,
    rng: np.random.Generator = None,
    action_type: str = 'forward',
    strang_M: bool = False,
) -> tuple[Z2GaugeConfig, float, float, int]:
    """One Metropolis sweep over all links in random order. Returns updated U,
    final det M, final S_g, and number of accepts.

    Weight: w = e^(-S_g) × |det M|.  Accept on |w_new / w_cur|.
    Sign of det M is tracked separately for reweighting in the observable
    estimator (sign-corrected average).
    """
    if K_E is None:
        K_E = K
    if K_M is None:
        K_M = K
    det_fn = _resolve_det_fn(action_type)
    links = link_list(geom)
    rng.shuffle(links)
    n_accept = 0
    cur_det = det_fn(geom, U)
    cur_Sg = gauge_action(geom, U, K=K, K_E=K_E, K_M=K_M, strang_M=strang_M)
    for link_type, t, x, y in links:
        flip_link(U, link_type, t, x, y)
        new_det = det_fn(geom, U)
        new_Sg = gauge_action(geom, U, K=K, K_E=K_E, K_M=K_M, strang_M=strang_M)
        # Use |det M| in the weight; the sign is recorded for reweighting.
        if abs(cur_det) > 0:
            abs_det_ratio = abs(new_det) / abs(cur_det)
        else:
            abs_det_ratio = 1.0 if abs(new_det) > 0 else 0.0
        log_ratio = -(new_Sg - cur_Sg) + (np.log(abs_det_ratio)
                                          if abs_det_ratio > 0 else -np.inf)
        ratio = np.exp(log_ratio) if log_ratio > -700 else 0.0
        if ratio >= 1 or rng.random() < ratio:
            cur_det, cur_Sg = new_det, new_Sg
            n_accept += 1
        else:
            flip_link(U, link_type, t, x, y)  # revert
    return U, cur_det, cur_Sg, n_accept


def run_metropolis(
    geom: LatticeGeometry,
    K: float = 1.0, K_E: float = None, K_M: float = None,
    n_sweeps: int = 100,
    n_warmup: int = 50,
    seed: int = 2026,
    record_every: int = 1,
    cold_start: bool = False,
    action_type: str = 'forward',
    plaq_flip_every: int = 0,
    strang_M: bool = False,
) -> MCMCResult:
    """Run Z₂ link Metropolis MC, recording config + observables.

    Supports separate K_E (electric, temporal plaquettes) and K_M (magnetic,
    spatial plaquettes) for matching z2_setup's g_e=1, g_m=0.5 conventions.

    Sign of det M is tracked in `sign_history` for reweighting.

    cold_start=True: initialize with all U_link=+1 (ordered config). Useful
    for ordered phases (large K) where random initial → low acceptance.
    """
    if K_E is None:
        K_E = K
    if K_M is None:
        K_M = K
    rng = np.random.default_rng(seed)
    if cold_start:
        U = Z2GaugeConfig.trivial(geom)
    else:
        U = Z2GaugeConfig.random(geom, rng)

    n_links_per_sweep = len(link_list(geom))

    configs: List[Z2GaugeConfig] = []
    det_M_history = []
    S_g_history = []
    sign_history = []
    n_accept_total = 0
    n_attempt_total = 0

    t0 = time.time()
    det_fn = _resolve_det_fn(action_type)
    for sweep in range(n_warmup + n_sweeps):
        U, det_M, S_g, n_acc = heatbath_sweep(
            geom, U, K=K, K_E=K_E, K_M=K_M, rng=rng, action_type=action_type,
            strang_M=strang_M)
        n_accept_total += n_acc
        n_attempt_total += n_links_per_sweep
        # Optional composite move: plaquette-flip Metropolis every K sweeps.
        # Flipping all 4 links of a plaquette preserves that plaquette's S_g
        # contribution (4 sign flips cancel), changes its 4 neighbors' plaqs,
        # and changes det M.  Reduces autocorrelation in the gauge sector.
        if plaq_flip_every > 0 and (sweep + 1) % plaq_flip_every == 0:
            U, det_M, S_g, _ = plaquette_flip_sweep(
                geom, U, K=K, K_E=K_E, K_M=K_M, rng=rng,
                action_type=action_type, strang_M=strang_M)
        if sweep >= n_warmup and (sweep - n_warmup) % record_every == 0:
            U_copy = Z2GaugeConfig(
                geom=geom,
                U_x=U.U_x.copy(),
                U_y=U.U_y.copy(),
                U_t=U.U_t.copy(),
            )
            configs.append(U_copy)
            det_M_history.append(det_M)
            S_g_history.append(S_g)
            sign_history.append(1 if det_M > 0 else (-1 if det_M < 0 else 0))
    walltime = time.time() - t0
    sign_arr = np.array(sign_history)
    return MCMCResult(
        configs=configs,
        det_M_history=np.array(det_M_history),
        S_g_history=np.array(S_g_history),
        sign_history=sign_arr,
        accept_rate=n_accept_total / max(n_attempt_total, 1),
        walltime=walltime,
        avg_sign=float(sign_arr.mean()) if len(sign_arr) else 0.0,
    )


def _chain_worker(args):
    """Worker for run_metropolis_parallel — one independent MC chain.

    Pins BLAS to 1 thread inside the worker (threadpoolctl) so the K-way
    multiprocessing parallelism is not fighting BLAS threading on small
    matrices.  Triggers Numba JIT warm-up so first-config cost is paid
    inside the worker rather than charged to the chain timer.
    """
    (geom_args, K, K_E, K_M, n_sweeps, n_warmup, seed,
     record_every, cold_start, action_type, plaq_flip_every, strang_M) = args
    # Pin BLAS in worker — net positive at small matrix sizes (V_4 ≤ ~50).
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        threadpool_limits = None
    # Numba JIT warm-up (eagerly compile kernels in this worker process).
    try:
        from action_z2_kernels import warm_up
        warm_up()
    except Exception:
        pass
    geom = LatticeGeometry(**geom_args)
    if threadpool_limits is not None:
        with threadpool_limits(limits=1):
            return run_metropolis(
                geom, K=K, K_E=K_E, K_M=K_M,
                n_sweeps=n_sweeps, n_warmup=n_warmup,
                seed=seed, record_every=record_every, cold_start=cold_start,
                action_type=action_type, plaq_flip_every=plaq_flip_every,
                strang_M=strang_M,
            )
    return run_metropolis(
        geom, K=K, K_E=K_E, K_M=K_M, n_sweeps=n_sweeps, n_warmup=n_warmup,
        seed=seed, record_every=record_every, cold_start=cold_start,
        action_type=action_type, plaq_flip_every=plaq_flip_every,
        strang_M=strang_M,
    )


def run_metropolis_parallel(
    geom: LatticeGeometry,
    n_chains: int = 4,
    K: float = 1.0, K_E: float = None, K_M: float = None,
    n_sweeps_per_chain: int = 500,
    n_warmup: int = 2000,
    seed: int = 2026,
    record_every: int = 1,
    cold_start: bool = False,
    action_type: str = 'forward',
    plaq_flip_every: int = 0,
    strang_M: bool = False,
) -> MCMCResult:
    """Run K independent MC chains in parallel processes and concatenate
    their recorded configs into a single MCMCResult.

    Each chain gets a different seed (seed + chain_idx).  BLAS is pinned to
    1 thread per worker (threadpoolctl) to avoid thread oversubscription
    on small matrices.  Numba JIT is warmed up eagerly in each worker.

    Returns the concatenated MCMCResult.  walltime is wall-clock of the
    full parallel run (not the sum of chain times).
    """
    import multiprocessing as mp
    if K_E is None:
        K_E = K
    if K_M is None:
        K_M = K
    geom_args = dict(Lx=geom.Lx, Ly=geom.Ly, N_E=geom.N_E, m=geom.m)
    args_list = [
        (geom_args, K, K_E, K_M, n_sweeps_per_chain, n_warmup,
         seed + chain_idx, record_every, cold_start, action_type,
         plaq_flip_every, strang_M)
        for chain_idx in range(n_chains)
    ]
    t0 = time.time()
    with mp.Pool(processes=n_chains) as pool:
        results = pool.map(_chain_worker, args_list)
    walltime = time.time() - t0

    # Concatenate all chain results
    configs = []
    det_M_history = []
    S_g_history = []
    sign_history = []
    n_accept_total = 0
    n_attempt_total = 0
    for r in results:
        configs.extend(r.configs)
        det_M_history.append(r.det_M_history)
        S_g_history.append(r.S_g_history)
        sign_history.append(r.sign_history)
        n_accept_total += int(r.accept_rate * len(r.configs))
        n_attempt_total += len(r.configs)
    sign_arr = np.concatenate(sign_history) if sign_history else np.array([])
    return MCMCResult(
        configs=configs,
        det_M_history=np.concatenate(det_M_history),
        S_g_history=np.concatenate(S_g_history),
        sign_history=sign_arr,
        accept_rate=(sum(r.accept_rate * len(r.configs) for r in results)
                     / max(sum(len(r.configs) for r in results), 1)),
        walltime=walltime,
        avg_sign=float(sign_arr.mean()) if len(sign_arr) else 0.0,
    )


def average_plaquette(geom: LatticeGeometry, U: Z2GaugeConfig) -> float:
    """⟨P⟩ = (1/N_plaq) Σ_plaq ∏_link σ. Range: [-1, 1]."""
    counts = count_links_and_plaquettes(geom)
    n_plaq = counts['total_plaq']
    Sg_at_K1 = gauge_action(geom, U, K=1.0)
    return -Sg_at_K1 / n_plaq


def _self_test():
    geom = LatticeGeometry(Lx=2, Ly=2, N_E=4, m=0.5)
    K_values = [0.0, 0.5, 1.0, 1.5, 2.0]
    print(f"Geometry: L_x={geom.Lx}, L_y={geom.Ly}, N_E={geom.N_E}, m={geom.m}")
    print(f"V_3 = {geom.V_3}, V_4 = {geom.V_4}")
    print(f"Links: {len(link_list(geom))}, Plaquettes: {count_links_and_plaquettes(geom)['total_plaq']}")
    print()
    print(f"  {'K':>5} | {'accept':>7} | {'⟨P⟩':>9} | {'⟨det M⟩':>12} | {'walltime (s)':>12}")
    print("  " + "-" * 60)
    for K in K_values:
        result = run_metropolis(geom, K=K, n_sweeps=200, n_warmup=50, seed=2026)
        plaq_vals = [average_plaquette(geom, U) for U in result.configs]
        avg_P = np.mean(plaq_vals)
        std_P = np.std(plaq_vals) / np.sqrt(len(plaq_vals))
        avg_det = np.mean(result.det_M_history)
        print(f"  {K:>5.2f} | {result.accept_rate:>7.3f} | "
              f"{avg_P:>+5.3f}±{std_P:.3f}  | {avg_det:>+12.4e} | {result.walltime:>12.1f}")


if __name__ == '__main__':
    _self_test()
