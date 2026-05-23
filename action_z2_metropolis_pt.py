"""
action_z2_metropolis_pt.py — Parallel-tempering (replica-exchange) MC across K_E
for Z_2 gauge + staggered fermions with OBC.

Motivation
----------
At high Suzuki K_E (e.g. K_E=1.387, corresponding to a_tau=0.0625 via
    K_E = -(1/2) log tanh(a_tau * g_E_Ham)
with g_E_Ham=1) the single-link heat-bath sampler in `action_z2_metropolis.py`
has catastrophic mixing: accept rates ~0.008, ~15 unique configs in 200 records,
chain-mean variance 5-13× across chains.  Standard fix: replica exchange.

Algorithm
---------
* N_replicas chains, each at its own K_E_i, with K_E_0 < K_E_1 < ... < K_E_{N-1}.
* K_M is held fixed across all replicas (only K_E governs the swap probability).
* Each replica advances `swap_every` heat-bath sweeps; then we attempt one swap
  per adjacent pair (alternating even/odd parity scheme, à la Metropolis).
* Acceptance for swap of replicas a, b (K_E_a < K_E_b):
      Δ = (K_E_a - K_E_b) * (S_E(U_b) - S_E(U_a))
  where S_E(U) = - Σ_{xτ,yτ} plaquette is the (sign-flipped) electric plaquette
  sum, so that S_g(U; K_E, K_M) = K_E * S_E(U) + K_M * S_M(U) up to a sign
  convention. Concretely, with S_g(U; K_E, K_M) = -K_E*P_E - K_M*P_M we have
  K_E·∂(P_E)/∂K_E = ... the standard PT detailed balance gives:
      p_swap = min(1, exp[(K_E_b - K_E_a) * (P_E(U_a) - P_E(U_b))])
  where P_E(U) = Σ electric plaquettes (signed). Equivalently
      p_swap = min(1, exp[(K_E_a - K_E_b) * (S_E(U_b) - S_E(U_a))])
  with S_E(U) := -P_E(U) (matching the action's sign).
  The |det M|^2 factor cancels because it is identical at both K_E values
  (the fermion piece is K_E-independent).  Sign(det M) also stays with the
  config and so is irrelevant for the swap weight (|w| only).

Public API
----------
* `run_metropolis_pt(geom, K_E_ladder, K_M, n_sweeps, ...)` → returns
    (results: List[MCMCResult], swap_stats: dict).  results[i] is the chain at
    K_E_ladder[i]; results[-1] is the physics chain (highest K_E).

* `verify_pt(geom, K_E_target, K_M, ...)` → runs both single-chain MC and
    PT at the same target K_E and reports the comparison the brief asks for.

Notes
-----
* Uses `heatbath_sweep` from `action_z2_metropolis.py` for the per-replica
  inner MC (the brief's constraint).
* Replica MC steps for each block of `swap_every` sweeps are parallelised
  across replicas with `multiprocessing.Pool` (BLAS pinned to 1 thread in each
  worker).
* Stays staggered everywhere.
"""

from __future__ import annotations
import time
import numpy as np
import multiprocessing as mp
from dataclasses import dataclass
from typing import List, Tuple

from action_z2_staggered import (
    LatticeGeometry, Z2GaugeConfig, gauge_action,
)
from action_z2_metropolis import (
    MCMCResult, heatbath_sweep, link_list, plaquette_flip_sweep,
    _resolve_det_fn,
)


# ---------------------------------------------------------------------------
# Helper: electric plaquette sum
# ---------------------------------------------------------------------------
def electric_plaquette_sum(geom: LatticeGeometry, U: Z2GaugeConfig) -> float:
    """Σ_{xτ,yτ} ∏ σ — the bare (unweighted) electric plaquette sum P_E(U).

    Implemented via the existing kernel by setting K_E=-1, K_M=0:
        gauge_action(K_E=-1, K_M=0) = -(-1) * P_E = +P_E.
    """
    return gauge_action(geom, U, K=0.0, K_E=-1.0, K_M=0.0)


# ---------------------------------------------------------------------------
# Replica worker — advances ONE chain by `n_sweeps` heat-bath sweeps.
# ---------------------------------------------------------------------------
def _replica_advance_worker(args):
    """Advance one replica by `n_sweeps` heat-bath sweeps.

    Inputs: (geom_args, U_state, K_E, K_M, n_sweeps, rng_state, action_type,
             plaq_flip_every, record, record_every)

    Returns: (U_state_out, det_M, S_g, P_E, rng_state_out, records)
      where records is a tuple of lists: (configs, det_M_hist, S_g_hist,
      sign_hist, P_E_hist) — only filled when record=True.
    """
    (geom_args, U_state, K_E, K_M, n_sweeps,
     rng_state, action_type, plaq_flip_every,
     record, record_every) = args

    # Pin BLAS to 1 thread and warm up Numba (same pattern as
    # `_chain_worker` in `action_z2_metropolis.py`).
    try:
        from threadpoolctl import threadpool_limits
        limiter = threadpool_limits(limits=1)
    except ImportError:
        limiter = None
    try:
        from action_z2_kernels import warm_up
        warm_up()
    except Exception:
        pass

    geom = LatticeGeometry(**geom_args)
    U_x, U_y, U_t = U_state
    U = Z2GaugeConfig(
        geom=geom, U_x=U_x.copy(), U_y=U_y.copy(), U_t=U_t.copy(),
    )
    rng = np.random.default_rng()
    rng.bit_generator.state = rng_state

    det_fn = _resolve_det_fn(action_type)

    configs, det_M_hist, S_g_hist, sign_hist, P_E_hist = [], [], [], [], []

    last_det = det_fn(geom, U)
    last_Sg = gauge_action(geom, U, K=0.0, K_E=K_E, K_M=K_M)

    for sweep in range(n_sweeps):
        U, last_det, last_Sg, _n_acc = heatbath_sweep(
            geom, U, K=0.0, K_E=K_E, K_M=K_M, rng=rng,
            action_type=action_type,
        )
        if plaq_flip_every > 0 and (sweep + 1) % plaq_flip_every == 0:
            U, last_det, last_Sg, _ = plaquette_flip_sweep(
                geom, U, K=0.0, K_E=K_E, K_M=K_M, rng=rng,
                action_type=action_type,
            )
        if record and (sweep % record_every == 0):
            configs.append(Z2GaugeConfig(
                geom=geom,
                U_x=U.U_x.copy(), U_y=U.U_y.copy(), U_t=U.U_t.copy(),
            ))
            det_M_hist.append(last_det)
            S_g_hist.append(last_Sg)
            sign_hist.append(1 if last_det > 0 else (-1 if last_det < 0 else 0))
            P_E_hist.append(electric_plaquette_sum(geom, U))

    if limiter is not None:
        try:
            limiter.__exit__(None, None, None)
        except Exception:
            pass

    U_state_out = (U.U_x, U.U_y, U.U_t)
    rng_state_out = rng.bit_generator.state
    records = (configs, det_M_hist, S_g_hist, sign_hist, P_E_hist)
    return (U_state_out, last_det, last_Sg,
            electric_plaquette_sum(geom, U), rng_state_out, records)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
@dataclass
class PTSwapStats:
    """Bookkeeping for replica-exchange acceptance."""
    n_attempts: np.ndarray  # shape (N-1,) — adjacent pairs
    n_accepts: np.ndarray   # shape (N-1,)

    @property
    def rates(self) -> np.ndarray:
        return self.n_accepts / np.maximum(self.n_attempts, 1)


def run_metropolis_pt(
    geom: LatticeGeometry,
    K_E_ladder: List[float],
    K_M: float,
    n_sweeps: int = 1000,
    n_warmup: int = 500,
    swap_every: int = 5,
    seed: int = 2026,
    record_every: int = 1,
    action_type: str = 'forward',
    plaq_flip_every: int = 0,
    cold_start: bool = False,
    n_workers: int | None = None,
    verbose: bool = False,
) -> Tuple[List[MCMCResult], PTSwapStats]:
    """Run parallel-tempered Z_2 heat-bath MC across the K_E ladder.

    The ladder is sorted ascending; replica i runs at K_E_ladder[i] with K_M
    common across replicas.  Each "block" advances every replica by
    `swap_every` heat-bath sweeps (in parallel processes), then attempts one
    swap per adjacent pair using even/odd-parity Metropolis exchange.

    Records configs only on the post-warmup blocks.

    Returns
    -------
    results : list of MCMCResult, one per replica (ordered by K_E ascending)
    swap_stats : PTSwapStats
    """
    K_E_ladder = list(map(float, K_E_ladder))
    if any(K_E_ladder[i] >= K_E_ladder[i + 1] for i in range(len(K_E_ladder) - 1)):
        raise ValueError("K_E_ladder must be strictly ascending.")
    N = len(K_E_ladder)
    if n_workers is None:
        n_workers = min(N, 16)

    geom_args = dict(Lx=geom.Lx, Ly=geom.Ly, N_E=geom.N_E, m=geom.m)

    # Initialise each replica.  Mix cold (high-K_E end) + random (low-K_E end)
    # for fastest equilibration.  Each replica gets its own RNG.
    init_rng = np.random.default_rng(seed)
    replicas: list[Z2GaugeConfig] = []
    rng_states: list[dict] = []
    for i in range(N):
        rng_i = np.random.default_rng(seed + 1009 * (i + 1))
        if cold_start:
            U_i = Z2GaugeConfig.trivial(geom)
        else:
            U_i = Z2GaugeConfig.random(geom, rng_i)
        replicas.append(U_i)
        rng_states.append(rng_i.bit_generator.state)

    # Swap-attempt parity counter (alternate even/odd offset).
    swap_attempts = np.zeros(N - 1, dtype=np.int64)
    swap_accepts = np.zeros(N - 1, dtype=np.int64)
    swap_rng = np.random.default_rng(seed + 7919)

    # Records (per replica)
    rec_configs: list[list[Z2GaugeConfig]] = [[] for _ in range(N)]
    rec_det:     list[list[float]] = [[] for _ in range(N)]
    rec_Sg:      list[list[float]] = [[] for _ in range(N)]
    rec_sign:    list[list[int]] = [[] for _ in range(N)]
    rec_PE:      list[list[float]] = [[] for _ in range(N)]

    t_start = time.time()

    total_sweeps = n_warmup + n_sweeps
    n_blocks = (total_sweeps + swap_every - 1) // swap_every

    pool = mp.Pool(processes=n_workers)
    try:
        sweeps_done = 0
        for block in range(n_blocks):
            this_block_sweeps = min(swap_every, total_sweeps - sweeps_done)
            recording = sweeps_done >= n_warmup

            args_list = [
                (geom_args,
                 (replicas[i].U_x, replicas[i].U_y, replicas[i].U_t),
                 K_E_ladder[i], K_M,
                 this_block_sweeps,
                 rng_states[i],
                 action_type, plaq_flip_every,
                 recording, record_every)
                for i in range(N)
            ]
            block_out = pool.map(_replica_advance_worker, args_list)

            # Unpack and update state
            P_E_now = np.zeros(N)
            for i, (U_state, det_M, S_g, P_E, rng_state, records) in enumerate(block_out):
                Ux, Uy, Ut = U_state
                replicas[i] = Z2GaugeConfig(
                    geom=geom, U_x=Ux, U_y=Uy, U_t=Ut,
                )
                rng_states[i] = rng_state
                P_E_now[i] = P_E
                if recording:
                    cfgs, dets, sgs, signs, pes = records
                    rec_configs[i].extend(cfgs)
                    rec_det[i].extend(dets)
                    rec_Sg[i].extend(sgs)
                    rec_sign[i].extend(signs)
                    rec_PE[i].extend(pes)

            sweeps_done += this_block_sweeps

            # Swap step: try adjacent pairs (a, a+1) with parity alternation
            #   block even: try a = 0, 2, 4, ...
            #   block odd : try a = 1, 3, 5, ...
            # This is the canonical safe ordering (no double-swap in one step).
            parity = block % 2
            for a in range(parity, N - 1, 2):
                b = a + 1
                # log p_swap = (K_E_a - K_E_b)*(S_E(U_b) - S_E(U_a))
                # with S_E := -P_E = +gauge_action_kernel_E.
                # Equivalently:  Δ = (K_E_b - K_E_a)*(P_E(U_a) - P_E(U_b))
                # accept iff random < exp(Δ).
                dK = K_E_ladder[b] - K_E_ladder[a]
                dPE = P_E_now[a] - P_E_now[b]
                log_p = dK * dPE
                swap_attempts[a] += 1
                if log_p >= 0.0 or swap_rng.random() < np.exp(log_p):
                    # Accept: swap gauge configs.
                    replicas[a], replicas[b] = replicas[b], replicas[a]
                    P_E_now[a], P_E_now[b] = P_E_now[b], P_E_now[a]
                    swap_accepts[a] += 1

            if verbose and (block % max(1, n_blocks // 10) == 0):
                rates = swap_accepts / np.maximum(swap_attempts, 1)
                print(f"  block {block + 1:4d}/{n_blocks}  "
                      f"swap rates={np.round(rates, 3).tolist()}")

    finally:
        pool.close()
        pool.join()

    walltime = time.time() - t_start

    # Wrap each replica's record into an MCMCResult.
    results: list[MCMCResult] = []
    for i in range(N):
        sign_arr = np.array(rec_sign[i], dtype=int)
        results.append(MCMCResult(
            configs=rec_configs[i],
            det_M_history=np.array(rec_det[i], dtype=float),
            S_g_history=np.array(rec_Sg[i], dtype=float),
            sign_history=sign_arr,
            accept_rate=float('nan'),    # per-link accept not aggregated here
            walltime=walltime,
            avg_sign=float(sign_arr.mean()) if len(sign_arr) else 0.0,
        ))

    swap_stats = PTSwapStats(
        n_attempts=swap_attempts,
        n_accepts=swap_accepts,
    )
    return results, swap_stats


# ---------------------------------------------------------------------------
# Verification helpers — single-chain vs PT at a target K_E
# ---------------------------------------------------------------------------
def _config_fingerprint(U: Z2GaugeConfig) -> bytes:
    """Stable hashable fingerprint of a Z2GaugeConfig."""
    return (U.U_x.tobytes() + b'|' + U.U_y.tobytes()
            + b'|' + U.U_t.tobytes())


def unique_config_fraction(configs: list[Z2GaugeConfig]) -> float:
    if len(configs) == 0:
        return 0.0
    fps = {_config_fingerprint(U) for U in configs}
    return len(fps) / len(configs)


def average_PE_per_plaq(geom: LatticeGeometry, configs: list[Z2GaugeConfig]) -> float:
    """⟨P_E⟩ averaged over configs, normalised per electric plaquette.

    Returns ⟨ (Σ_E P_E) / N_E_plaq ⟩, in [-1, +1].
    """
    if len(configs) == 0:
        return float('nan')
    n_E_plaq = ((geom.N_E - 1) * (geom.Lx - 1) * geom.Ly
                + (geom.N_E - 1) * geom.Lx * (geom.Ly - 1))
    vals = [electric_plaquette_sum(geom, U) / max(n_E_plaq, 1) for U in configs]
    return float(np.mean(vals))


def verify_pt(
    geom: LatticeGeometry,
    K_E_target: float,
    K_M: float,
    K_E_ladder: List[float] | None = None,
    n_sweeps: int = 1000,
    n_warmup: int = 500,
    swap_every: int = 5,
    n_single_chains: int = 8,
    seed: int = 2026,
    action_type: str = 'forward',
    plaq_flip_every: int = 0,
) -> dict:
    """Run single-chain heat-bath MC vs PT at the same K_E_target, K_M.

    The single-chain comparison uses `n_single_chains` independent chains at
    K_E_target (same wall-clock parallelism as the PT ladder).  PT uses the
    supplied ladder (highest entry = K_E_target).

    Returns a dict of comparison metrics.
    """
    if K_E_ladder is None:
        # Default ladder: linear, ~8 replicas, low end at K_E ~ 0.40.
        K_E_ladder = list(np.linspace(0.40, K_E_target, 8))
    if abs(K_E_ladder[-1] - K_E_target) > 1e-9:
        raise ValueError("Top of K_E ladder must equal K_E_target.")

    # ----- 1.  Single-chain (n_single_chains parallel chains at K_E_target) -----
    from action_z2_metropolis import run_metropolis_parallel
    t0 = time.time()
    n_chains = max(n_single_chains, 1)
    # Run each chain via run_metropolis_parallel concatenates, but we
    # want per-chain spread → use individual workers.  Simplest: call the
    # underlying _chain_worker via the parallel pool ourselves.
    from action_z2_metropolis import _chain_worker
    geom_args = dict(Lx=geom.Lx, Ly=geom.Ly, N_E=geom.N_E, m=geom.m)
    args_list = [
        (geom_args, 0.0, K_E_target, K_M,
         n_sweeps, n_warmup, seed + 13 * (i + 1),
         1, False, action_type, plaq_flip_every)
        for i in range(n_chains)
    ]
    with mp.Pool(processes=min(n_chains, 16)) as pool:
        single_results = pool.map(_chain_worker, args_list)
    single_walltime = time.time() - t0

    # ----- 2.  PT run -----
    pt_results, swap_stats = run_metropolis_pt(
        geom, K_E_ladder=K_E_ladder, K_M=K_M,
        n_sweeps=n_sweeps, n_warmup=n_warmup,
        swap_every=swap_every, seed=seed,
        action_type=action_type, plaq_flip_every=plaq_flip_every,
        verbose=False,
    )
    pt_walltime = pt_results[-1].walltime  # all replicas share walltime

    # ----- 3.  Metrics -----
    # ⟨P_E⟩ per chain (single) and per replica (PT, just the top replica;
    #   spread across PT chains comes from running multiple PT instances —
    #   here we report PT block-jackknife as an ersatz spread.)
    n_E_plaq = ((geom.N_E - 1) * (geom.Lx - 1) * geom.Ly
                + (geom.N_E - 1) * geom.Lx * (geom.Ly - 1))

    sc_PE_per_chain = []
    sc_unique_per_chain = []
    for r in single_results:
        sc_PE_per_chain.append(average_PE_per_plaq(geom, r.configs))
        sc_unique_per_chain.append(unique_config_fraction(r.configs))
    sc_PE_per_chain = np.array(sc_PE_per_chain)
    sc_unique_per_chain = np.array(sc_unique_per_chain)

    # PT: split top-replica configs into n_chains equal blocks for spread est.
    top = pt_results[-1]
    n_top = len(top.configs)
    block_size = max(1, n_top // n_chains)
    pt_PE_blocks = []
    pt_unique_blocks = []
    for i in range(n_chains):
        block = top.configs[i * block_size: (i + 1) * block_size]
        if len(block) == 0:
            continue
        pt_PE_blocks.append(average_PE_per_plaq(geom, block))
        pt_unique_blocks.append(unique_config_fraction(block))
    pt_PE_blocks = np.array(pt_PE_blocks)
    pt_unique_blocks = np.array(pt_unique_blocks)

    pt_unique_top = unique_config_fraction(top.configs)

    return dict(
        K_E_ladder=list(map(float, K_E_ladder)),
        K_M=float(K_M),
        n_sweeps=n_sweeps,
        n_warmup=n_warmup,
        swap_every=swap_every,
        swap_attempts=swap_stats.n_attempts.tolist(),
        swap_accepts=swap_stats.n_accepts.tolist(),
        swap_rates=swap_stats.rates.tolist(),
        single_chain_PE=sc_PE_per_chain.tolist(),
        single_chain_PE_mean=float(sc_PE_per_chain.mean()),
        single_chain_PE_spread=float(sc_PE_per_chain.max() - sc_PE_per_chain.min()),
        single_chain_PE_std=float(sc_PE_per_chain.std(ddof=1) if len(sc_PE_per_chain) > 1 else 0.0),
        single_chain_unique=sc_unique_per_chain.tolist(),
        pt_PE_top_replica_blocks=pt_PE_blocks.tolist(),
        pt_PE_top_replica_mean=float(pt_PE_blocks.mean()) if len(pt_PE_blocks) else float('nan'),
        pt_PE_top_replica_spread=float(pt_PE_blocks.max() - pt_PE_blocks.min()) if len(pt_PE_blocks) else float('nan'),
        pt_PE_top_replica_std=float(pt_PE_blocks.std(ddof=1)) if len(pt_PE_blocks) > 1 else 0.0,
        pt_unique_top_replica=float(pt_unique_top),
        pt_unique_blocks=pt_unique_blocks.tolist(),
        single_chain_walltime=float(single_walltime),
        pt_walltime=float(pt_walltime),
    )


# ---------------------------------------------------------------------------
# CLI self-test
# ---------------------------------------------------------------------------
def _self_test():
    """Compare PT vs single-chain at K_E=1.387 (a_tau=0.0625)."""
    g_E_Ham = 1.0
    a_tau = 0.0625
    K_E_target = -0.5 * np.log(np.tanh(a_tau * g_E_Ham))
    K_M = a_tau * 0.5  # held fixed, matches the high-K_E target
    print(f"a_tau = {a_tau}, K_E_target = {K_E_target:.4f}, K_M = {K_M:.5f}")

    geom = LatticeGeometry(Lx=2, Ly=2, N_E=int(round(4.0 / a_tau)) + 1, m=a_tau * 0.5)
    print(f"geom: Lx={geom.Lx}, Ly={geom.Ly}, N_E={geom.N_E}, m={geom.m:.4f}")

    # 8-replica linear ladder; covers the freely-mixing low-K_E end (0.40)
    # up to the physics target K_E=1.387.  Top step 0.20 to keep swap rates
    # workable at the high-K_E end.
    K_E_ladder = list(np.linspace(0.40, K_E_target, 8))
    print(f"K_E ladder: {[f'{k:.3f}' for k in K_E_ladder]}")

    report = verify_pt(
        geom, K_E_target=K_E_target, K_M=K_M,
        K_E_ladder=K_E_ladder,
        n_sweeps=1000, n_warmup=500, swap_every=5,
        n_single_chains=8, seed=2026,
        action_type='forward', plaq_flip_every=5,
    )
    print()
    print("=" * 72)
    print("SUMMARY  (single-chain MC vs PT, both at K_E_target = "
          f"{K_E_target:.4f})")
    print("=" * 72)
    print(f"  swap accept rates per adjacent pair: "
          f"{[f'{r:.3f}' for r in report['swap_rates']]}")
    print(f"  ⟨P_E⟩ single-chain : mean={report['single_chain_PE_mean']:+.4f}, "
          f"spread={report['single_chain_PE_spread']:.4f}, "
          f"std={report['single_chain_PE_std']:.4f}")
    print(f"  ⟨P_E⟩ PT top-rep   : mean={report['pt_PE_top_replica_mean']:+.4f}, "
          f"spread={report['pt_PE_top_replica_spread']:.4f}, "
          f"std={report['pt_PE_top_replica_std']:.4f}")
    print(f"  unique-config fraction single-chain (per chain): "
          f"{[f'{u:.3f}' for u in report['single_chain_unique']]}")
    print(f"  unique-config fraction PT top-replica overall: "
          f"{report['pt_unique_top_replica']:.3f}")
    print(f"  walltime single-chain={report['single_chain_walltime']:.1f} s, "
          f"PT={report['pt_walltime']:.1f} s")


if __name__ == '__main__':
    _self_test()
