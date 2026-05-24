"""V_3=6 Trotter convergence scan: a_τ → 0 at β=2, m=0.5.

For each a_τ in {0.5, 0.25, 0.125}:
  - Run gauge-only MC in temporal gauge with W_ORDER=1.
  - Compute C(t) via slab accumulator (no dense O_t).

Reference C(t) at β=2 from sparse Krylov Hutchinson (a_τ-independent;
computed once).

Expectation: gap(a_τ, t) := C_lat(a_τ, t) − C_ED(t) shrinks linearly
in a_τ (clean Trotter convergence, geometry-independent of the
V_3=4 production case verified in Phase 38).
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "4"

import sys, time
import numpy as np

sys.path.insert(0, '/home/hlamm/Desktop/QC/logdet/m1_toy')

import epoq_classical_sampler as cs
cs.M_MASS = 0.5

from action_z2_staggered import LatticeGeometry
from action_z2_metropolis_pt import run_metropolis_pt
from action_minkowski_stitch import qc_layout_counts
from epoq_pipeline_direct_sampling import accumulate_C_direct_slab
from epoq_sparse_ed import compute_C_t_hutchinson


LX, LY = 2, 3
M_HAM, G_E_HAM, G_M_HAM, G_HOP = 0.5, 1.0, 0.5, 0.5
BETA = 2.0
W_ORDER = 1
A_TAUS = [0.5, 0.25, 0.125]
N_CHAINS = 2
N_SWEEPS = 5_000
N_WARMUP = 1_000
TIMES = [0.0, 0.5, 1.0]

ED_N_RANDOM = 64
ED_SEED = 20260524


def build_K_E_ladder(K_E_target, n_replicas=6):
    K_E_low = max(0.05, K_E_target - 0.6)
    return list(np.linspace(K_E_low, K_E_target, n_replicas))


def main():
    print("=" * 78)
    print(f"V_3=6 Trotter scan  (Lx={LX}, Ly={LY})")
    print(f"  β={BETA}, m={M_HAM}, "
          f"a_τ ∈ {A_TAUS}, {N_CHAINS} chain(s) × {N_SWEEPS} sweeps")
    print("=" * 78, flush=True)

    geom_h = LatticeGeometry(Lx=LX, Ly=LY, N_E=1, m=M_HAM)
    n_gauge, n_matter = qc_layout_counts(geom_h)
    NQ = n_gauge + n_matter
    DIM = 1 << NQ
    print(f"\nQubits: n_gauge={n_gauge}, n_matter={n_matter}, NQ={NQ}, DIM={DIM}")

    # --- Build sparse H + n_0 (used by both ED and slab accumulator) ---
    print("\nBuilding sparse H, n_0 ...", flush=True)
    t0 = time.time()
    H_sparse = cs.build_sparse_H_QC(geom_h, g_e=G_E_HAM, g_m=G_M_HAM,
                                    g_hop=G_HOP, m_mass=M_HAM)
    n0_sparse = cs.build_sparse_n0_at_site00(geom_h)
    print(f"  done in {time.time()-t0:.1f}s.  H nnz={H_sparse.nnz}", flush=True)

    # --- Sparse Hutchinson ED reference (one shot, a_τ-independent) ---
    print(f"\nSparse Hutchinson ED reference (n_random={ED_N_RANDOM}) ...",
          flush=True)
    t0 = time.time()
    C_ED, sigma_ED, _ = compute_C_t_hutchinson(
        H_sparse, n0_sparse, beta=BETA, times=TIMES,
        n_random=ED_N_RANDOM, seed=ED_SEED, progress=False,
    )
    print(f"  done in {time.time()-t0:.1f}s.", flush=True)
    print(f"  ED C(t):")
    for t in TIMES:
        print(f"    C({t}) = {C_ED[t]:+.5f} ± {sigma_ED[t]:.5f}", flush=True)

    # --- For each a_τ: MC + slab accumulator ---
    results = {}   # a_tau -> {t: (mean, sem, gap)}
    for a_tau in A_TAUS:
        print(f"\n{'='*60}")
        print(f"a_τ = {a_tau}  (N_E = {int(round(BETA/a_tau))+1})")
        print(f"{'='*60}", flush=True)
        N_E = int(round(BETA / a_tau)) + 1
        K_E = -0.5 * np.log(np.tanh(a_tau * G_E_HAM))
        K_M = a_tau * G_M_HAM
        m_action = a_tau * M_HAM
        geom_mc = LatticeGeometry(Lx=LX, Ly=LY, N_E=N_E, m=m_action)
        K_E_ladder = build_K_E_ladder(K_E)

        C_chains = {t: [] for t in TIMES}
        t_total = time.time()
        for ci in range(N_CHAINS):
            seed = 20260524 + ci * 11111
            print(f"  chain {ci+1}/{N_CHAINS} ...", flush=True)
            t0 = time.time()
            pt_results, _ = run_metropolis_pt(
                geom_mc, K_E_ladder=K_E_ladder, K_M=K_M,
                n_sweeps=N_SWEEPS, n_warmup=N_WARMUP, swap_every=5,
                seed=seed, action_type='gauge_only', n_workers=6,
                temporal_gauge=True,
            )
            configs = pt_results[-1].configs
            t_mc = time.time() - t0
            print(f"    MC: {t_mc:.0f}s, {len(configs)} configs", flush=True)

            t0 = time.time()
            C, denom, num, _ = accumulate_C_direct_slab(
                geom_mc, configs, a_tau=a_tau, times=TIMES,
                H_sparse=H_sparse,
                m_obs=M_HAM, g_hop=G_HOP, w_order=W_ORDER,
                progress=False,
            )
            t_acc = time.time() - t0
            print(f"    slab accumulate: {t_acc:.0f}s, Tr[ρ_H]={denom:+.3e}",
                  flush=True)
            for t in TIMES:
                C_chains[t].append(C[t])
            print(f"    C(t): "
                  + "  ".join(f"C({t})={C[t]:+.5f}" for t in TIMES), flush=True)

        print(f"\n  a_τ={a_tau} cross-chain (walltime {time.time()-t_total:.0f}s):")
        results[a_tau] = {}
        for t in TIMES:
            vals = np.array(C_chains[t])
            mean = vals.mean()
            sem = vals.std(ddof=1) / np.sqrt(N_CHAINS) if N_CHAINS > 1 else 0.0
            gap = mean - C_ED[t]
            results[a_tau][t] = (mean, sem, gap)
            print(f"    t={t}: C_lat={mean:+.5f}"
                  + (f"±{sem:.5f}" if N_CHAINS > 1 else "")
                  + f"  C_ED={C_ED[t]:+.5f}±{sigma_ED[t]:.5f}  gap={gap:+.5f}",
                  flush=True)

    # --- Trotter convergence table ---
    print("\n" + "=" * 78)
    print("TROTTER CONVERGENCE TABLE")
    print(f"  gap(a_τ, t) := C_lat(a_τ, t) − C_ED(t)")
    print(f"  ratio = gap(this a_τ) / gap(prev a_τ); O(a_τ) → ratio ≈ 0.5 per halving")
    print("=" * 78)
    print(f"\n{'t':>5} | " + " | ".join(f"a_τ={a:>5} ({'gap':>8})" for a in A_TAUS)
          + f" | {'gap@0.25 / gap@0.5':>20} | {'gap@0.125 / gap@0.25':>22}")
    for t in TIMES:
        gaps = [results[a][t][2] for a in A_TAUS]
        ratios = []
        for k in range(1, len(gaps)):
            r = gaps[k] / gaps[k-1] if abs(gaps[k-1]) > 1e-12 else float('nan')
            ratios.append(r)
        row = f"  {t:>5.2f} | " + " | ".join(
            f"a_τ={a}  ({g:+8.5f})" for a, g in zip(A_TAUS, gaps))
        for r in ratios:
            row += f"  | {r:>20.3f}"
        print(row)

    print(f"\nIf gaps shrink as O(a_τ): ratios should be ~0.5 per a_τ halving.")
    print(f"Larger ratios indicate sub-leading terms still dominant; "
          f"smaller ratios → MC statistics noise floor.")


if __name__ == "__main__":
    main()
