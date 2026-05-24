"""V_3=6 a_τ=0.125 high-stat diagnostic for the suspected sign flip at t=1.0.

Prior 2-chain × 5k-sweep run (commit e389ea0, re-interpreted in commit b44659f)
gave gap = +0.0030 ± 0.00055 at t=1.0 a_τ=0.125 — opposite sign to the
gap pattern at coarser a_τ.  This script reruns with 4 chains × 30k sweeps
to either confirm or refute the sign flip.

Dense ED reference (sigma_ED = 0) is hardcoded from commit b44659f:
  C_ED(0.0) = +0.274874
  C_ED(0.5) = +0.241651
  C_ED(1.0) = +0.206345
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


LX, LY = 2, 3
M_HAM, G_E_HAM, G_M_HAM, G_HOP = 0.5, 1.0, 0.5, 0.5
BETA = 2.0
W_ORDER = 1
A_TAU = 0.125
N_CHAINS = 4
N_SWEEPS = 30_000
N_WARMUP = 3_000
TIMES = [0.0, 0.5, 1.0]

C_ED_DENSE = {0.0: +0.274874, 0.5: +0.241651, 1.0: +0.206345}


def build_K_E_ladder(K_E_target, n_replicas=6):
    K_E_low = max(0.05, K_E_target - 0.6)
    return list(np.linspace(K_E_low, K_E_target, n_replicas))


def main():
    print("=" * 78)
    print(f"V_3=6 a_τ=0.125 SIGN-FLIP DIAGNOSTIC")
    print(f"  β={BETA}, m={M_HAM}, {N_CHAINS} chains × {N_SWEEPS} sweeps")
    print(f"  Dense ED ref (σ=0): "
          + "  ".join(f"C_ED({t})={C_ED_DENSE[t]:+.6f}" for t in TIMES))
    print("=" * 78, flush=True)

    geom_h = LatticeGeometry(Lx=LX, Ly=LY, N_E=1, m=M_HAM)
    print(f"\nBuilding sparse H ...", flush=True)
    t0 = time.time()
    H_sparse = cs.build_sparse_H_QC(geom_h, g_e=G_E_HAM, g_m=G_M_HAM,
                                    g_hop=G_HOP, m_mass=M_HAM)
    print(f"  done in {time.time()-t0:.1f}s (nnz={H_sparse.nnz})", flush=True)

    N_E = int(round(BETA / A_TAU)) + 1
    K_E = -0.5 * np.log(np.tanh(A_TAU * G_E_HAM))
    K_M = A_TAU * G_M_HAM
    m_action = A_TAU * M_HAM
    geom_mc = LatticeGeometry(Lx=LX, Ly=LY, N_E=N_E, m=m_action)
    K_E_ladder = build_K_E_ladder(K_E)
    print(f"\nMC geom: N_E={N_E}, K_E={K_E:.4f}, K_M={K_M:.4f}, "
          f"m_action={m_action:.4f}", flush=True)

    C_chains = {t: [] for t in TIMES}
    t_total = time.time()
    for ci in range(N_CHAINS):
        seed = 20260524 + ci * 11111
        print(f"\n  chain {ci+1}/{N_CHAINS} (seed={seed}) ...", flush=True)
        tc = time.time()
        pt_results, _ = run_metropolis_pt(
            geom_mc, K_E_ladder=K_E_ladder, K_M=K_M,
            n_sweeps=N_SWEEPS, n_warmup=N_WARMUP, swap_every=5,
            seed=seed, action_type='gauge_only', n_workers=6,
            temporal_gauge=True,
        )
        configs = pt_results[-1].configs
        t_mc = time.time() - tc
        print(f"    MC done in {t_mc:.0f}s, {len(configs)} configs", flush=True)

        tc = time.time()
        C, denom, num, _ = accumulate_C_direct_slab(
            geom_mc, configs, a_tau=A_TAU, times=TIMES,
            H_sparse=H_sparse,
            m_obs=M_HAM, g_hop=G_HOP, w_order=W_ORDER,
            progress=True, progress_every=64,
        )
        t_acc = time.time() - tc
        print(f"    slab accumulate done in {t_acc:.0f}s, "
              f"Tr[ρ_H]={denom:+.3e}", flush=True)
        for t in TIMES:
            C_chains[t].append(C[t])
        print(f"    chain {ci+1} C(t):  "
              + "  ".join(f"C({t})={C[t]:+.6f}" for t in TIMES), flush=True)

    total_wall = time.time() - t_total
    print(f"\n\nAll chains done in {total_wall:.0f}s ({total_wall/60:.1f} min).")
    print(f"\n{'='*78}")
    print(f"CROSS-CHAIN AT V_3=6 a_τ=0.125 (N={N_CHAINS} chains)")
    print(f"{'='*78}")
    for t in TIMES:
        vals = np.array(C_chains[t])
        mean = float(vals.mean())
        sem = float(vals.std(ddof=1) / np.sqrt(N_CHAINS))
        gap = mean - C_ED_DENSE[t]
        sigma_gap = sem   # σ_ED = 0
        nsig = gap / sigma_gap if sigma_gap > 0 else 0.0
        print(f"  t={t:4.2f}: C_lat={mean:+.6f}±{sem:.6f}  "
              f"C_ED={C_ED_DENSE[t]:+.6f}  gap={gap:+.6f}  ({nsig:+.2f}σ)")
        print(f"          per-chain: " + ", ".join(f"{v:+.5f}" for v in vals))

    print(f"\nCompare to prior 2-chain × 5k-sweep gaps (commit b44659f):")
    prior = {0.0: (-0.00156, 0.00121),
             0.5: (+0.00008, 0.00021),
             1.0: (+0.00297, 0.00055)}
    print(f"  t=0.0:  prior gap = {prior[0.0][0]:+.5f} ± {prior[0.0][1]:.5f}")
    print(f"  t=0.5:  prior gap = {prior[0.5][0]:+.5f} ± {prior[0.5][1]:.5f}")
    print(f"  t=1.0:  prior gap = {prior[1.0][0]:+.5f} ± {prior[1.0][1]:.5f}")
    print(f"\nIf t=1.0 gap remains ≈ +0.003 with tighter error: sign flip is real.")
    print(f"If it shifts toward zero / negative: prior was MC noise undersampled.")


if __name__ == "__main__":
    main()
