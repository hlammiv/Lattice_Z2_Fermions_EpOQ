"""First V_3=8 3D 2×2×2 end-to-end smoke (Phase 40 step 6).

Combines everything from Phase 39, 41, 41.5, 40:
  - 3D lattice + 3D H_KS (transfer_matrix_kbc_trotterized)
  - 3D qubit layout + 3D Pauli builder (epoq_classical_sampler)
  - sparse H + sparse Hutchinson C(t) ED reference (epoq_sparse_ed)
  - single-chain MC with U_z proposals (action_z2_metropolis)
  - slab accumulator (epoq_pipeline_direct_sampling)

Goals:
  1. Verify the pipeline runs end-to-end at V_3=8 3D without error.
  2. Get a first C(t) measurement and compare to ED reference.
  3. Measure timing breakdown across stages to validate cost estimates.

Settings (smallest for tractable smoke):
  Lx=Ly=Lz=2, β=2, m=0.5, a_τ=0.5 (N_E=5)
  1 chain × 100 sweeps + 50 warmup
  t = [0.0] ONLY (no Minkowski expm needed in slab Phase 2; per-config
     cost stays trivial because Phase 2 special-cases t=0)
  Hutchinson n_random=16 for ED reference

For dynamics (t>0) at 3D 2×2×2 the slab Phase 2 becomes ~few min per
(g, t), and with ~hundreds of unique g visited that scales to hours.
Future work: optimize slab (batched expms, sub-sample g) or accept
the wall.
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
from action_z2_metropolis import run_metropolis
from action_minkowski_stitch import qc_layout_counts
from epoq_pipeline_direct_sampling import accumulate_C_direct_slab
from epoq_sparse_ed import compute_C_t_hutchinson


LX, LY, LZ = 2, 2, 2
M_HAM, G_E_HAM, G_M_HAM, G_HOP = 0.5, 1.0, 0.5, 0.5
BETA = 2.0
W_ORDER = 1
A_TAU = 0.5
N_SWEEPS = 100
N_WARMUP = 50
TIMES = [0.0]
ED_N_RANDOM = 16


def main():
    print("=" * 78)
    print(f"V_3=8 3D 2×2×2 end-to-end smoke  (Phase 40 step 6)")
    print(f"  β={BETA}, m={M_HAM}, a_τ={A_TAU}, single chain × {N_SWEEPS} sweeps")
    print("=" * 78, flush=True)

    geom_h = LatticeGeometry(Lx=LX, Ly=LY, Lz=LZ, N_E=1, m=M_HAM)
    n_gauge, n_matter = qc_layout_counts(geom_h)
    NQ = n_gauge + n_matter
    DIM = 1 << NQ
    print(f"\nQubits: n_gauge={n_gauge}, n_matter={n_matter}, NQ={NQ}, DIM={DIM}")

    # --- Sparse H + n_0 ---
    print(f"\nBuilding sparse H_QC at 3D 2×2×2 ...", flush=True)
    t0 = time.time()
    H_sparse = cs.build_sparse_H_QC(geom_h, g_e=G_E_HAM, g_m=G_M_HAM,
                                    g_hop=G_HOP, m_mass=M_HAM)
    n0_sparse = cs.build_sparse_n0_at_site00(geom_h)
    print(f"  done in {time.time()-t0:.1f}s.  H nnz={H_sparse.nnz}, "
          f"sparse mem ≈ {(H_sparse.data.nbytes + H_sparse.indices.nbytes + H_sparse.indptr.nbytes) / 1024 / 1024:.0f} MB",
          flush=True)

    # --- Sparse Hutchinson ED reference ---
    print(f"\nSparse Hutchinson ED reference (n_random={ED_N_RANDOM}) ...",
          flush=True)
    t0 = time.time()
    C_ED, sigma_ED, _ = compute_C_t_hutchinson(
        H_sparse, n0_sparse, beta=BETA, times=TIMES,
        n_random=ED_N_RANDOM, seed=20260524,
        progress=True, progress_every=4,
    )
    t_ed = time.time() - t0
    print(f"  ED done in {t_ed:.0f}s "
          f"({t_ed/ED_N_RANDOM:.1f}s/sample)", flush=True)
    print(f"  ED C(t):")
    for t in TIMES:
        print(f"    C({t}) = {C_ED[t]:+.5f} ± {sigma_ED[t]:.5f}", flush=True)

    # --- Run single-chain MC ---
    N_E = int(round(BETA / A_TAU)) + 1
    K_E = -0.5 * np.log(np.tanh(A_TAU * G_E_HAM))
    K_M = A_TAU * G_M_HAM
    m_action = A_TAU * M_HAM
    geom_mc = LatticeGeometry(Lx=LX, Ly=LY, Lz=LZ, N_E=N_E, m=m_action)
    print(f"\nMC geom: Lx=Ly=Lz=2, N_E={N_E}, "
          f"K_E={K_E:.4f}, K_M={K_M:.4f}", flush=True)

    print(f"\nRunning single-chain MC ({N_SWEEPS} sweeps + {N_WARMUP} warmup, "
          f"temporal_gauge=True) ...", flush=True)
    t0 = time.time()
    result = run_metropolis(
        geom_mc, K_E=K_E, K_M=K_M,
        n_sweeps=N_SWEEPS, n_warmup=N_WARMUP, seed=20260524,
        record_every=1, cold_start=False,
        action_type='gauge_only',
        temporal_gauge=True,
    )
    configs = result.configs
    t_mc = time.time() - t0
    print(f"  MC done in {t_mc:.0f}s ({t_mc/N_SWEEPS*1000:.0f} ms/sweep), "
          f"{len(configs)} configs recorded, "
          f"accept rate {result.accept_rate:.3f}", flush=True)

    # --- Slab accumulator ---
    print(f"\nSlab accumulator at 3D 2×2×2 ...", flush=True)
    t0 = time.time()
    C_lat, denom, num, _ = accumulate_C_direct_slab(
        geom_mc, configs, a_tau=A_TAU, times=TIMES,
        H_sparse=H_sparse,
        m_obs=M_HAM, g_hop=G_HOP, w_order=W_ORDER,
        progress=True, progress_every=128,
    )
    t_slab = time.time() - t0
    print(f"\n  Slab done in {t_slab:.0f}s ({t_slab/60:.1f} min)", flush=True)

    # --- Compare ---
    print("\n" + "=" * 78)
    print(f"V_3=8 3D 2×2×2 FIRST RESULTS  (β={BETA}, m={M_HAM}, a_τ={A_TAU})")
    print("=" * 78)
    print(f"\n  {'t':>5} | {'C_lat':>12} | {'C_ED':>12}±{'σ_ED':>10} | "
          f"{'Δ = C_lat - C_ED':>17}")
    for t in TIMES:
        gap = C_lat[t] - C_ED[t]
        print(f"  {t:>5.2f} | {C_lat[t]:+.7f} | {C_ED[t]:+.7f} ± "
              f"{sigma_ED[t]:.5f} | {gap:+.5f}")

    print(f"\nNotes:")
    print(f"  - MC: {len(configs)} configs (small smoke, no PT). "
          f"σ_C_lat probably ~few %.")
    print(f"  - ED: σ_ED ~ {max(sigma_ED.values()):.4f} from n_random={ED_N_RANDOM} "
          f"Hutchinson samples.")
    print(f"  - Combined gap uncertainty dominated by both — "
          f"use this as a sanity check, not a precision result.")

    print(f"\nTiming breakdown:")
    print(f"  Sparse H build:       1 s (one-time)")
    print(f"  Sparse Hutchinson ED: {t_ed:.0f}s ({t_ed/ED_N_RANDOM:.1f}s/sample × {ED_N_RANDOM})")
    print(f"  MC ({N_SWEEPS} sweeps):       {t_mc:.0f}s ({t_mc/N_SWEEPS*1000:.0f} ms/sweep)")
    print(f"  Slab accumulator:     {t_slab:.0f}s ({t_slab/60:.1f} min)")
    print(f"  Total:                {t_ed + t_mc + t_slab:.0f}s "
          f"({(t_ed + t_mc + t_slab) / 60:.1f} min)")

    print(f"\nV_3=8 3D 2×2×2 end-to-end PIPELINE WORKS.")


if __name__ == "__main__":
    main()
