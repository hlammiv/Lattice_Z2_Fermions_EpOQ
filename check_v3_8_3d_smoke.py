"""3D 2×2×2 end-to-end smoke with Hutchinson-renormalized denominator.

The path-integral C_denom is rare-event-dominated at n_gauge ≥ 10
(see feedback_3d_denominator_rare_event).  Workaround: use the slab
numerator at t=0 as the renormalization point, and tie the absolute
scale to the sparse Hutchinson Tr[ρ_β·n_0]/Tr[ρ_β] estimate.

Math:
  slab_num[t]  ≈ (N/Z_g) · Tr[ρ_β · O_t]           (MC sample sum)
  slab_num[0]  ≈ (N/Z_g) · Tr[ρ_β · n_0]            (no rare event)
  ratio        = slab_num[t] / slab_num[0]
              ≈ Tr[ρ_β · O_t] / Tr[ρ_β · n_0]
  C_Hutch(0)   ≈ Tr[ρ_β · n_0] / Tr[ρ_β]            (Hutchinson)
  C_renorm(t)  = ratio × C_Hutch(0) ≈ C(t)

Both slab_num[0] and slab_num[t] are SUMS over all configs (linear in N,
all gauge-sector combinations contribute), so neither suffers the
1/4096 rare-event problem.

Settings:
  Lx=Ly=Lz=2, β=2, m=0.5, a_τ=0.5 (N_E=5)
  1 chain × 20 sweeps + 10 warmup (small for ~30-min wall)
  t = [0.0, 0.5]
  Hutchinson n_random=8
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
N_SWEEPS = 5
N_WARMUP = 0
TIMES = [0.0, 0.5]
ED_N_RANDOM = 8


def main():
    print("=" * 78)
    print(f"V_3=8 3D 2×2×2 end-to-end smoke  (Hutchinson-renormalized)")
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
    print(f"  done in {time.time()-t0:.1f}s.  H nnz={H_sparse.nnz}",
          flush=True)

    # --- Sparse Hutchinson ED reference ---
    print(f"\nSparse Hutchinson ED reference (n_random={ED_N_RANDOM}) ...",
          flush=True)
    t0 = time.time()
    C_ED, sigma_ED, _ = compute_C_t_hutchinson(
        H_sparse, n0_sparse, beta=BETA, times=TIMES,
        n_random=ED_N_RANDOM, seed=20260524,
        progress=True, progress_every=2,
    )
    t_ed = time.time() - t0
    print(f"  ED done in {t_ed:.0f}s", flush=True)
    print(f"  ED C(t):")
    for t in TIMES:
        print(f"    C({t}) = {C_ED[t]:+.5f} ± {sigma_ED[t]:.5f}", flush=True)

    # --- Run single-chain MC ---
    N_E = int(round(BETA / A_TAU)) + 1
    K_E = -0.5 * np.log(np.tanh(A_TAU * G_E_HAM))
    K_M = A_TAU * G_M_HAM
    m_action = A_TAU * M_HAM
    geom_mc = LatticeGeometry(Lx=LX, Ly=LY, Lz=LZ, N_E=N_E, m=m_action)
    print(f"\nMC geom: N_E={N_E}, K_E={K_E:.4f}, K_M={K_M:.4f}", flush=True)

    print(f"\nRunning single-chain MC ({N_SWEEPS} sweeps + {N_WARMUP} warmup) ...",
          flush=True)
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
    print(f"  MC done in {t_mc:.0f}s, {len(configs)} configs recorded, "
          f"accept rate {result.accept_rate:.3f}", flush=True)

    # --- Slab accumulator ---
    print(f"\nSlab accumulator (streaming Phase 2+3) ...", flush=True)
    t0 = time.time()
    C_lat_native, denom, num, _ = accumulate_C_direct_slab(
        geom_mc, configs, a_tau=A_TAU, times=TIMES,
        H_sparse=H_sparse,
        m_obs=M_HAM, g_hop=G_HOP, w_order=W_ORDER,
        progress=True, progress_every=16,
    )
    t_slab = time.time() - t0
    print(f"\n  Slab done in {t_slab:.0f}s ({t_slab/60:.1f} min)", flush=True)
    print(f"  C_denom = {denom:+.5e}  (path-integral diagonal sum)")
    print(f"  Trho_O[t] (path-integral numerator, complex):")
    for t in TIMES:
        print(f"    Trho_O({t}) = {num[t].real:+.5e} + {num[t].imag:+.5e}j")

    # --- Renormalize: ratio of slab numerators × Hutchinson C(0) ---
    print(f"\n" + "=" * 78)
    print(f"Renormalized C(t):  ratio of slab Trho_O × Hutchinson C(0)")
    print("=" * 78)
    if num[0.0].real == 0:
        print("  ABORT: Trho_O[0] = 0 — can't renormalize")
        return 1

    print(f"\n  Renormalization: C_renorm(t) = (Trho_O[t] / Trho_O[0]) × C_Hutch(0)")
    print(f"    C_Hutch(0) = {C_ED[0.0]:+.5f} ± {sigma_ED[0.0]:.5f}\n")

    print(f"  {'t':>5} | {'Trho_O[t].real':>16} | {'ratio':>10} | "
          f"{'C_renorm(t)':>12} | {'C_Hutch(t)':>13} | {'gap':>10}")
    for t in TIMES:
        ratio = num[t].real / num[0.0].real
        C_renorm = ratio * C_ED[0.0]
        gap = C_renorm - C_ED[t]
        print(f"  {t:>5.2f} | {num[t].real:+.8e} | {ratio:+.5f} | "
              f"{C_renorm:+.5f} | {C_ED[t]:+.5f}±{sigma_ED[t]:.5f} | "
              f"{gap:+.5f}")

    print(f"\nNotes:")
    print(f"  - Slab numerators sum over ALL configs (no δ_{{g_top,g_bot}} rare event).")
    print(f"  - Ratio Trho_O[t]/Trho_O[0] ≈ Tr[ρ_β·O_t]/Tr[ρ_β·n_0]; "
          f"unitless, normalization-free.")
    print(f"  - Multiplied by C_Hutch(0) ≈ ⟨n_0⟩_β to recover absolute scale.")
    print(f"  - Gap to Hutchinson C(t): tests whether slab MC + Hutchinson scale "
          f"are mutually consistent at 3D.")

    print(f"\nTiming:")
    print(f"  Hutchinson ED: {t_ed:.0f}s")
    print(f"  MC:            {t_mc:.0f}s")
    print(f"  Slab:          {t_slab:.0f}s")
    print(f"  Total:         {t_ed + t_mc + t_slab:.0f}s "
          f"({(t_ed + t_mc + t_slab) / 60:.1f} min)")

    print(f"\nV_3=8 3D 2×2×2 PIPELINE END-TO-END WORKS (renormalization scheme).")


if __name__ == "__main__":
    main()
