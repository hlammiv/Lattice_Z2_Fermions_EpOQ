"""Validate accumulate_C_direct_slab (sparse + per-gauge-sector cache) against
accumulate_C_direct (dense O_t) at V_3=6.

Both run on the SAME MC configs.  Expected agreement: Krylov tolerance
on the sparse expm_multiply (default scipy: backward-stable, ~ machine precision
for our small-β setup), times an O(N_configs) accumulation → relative
error on C(t) should be O(1e-10) or better.

This validates the slab path before relying on it for V_3 > 6 where dense
O_t no longer fits in RAM.
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "4"

import sys, time
import numpy as np
from scipy.sparse.linalg import expm_multiply

sys.path.insert(0, '/home/hlamm/Desktop/QC/logdet/m1_toy')

import epoq_classical_sampler as cs
cs.M_MASS = 0.5

from action_z2_staggered import LatticeGeometry
from action_z2_metropolis_pt import run_metropolis_pt
from action_minkowski_stitch import qc_layout_counts
from epoq_pipeline_direct_sampling import (
    accumulate_C_direct,
    accumulate_C_direct_slab,
)


LX, LY = 2, 3
M_HAM, G_E_HAM, G_M_HAM, G_HOP = 0.5, 1.0, 0.5, 0.5
BETA = 2.0
W_ORDER = 1
A_TAU = 0.5
N_SWEEPS = 5_000
N_WARMUP = 1_000
N_CHAINS = 1
TIMES = [0.0, 0.5, 1.0]


def build_K_E_ladder(K_E_target, n_replicas=6):
    K_E_low = max(0.05, K_E_target - 0.6)
    return list(np.linspace(K_E_low, K_E_target, n_replicas))


def build_n0_dense(geom):
    """Dense n_0 = (I - Z_{q_m(0,0)}) / 2 over full Hilbert space."""
    n_g, n_m = qc_layout_counts(geom)
    nq = n_g + n_m
    dim = 1 << nq
    q = n_g
    diag = np.array([(0.0 if ((i >> q) & 1) == 0 else 1.0) for i in range(dim)],
                    dtype=complex)
    return np.diag(diag).astype(complex)


def main():
    print("=" * 78)
    print(f"Slab vs dense accumulator at V_3=6  (Lx={LX}, Ly={LY})")
    print(f"  β={BETA}, m={M_HAM}, a_τ={A_TAU}, "
          f"{N_CHAINS} chain × {N_SWEEPS} sweeps")
    print("=" * 78, flush=True)

    geom_pauli = LatticeGeometry(Lx=LX, Ly=LY, N_E=1, m=M_HAM)
    n_gauge, n_matter = qc_layout_counts(geom_pauli)
    NQ = n_gauge + n_matter
    DIM = 1 << NQ
    print(f"\nQubits: n_gauge={n_gauge}, n_matter={n_matter}, NQ={NQ}, "
          f"DIM={DIM}", flush=True)

    # --- Sparse H and n0 ---
    print("\nBuilding sparse H and n_0 ...", flush=True)
    t0 = time.time()
    H_sparse = cs.build_sparse_H_QC(geom_pauli, g_e=G_E_HAM, g_m=G_M_HAM,
                                    g_hop=G_HOP, m_mass=M_HAM)
    n0_sparse = cs.build_sparse_n0_at_site00(geom_pauli)
    print(f"  done in {time.time()-t0:.1f}s.  H nnz={H_sparse.nnz}, "
          f"n_0 nnz={n0_sparse.nnz}", flush=True)

    # --- Dense O_total[t] via sparse expm on identity (no dense expm!) ---
    print("\nBuilding dense O_total[t] via sparse expm on dense identity ...",
          flush=True)
    n0_dense = build_n0_dense(geom_pauli)
    O_total = {}
    t0 = time.time()
    for t in TIMES:
        if t == 0.0:
            # U(0) = I → O_t = n0 · n0 = n0
            O_total[t] = n0_dense @ n0_dense
        else:
            tt = time.time()
            # U(t) = e^{-iHt} as a (DIM, DIM) dense matrix via sparse expm_multiply
            U_t = expm_multiply(-1j * t * H_sparse, np.eye(DIM, dtype=complex))
            UOU = U_t.conj().T @ n0_dense @ U_t
            O_total[t] = UOU @ n0_dense
            print(f"  O_total[{t}]:  {time.time()-tt:.1f}s "
                  f"(sparse expm + matmuls)", flush=True)
    print(f"  total: {time.time()-t0:.1f}s for {len(TIMES)} times.", flush=True)

    # --- Run MC ---
    N_E = int(round(BETA / A_TAU)) + 1
    K_E = -0.5 * np.log(np.tanh(A_TAU * G_E_HAM))
    K_M = A_TAU * G_M_HAM
    m_action = A_TAU * M_HAM
    geom_mc = LatticeGeometry(Lx=LX, Ly=LY, N_E=N_E, m=m_action)
    K_E_ladder = build_K_E_ladder(K_E)
    print(f"\nRunning MC at V_3=6, N_E={N_E}, "
          f"K_E={K_E:.4f}, K_M={K_M:.4f} ...", flush=True)
    t0 = time.time()
    pt_results, _ = run_metropolis_pt(
        geom_mc, K_E_ladder=K_E_ladder, K_M=K_M,
        n_sweeps=N_SWEEPS, n_warmup=N_WARMUP, swap_every=5,
        seed=20260524, action_type='gauge_only', n_workers=6,
        temporal_gauge=True,
    )
    configs = pt_results[-1].configs
    print(f"  MC done in {time.time()-t0:.0f}s. {len(configs)} configs.",
          flush=True)

    # --- Dense accumulator ---
    print("\nDense O_t accumulator ...", flush=True)
    t0 = time.time()
    C_dense, denom_dense, num_dense, n_d = accumulate_C_direct(
        geom_mc, configs, a_tau=A_TAU, times=TIMES, O_total_dict=O_total,
        m_obs=M_HAM, g_hop=G_HOP, w_order=W_ORDER,
    )
    t_dense = time.time() - t0
    print(f"  dense: {t_dense:.0f}s on {n_d} configs", flush=True)

    # --- Slab accumulator ---
    print("\nSlab accumulator ...", flush=True)
    t0 = time.time()
    C_slab, denom_slab, num_slab, n_s = accumulate_C_direct_slab(
        geom_mc, configs, a_tau=A_TAU, times=TIMES,
        H_sparse=H_sparse, m_obs=M_HAM, g_hop=G_HOP, w_order=W_ORDER,
        progress=True, progress_every=32,
    )
    t_slab = time.time() - t0
    print(f"  slab: {t_slab:.0f}s on {n_s} configs", flush=True)

    # --- Compare ---
    print("\n" + "=" * 78)
    print("Comparison  (relative tol 1e-9 on traces, abs tol 1e-12 on C(t))")
    print("=" * 78)

    print(f"\nDenominator Tr[ρ_H]:")
    print(f"  dense = {denom_dense:+.15e}")
    print(f"  slab  = {denom_slab:+.15e}")
    print(f"  identical by construction (no expm involved): "
          f"|Δ| = {abs(denom_dense - denom_slab):.3e}")

    print(f"\nTr[ρ_H · O_t]:")
    print(f"  {'t':>5} | {'dense (Re)':>18} | {'slab (Re)':>18} | "
          f"{'|Δ|':>10} | {'|Δ|/|val|':>10}")
    max_rel = 0.0
    for t in TIMES:
        dr = num_dense[t].real; sr = num_slab[t].real
        d = abs(dr - sr)
        rel = d / max(abs(dr), 1e-30)
        max_rel = max(max_rel, rel)
        print(f"  {t:>5.2f} | {dr:+.10e} | {sr:+.10e} | "
              f"{d:.3e} | {rel:.3e}")
    print(f"  max rel diff: {max_rel:.3e}  "
          f"({'PASS' if max_rel < 1e-9 else 'WARN'})")

    print(f"\nFinal C(t):")
    print(f"  {'t':>5} | {'dense':>15} | {'slab':>15} | {'|Δ|':>10}")
    max_C = 0.0
    for t in TIMES:
        d = abs(C_dense[t] - C_slab[t])
        max_C = max(max_C, d)
        print(f"  {t:>5.2f} | {C_dense[t]:+.10e} | {C_slab[t]:+.10e} | {d:.3e}")
    print(f"  max |Δ|: {max_C:.3e}  ({'PASS' if max_C < 1e-12 else 'WARN'})")

    print(f"\nTiming: dense {t_dense:.0f}s vs slab {t_slab:.0f}s on "
          f"{n_d} configs.")
    print(f"  At V_3=6 with dense O_t buildable in ~40 sec, dense path is "
          f"competitive.\n"
          f"  Slab payoff scales with V_3 — dense O_t is infeasible at V_3=8.")

    all_pass = (max_rel < 1e-9) and (max_C < 1e-12)
    print("\n" + ("ALL PASS" if all_pass else "WARN(S) ABOVE — check Krylov tolerance"))
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
