"""Re-interpret the V_3=6 Trotter scan gaps using a tight (dense eigh) ED
reference instead of the Hutchinson σ ~ 0.007 used in the original scan.

MC C_lat values are hardcoded from the prior scan run (commit e389ea0).
ED is recomputed once via dense eigh + eigenbasis trace (σ_ED = 0).

This avoids a ~70 min MC re-run while giving the methodology-paper-quality
Trotter convergence plot.
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
from action_minkowski_stitch import qc_layout_counts


# --- Geometry and couplings (must match check_v3_6_trotter_scan.py) ---
LX, LY = 2, 3
M_HAM, G_E_HAM, G_M_HAM, G_HOP = 0.5, 1.0, 0.5, 0.5
BETA = 2.0
TIMES = [0.0, 0.5, 1.0]


# --- C_lat from commit e389ea0 (2 chains × 5000 sweeps each, gauge_only) ---
# Format: a_tau -> t -> list of per-chain C(t)
C_LAT = {
    0.5:   {0.0: [+0.26999, +0.27363],
            0.5: [+0.23617, +0.24071],
            1.0: [+0.19895, +0.20589]},
    0.25:  {0.0: [+0.27226, +0.27279],
            0.5: [+0.23802, +0.23999],
            1.0: [+0.20045, +0.20333]},
    0.125: {0.0: [+0.27452, +0.27210],
            0.5: [+0.24194, +0.24152],
            1.0: [+0.20876, +0.20987]},
}
A_TAUS = sorted(C_LAT.keys(), reverse=True)


P2 = {
    'I': np.eye(2, dtype=complex),
    'X': np.array([[0, 1], [1, 0]], dtype=complex),
    'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
    'Z': np.diag([1, -1]).astype(complex),
}


def pauli_string(factors, nq):
    by_q = {q: P2['I'] for q in range(nq)}
    for q, ax in factors:
        by_q[q] = P2[ax]
    r = by_q[nq - 1]
    for q in range(nq - 2, -1, -1):
        r = np.kron(r, by_q[q])
    return r


def main():
    print("=" * 78)
    print(f"V_3=6 Trotter scan: re-interpret with dense eigh ED (σ_ED = 0)")
    print(f"  β={BETA}, m={M_HAM}, MC values from commit e389ea0")
    print("=" * 78, flush=True)

    geom_h = LatticeGeometry(Lx=LX, Ly=LY, N_E=1, m=M_HAM)
    n_gauge, n_matter = qc_layout_counts(geom_h)
    NQ = n_gauge + n_matter
    DIM = 1 << NQ

    print(f"\nDIM = {DIM} ({NQ} qubits).  Dense eigh path...", flush=True)
    t0 = time.time()
    terms = cs.build_pauli_terms_general(
        geom_h, g_e=G_E_HAM, g_m=G_M_HAM, g_hop=G_HOP, m_mass=M_HAM)
    H = sum(c * pauli_string(f, NQ) for c, f in terms)
    H = (H + H.conj().T) / 2
    print(f"  H built ({len(terms)} terms): {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    eigs, V_eig = np.linalg.eigh(H)
    print(f"  eigh: {time.time()-t0:.1f}s.  "
          f"E_0={eigs[0]:+.5f}, E_max={eigs[-1]:+.5f}", flush=True)

    n0_op = 0.5 * np.eye(DIM, dtype=complex) - 0.5 * pauli_string(
        [(n_gauge, "Z")], NQ)
    n0_eig = V_eig.conj().T @ n0_op @ V_eig
    w_beta = np.exp(-BETA * eigs)
    Z_beta = w_beta.sum()
    C_ED = {}
    for t in TIMES:
        phase = np.exp(-1j * eigs * t)
        UOU_eig = (phase.conj()[:, None] * n0_eig) * phase[None, :]
        O_eig = UOU_eig @ n0_eig
        C_ED[t] = float((w_beta * np.diag(O_eig).real).sum() / Z_beta)

    print(f"\nDense ED  (σ_ED = 0):")
    for t in TIMES:
        print(f"  C_ED({t}) = {C_ED[t]:+.6f}", flush=True)

    print(f"\nMC C_lat (mean ± SEM across 2 chains × 5k sweeps):")
    means = {a: {t: np.mean(C_LAT[a][t]) for t in TIMES} for a in A_TAUS}
    sems = {a: {t: np.std(C_LAT[a][t], ddof=1) / np.sqrt(2) for t in TIMES}
            for a in A_TAUS}
    for a in A_TAUS:
        print(f"  a_τ={a}: "
              + "  ".join(f"C({t})={means[a][t]:+.5f}±{sems[a][t]:.5f}"
                          for t in TIMES))

    print(f"\nGaps  (gap = C_lat - C_ED;  σ_gap = σ_C_lat since σ_ED = 0):")
    print(f"  {'t':>5} | " + "  ".join(f"a_τ={a:>5}" for a in A_TAUS))
    for t in TIMES:
        gaps = {a: means[a][t] - C_ED[t] for a in A_TAUS}
        print(f"  {t:>5.2f} | "
              + "  ".join(f"{gaps[a]:+8.5f}±{sems[a][t]:.5f}" for a in A_TAUS))

    print(f"\nTrotter ratios  (each / preceding):")
    print(f"  {'t':>5} | " + "  ".join(
        f"{a1}/{a0}".rjust(13) for a0, a1 in zip(A_TAUS[:-1], A_TAUS[1:])))
    for t in TIMES:
        gaps = {a: means[a][t] - C_ED[t] for a in A_TAUS}
        sems_t = {a: sems[a][t] for a in A_TAUS}
        ratios = []
        for a0, a1 in zip(A_TAUS[:-1], A_TAUS[1:]):
            g0 = gaps[a0]; g1 = gaps[a1]
            if abs(g0) > 3 * sems_t[a0]:
                ratios.append(f"{g1/g0:+.3f}")
            else:
                ratios.append(f"(g0 < 3σ)")
        print(f"  {t:>5.2f} | " + "  ".join(r.rjust(13) for r in ratios))

    print(f"\nIf gap ~ O(a_τ):  ratio per halving ≈ 0.5.")
    print(f"If gap ~ O(a_τ²): ratio per halving ≈ 0.25.")
    print(f"Combined MC σ alone: σ_gap_total ≈ √(σ²(a_τ_a) + σ²(a_τ_b)) per ratio.")


if __name__ == "__main__":
    main()
