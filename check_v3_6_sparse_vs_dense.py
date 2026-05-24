"""Validate sparse-Krylov Hutchinson C(t) against dense eigh at V_3=6.

At V_3=6 (dim 8192) we can do both — dense ED is the truth and sparse
ED is the candidate for scaling past V_3=6.  At larger V_3 only sparse
is feasible, so this is the only place we get a head-to-head test.

What we check:
  1. sparse H == dense H (max diff)
  2. C_dense(t) from eigh + eigenbasis trace (point estimate)
  3. C_sparse(t) from Hutchinson with various n_random ∈ {16, 32, 64}
     → mean ± σ_C
  4. Pull: |C_sparse − C_dense| / σ_C  should be O(1) (a few σ)

Couplings: same as check_v3_6.py (β=2, m=0.5).
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
from epoq_sparse_ed import compute_C_t_hutchinson


LX, LY = 2, 3
M_HAM, G_E_HAM, G_M_HAM, G_HOP = 0.5, 1.0, 0.5, 0.5
BETA = 2.0
TIMES = [0.0, 0.5, 1.0]

# Sparse and dense Pauli builders for the dense equivalence check
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
    print(f"Sparse-Krylov ED vs dense eigh at V_3=6  (Lx={LX}, Ly={LY})")
    print(f"  β={BETA}, m={M_HAM}, n_0 at site (0,0)")
    print("=" * 78, flush=True)

    geom = LatticeGeometry(Lx=LX, Ly=LY, N_E=1, m=M_HAM)
    n_gauge, n_matter = qc_layout_counts(geom)
    NQ = n_gauge + n_matter
    DIM = 1 << NQ
    print(f"\nQubits: n_gauge={n_gauge}, n_matter={n_matter}, NQ={NQ}, DIM={DIM}")

    # --- Sparse H ---
    print("\nBuilding sparse H...", flush=True)
    t0 = time.time()
    H_sparse = cs.build_sparse_H_QC(geom, g_e=G_E_HAM, g_m=G_M_HAM,
                                    g_hop=G_HOP, m_mass=M_HAM)
    t_sparse_build = time.time() - t0
    nnz = H_sparse.nnz
    sparse_mem_mb = (H_sparse.data.nbytes
                     + H_sparse.indices.nbytes
                     + H_sparse.indptr.nbytes) / 1024 / 1024
    print(f"  done in {t_sparse_build:.1f}s.  nnz={nnz}, "
          f"sparse mem ≈ {sparse_mem_mb:.1f} MB", flush=True)
    n0_sparse = cs.build_sparse_n0_at_site00(geom)
    print(f"  n_0 sparse: nnz={n0_sparse.nnz} (= matter occ at site (0,0))",
          flush=True)

    # --- Dense H for equivalence + dense ED truth ---
    print("\nBuilding dense H (only at V_3=6; serves as truth)...", flush=True)
    t0 = time.time()
    terms = cs.build_pauli_terms_general(
        geom, g_e=G_E_HAM, g_m=G_M_HAM, g_hop=G_HOP, m_mass=M_HAM)
    H_dense = sum(c * pauli_string(f, NQ) for c, f in terms)
    H_dense = (H_dense + H_dense.conj().T) / 2
    t_dense_build = time.time() - t0
    print(f"  done in {t_dense_build:.1f}s.", flush=True)

    # --- Sparse == dense ---
    print("\nEquivalence check sparse vs dense:")
    diff = np.max(np.abs(H_sparse.toarray() - H_dense))
    print(f"  max|H_sparse − H_dense| = {diff:.3e}    "
          f"({'PASS' if diff < 1e-12 else 'FAIL'})", flush=True)

    # --- Dense ED via eigh + eigenbasis (truth) ---
    print("\nDense ED reference (eigh + eigenbasis)...", flush=True)
    t0 = time.time()
    eigs, V_eig = np.linalg.eigh(H_dense)
    print(f"  eigh done in {time.time()-t0:.1f}s. "
          f"E_0={eigs[0]:+.4f}, E_max={eigs[-1]:+.4f}", flush=True)

    n0_op = 0.5 * np.eye(DIM, dtype=complex) - 0.5 * pauli_string(
        [(n_gauge, "Z")], NQ)
    n0_eig = V_eig.conj().T @ n0_op @ V_eig
    w_beta = np.exp(-BETA * eigs)
    Z_beta = w_beta.sum()
    C_dense = {}
    for t in TIMES:
        phase = np.exp(-1j * eigs * t)
        UOU_eig = (phase.conj()[:, None] * n0_eig) * phase[None, :]
        O_eig = UOU_eig @ n0_eig
        C_dense[t] = float((w_beta * np.diag(O_eig).real).sum() / Z_beta)
    print(f"  Dense C(t):   "
          + "  ".join(f"C({t})={C_dense[t]:+.5f}" for t in TIMES),
          flush=True)

    # --- Sparse Hutchinson at various n_random ---
    print(f"\nSparse Hutchinson C(t) (β={BETA}):")
    nrandom_list = [16, 32, 64]
    results = {}
    for nr in nrandom_list:
        print(f"\n  n_random = {nr}:", flush=True)
        t0 = time.time()
        C, sigma_C, raw = compute_C_t_hutchinson(
            H_sparse, n0_sparse, beta=BETA, times=TIMES,
            n_random=nr, seed=20260524,
            progress=True, progress_every=8,
        )
        t_hutch = time.time() - t0
        print(f"  {nr} samples in {t_hutch:.1f}s "
              f"({t_hutch/nr:.2f}s/sample)", flush=True)
        results[nr] = (C, sigma_C, t_hutch)

    # --- Pull comparison ---
    print("\n" + "=" * 78)
    print(f"Comparison C_sparse − C_dense  (|pull| = |Δ|/σ_C should be O(few))")
    print("=" * 78)
    for nr in nrandom_list:
        C_sp, sigma_C, _ = results[nr]
        print(f"\nn_random = {nr}:")
        print(f"  {'t':>5} | {'C_sparse':>11} | {'σ_C':>9} | {'C_dense':>11} | "
              f"{'Δ':>10} | {'pull':>7}")
        max_pull = 0.0
        for t in TIMES:
            d = C_sp[t] - C_dense[t]
            pull = d / sigma_C[t] if sigma_C[t] > 0 else 0.0
            max_pull = max(max_pull, abs(pull))
            print(f"  {t:>5.2f} | {C_sp[t]:+.7f} | {sigma_C[t]:.3e} | "
                  f"{C_dense[t]:+.7f} | {d:+.3e} | {pull:+.2f}")
        print(f"  max |pull| = {max_pull:.2f}  "
              f"({'PASS' if max_pull < 4 else 'WARN'})")


if __name__ == "__main__":
    main()
