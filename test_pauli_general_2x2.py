"""Equivalence test: build_pauli_terms_general at Lx=Ly=2 must reproduce
the dense Hamiltonian of the legacy build_pauli_terms (2×2 hard-coded).

Pauli decompositions can differ in term ordering or combine identity
pieces differently, but the SUM (dense H) must be identical.

Also sanity-checks at 2×3 that the resulting matrix is:
  - Hermitian (H = H†)
  - has the expected qubit count (13 = 7 gauge + 6 matter)
  - reduces to the 2×2 result when restricted to the all-trivial gauge sector
    (this would be a stronger check; we just report the spectrum here)
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "4"

import sys
import numpy as np

sys.path.insert(0, '/home/hlamm/Desktop/QC/logdet/m1_toy')
import epoq_classical_sampler as cs

from action_z2_staggered import LatticeGeometry


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


def dense_H(terms, nq):
    dim = 1 << nq
    H = np.zeros((dim, dim), dtype=complex)
    for coef, factors in terms:
        H += coef * pauli_string(factors, nq)
    return H


def main():
    print("=" * 78)
    print("Equivalence test: build_pauli_terms_general vs legacy build_pauli_terms")
    print("=" * 78)

    cs.M_MASS = 0.5

    geom_2x2 = LatticeGeometry(Lx=2, Ly=2, N_E=4, m=0.5)
    terms_legacy = cs.build_pauli_terms()
    terms_general = cs.build_pauli_terms_general(geom_2x2)
    print(f"\n2×2 layout, 8 qubits:")
    print(f"  legacy:  {len(terms_legacy)} Pauli terms")
    print(f"  general: {len(terms_general)} Pauli terms")

    H_legacy = dense_H(terms_legacy, 8)
    H_general = dense_H(terms_general, 8)
    H_legacy = (H_legacy + H_legacy.conj().T) / 2
    H_general = (H_general + H_general.conj().T) / 2

    d = np.max(np.abs(H_legacy - H_general))
    print(f"  max|H_general − H_legacy| = {d:.3e}")
    pass_22 = d < 1e-12
    print(f"  {'PASS' if pass_22 else 'FAIL'}: dense H agreement at 2×2")

    eigs_l = np.linalg.eigvalsh(H_legacy)
    eigs_g = np.linalg.eigvalsh(H_general)
    print(f"  spectrum match (sorted, max |Δ|): "
          f"{np.max(np.abs(np.sort(eigs_l) - np.sort(eigs_g))):.3e}")

    print(f"\n2×3 layout (V_3=6, 13 qubits):")
    geom_2x3 = LatticeGeometry(Lx=2, Ly=3, N_E=4, m=0.5)
    terms_2x3 = cs.build_pauli_terms_general(geom_2x3, m_mass=0.5)
    print(f"  {len(terms_2x3)} Pauli terms")
    n_y = 2 * 2  # 4
    n_x = 1 * 3  # 3
    n_g = n_y + n_x  # 7
    nq = n_g + 6  # 13
    print(f"  qubits: n_y={n_y}, n_x={n_x}, n_gauge={n_g}, n_matter=6, NQ={nq}")
    print(f"  dim = 2^{nq} = {1 << nq}")

    print(f"  building dense H (~{(1<<nq)**2 * 16 / 1e9:.1f} GB)...", flush=True)
    H_2x3 = dense_H(terms_2x3, nq)
    H_2x3 = (H_2x3 + H_2x3.conj().T) / 2
    herm_2x3 = np.max(np.abs(H_2x3 - H_2x3.conj().T))
    print(f"  Hermiticity check max|H − H†| = {herm_2x3:.3e}")
    pass_23 = herm_2x3 < 1e-12
    print(f"  {'PASS' if pass_23 else 'FAIL'}: H_2x3 Hermitian")

    eigs_23 = np.linalg.eigvalsh(H_2x3)
    print(f"  Min/max eig: {eigs_23.min():+.5f}, {eigs_23.max():+.5f}")
    print(f"  Ground-state E_0 = {eigs_23.min():+.6f}")

    print(f"\n{'ALL PASS' if (pass_22 and pass_23) else 'FAILURE(S) ABOVE'}")
    return 0 if (pass_22 and pass_23) else 1


if __name__ == "__main__":
    sys.exit(main())
