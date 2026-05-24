"""Phase 40 step 3 test: build_H_lat_slice + build_pauli_terms_general
agree at 3D (2x2x2) on the trivial-gauge sector.

Strategy:
  1. Lattice path: build_H_lat_slice at trivial 3D gauge → V_3=8 = dim 256
     matrix in Fock-lex basis.  Verify Hermitian, sensible spectrum.
  2. QC path: build_sparse_H_QC at 3D (NQ=20, dim 2^20=1M sparse), then
     restrict to all-zero gauge bit sector → 256×256 dense submatrix.
  3. Compare spectra (must agree modulo Fock-lex ↔ matter-bit permutation).
     The bitcount of each Fock state determines the matter occupation
     vector, which corresponds to a unique 8-bit matter index in the
     QC basis (with the appropriate Fock-lex ↔ q_m(x,y,z) bit reordering).

If spectra agree, the lattice and QC sides are consistent at 3D.
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "4"

import sys
import numpy as np

sys.path.insert(0, '/home/hlamm/Desktop/QC/logdet/m1_toy')

import epoq_classical_sampler as cs
cs.M_MASS = 0.5

from action_z2_staggered import LatticeGeometry, Z2GaugeConfig
from action_minkowski_stitch import qc_layout_counts
from transfer_matrix_kbc_trotterized import build_H_lat_slice


def main():
    print("=" * 70)
    print("Phase 40 step 3: 3D lattice H_KS vs Pauli H_QC trivial-gauge sector")
    print("=" * 70)

    geom_3d = LatticeGeometry(Lx=2, Ly=2, Lz=2, N_E=1, m=0.5)
    Lx, Ly, Lz = 2, 2, 2
    V3 = 8
    dim_fock = 1 << V3
    n_gauge, n_matter = qc_layout_counts(geom_3d)
    NQ = n_gauge + n_matter
    DIM = 1 << NQ
    print(f"\n3D 2×2×2: V_3={V3}, n_gauge={n_gauge}, n_matter={n_matter}, "
          f"NQ={NQ}, DIM={DIM}")

    # --- 1. Lattice H_lat_slice at trivial gauge ---
    U_3d = Z2GaugeConfig.trivial(geom_3d)
    print(f"\nLattice H_lat_slice (trivial 3D) ...", flush=True)
    H_lat = build_H_lat_slice(
        geom_3d, U_3d.U_x[0], U_3d.U_y[0], m=0.5,
        K_E=1.0, K_M=0.5, g_hop=0.5, U_z_slice=U_3d.U_z[0])
    print(f"  shape: {H_lat.shape}")
    herm_err = np.max(np.abs(H_lat - H_lat.conj().T))
    print(f"  max|H - H†| = {herm_err:.3e}")
    assert herm_err < 1e-12
    eigs_lat = np.linalg.eigvalsh(H_lat).real
    print(f"  Spectrum: min={eigs_lat.min():+.5f}, max={eigs_lat.max():+.5f}, "
          f"len={len(eigs_lat)}")

    # --- 2. QC sparse H restricted to trivial gauge sector ---
    print(f"\nQC sparse H at 3D 2×2×2 (dim {DIM})...", flush=True)
    import time
    t0 = time.time()
    H_sparse = cs.build_sparse_H_QC(geom_3d, g_e=1.0, g_m=0.5,
                                     g_hop=0.5, m_mass=0.5)
    print(f"  built in {time.time()-t0:.1f}s.  nnz={H_sparse.nnz}",
          flush=True)
    # Trivial gauge sector indices: gauge bits all zero.  In bit layout,
    # gauge occupies bits 0..n_gauge-1; matter occupies bits n_gauge..NQ-1.
    # All-zero gauge: state index = matter << n_gauge, matter ∈ [0, 2^V_3).
    matter_indices = np.arange(dim_fock, dtype=np.int64) << n_gauge
    # Extract 256x256 submatrix
    print(f"  Extracting all-zero gauge sector submatrix ...", flush=True)
    t0 = time.time()
    H_qc_sector = H_sparse[matter_indices, :][:, matter_indices].toarray()
    print(f"  done in {time.time()-t0:.1f}s.", flush=True)
    H_qc_sector = (H_qc_sector + H_qc_sector.conj().T) / 2.0
    eigs_qc = np.linalg.eigvalsh(H_qc_sector).real
    print(f"  Spectrum: min={eigs_qc.min():+.5f}, max={eigs_qc.max():+.5f}, "
          f"len={len(eigs_qc)}")

    # --- 3. Compare spectra (sorted; permutation invariance) ---
    diff_spec = np.max(np.abs(np.sort(eigs_lat) - np.sort(eigs_qc)))
    print(f"\nmax |Δ sorted eigvals|: {diff_spec:.3e}", flush=True)
    if diff_spec < 1e-10:
        print("  PASS: lattice and QC trivial-gauge spectra match")
    else:
        print("  FAIL: spectra differ — check qubit layout / hopping conventions")
        return 1

    # --- 4. Random 3D config: lattice H_lat_slice is Hermitian ---
    print(f"\nRandom 3D config: lattice H_lat_slice Hermitian check ...",
          flush=True)
    rng = np.random.default_rng(7)
    U_rand_3d = Z2GaugeConfig.random(geom_3d, rng)
    H_rand = build_H_lat_slice(
        geom_3d, U_rand_3d.U_x[0], U_rand_3d.U_y[0], m=0.5,
        K_E=1.0, K_M=0.5, g_hop=0.5, U_z_slice=U_rand_3d.U_z[0])
    herm_err_rand = np.max(np.abs(H_rand - H_rand.conj().T))
    eigs_rand = np.linalg.eigvalsh(H_rand).real
    print(f"  max|H - H†| = {herm_err_rand:.3e}, "
          f"E_min={eigs_rand.min():+.5f}, E_max={eigs_rand.max():+.5f}")
    assert herm_err_rand < 1e-12

    # --- 5. 2D backward compat: 2x2 builder still matches legacy ---
    geom_2d = LatticeGeometry(Lx=2, Ly=2, N_E=1, m=0.5)
    terms_general_2d = cs.build_pauli_terms_general(geom_2d,
                                                     g_e=1.0, g_m=0.5,
                                                     g_hop=0.5, m_mass=0.5)
    terms_legacy_2d = cs.build_pauli_terms()
    # Same number of terms expected
    assert len(terms_general_2d) == len(terms_legacy_2d), (
        f"Pauli term count: general {len(terms_general_2d)} vs "
        f"legacy {len(terms_legacy_2d)}"
    )
    print(f"\n2D Pauli builder (Lz=1) still gives {len(terms_general_2d)} terms "
          f"(matches legacy).  PASS")

    print("\n" + "=" * 70)
    print("Phase 40 step 3 — ALL PASS")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
