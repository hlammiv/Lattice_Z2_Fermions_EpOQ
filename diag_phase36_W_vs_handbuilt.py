"""Phase 36: at a single fixed gauge config U, compare
  (1) lattice W = T_F[g(1)]·T_F[g(0)] from compute_combined_weight_trotter
  (2) independently-built W = expm(-a_τH_F[g(1)])·expm(-a_τH_F[g(0)])
      with H_F[g] constructed directly from cs.build_pauli_terms by
      replacing Z_l on link qubits with σ_l(g) c-numbers.

If (1) ≈ (2) → no bug in T_F; structural error is in the path-integral
derivation (gauge transitions, action structure).
If (1) ≠ (2) → implementation bug in build_H_lat_slice or
compute_combined_weight_trotter.
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import sys
import numpy as np
from scipy.linalg import expm

sys.path.insert(0, '/home/hlamm/Desktop/QC/logdet/m1_toy')

import epoq_classical_sampler as cs
cs.M_MASS = 0.5

from action_z2_staggered import LatticeGeometry, Z2GaugeConfig
from transfer_matrix_kbc_trotterized import compute_combined_weight_trotter, build_T_F_trotter
from action_corner_direct_v3 import _fock_index_to_psi_map

A_TAU = 1.0
M_HAM = 0.5
G_HOP = 0.5


def pauli_op_4q(factors):
    """Build 16x16 Pauli string on 4 matter qubits (q0..q3 in matter-only space).
    q0 = LSB of the matter int (like q4 in the full 8-qubit space).
    """
    P2 = {'I': np.eye(2, dtype=complex),
          'X': np.array([[0, 1], [1, 0]], dtype=complex),
          'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
          'Z': np.diag([1, -1]).astype(complex)}
    by_q = {q: P2['I'] for q in range(4)}
    for q, ax in factors:
        by_q[q] = P2[ax]
    r = by_q[3]
    for q in range(2, -1, -1):
        r = np.kron(r, by_q[q])
    return r


def build_H_F_independent(U_x_slice, U_y_slice):
    """Build 16x16 H_F[g] at a given spatial gauge eigenstate by replacing
    Z_l on link qubits with σ_l(g) in build_pauli_terms.

    Conventions follow epoq_classical_sampler:
      - Matter qubit q4 corresponds to site (0,0); q5 to (0,1); etc.
      - In the 4-q matter-only space, "matter qubit q_(4+a)" → q_a (site a).
      - Gauge link qubits 0..3 → c-numbers σ_l.

    Link gauge values σ_l (from U_x/U_y slices):
      q0 = U_y[0, 0]  (y-link at x=0)
      q1 = U_y[1, 0]  (y-link at x=1)
      q2 = U_x[0, 0]  (x-link at y=0)
      q3 = U_x[0, 1]  (x-link at y=1)
    (Same as gauge_qc_bits_from_slice; σ=+1→bit 0, σ=-1→bit 1.)
    """
    sigma = {
        0: float(U_y_slice[0, 0]),
        1: float(U_y_slice[1, 0]),
        2: float(U_x_slice[0, 0]),
        3: float(U_x_slice[0, 1]),
    }

    H = np.zeros((16, 16), dtype=complex)
    const_shift = 0.0

    terms = cs.build_pauli_terms()
    for coef, ops in terms:
        # Separate gauge ops (q<4) from matter ops (q>=4).
        gauge_z_factor = 1.0
        matter_ops_raw = []  # list of (q_in_8q, axis)
        skip = False
        for q, axis in ops:
            if q < 4:
                # Must be Z for a gauge-only insertion; X means H_E term
                if axis == 'X':
                    skip = True
                    break
                elif axis == 'Z':
                    gauge_z_factor *= sigma[q]
                else:
                    raise ValueError(f"unexpected gauge op {axis} on q{q}")
            else:
                # Matter qubit q (>= 4); map to matter-only space q-4
                matter_ops_raw.append((q - 4, axis))
        if skip:
            # Pure-gauge H_E term (single X on link qubit) — skip for H_F
            continue
        # Skip pure-magnetic H_M term (Z_0Z_1Z_2Z_3 on gauge, no matter ops)
        if not matter_ops_raw and gauge_z_factor != 0.0:
            if all(q < 4 and a == 'Z' for q, a in ops) and len(ops) >= 2:
                continue
        eff_coef = coef * gauge_z_factor
        if not matter_ops_raw:
            # Identity term (constant shift on matter)
            const_shift += eff_coef.real
            continue
        H += eff_coef * pauli_op_4q(matter_ops_raw)
    H += const_shift * np.eye(16, dtype=complex)
    H = (H + H.conj().T) / 2
    return H


def main():
    print("=" * 80)
    print("Phase 36: lattice W vs independently-built W from build_pauli_terms")
    print(f"  m={M_HAM}, a_τ={A_TAU}")
    print("=" * 80, flush=True)

    geom = LatticeGeometry(Lx=2, Ly=2, N_E=3, m=A_TAU * M_HAM)

    # Pick a non-trivial gauge config so we can spot bugs that vanish on trivial
    rng = np.random.default_rng(42)
    U = Z2GaugeConfig.random(geom, rng)
    print(f"\nUsing gauge config with U_x={U.U_x.flatten()}, U_y={U.U_y.flatten()}", flush=True)

    # --- (A) lattice path: T_F[g(0)] and T_F[g(1)] from build_T_F_trotter ---
    print(f"\n[A] Lattice path: build_T_F_trotter at each slice (lex basis)")
    T_F_lat_0_lex = build_T_F_trotter(
        geom, U.U_x[0, :, :], U.U_y[0, :, :],
        a_tau=A_TAU, m=M_HAM, K_E=0.0, K_M=0.0, g_hop=G_HOP)
    T_F_lat_1_lex = build_T_F_trotter(
        geom, U.U_x[1, :, :], U.U_y[1, :, :],
        a_tau=A_TAU, m=M_HAM, K_E=0.0, K_M=0.0, g_hop=G_HOP)

    W_lat_lex = T_F_lat_1_lex @ T_F_lat_0_lex

    # Re-index to psi basis using _fock_index_to_psi_map
    idx_to_psi = _fock_index_to_psi_map(4)
    def lex_to_psi(M_lex):
        M_psi = np.zeros_like(M_lex)
        for li in range(16):
            for lj in range(16):
                M_psi[idx_to_psi[li], idx_to_psi[lj]] = M_lex[li, lj]
        return M_psi
    W_lat = lex_to_psi(W_lat_lex)
    T_F_lat_0 = lex_to_psi(T_F_lat_0_lex)
    T_F_lat_1 = lex_to_psi(T_F_lat_1_lex)
    print(f"  W_lat (psi basis): shape {W_lat.shape}, trace = {np.trace(W_lat).real:.5f}")

    # --- (B) independent path: H_F[g] from build_pauli_terms ---
    print(f"\n[B] Independent path: H_F[g] from cs.build_pauli_terms (matter q0=LSB)")
    H_F_indep_0 = build_H_F_independent(U.U_x[0, :, :], U.U_y[0, :, :])
    H_F_indep_1 = build_H_F_independent(U.U_x[1, :, :], U.U_y[1, :, :])
    T_F_indep_0 = expm(-A_TAU * H_F_indep_0)
    T_F_indep_1 = expm(-A_TAU * H_F_indep_1)
    W_indep = T_F_indep_1 @ T_F_indep_0
    print(f"  W_indep (psi basis): shape {W_indep.shape}, trace = {np.trace(W_indep).real:.5f}")

    # --- Compare ---
    print(f"\n[Comparisons]")
    print(f"  ‖T_F_lat_0 − T_F_indep_0‖_F = {np.linalg.norm(T_F_lat_0 - T_F_indep_0):.4e}")
    print(f"  ‖T_F_lat_1 − T_F_indep_1‖_F = {np.linalg.norm(T_F_lat_1 - T_F_indep_1):.4e}")
    print(f"  ‖W_lat − W_indep‖_F          = {np.linalg.norm(W_lat - W_indep):.4e}")
    print(f"  max|T_F_lat_0 − T_F_indep_0| = {np.max(np.abs(T_F_lat_0 - T_F_indep_0)):.4e}")
    print(f"  max|W_lat − W_indep|         = {np.max(np.abs(W_lat - W_indep)):.4e}")

    # Inspect H_F directly
    from transfer_matrix_kbc_trotterized import build_H_lat_slice
    H_lat_0_lex = build_H_lat_slice(geom, U.U_x[0, :, :], U.U_y[0, :, :],
                                     m=M_HAM, K_E=0.0, K_M=0.0, g_hop=G_HOP)
    H_lat_0 = lex_to_psi(H_lat_0_lex)
    print(f"\n  ‖H_lat[0] − H_F_indep[0]‖_F = {np.linalg.norm(H_lat_0 - H_F_indep_0):.4e}")
    print(f"  max|H_lat[0] − H_F_indep[0]| = {np.max(np.abs(H_lat_0 - H_F_indep_0)):.4e}")

    # Spectrum comparison (gauge-invariant: eigenvalues should agree)
    eigs_lat = np.linalg.eigvalsh(H_lat_0)
    eigs_indep = np.linalg.eigvalsh(H_F_indep_0)
    eigs_lat.sort()
    eigs_indep.sort()
    print(f"\n  Spectrum H_lat[0]:  {eigs_lat.real}")
    print(f"  Spectrum H_indep[0]: {eigs_indep.real}")
    print(f"  max |eig diff| = {np.max(np.abs(eigs_lat - eigs_indep)):.4e}")


if __name__ == "__main__":
    main()
