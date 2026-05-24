"""Phase 37: at the gauge sector ONLY, compare:
  (A) lattice gauge transition operator T_E_lat[g', g] from the Wilson
      temporal plaquette action, summed over U_t variables for fixed
      spatial (g, g'). Also compute its temporal-gauge version (U_t=+1).
  (B) Hamiltonian transition matrix element ⟨g'|e^{-a_τH_E}|g⟩ =
      ∏_l (cosh or sinh based on per-link flip).

Per Suzuki, (A) should equal (B) up to a gauge-independent normalization
constant (per-link factor c per transition).

If they MATCH up to constant → gauge sector is correctly mapped; bug is
elsewhere.
If they DON'T match → Wilson plaquette + U_t integration doesn't give
per-link Suzuki bonds, contrary to my analysis.
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import sys
import numpy as np
from scipy.linalg import expm

sys.path.insert(0, '/home/hlamm/Desktop/QC/logdet/m1_toy')

A_TAU = 1.0
G_E = 1.0
K_E = -0.5 * np.log(np.tanh(A_TAU * G_E))

# 16-dim gauge sector (4 link qubits)
# Use 4-bit gauge state convention: bit i (LSB-first) = σ_i.
# σ_i = (1 - 2*bit_i) ∈ {+1, -1}.
def bit_to_sigma(state, i):
    return 1.0 - 2.0 * ((state >> i) & 1)


def hamiltonian_TE_4link():
    """16x16 matrix of e^{-a_τH_E} where H_E = -g_E·Σ_l X_l on 4 link qubits.

    Per link l: e^{a_τg_E·X_l}. Matrix element ⟨σ'|...|σ⟩ = cosh(a_τg_E) for
    σ'=σ, sinh(a_τg_E) for σ'=-σ. For 4 independent links → 16x16 tensor product.
    """
    # 2x2 single-link matrix
    cosh_val = np.cosh(A_TAU * G_E)
    sinh_val = np.sinh(A_TAU * G_E)
    single = np.array([[cosh_val, sinh_val], [sinh_val, cosh_val]], dtype=complex)
    # Tensor 4 of them, qubit 0 = LSB
    T = single
    for _ in range(3):
        T = np.kron(single, T)
    return T


def lattice_TE_temporal_gauge_one_transition():
    """16x16 matrix where M[g', g] = product over 4 links of e^{K_E·σ_l(g)·σ_l(g')}.

    This is the lattice contribution to ⟨g'|T_E|g⟩ for one transition in
    TEMPORAL GAUGE (U_t = +1), with no K_M.  No normalization included.
    """
    T = np.zeros((16, 16), dtype=complex)
    for g in range(16):
        for gp in range(16):
            S = 0.0
            for l in range(4):
                σ = bit_to_sigma(g, l)
                σp = bit_to_sigma(gp, l)
                S += σ * σp
            T[gp, g] = np.exp(K_E * S)
    return T


def lattice_TE_free_U_t_one_transition():
    """16x16 matrix from Wilson 4-link temporal plaquettes, with U_t INTEGRATED.

    For one transition, the 4 spatial-temporal plaquettes are:
      xτ at y=0: u_x(t,0,0) · u_t(t,1,0) · u_x(t+1,0,0) · u_t(t,0,0)
      xτ at y=1: u_x(t,0,1) · u_t(t,1,1) · u_x(t+1,0,1) · u_t(t,0,1)
      yτ at x=0: u_y(t,0,0) · u_t(t,0,1) · u_y(t+1,0,0) · u_t(t,0,0)
      yτ at x=1: u_y(t,1,0) · u_t(t,1,1) · u_y(t+1,1,0) · u_t(t,1,0)

    Sum over 4 U_t variables (16 configs).  No K_M, no normalization.

    Gauge convention (matches gauge_qc_bits_from_slice):
      bit 0 = U_y[0, 0]   (q0)
      bit 1 = U_y[1, 0]   (q1)
      bit 2 = U_x[0, 0]   (q2)
      bit 3 = U_x[0, 1]   (q3)
    """
    T = np.zeros((16, 16), dtype=complex)
    for g in range(16):
        # σ_q's of g (slice t):
        σ_y00_t = bit_to_sigma(g, 0)
        σ_y10_t = bit_to_sigma(g, 1)
        σ_x00_t = bit_to_sigma(g, 2)
        σ_x01_t = bit_to_sigma(g, 3)
        for gp in range(16):
            σ_y00_tp = bit_to_sigma(gp, 0)
            σ_y10_tp = bit_to_sigma(gp, 1)
            σ_x00_tp = bit_to_sigma(gp, 2)
            σ_x01_tp = bit_to_sigma(gp, 3)
            # Spatial bond products A_i (for the 4 spatial bonds at t→t+1)
            A_xτ_y0 = σ_x00_t * σ_x00_tp   # u_x at y=0
            A_xτ_y1 = σ_x01_t * σ_x01_tp   # u_x at y=1
            A_yτ_x0 = σ_y00_t * σ_y00_tp   # u_y at x=0
            A_yτ_x1 = σ_y10_t * σ_y10_tp   # u_y at x=1
            # Sum over 4 U_t variables: u_t(t,0,0), u_t(t,0,1), u_t(t,1,0), u_t(t,1,1)
            S_sum = 0.0
            for σ_t000 in (-1, 1):
                for σ_t001 in (-1, 1):
                    for σ_t010 in (-1, 1):
                        for σ_t011 in (-1, 1):
                            # Plaquettes (with U_t):
                            P_xτ_y0 = A_xτ_y0 * σ_t010 * σ_t000  # σ_t(1,0) and σ_t(0,0)
                            P_xτ_y1 = A_xτ_y1 * σ_t011 * σ_t001  # σ_t(1,1) and σ_t(0,1)
                            P_yτ_x0 = A_yτ_x0 * σ_t001 * σ_t000  # σ_t(0,1) and σ_t(0,0)
                            P_yτ_x1 = A_yτ_x1 * σ_t011 * σ_t010  # σ_t(1,1) and σ_t(1,0)
                            S_sum += np.exp(K_E * (P_xτ_y0 + P_xτ_y1 + P_yτ_x0 + P_yτ_x1))
            T[gp, g] = S_sum
    return T


def main():
    print("=" * 80)
    print("Phase 37: gauge-sector T_E comparison")
    print(f"  a_τ={A_TAU}, g_E={G_E}, K_E={K_E:.4f}")
    print("=" * 80)

    print(f"\n[A1] Hamiltonian T_E = ∏_l e^{{a_τg_E X_l}}, per-link cosh/sinh")
    T_ham = hamiltonian_TE_4link()
    print(f"  Z (trace) = {np.trace(T_ham).real:.4f}")
    print(f"  Symmetric? {np.allclose(T_ham, T_ham.T)}")

    print(f"\n[A2] Lattice T_E (temporal gauge, no U_t fluctuation)")
    T_lat_temp = lattice_TE_temporal_gauge_one_transition()
    print(f"  Z (trace) = {np.trace(T_lat_temp).real:.4f}")

    print(f"\n[A3] Lattice T_E (free U_t, integrated over 4 U_t vars)")
    T_lat_free = lattice_TE_free_U_t_one_transition()
    print(f"  Z (trace) = {np.trace(T_lat_free).real:.4f}")

    # Per-link Suzuki: Hamiltonian = c^4 · lattice (per transition)
    # c per link = sqrt(cosh·sinh) = sqrt(sinh(2a_τg_E)/2)
    c_per_link = np.sqrt(np.cosh(A_TAU * G_E) * np.sinh(A_TAU * G_E))
    print(f"\n  Expected Suzuki normalization: c_per_link^4 = {c_per_link**4:.4f}")
    print(f"  Z_ham / Z_lat_temp = {np.trace(T_ham).real / np.trace(T_lat_temp).real:.4f}")
    print(f"  Z_ham / Z_lat_free = {np.trace(T_ham).real / np.trace(T_lat_free).real:.4f}")

    print(f"\n[Element-wise comparison: Ham vs Lat_temp]")
    # Ratio: Ham/Lat_temp (should be constant if proportional)
    ratio = T_ham / np.maximum(T_lat_temp.real, 1e-30)
    ratio_vals = ratio.flatten().real
    print(f"  ratio (Ham[i,j] / Lat_temp[i,j]) min={ratio_vals.min():.4f}, max={ratio_vals.max():.4f}")
    print(f"  ratio std/mean = {ratio_vals.std()/ratio_vals.mean():.4e}")

    # Normalized comparison
    T_ham_n = T_ham / np.trace(T_ham).real
    T_lat_temp_n = T_lat_temp / np.trace(T_lat_temp).real
    T_lat_free_n = T_lat_free / np.trace(T_lat_free).real
    print(f"\n  ‖T_ham_n − T_lat_temp_n‖_F = {np.linalg.norm(T_ham_n - T_lat_temp_n):.4e}")
    print(f"  ‖T_ham_n − T_lat_free_n‖_F = {np.linalg.norm(T_ham_n - T_lat_free_n):.4e}")
    print(f"  ‖T_lat_temp_n − T_lat_free_n‖_F = {np.linalg.norm(T_lat_temp_n - T_lat_free_n):.4e}")

    # Print a few representative matrix elements
    print(f"\n[Few elements: T_ham vs T_lat (raw)]")
    print(f"  diag g=0:    ham={T_ham[0,0].real:.4f}, temp={T_lat_temp[0,0].real:.4f}, free={T_lat_free[0,0].real:.4f}")
    print(f"  diag g=5:    ham={T_ham[5,5].real:.4f}, temp={T_lat_temp[5,5].real:.4f}, free={T_lat_free[5,5].real:.4f}")
    print(f"  diag g=15:   ham={T_ham[15,15].real:.4f}, temp={T_lat_temp[15,15].real:.4f}, free={T_lat_free[15,15].real:.4f}")
    print(f"  off g=0,gp=1: ham={T_ham[1,0].real:.4f}, temp={T_lat_temp[1,0].real:.4f}, free={T_lat_free[1,0].real:.4f}")
    print(f"  off g=0,gp=15: ham={T_ham[15,0].real:.4f}, temp={T_lat_temp[15,0].real:.4f}, free={T_lat_free[15,0].real:.4f}")


if __name__ == "__main__":
    main()
