"""
M_act_5: stitch corner samples from the classical action MC to the QC
Minkowski leg.

For each sampled (U_full_lattice, Ψ_i, Ψ_j) from the Euclidean classical-MC
pipeline:
  - Extract gauge fields at the two temporal boundaries (t=0 and t=N_E−1)
  - Map (gauge_boundary, fermion_Ψ) into 8-qubit computational basis index
  - Compute ⟨Ψ_j|U_M†(t) O U_M(t)|Ψ_i⟩ via Trotter on H (reuse `qs` module)

Convention mapping (classical lattice ↔ z2_setup.py 8-qubit basis):
  q0 ↔ U_y[t, 0, 0]      (link (0,0)−(0,1)),  σ=+1 → bit=0, σ=−1 → bit=1
  q1 ↔ U_y[t, 1, 0]      (link (1,0)−(1,1))
  q2 ↔ U_x[t, 0, 0]      (link (0,0)−(1,0))
  q3 ↔ U_x[t, 0, 1]      (link (0,1)−(1,1))
  q4..q7 ↔ matter at (0,0), (0,1), (1,0), (1,1)  [bits 0..3 of Ψ]
"""
from __future__ import annotations
import sys
import numpy as np
sys.path.insert(0, '/home/hlamm/Desktop/QC/logdet/m1_toy')
import epoq_quantum_simulator as qs

from action_z2_staggered import LatticeGeometry, Z2GaugeConfig


def gauge_qc_bits_from_slice(U: Z2GaugeConfig, t_slice: int) -> int:
    """Extract 4-bit QC link-qubit value from gauge config at temporal slice t_slice.

    Layout per z2_setup.py convention (2×2 OBC only):
      q0 = U_y[t, 0, 0],  q1 = U_y[t, 1, 0],
      q2 = U_x[t, 0, 0],  q3 = U_x[t, 0, 1].
    Encoding: σ = +1 → bit 0,  σ = −1 → bit 1.

    For general Lx × Ly use gauge_qc_bits_general instead — this function
    is kept for backward-compatible 2×2 production scripts (gives the
    same bit string as the general version on 2×2).
    """
    def bit(σ):
        return 0 if σ == 1 else 1
    b0 = bit(U.U_y[t_slice, 0, 0])
    b1 = bit(U.U_y[t_slice, 1, 0])
    b2 = bit(U.U_x[t_slice, 0, 0])
    b3 = bit(U.U_x[t_slice, 0, 1])
    return b0 | (b1 << 1) | (b2 << 2) | (b3 << 3)


def qc_layout_counts(geom: LatticeGeometry) -> tuple[int, int]:
    """Return (n_gauge_qubits, n_matter_qubits) for the general qubit layout.

    Layout (extends the 2×2 convention; identical bit positions at Lx=Ly=2):
      - y-link qubits, positions [0, N_y):     q = x * (Ly - 1) + y
                                                for x ∈ [0,Lx), y ∈ [0,Ly-1)
      - x-link qubits, positions [N_y, N_g):   q = N_y + x * Ly + y
                                                for x ∈ [0,Lx-1), y ∈ [0,Ly)
      - matter qubits, positions [N_g, NQ):    q = N_g + x * Ly + y
                                                for x ∈ [0,Lx), y ∈ [0,Ly)

    where N_y = Lx · (Ly-1), N_x = (Lx-1) · Ly, N_g = N_y + N_x.
    """
    Lx, Ly = geom.Lx, geom.Ly
    n_y = Lx * (Ly - 1)
    n_x = (Lx - 1) * Ly
    return (n_y + n_x, Lx * Ly)


def gauge_qc_bits_general(U: Z2GaugeConfig, t_slice: int,
                          geom: LatticeGeometry) -> int:
    """Generalized 2×2 → arbitrary Lx × Ly OBC gauge-bit extractor.

    Bit positions match qc_layout_counts() — y-links first (varying y inner,
    x outer), then x-links (varying y inner, x outer).  Encoding σ=+1→0,
    σ=−1→1.  Equivalent to gauge_qc_bits_from_slice at Lx=Ly=2 (bit-for-bit).
    """
    Lx, Ly = geom.Lx, geom.Ly
    n_y = Lx * (Ly - 1)
    bits = 0
    for x in range(Lx):
        for y in range(Ly - 1):
            pos = x * (Ly - 1) + y
            b = 0 if U.U_y[t_slice, x, y] == 1 else 1
            bits |= b << pos
    for x in range(Lx - 1):
        for y in range(Ly):
            pos = n_y + x * Ly + y
            b = 0 if U.U_x[t_slice, x, y] == 1 else 1
            bits |= b << pos
    return bits


def qc_basis_index(geom: LatticeGeometry, U: Z2GaugeConfig,
                   t_slice: int, psi_fermion: int) -> int:
    """Build 8-bit QC computational basis index for state |gauge ⊗ fermion⟩."""
    assert geom.Lx == 2 and geom.Ly == 2, "stitch hard-coded for 2×2 spatial"
    gauge_bits = gauge_qc_bits_from_slice(U, t_slice)
    # Fermion bits go into q4..q7, taking bits 0..3 of psi_fermion in order
    fermion_bits = psi_fermion & 0xF
    return gauge_bits | (fermion_bits << 4)


def matrix_element_minkowski(
    geom: LatticeGeometry, U: Z2GaugeConfig,
    psi_i: int, psi_j: int,
    t: float, n_trotter: int = 200,
) -> complex:
    """⟨Ψ_j|U_M†(t) O U_M(t)|Ψ_i⟩ where O = n_0 (occupation at site (0,0))
    and the boundary states use gauge at t=0 and t=N_E−1 respectively.
    """
    i_bit = qc_basis_index(geom, U, 0, psi_i)
    j_bit = qc_basis_index(geom, U, geom.N_E - 1, psi_j)
    n0_obs = [(0.5 + 0j, []), (-0.5 + 0j, [(4, 'Z')])]
    return qs.observable_matrix_element(j_bit, i_bit, n0_obs, t, n_trotter=n_trotter)


def static_observable_diagonal(
    geom: LatticeGeometry, U: Z2GaugeConfig, psi: int,
) -> int:
    """O_i = ⟨Ψ_i|n_0|Ψ_i⟩ = matter occupation at site (0,0) in Ψ.

    Bit 0 of psi is matter at (0,0).
    """
    return (psi >> 0) & 1


def _self_test():
    geom = LatticeGeometry(Lx=2, Ly=2, N_E=4, m=0.5)
    rng = np.random.default_rng(2026)
    U = Z2GaugeConfig.random(geom, rng)

    # Check gauge slice bit-encoding for one slice
    bits = gauge_qc_bits_from_slice(U, 0)
    print(f"Gauge slice at t=0 ↔ 4-bit QC link config: {bits:04b}")
    # Each link's σ value
    print(f"  U_y[0,0,0] = {U.U_y[0,0,0]:+d} → q0 bit = {bits & 1}")
    print(f"  U_y[0,1,0] = {U.U_y[0,1,0]:+d} → q1 bit = {(bits >> 1) & 1}")
    print(f"  U_x[0,0,0] = {U.U_x[0,0,0]:+d} → q2 bit = {(bits >> 2) & 1}")
    print(f"  U_x[0,0,1] = {U.U_x[0,0,1]:+d} → q3 bit = {(bits >> 3) & 1}")

    # Try a few Ψ_i, Ψ_j matrix elements
    print("\nMatrix elements at t = 0 (should give ⟨Ψ_j|O|Ψ_i⟩):")
    print(f"  {'Ψ_i':>6} | {'Ψ_j':>6} | {'i_bit':>6} | {'j_bit':>6} | {'value':>15}")
    for psi_i, psi_j in [(0b0001, 0b0001), (0b0010, 0b0010), (0b0001, 0b0010), (0b0000, 0b0000)]:
        me = matrix_element_minkowski(geom, U, psi_i, psi_j, t=0.0)
        i_bit = qc_basis_index(geom, U, 0, psi_i)
        j_bit = qc_basis_index(geom, U, geom.N_E - 1, psi_j)
        print(f"  {psi_i:>06b} | {psi_j:>06b} | {i_bit:>6d} | {j_bit:>6d} | "
              f"{me.real:+.4f}{me.imag:+.4f}j")

    print("\nAt t=0, U†OU = O. So ⟨Ψ_j|O|Ψ_i⟩ depends on whether the gauge\n"
          "content at t=0 matches that at t=N_E−1 (since O is diagonal in matter\n"
          "but acts as I on gauge).")

    bits_top = gauge_qc_bits_from_slice(U, 0)
    bits_bot = gauge_qc_bits_from_slice(U, geom.N_E - 1)
    print(f"\nGauge t=0:        bits = {bits_top:04b}")
    print(f"Gauge t=N_E−1:    bits = {bits_bot:04b}")
    print(f"Same?  {bits_top == bits_bot}")
    print("If different, gauge mismatch suppresses matrix elements.")


if __name__ == '__main__':
    _self_test()
