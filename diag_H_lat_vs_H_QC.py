"""
diag_H_lat_vs_H_QC.py — diagnostic: compare action-extracted H_lat
to z2_setup's H_QC.

Setup: V_3=4 spatial (2x2 OBC), trivial gauge config U=I.

H_lat from action:
    1. Build M-K T̂_F² (256x256, positive-def).
    2. Take matrix log: H_lat_aux = −log(T̂_F²)/(2·a_τ) on (χ, φ) Fock.
    3. Project to φ=0 subspace: H_lat_chi (16x16) acting on χ-Fock only.
    4. Compute thermal observables (β=4).

H_QC from z2_setup:
    1. Build_pauli_terms → 256x256 H (gauge + fermion qubits).
    2. Restrict to "all-up gauge eigenstate" sector (Z_l = +1 for all l): 16x16
       on fermion qubits.
    3. Thermal observables (β=4).

Diagnostic comparisons:
    (a) Spectrum of H_lat_chi vs H_QC_fermion_sector
    (b) Thermal ⟨n̂_0⟩ from each at β=4
    (c) Ground states overlap

A large discrepancy indicates the action's discretization gives a different
lattice Hamiltonian than z2_setup intends.
"""
from __future__ import annotations
import sys
import numpy as np
from scipy.linalg import expm, logm

sys.path.insert(0, '/home/hlamm/Desktop/QC/logdet/m1_toy')
from action_z2_staggered import LatticeGeometry, Z2GaugeConfig
from transfer_matrix_mk import (
    build_fermion_ops, mk_T_F_single_step, project_to_phi_vacuum,
)


def build_H_lat_chi_from_action(geom: LatticeGeometry, U: Z2GaugeConfig,
                                a_tau: float = 1.0,
                                phi_treatment: str = 'trace'):
    """Extract H_lat on chi-Fock (16x16) from action's M-K T̂_F².

    phi_treatment:
      'project': project to |φ=0⟩ subspace (= specific boundary projection)
      'trace':   partial trace over φ — gives the effective KS Hamiltonian
                 on χ-only space (correct continuum limit).
    """
    chi_ops, phi_ops = build_fermion_ops(geom.V_3)
    T_step = mk_T_F_single_step((geom.Lx, geom.Ly), chi_ops, phi_ops,
                                U.U_x, U.U_y, 0, geom.m)
    T_sq = T_step @ T_step
    T_sq_H = (T_sq + T_sq.conj().T) / 2

    if phi_treatment == 'project':
        # Project (T̂_F²) to phi=0 directly, then take log
        T_sq_chi = project_to_phi_vacuum(T_sq_H, geom.V_3)
    elif phi_treatment == 'trace':
        # Partial trace over phi-Fock (modes V_3..2V_3-1).
        # State index = chi_bits | (phi_bits << V_3).
        # Sum over phi_bits for diagonal in phi.
        V3 = geom.V_3
        dim_chi = 1 << V3
        dim_phi = 1 << V3
        T_sq_chi = np.zeros((dim_chi, dim_chi), dtype=complex)
        for chi_i in range(dim_chi):
            for chi_j in range(dim_chi):
                # Σ_phi T̂_F²[chi_i, phi; chi_j, phi]
                acc = 0.0 + 0.0j
                for phi_b in range(dim_phi):
                    idx_i = chi_i | (phi_b << V3)
                    idx_j = chi_j | (phi_b << V3)
                    acc += T_sq_H[idx_i, idx_j]
                T_sq_chi[chi_i, chi_j] = acc
    else:
        raise ValueError(phi_treatment)

    T_sq_chi_H = (T_sq_chi + T_sq_chi.conj().T) / 2
    H_lat_chi = -logm(T_sq_chi_H) / (2 * a_tau)
    H_lat_chi = (H_lat_chi + H_lat_chi.conj().T) / 2
    return H_lat_chi


def build_H_QC_fermion_sector():
    """Build z2_setup H_QC and restrict to all-up gauge eigenstate sector."""
    NQ = 8
    DIM = 256
    P2 = {
        'I': np.eye(2, dtype=complex), 'X': np.array([[0,1],[1,0]], dtype=complex),
        'Y': np.array([[0,-1j],[1j,0]], dtype=complex), 'Z': np.diag([1,-1]).astype(complex),
    }
    def pauli(factors):
        by_q = {q: P2['I'] for q in range(NQ)}
        for q, ax in factors: by_q[q] = P2[ax]
        result = by_q[NQ-1]
        for q in range(NQ-2, -1, -1):
            result = np.kron(result, by_q[q])
        return result

    import epoq_classical_sampler as cs
    H_dense = sum(c * pauli(fac) for c, fac in cs.build_pauli_terms())
    H_dense = (H_dense + H_dense.conj().T) / 2

    # Project to all-up gauge eigenstate: |Z_l=+1⟩ for l=0..3.
    # State has bits 0-3 (link qubits) all 0, bits 4-7 (matter) free.
    # Basis: states 0..15 correspond to (link bits = 0, matter bits 0..3).
    # We need to project H_dense onto this 16-dim subspace.
    # All-up gauge: gauge link state = |0,0,0,0⟩ for X-basis qubits.
    # But link qubits in z2_setup are |0⟩ = +1 eigenstate of Z (= "ferromagnetic").
    # H_QC uses Z_l, X_l as Pauli operators.  Z eigenstate |+⟩ = bit 0 of qubit 0..3.
    # State index = (matter bits) << 4 | (link bits). All-up gauge = link bits 0.
    # So projected H is rows/cols [0, 16, 32, ..., 240] (link bits = 0).
    indices = [matter << 4 for matter in range(16)]
    H_sector = H_dense[np.ix_(indices, indices)]
    H_sector = (H_sector + H_sector.conj().T) / 2

    # Build n_0 in same basis (bit 4 = matter at site 0)
    n_0_dense = 0.5 * np.eye(DIM, dtype=complex) - 0.5 * pauli([(4, 'Z')])
    n_0_sector = n_0_dense[np.ix_(indices, indices)]
    return H_sector, n_0_sector


def thermal_n0(H, n_op, beta):
    rho = expm(-beta * H)
    Z = np.trace(rho).real
    return (np.trace(rho @ n_op) / Z).real


def main():
    print("=" * 78)
    print("H_lat (from action) vs H_QC (z2_setup) — trivial U, V_3=4")
    print("=" * 78)
    geom = LatticeGeometry(Lx=2, Ly=2, N_E=5, m=0.5)
    U = Z2GaugeConfig.trivial(geom)

    print("\nBuilding H_lat from M-K T̂_F² (PARTIAL TRACE over φ)...")
    H_lat = build_H_lat_chi_from_action(geom, U, a_tau=1.0, phi_treatment='trace')
    print(f"  shape: {H_lat.shape}")
    print(f"  Hermitian? max|H-H†| = {np.max(np.abs(H_lat - H_lat.conj().T)):.2e}")
    eigs_lat = np.linalg.eigvalsh((H_lat + H_lat.conj().T) / 2)
    print(f"  spectrum: {sorted(eigs_lat.real.tolist())}")

    print("\nFor comparison: H_lat with φ=0 PROJECTION (what pipeline uses)...")
    H_lat_proj = build_H_lat_chi_from_action(geom, U, a_tau=1.0, phi_treatment='project')
    eigs_proj = np.linalg.eigvalsh((H_lat_proj + H_lat_proj.conj().T) / 2)
    print(f"  spectrum: {sorted(eigs_proj.real.tolist())}")

    print("\nBuilding H_QC fermion sector from z2_setup...")
    H_QC, n_0 = build_H_QC_fermion_sector()
    print(f"  shape: {H_QC.shape}")
    print(f"  Hermitian? max|H-H†| = {np.max(np.abs(H_QC - H_QC.conj().T)):.2e}")
    eigs_QC = np.linalg.eigvalsh(H_QC)
    print(f"  spectrum: {sorted(eigs_QC.real.tolist())}")

    print("\nThermal ⟨n̂_0⟩ at β=4:")
    n_op_lat = np.diag([(i >> 0) & 1 for i in range(16)]).astype(complex)
    n0_lat = thermal_n0(H_lat, n_op_lat, beta=4.0)
    n0_lat_proj = thermal_n0(H_lat_proj, n_op_lat, beta=4.0)
    n0_QC = thermal_n0(H_QC, n_0, beta=4.0)
    print(f"  from H_lat (action, φ traced):  {n0_lat:+.6f}  ← should match H_QC")
    print(f"  from H_lat (action, φ=0 proj):  {n0_lat_proj:+.6f}  ← pipeline observable")
    print(f"  from H_QC (z2_setup all-up gauge): {n0_QC:+.6f}")

    print("\nGround states:")
    _, evecs_lat = np.linalg.eigh((H_lat + H_lat.conj().T) / 2)
    _, evecs_QC = np.linalg.eigh(H_QC)
    psi0_lat = evecs_lat[:, 0]
    psi0_QC = evecs_QC[:, 0]
    overlap = abs(np.vdot(psi0_lat, psi0_QC))
    print(f"  |⟨ground_lat|ground_QC⟩| = {overlap:.6f}")
    print(f"  ground state ⟨n̂_0⟩ from H_lat: {(psi0_lat.conj() @ n_op_lat @ psi0_lat).real:+.6f}")
    print(f"  ground state ⟨n̂_0⟩ from H_QC:  {(psi0_QC.conj() @ n_0 @ psi0_QC).real:+.6f}")


if __name__ == "__main__":
    main()
