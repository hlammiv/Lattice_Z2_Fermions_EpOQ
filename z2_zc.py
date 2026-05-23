"""
2+1d Z₂ + staggered fermion: Z-C unitary equivalence.

Goal: build a unitary U_ZC that maps the 16-dim physical subspace into a
16-dim "matter eliminated" subspace, and verify that the Hamiltonian
restricted to physical agrees with the Z-C-rotated H restricted to
matter-eliminated.

For 2+1d Z₂, the Gauss-law relation in the σ_x_link basis is
    n(i,j) = ⊕_{links L at (i,j)} e_L^(x)   (mod 2)
where e_L^(x) ∈ {0,1} is the σ_x_link bit value. So matter content is
fully determined by link content (in σ_x basis). Z-C absorbs matter into
the gauge field by setting matter = 0 and using link σ_x to encode it.
"""

import numpy as np
from numpy import kron
from numpy.linalg import eigvalsh, eigh

# Pauli operators
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H_pauli = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
N_op = np.diag([0, 1]).astype(complex)
ann = np.array([[0, 1], [0, 0]], dtype=complex)

NQ = 8
dim = 2 ** NQ

def single_qubit_op(op, q):
    result = np.eye(1, dtype=complex)
    for i in range(NQ):
        result = kron(result, op if i == q else I2)
    return result

X_q = [single_qubit_op(X, q) for q in range(NQ)]
Z_q = [single_qubit_op(Z, q) for q in range(NQ)]
N_q = [single_qubit_op(N_op, q) for q in range(NQ)]

def fermion(idx_in_matter):
    q = 4 + idx_in_matter
    op = np.eye(dim, dtype=complex)
    for b in range(idx_in_matter):
        op = op @ Z_q[4 + b]
    op = op @ single_qubit_op(ann, q)
    return op

psi = [fermion(a) for a in range(4)]
psi_dag = [p.conj().T for p in psi]

links_at_site = {
    (0, 0): [0, 2], (0, 1): [0, 3], (1, 0): [1, 2], (1, 1): [1, 3]
}
parity = {(0, 0): +1, (0, 1): -1, (1, 0): -1, (1, 1): +1}
matter_qubit = {(0, 0): 4, (0, 1): 5, (1, 0): 6, (1, 1): 7}
I_full = np.eye(dim, dtype=complex)

def gauss_op(site):
    G = I_full.copy()
    for ql in links_at_site[site]:
        G = G @ X_q[ql]
    sign_op = I_full - 2 * N_q[matter_qubit[site]]
    if parity[site] == -1:
        sign_op = -sign_op
    return G @ sign_op

G_sites = {s: gauss_op(s) for s in [(0,0), (0,1), (1,0), (1,1)]}

# --- Hamiltonian (same as z2_setup.py) ---
g_e, g_m, g_hop, m_mass = 1.0, 0.5, 0.5, 0.5
H_E = -g_e * sum(X_q[q] for q in range(4))
H_M = -g_m * (Z_q[0] @ Z_q[3] @ Z_q[1] @ Z_q[2])

hop_pairs = [(0, 0, 1), (1, 2, 3), (2, 0, 2), (3, 1, 3)]
H_hop = np.zeros((dim, dim), dtype=complex)
for q_link, a, b in hop_pairs:
    term = psi_dag[a] @ Z_q[q_link] @ psi[b]
    H_hop += g_hop * (term + term.conj().T)

H_mass = np.zeros((dim, dim), dtype=complex)
for site, q_m in matter_qubit.items():
    H_mass += m_mass * parity[site] * N_q[q_m]

H = H_E + H_M + H_hop + H_mass
H = (H + H.conj().T) / 2

# --- Physical subspace and reference spectrum ---
P_phys = I_full.copy()
for s in [(0,0), (0,1), (1,0), (1,1)]:
    P_phys = P_phys @ (I_full + G_sites[s]) / 2
P_phys = (P_phys + P_phys.conj().T) / 2

evals_P, evecs_P = eigh(P_phys)
phys_basis = evecs_P[:, evals_P > 0.5]
H_phys = phys_basis.conj().T @ H @ phys_basis
H_phys = (H_phys + H_phys.conj().T) / 2
eigs_phys = eigvalsh(H_phys)

print(f"Physical-subspace eigvalues (first 8): {eigs_phys[:8].round(4)}")

# --- Z-C construction ---
# Strategy: in the σ_x_link basis, the Gauss law constraint becomes
#   n(i,j) = ⊕ e_L (link bits in σ_x basis)
# Z-C in this basis is a multi-CNOT network: condition on σ_x_link bits,
# X the matter qubit accordingly.
#
# In computational basis, U_ZC = R · U_ZC^{(x)} · R^†, where R is the
# Hadamard transform on the 4 link qubits (bringing them from σ_z to σ_x basis).

# Build Hadamard transform on the 4 link qubits
R = I_full.copy()
# Apply Hadamard on each link qubit q=0..3
for q_link in range(4):
    H_q = single_qubit_op(H_pauli, q_link)
    R = R @ H_q

# In the σ_x_link basis, the Z-C operation is a multi-controlled-NOT:
# For each matter site, XOR with the sum of link σ_x bits at that site.
# This is implementable as a chain of CNOTs from link qubits to matter qubits.

# CNOT(control, target) = I if control=0, X on target if control=1
def cnot(control_q, target_q):
    """CNOT with control on qubit `control_q`, target on `target_q`."""
    # Project control onto |0⟩, multiply by I; project onto |1⟩, multiply by X on target
    P0 = (I2 + Z) / 2   # |0⟩⟨0|
    P1 = (I2 - Z) / 2   # |1⟩⟨1|
    op = np.eye(1, dtype=complex)
    for q in range(NQ):
        if q == control_q:
            op_q = P0   # initialize for the |0⟩⟨0| part
        elif q == target_q:
            op_q = I2
        else:
            op_q = I2
        op = kron(op, op_q)
    # That was the I part. Now add the X part.
    op2 = np.eye(1, dtype=complex)
    for q in range(NQ):
        if q == control_q:
            op_q = P1
        elif q == target_q:
            op_q = X
        else:
            op_q = I2
        op2 = kron(op2, op_q)
    return op + op2

# Build U_ZC^{(x)} as CNOTs from link bits to matter qubits per Gauss law
# n(i,j) = ⊕ e_L for links L at (i,j)
# So: matter qubit M_(i,j) gets XOR-flipped by each link qubit at site (i,j).
U_ZC_x = I_full.copy()
for site in [(0,0), (0,1), (1,0), (1,1)]:
    target_q = matter_qubit[site]
    for control_q in links_at_site[site]:
        U_ZC_x = cnot(control_q, target_q) @ U_ZC_x

# Convert to computational basis: U_ZC = R · U_ZC^{(x)} · R^†
U_ZC = R @ U_ZC_x @ R.conj().T
err_unit = np.linalg.norm(U_ZC @ U_ZC.conj().T - I_full)
print(f"||U_ZC U_ZC† - I|| = {err_unit:.3e}  (should be 0)")

# --- Apply Z-C to H and check spectral equivalence on physical subspace ---
H_ZC = U_ZC.conj().T @ H @ U_ZC
H_ZC = (H_ZC + H_ZC.conj().T) / 2

# After Z-C, physical states should have matter = something simple.
# Check: what's the matter content of Z-C-rotated physical basis?
phys_basis_ZC = U_ZC.conj().T @ phys_basis

# Compute ⟨total matter occupation⟩ in the rotated physical states
total_n = N_q[4] + N_q[5] + N_q[6] + N_q[7]
total_n_in_rotated = phys_basis_ZC.conj().T @ total_n @ phys_basis_ZC
mean_n_per_state = np.real(np.diagonal(total_n_in_rotated))
print()
print(f"Mean ⟨N_total⟩ per Z-C-rotated physical state:")
print(f"  range: [{mean_n_per_state.min():.4f}, {mean_n_per_state.max():.4f}]")
print("  (should be 0 for all if Z-C eliminates matter)")

# Spectrum of H_ZC restricted to "matter=|0,0,0,0⟩" sector
P_matter0 = I_full.copy()
for q_m in range(4, 8):
    P_matter0 = P_matter0 @ (I_full - N_q[q_m])
P_matter0 = (P_matter0 + P_matter0.conj().T) / 2

evals_Pm, evecs_Pm = eigh(P_matter0)
matter0_basis = evecs_Pm[:, evals_Pm > 0.5]
print(f"\nMatter=|0,0,0,0⟩ subspace dimension: {matter0_basis.shape[1]} (= 2^4 = 16)")

H_ZC_m0 = matter0_basis.conj().T @ H_ZC @ matter0_basis
H_ZC_m0 = (H_ZC_m0 + H_ZC_m0.conj().T) / 2
eigs_ZC_m0 = eigvalsh(H_ZC_m0)

print(f"\nLowest eigenvalues of H_ZC in matter=0 sector: {eigs_ZC_m0[:8].round(4)}")
print(f"Lowest eigenvalues of H_phys (original):       {eigs_phys[:8].round(4)}")
print()

if np.allclose(np.sort(eigs_ZC_m0), np.sort(eigs_phys)):
    print("✓ Z-C spectral equivalence in 2+1d Z₂: confirmed.")
else:
    print(f"  Max diff: {np.max(np.abs(np.sort(eigs_ZC_m0) - np.sort(eigs_phys))):.3e}")
