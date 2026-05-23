"""
2+1d Z₂ Z-C: corrected — the Z-C target is the staggered vacuum
|0,1,1,0⟩_matter (filled odd-parity sites), not |0,0,0,0⟩.

This follows from the staggered Gauss-law structure:
    n(i,j) = Σ_L e_L^x         for even sites (parity +1)
    n(i,j) = Σ_L e_L^x ⊕ 1     for odd sites  (parity -1)

The CNOT network from links → matter sites eliminates the Σ_L e_L^x part,
leaving:
    matter_final = 0  for even sites
    matter_final = 1  for odd sites
which is the staggered vacuum configuration |0,1,1,0⟩.

To get matter=|0,0,0,0⟩ instead, apply X on odd-parity matter qubits after
the CNOT network.
"""

import numpy as np
from numpy import kron
from numpy.linalg import eigvalsh, eigh

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

def fermion(idx):
    q = 4 + idx
    op = np.eye(dim, dtype=complex)
    for b in range(idx):
        op = op @ Z_q[4 + b]
    op = op @ single_qubit_op(ann, q)
    return op

psi = [fermion(a) for a in range(4)]
psi_dag = [p.conj().T for p in psi]

links_at_site = {(0, 0): [0, 2], (0, 1): [0, 3], (1, 0): [1, 2], (1, 1): [1, 3]}
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

# Physical projector and reference
P_phys = I_full.copy()
for s in [(0,0), (0,1), (1,0), (1,1)]:
    P_phys = P_phys @ (I_full + G_sites[s]) / 2
P_phys = (P_phys + P_phys.conj().T) / 2

evals_P, evecs_P = eigh(P_phys)
phys_basis = evecs_P[:, evals_P > 0.5]
H_phys = phys_basis.conj().T @ H @ phys_basis
H_phys = (H_phys + H_phys.conj().T) / 2
eigs_phys = eigvalsh(H_phys)

# Hadamard on link qubits (rotate σ_z basis → σ_x basis on links)
R = I_full.copy()
for q_link in range(4):
    R = R @ single_qubit_op(H_pauli, q_link)

# CNOT in σ_x_link basis (computational-basis CNOT after rotation)
def cnot_full(control_q, target_q):
    P0 = (I2 + Z) / 2
    P1 = (I2 - Z) / 2
    op_I = np.eye(1, dtype=complex)
    op_X = np.eye(1, dtype=complex)
    for q in range(NQ):
        if q == control_q:
            op_I = kron(op_I, P0)
            op_X = kron(op_X, P1)
        elif q == target_q:
            op_I = kron(op_I, I2)
            op_X = kron(op_X, X)
        else:
            op_I = kron(op_I, I2)
            op_X = kron(op_X, I2)
    return op_I + op_X

U_ZC_x = I_full.copy()
for site in [(0,0), (0,1), (1,0), (1,1)]:
    target_q = matter_qubit[site]
    for control_q in links_at_site[site]:
        U_ZC_x = cnot_full(control_q, target_q) @ U_ZC_x

# Add X gates on odd-parity matter qubits to map to |0,0,0,0⟩ instead of |0,1,1,0⟩
U_offset = I_full.copy()
for site, q_m in matter_qubit.items():
    if parity[site] == -1:
        U_offset = U_offset @ X_q[q_m]

# Two versions: without X offset → maps to |0,1,1,0⟩; with offset → maps to |0,0,0,0⟩
U_ZC_no_offset = R @ U_ZC_x @ R.conj().T
U_ZC_with_offset = U_offset @ U_ZC_no_offset

for label, U in [("no offset", U_ZC_no_offset), ("with offset", U_ZC_with_offset)]:
    print(f"\n--- U_ZC version: {label} ---")
    err = np.linalg.norm(U @ U.conj().T - I_full)
    print(f"  ||U U† - I|| = {err:.3e}")

    # Check matter content of Z-C-rotated physical states
    phys_rot = U.conj().T @ phys_basis
    total_n = sum(N_q[4 + a] for a in range(4))
    n_diag = phys_rot.conj().T @ total_n @ phys_rot
    n_vals = np.real(np.diagonal(n_diag))
    print(f"  ⟨N_total⟩ in rotated physical states: [{n_vals.min():.4f}, {n_vals.max():.4f}]")

    # Check the matter sector containing them
    n_a_diag = []
    for a in range(4):
        op = N_q[4 + a]
        d = phys_rot.conj().T @ op @ phys_rot
        n_a_diag.append(np.real(np.diagonal(d)).mean())
    print(f"  Mean ⟨n_a⟩ per matter site (a=0..3): {[f'{v:.3f}' for v in n_a_diag]}")

# --- Verify spectral equivalence on the appropriate matter sector ---
# The "with offset" version targets matter=|0,0,0,0⟩.
H_ZC = U_ZC_with_offset.conj().T @ H @ U_ZC_with_offset
H_ZC = (H_ZC + H_ZC.conj().T) / 2

# Project onto matter=|0,0,0,0⟩
P_m0 = I_full.copy()
for q_m in range(4, 8):
    P_m0 = P_m0 @ (I_full - N_q[q_m])
P_m0 = (P_m0 + P_m0.conj().T) / 2

evals_Pm, evecs_Pm = eigh(P_m0)
m0_basis = evecs_Pm[:, evals_Pm > 0.5]
H_ZC_m0 = m0_basis.conj().T @ H_ZC @ m0_basis
H_ZC_m0 = (H_ZC_m0 + H_ZC_m0.conj().T) / 2
eigs_ZC = eigvalsh(H_ZC_m0)

print()
print("=" * 60)
print("Spectrum comparison")
print("=" * 60)
print(f"H_phys (physical sector):                  {eigs_phys.round(4)}")
print(f"H_ZC restricted to matter=|0,0,0,0⟩:        {eigs_ZC.round(4)}")
max_diff = np.max(np.abs(np.sort(eigs_phys) - np.sort(eigs_ZC)))
print(f"\nMax |Δ eigenvalue| = {max_diff:.3e}")
if max_diff < 1e-10:
    print("✓ Z-C spectral equivalence verified for 2+1d Z₂ + staggered.")
