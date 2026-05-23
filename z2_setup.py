"""
2+1d Z₂ LGT + staggered fermions on a 2×2 OBC spatial lattice.

The M4 demonstration target: the smallest non-trivial 2+1d Z₂ system
that exercises everything beyond 1+1d Schwinger:
  - Genuine 2D lattice with plaquette (magnetic) interaction
  - Z₂ gauge group (discrete, no charge-ladder issues)
  - Staggered fermions with 2D parity (-1)^{x+y}
  - Gauss law constraints at every site

Lattice geometry (2×2 OBC):
  Sites: (0,0), (0,1), (1,0), (1,1) — labels M0, M1, M2, M3
  Links: L_h0 = (0,0)-(0,1)  [hor row 0]
         L_h1 = (1,0)-(1,1)  [hor row 1]
         L_v0 = (0,0)-(1,0)  [vert col 0]
         L_v1 = (0,1)-(1,1)  [vert col 1]
  Plaquette: L_h0 · L_v1 · L_h1 · L_v0 (going around the square)

Qubit ordering (8 qubits total):
  q0=L_h0, q1=L_h1, q2=L_v0, q3=L_v1,
  q4=M_00, q5=M_01, q6=M_10, q7=M_11

This setup builds the Hamiltonian, identifies the Gauss-law-physical
subspace, and verifies the dimension counts.
"""

import numpy as np
from numpy import kron
from numpy.linalg import eigvalsh, matrix_rank

# Pauli basis
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
N_op = np.diag([0, 1]).astype(complex)        # number operator
ann = np.array([[0, 1], [0, 0]], dtype=complex)  # σ- (lowering)
cre = ann.conj().T

NQ = 8
dim = 2 ** NQ

def single_qubit_op(op, q):
    """Embed 2×2 op at qubit q in the full 2^NQ Hilbert space."""
    result = np.eye(1, dtype=complex)
    for i in range(NQ):
        result = kron(result, op if i == q else I2)
    return result

# Pre-build all single-qubit operators we need
print(f"Building operators... ({NQ} qubits, dim={dim})")
X_q = [single_qubit_op(X, q) for q in range(NQ)]
Z_q = [single_qubit_op(Z, q) for q in range(NQ)]
N_q = [single_qubit_op(N_op, q) for q in range(NQ)]

# --- Fermion operators with Jordan-Wigner strings -------------------------
# Order matter qubits as 4, 5, 6, 7 (linear).
# ψ_a = (∏_{b<a in matter ordering} Z_b) · σ-_a
# For our matter ordering M_00=q4, M_01=q5, M_10=q6, M_11=q7:
def fermion(idx_in_matter):
    """Annihilation operator for matter site, with JW string."""
    q = 4 + idx_in_matter
    op = np.eye(dim, dtype=complex)
    for b in range(idx_in_matter):
        op = op @ Z_q[4 + b]
    op = op @ single_qubit_op(ann, q)
    return op

psi = [fermion(a) for a in range(4)]
psi_dag = [p.conj().T for p in psi]

# --- Gauss law operators --------------------------------------------------
# G(i,j) = ∏_{links L touching (i,j)} σ_x^L · (-1)^Q(i,j)
# Staggered charge: Q(x,y) = n(x,y) - (1 - (-1)^{x+y})/2
# Parity (-1)^{x+y}: (0,0)→+1, (0,1)→-1, (1,0)→-1, (1,1)→+1
# So Q(x,y) = n(x,y) - 0  for even sites (x+y even)
#           = n(x,y) - 1  for odd sites (x+y odd)
matter_qubit_of_site = {(0, 0): 4, (0, 1): 5, (1, 0): 6, (1, 1): 7}
links_at_site = {
    (0, 0): [0, 2],   # L_h0 (q0), L_v0 (q2)
    (0, 1): [0, 3],   # L_h0 (q0), L_v1 (q3)
    (1, 0): [1, 2],   # L_h1 (q1), L_v0 (q2)
    (1, 1): [1, 3],   # L_h1 (q1), L_v1 (q3)
}
parity = {(0, 0): +1, (0, 1): -1, (1, 0): -1, (1, 1): +1}
n_offset = {p: 0 if p == +1 else 1 for p in [+1, -1]}   # charge offset

I_full = np.eye(dim, dtype=complex)

def gauss_op(site):
    q_links = links_at_site[site]
    q_matter = matter_qubit_of_site[site]
    # Product of σ_x on touching links
    G = I_full.copy()
    for ql in q_links:
        G = G @ X_q[ql]
    # (-1)^Q = (-1)^(n - offset).  Since n ∈ {0,1}, (-1)^n = 1 - 2n
    # For (-1)^{n - offset}: if offset = 0, (-1)^n = 1 - 2n; if offset = 1, -(1 - 2n) = -1 + 2n
    # Operator form: (-1)^n = I - 2N_q[q_matter]
    sign_op = I_full - 2 * N_q[q_matter]
    if parity[site] == -1:
        sign_op = -sign_op   # extra -1 from offset
    G = G @ sign_op
    return G

G_sites = {site: gauss_op(site) for site in [(0,0), (0,1), (1,0), (1,1)]}

# --- Hamiltonian ---------------------------------------------------------
g_e = 1.0    # electric coupling (E² term, but for Z₂ we use σ_x for kinetic)
g_m = 0.5    # magnetic coupling (plaquette term)
g_hop = 0.5
m_mass = 0.5

# Electric: H_E = -g_e Σ_links σ_x^L (in Z₂ this is the "kinetic" term)
H_E_op = -g_e * sum(X_q[q] for q in range(4))   # links are q=0..3

# Magnetic plaquette: ∏ σ_z on the 4 links of the 1 plaquette
H_M_op = -g_m * (Z_q[0] @ Z_q[3] @ Z_q[1] @ Z_q[2])   # L_h0 · L_v1 · L_h1 · L_v0

# Fermion hopping: ε Σ_links (ψ†_x σ_z^L ψ_y + h.c.)
# Pairings: L_h0 connects M_00 (a=0) and M_01 (a=1), link qubit q=0
#           L_h1 connects M_10 (a=2) and M_11 (a=3), link qubit q=1
#           L_v0 connects M_00 (a=0) and M_10 (a=2), link qubit q=2
#           L_v1 connects M_01 (a=1) and M_11 (a=3), link qubit q=3
hop_pairs = [
    (0, 0, 1),  # link q=0, matter a=0 ↔ a=1
    (1, 2, 3),
    (2, 0, 2),
    (3, 1, 3),
]
H_hop_op = np.zeros((dim, dim), dtype=complex)
for q_link, a, b in hop_pairs:
    term = psi_dag[a] @ Z_q[q_link] @ psi[b]
    H_hop_op += g_hop * (term + term.conj().T)

# Staggered mass: m Σ_sites (-1)^{x+y} n(x,y)
H_mass_op = np.zeros((dim, dim), dtype=complex)
for site, q_matter in matter_qubit_of_site.items():
    H_mass_op += m_mass * parity[site] * N_q[q_matter]

H_total = H_E_op + H_M_op + H_hop_op + H_mass_op
H_total = (H_total + H_total.conj().T) / 2

print(f"H built. ||H - H†|| = {np.linalg.norm(H_total - H_total.conj().T):.3e}")

# --- Identify physical subspace ------------------------------------------
# Physical states satisfy G(site) |ψ⟩ = +|ψ⟩ for all sites.
# Project: P_phys = ∏_sites (I + G_site)/2

P_phys = I_full.copy()
for site in [(0,0), (0,1), (1,0), (1,1)]:
    P_phys = P_phys @ (I_full + G_sites[site]) / 2
P_phys = (P_phys + P_phys.conj().T) / 2

# Verify P_phys is a projector
err = np.linalg.norm(P_phys @ P_phys - P_phys)
print(f"||P²-P|| = {err:.3e}")
phys_dim = int(round(np.trace(P_phys).real))
print(f"Physical subspace dimension: {phys_dim} (out of {dim})")

# Expected: 4 Gauss constraints, but 1 redundant (global product) → 3 indep
# Physical dim = dim / 2^3 = 256 / 8 = 32. ✓ if 32.

# Diagonalize in physical subspace
# Build basis for physical subspace
evals_P, evecs_P = np.linalg.eigh(P_phys)
phys_basis = evecs_P[:, evals_P > 0.5]
print(f"Verified physical basis dim: {phys_basis.shape[1]}")

H_phys = phys_basis.conj().T @ H_total @ phys_basis
H_phys = (H_phys + H_phys.conj().T) / 2
eigs_phys = eigvalsh(H_phys)
print(f"\nLowest 6 eigenvalues of H in physical subspace:")
for e in eigs_phys[:6]:
    print(f"  E = {e:+.4f}")
print(f"Ground-state energy: {eigs_phys[0]:.6f}")
