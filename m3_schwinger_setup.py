"""
M3 sandbox setup: 1+1d Schwinger model (U(1) gauge + staggered fermion)
on Nx=2 sites with one link, open boundary conditions.

Goal: build the full (gauge × fermion) Hilbert space, identify the physical
subspace via the Gauss law, then in m3_zc.py apply the Z-C unitary and verify
the resulting basis is fermion-free.

Conventions follow 1905.00652:
  - Sites x ∈ {0, 1}, link x ∈ {0} (between sites 0 and 1).
  - U(1) electric field E_x ∈ Z; truncated to {-E_max, ..., +E_max}.
  - Staggered charge: Q(x) = n(x) - (1 - (-1)^x)/2.
    So Q(0) = n(0) (even site, no charge offset for n=0).
       Q(1) = n(1) - 1 (odd site, vacuum has n=1 by staggered convention,
                       but standard convention sets vacuum at n=0 on odd → Q=-1.)
    [Note: there are 2 conventions in the literature; sticking with the one
     where vacuum has alternating filled/empty for the staggered ground state.
     We'll let Gauss law pick out the consistent physical states.]
  - Gauss law (OBC): D(x) - Q(x) = 0 with D(0) = E_0, D(1) = -E_0.
"""

import numpy as np
from numpy import kron

# --- single-site / single-link operators -----------------------------------
E_max = 2
dim_E = 2 * E_max + 1    # E ∈ {-2, -1, 0, 1, 2}
E_vals = np.arange(-E_max, E_max + 1)

# Gauge field operators on the link
E_op = np.diag(E_vals).astype(complex)
# U = e^{iA} raises E by 1: U |E⟩ = |E+1⟩
U_op = np.zeros((dim_E, dim_E), dtype=complex)
for k in range(dim_E - 1):
    U_op[k + 1, k] = 1.0    # raise
# (U is the "ladder" on the truncated rep; not exactly unitary at boundaries
#  due to truncation. For checking structure we don't care about boundary.)

# Fermion: hard-core occupation at each site, dim_F_site = 2
dim_F = 2
n_op = np.diag([0, 1]).astype(complex)        # number operator
psi = np.array([[0, 1], [0, 0]], dtype=complex)    # ψ: annihilates
psi_dag = psi.conj().T

I_E = np.eye(dim_E, dtype=complex)
I_F = np.eye(dim_F, dtype=complex)

# --- full Hilbert space: |E⟩ ⊗ |n_0⟩ ⊗ |n_1⟩ -------------------------------
dim_total = dim_E * dim_F * dim_F
print(f"Full Hilbert space dimension: {dim_total} (= {dim_E} × {dim_F}² for E × n0 × n1)")

def build_op(O_E, O_n0, O_n1):
    return kron(kron(O_E, O_n0), O_n1)

E_full = build_op(E_op, I_F, I_F)
n0_full = build_op(I_E, n_op, I_F)
n1_full = build_op(I_E, I_F, n_op)

# --- Gauss law operators ---------------------------------------------------
# Q(x) for staggered: Q(0) = n(0), Q(1) = n(1) - 1
Q0 = n0_full
Q1 = n1_full - build_op(I_E, I_F, I_F)   # n(1) - 1

# D(x) at OBC: D(0) = E_0 - E_{-1} = E_0 (no left edge)
#              D(1) = E_1 - E_0 = -E_0 (no right edge)
D0 = E_full
D1 = -E_full

G0 = D0 - Q0
G1 = D1 - Q1

# --- identify physical subspace --------------------------------------------
# Physical states satisfy G(x)|ψ⟩ = 0 for all x.
# In product basis they are simultaneous null-eigenstates of G0, G1.

# Build basis labels
basis_labels = []
for iE in range(dim_E):
    for n0 in range(2):
        for n1 in range(2):
            basis_labels.append((E_vals[iE], n0, n1))

# Check which product states satisfy Gauss
phys_indices = []
print("\nProduct basis states satisfying Gauss law:")
print(f"  {'idx':>4} | {'E':>3} | {'n0':>3} | {'n1':>3} | {'G0':>4} | {'G1':>4}")
for idx, (E, n0, n1) in enumerate(basis_labels):
    Q0_val = n0
    Q1_val = n1 - 1
    g0 = E - Q0_val
    g1 = -E - Q1_val
    if g0 == 0 and g1 == 0:
        phys_indices.append(idx)
        print(f"  {idx:>4} | {E:>3} | {n0:>3} | {n1:>3} | {g0:>4} | {g1:>4}  ← physical")

print(f"\nPhysical subspace dimension: {len(phys_indices)} out of {dim_total}")
print("(For Nx=2 OBC Schwinger: charge neutrality forces n0 + n1 = 1 with E = n0)")

# --- Hamiltonian (for reference; not used in this checkpoint) --------------
# H = (g²/2) E² + ε(0) (ψ_dag(0) U ψ(1) + h.c.) + M [(-1)^0 n_0 + (-1)^1 n_1]
g_sq = 1.0
M_mass = 0.5
epsilon = 0.5

H_E = (g_sq / 2) * (E_full @ E_full)
# ψ†(0) U ψ(1): create at 0, lower E, annihilate at 1
psi_dag_0_U_psi_1 = build_op(U_op, psi_dag, psi)
H_int = epsilon * (psi_dag_0_U_psi_1 + psi_dag_0_U_psi_1.conj().T)
H_M = M_mass * (n0_full - n1_full)   # (-1)^x staggered mass

H_total = H_E + H_int + H_M

# Restrict to physical subspace and diagonalize
H_phys = H_total[np.ix_(phys_indices, phys_indices)]
eig = np.linalg.eigvalsh((H_phys + H_phys.conj().T) / 2)
print(f"\nPhysical-subspace eigenvalues of H_Schwinger (Nx=2, g²={g_sq}, M={M_mass}):")
for v in eig:
    print(f"  E = {v.real:+.4f}")

# --- save physical-subspace info for m3_zc.py ------------------------------
np.savez('m1_toy/m3_phys_subspace.npz',
         phys_indices=phys_indices,
         basis_labels=basis_labels,
         dim_E=dim_E,
         E_max=E_max,
         dim_F=dim_F)

print("\nSaved physical subspace data → m1_toy/m3_phys_subspace.npz")
print("Next: m3_zc.py — apply Z-C unitary and verify fermion-elimination.")
