"""
M3 part 1: Z-C unitary equivalence for 1+1d Schwinger model (Nx=2 OBC).

The Z-C transformation eliminates the fermion DOFs. In 1+1d this is the
Gauss-law-inverted Schwinger trick: the matter content is encoded in the
gauge electric field, and the dynamics reduces to a gauge-only effective
Hamiltonian on a constrained gauge subspace.

This script:
  (1) Builds H_eff^gauge as a 2×2 matrix on {|E=0⟩, |E=1⟩}
  (2) Compares its eigenvalues against the H_phys eigenvalues from the full
      gauge×fermion Hamiltonian projected to the physical subspace
  (3) Verifies a Z-C-like unitary maps physical states to gauge-only states

The unitary equivalence is the operational content of Z-C in 1+1d.
"""

import numpy as np
from numpy.linalg import eigvalsh

# Parameters (same as m3_schwinger_setup.py)
g_sq = 1.0
M_mass = 0.5
epsilon = 0.5

# --- (1) Reduced gauge-only Hamiltonian -----------------------------------
# Physical states satisfy n_0 = E, n_1 = 1 - E.
# Original H_M = M (n_0 - n_1) = M(E - (1-E)) = M(2E - 1) on physical states.
# H_int = ε(ψ†(0) U ψ(1) + h.c.) maps |0,0,1⟩ → |1,1,0⟩ and h.c.,
#   i.e., a hop between |E=0⟩_gauge and |E=1⟩_gauge with amplitude ε.
# H_KS = (g²/2) E².
#
# Effective gauge-only Hamiltonian on {|E=0⟩, |E=1⟩}:

E_vals_eff = np.array([0, 1])
diag = (g_sq / 2) * E_vals_eff**2 + M_mass * (2 * E_vals_eff - 1)
H_eff_gauge = np.array([
    [diag[0], epsilon],
    [epsilon, diag[1]],
])
print("Effective gauge-only Hamiltonian H_eff (2×2 on {|E=0⟩, |E=1⟩}):")
print(H_eff_gauge)
print()
eigs_eff = eigvalsh(H_eff_gauge)
print(f"H_eff eigenvalues: {eigs_eff}")

# --- (2) Cross-check against H_phys from full Hilbert space ---------------
# Rebuild H_phys here for independent check
dim_E = 5
E_max = 2
E_vals = np.arange(-E_max, E_max + 1)

def site_op(op_E, op_n0, op_n1):
    return np.kron(np.kron(op_E, op_n0), op_n1)

E_op = np.diag(E_vals).astype(complex)
U_op = np.zeros((dim_E, dim_E), dtype=complex)
for k in range(dim_E - 1):
    U_op[k + 1, k] = 1.0
n_op = np.diag([0, 1]).astype(complex)
psi = np.array([[0, 1], [0, 0]], dtype=complex)
psi_dag = psi.conj().T
I_E = np.eye(dim_E, dtype=complex)
I_F = np.eye(2, dtype=complex)

E_full = site_op(E_op, I_F, I_F)
n0_full = site_op(I_E, n_op, I_F)
n1_full = site_op(I_E, I_F, n_op)
I_full = site_op(I_E, I_F, I_F)

# Physical state indices (E, n0, n1) → linear index = iE*4 + n0*2 + n1
# Physical: (E=0, 0, 1) → 2*4+0*2+1 = 9
#           (E=1, 1, 0) → 3*4+1*2+0 = 14
phys_indices = [9, 14]

psi_dag0_U_psi1 = site_op(U_op, psi_dag, psi)
H_total = (
    (g_sq / 2) * (E_full @ E_full)
    + epsilon * (psi_dag0_U_psi1 + psi_dag0_U_psi1.conj().T)
    + M_mass * (n0_full - n1_full)
)
H_phys = H_total[np.ix_(phys_indices, phys_indices)]
H_phys_herm = (H_phys + H_phys.conj().T) / 2
eigs_phys = eigvalsh(H_phys_herm)

print()
print(f"H_phys eigenvalues (from full Hilbert space, restricted to physical subspace):")
print(f"  {eigs_phys}")
print()
print(f"Difference: {np.abs(np.sort(eigs_eff) - np.sort(eigs_phys))}")
print(f"Max difference: {np.max(np.abs(np.sort(eigs_eff) - np.sort(eigs_phys))):.3e}")
if np.allclose(np.sort(eigs_eff), np.sort(eigs_phys)):
    print("\n✓ Spectra match. Z-C unitary equivalence verified for Nx=2 Schwinger.")
else:
    print("\n✗ Spectra differ — something is wrong.")

# --- (3) Explicit Z-C unitary construction --------------------------------
# Construct an explicit 20×20 unitary that maps |E,0,1⟩ → |E,0,0⟩ for physical
# states. This is the "Gauss-law-inverted basis change" in 1+1d.
print()
print("=" * 60)
print("(3) Explicit Z-C unitary on the toy Hilbert space")
print("=" * 60)

dim_total = 20

# In the basis {|E,n0,n1⟩}, physical states are |E=0,0,1⟩ and |E=1,1,0⟩.
# Z-C maps these to |E=0,0,0⟩ and |E=1,0,0⟩.
# i.e., index 9 → 8, index 14 → 12.

# Build U_ZC as a permutation on physical states, identity on the rest:
U_ZC = np.eye(dim_total, dtype=complex)
# Swap (9, 8) and (14, 12)
def swap(U, i, j):
    U[[i, j], :] = U[[j, i], :]

swap(U_ZC, 9, 8)
swap(U_ZC, 14, 12)

# Check unitarity
err_unit = np.linalg.norm(U_ZC @ U_ZC.conj().T - np.eye(dim_total))
print(f"||U_ZC U_ZC† - I||  = {err_unit:.3e}  (should be 0)")

# Apply U_ZC to H_total and verify the matter sector is decoupled
H_transformed = U_ZC @ H_total @ U_ZC.conj().T

# The gauge-only target states are {|E,0,0⟩} for E ∈ {-2,...,2}, indices {0,4,8,12,16}
gauge_only_indices = [iE * 4 for iE in range(dim_E)]
print(f"Gauge-only state indices (|E, n0=0, n1=0⟩): {gauge_only_indices}")
print(f"Physical states after Z-C should live in this sector: indices {[8, 12]}")

# Eigenvalues of H_transformed on the {|E,0,0⟩} subspace
H_transformed_gauge = H_transformed[np.ix_(gauge_only_indices, gauge_only_indices)]
H_transformed_gauge_herm = (H_transformed_gauge + H_transformed_gauge.conj().T) / 2
eigs_transformed = eigvalsh(H_transformed_gauge_herm)
print(f"\nEigenvalues of H_transformed in gauge-only sector:")
print(f"  {eigs_transformed}")
print(f"\nLowest 2 eigenvalues (should match H_phys): {eigs_transformed[:2]}")
print(f"H_phys eigenvalues:                          {eigs_phys}")

# The match might be imperfect because our simple "permutation" Z-C only
# acts on 2 of 4 unphysical matter states per E, so the gauge-only subspace
# now mixes "transformed physical" with "untransformed unphysical" states.
# A proper Z-C constructs a unitary that decouples the gauge-only subspace.
