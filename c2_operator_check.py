"""
c.2: operator-level cross-check.

For free fermions, the Lagrangian determinant on a lattice with anti-periodic
temporal BC should match the operator-level finite-T partition function:
    det M_L(APBC)  =  Tr e^{-βH_F}  =  det(1 + e^{-βh})
where h is the single-particle Hamiltonian.

For 1+1d staggered fermions specifically, the standard Susskind Hamiltonian
form is:
    H_F = m Σ_x (-1)^x χ†(x) χ(x) + (i/2) Σ_x [χ†(x+1) χ(x) - χ†(x) χ(x+1)]

We compute det M_APBC directly (from v1) and Tr e^{-βh} from h, and check
their relation.  This is the bridge between the Lagrangian framework that
M2 PF estimator works in and the Hamiltonian framework that M3 Z-C and the
QC Minkowski leg work in.
"""

import numpy as np
from numpy.linalg import det
from scipy.linalg import expm

Nx, Nt, m_mass = 4, 8, 0.5
beta = Nt   # lattice units, a=1

def idx(t, x, Nx=Nx, Nt=Nt):
    return (t % Nt) * Nx + (x % Nx)

# Build the Lagrangian Dirac matrix (same as v1)
def build_M_APBC():
    N = Nx * Nt
    M = np.zeros((N, N), dtype=complex)
    for t in range(Nt):
        for x in range(Nx):
            M[idx(t, x), idx(t, x)] = m_mass
    for t in range(Nt):
        for x in range(Nx):
            sfwd = -1 if (t == Nt - 1) else 1
            sbwd = -1 if (t == 0) else 1
            M[idx(t, x), idx(t + 1, x)] += 0.5 * sfwd
            M[idx(t, x), idx(t - 1, x)] += -0.5 * sbwd
    for t in range(Nt):
        eta1 = (-1) ** t
        for x in range(Nx):
            M[idx(t, x), idx(t, x + 1)] += 0.5 * eta1
            M[idx(t, x), idx(t, x - 1)] += -0.5 * eta1
    return M

M_lag = build_M_APBC()
det_lag = det(M_lag)

# --- Hamiltonian single-particle h_staggered -----------------------------
# Susskind staggered: H_F = m Σ_x (-1)^x χ†χ + (i/2) Σ_x (χ†(x+1)χ(x) - h.c.)
# Single-particle matrix h (Nx × Nx):
h_stag = np.zeros((Nx, Nx), dtype=complex)
for x in range(Nx):
    h_stag[x, x] = m_mass * ((-1) ** x)
for x in range(Nx):
    h_stag[(x + 1) % Nx, x] += +1j / 2     # χ†(x+1) χ(x)
    h_stag[x, (x + 1) % Nx] += -1j / 2     # -χ†(x) χ(x+1)
# verify hermiticity
assert np.allclose(h_stag, h_stag.conj().T)

# Operator-level partition function
exp_minus_beta_h = expm(-beta * h_stag)
det_op_partition = det(np.eye(Nx, dtype=complex) + exp_minus_beta_h)

print(f"Lagrangian: 1+1d free staggered, Nx={Nx}, Nt={Nt}, m={m_mass}")
print(f"  det M_L(APBC) = {det_lag}")
print()
print(f"Operator-level: h_staggered eigenvalues = {np.sort(np.linalg.eigvalsh(h_stag))}")
print(f"  Tr e^{{-βh_F}} = det(1 + e^{{-βh}}) = {det_op_partition}")
print()
print(f"Ratio det_L / det_op = {det_lag / det_op_partition}")

# --- Try other obvious normalization candidates ---------------------------
print()
print("Trying simple normalization conventions:")
candidates = {
    "det_op": det_op_partition,
    "2 * det_op": 2 * det_op_partition,
    "det_op / 2": det_op_partition / 2,
    "det_op^2": det_op_partition ** 2,
    "(-1)^Nx * det_op": ((-1) ** Nx) * det_op_partition,
    "det_op / 2^Nx": det_op_partition / (2 ** Nx),
}
for label, val in candidates.items():
    print(f"  det_L / ({label}) = {det_lag / val}")

# --- Eigenvalue-spectrum approach: derive det M from M's eigenvalues ------
# Free staggered with APBC: Fourier-diagonalize spatially & temporally.
# Spatial momenta k_n = 2π n / Nx, n = 0,...,Nx-1
# Temporal frequencies (APBC): ω_n = (2n+1)π / Nt, n = 0,...,Nt-1
# For each (k, ω) pair, the staggered Dirac operator gives an eigenvalue.
# But staggered's (-1)^t phase couples consecutive temporal frequencies,
# so the spectrum is computed in a 2-cycle basis.

print()
print("Direct eigenvalue check:")
eigs_M = np.linalg.eigvals(M_lag)
det_via_eigs = np.prod(eigs_M)
print(f"  det M_L (direct) = {det_lag}")
print(f"  det M_L (prod eigvals) = {det_via_eigs}")
print(f"  Agreement: {np.isclose(det_lag, det_via_eigs)}")
print(f"\n  Lagrangian spectrum |M| values (first 8): {sorted(np.abs(eigs_M))[:8]}")

# --- Continuum-limit cross-check ------------------------------------------
# Take β large (Nt large) with m fixed; det(1+e^{-βh}) ≈ det(e^{-βh_+}) for
# the positive-energy sector ≈ ∏ e^{-βε_i} for positive ε_i.
print()
print("Continuum-limit sanity: take β→large, det(1+e^{-βh}) ≈ ∏ ε>0 e^{-βε}*(1+e^{-βε})")
beta_large = 20
exp_minus_betaL_h = expm(-beta_large * h_stag)
det_partition_large = det(np.eye(Nx, dtype=complex) + exp_minus_betaL_h)
eig_h = np.sort(np.linalg.eigvalsh(h_stag).real)
analytic_large = np.prod([1 + np.exp(-beta_large * eps) for eps in eig_h])
print(f"  β={beta_large}: det(1+e^{{-βh}}) = {det_partition_large:.6e}")
print(f"  Analytic from h-eigenvalues:    {analytic_large:.6e}")
print(f"  Agreement: {np.isclose(det_partition_large, analytic_large)}")
