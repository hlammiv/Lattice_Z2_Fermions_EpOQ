"""
M1 sandbox v1: structural Schur identity check for 1+1d free staggered fermion
on a Euclidean lattice (no Wick rotation yet).

Verifies: det M = det(M_E) * det(M_M - Sigma)
where the lattice is split into "Euclidean half" (first Nt_E timeslices) and
"Minkowski half" (remaining timeslices) via Schur complement on the temporal
blocking.

If this fails, the Schur structure itself is wrong.
If this passes, we know the structural identity is good, and the next step is
to add Wick-rotation factors on the M half to get the actual SK case.
"""

import numpy as np
from numpy.linalg import det, solve

# ----------------------------------------------------------------------------
# Lattice setup
# ----------------------------------------------------------------------------
Nx = 4          # spatial sites, periodic
Nt = 8          # total temporal sites, anti-periodic (thermal)
Nt_E = 4        # "Euclidean half" — first Nt_E timeslices
m = 0.5         # bare mass

assert 1 <= Nt_E < Nt

def idx(t, x):
    return (t % Nt) * Nx + (x % Nx)

N = Nx * Nt
M = np.zeros((N, N), dtype=complex)

# Mass term
for t in range(Nt):
    for x in range(Nx):
        M[idx(t, x), idx(t, x)] = m

# Staggered phases:
#   eta_0(x) = 1                  (temporal phase, by convention)
#   eta_1(t, x) = (-1)^t          (spatial phase: sum of mu' < 1 is just x_0 = t)

# Temporal hops with anti-periodic BC
for t in range(Nt):
    for x in range(Nx):
        # forward hop: (t,x) -> (t+1,x), coefficient +1/2 * eta_0 = +1/2
        sign_apbc_fwd = -1 if (t == Nt - 1) else 1   # wrap picks up APBC sign
        M[idx(t, x), idx(t + 1, x)] += 0.5 * sign_apbc_fwd

        # backward hop: (t,x) -> (t-1,x), coefficient -1/2 * eta_0(t-1) = -1/2
        sign_apbc_bwd = -1 if (t == 0) else 1
        M[idx(t, x), idx(t - 1, x)] += -0.5 * sign_apbc_bwd

# Spatial hops with periodic BC, staggered phase (-1)^t
for t in range(Nt):
    eta1 = (-1) ** t
    for x in range(Nx):
        # forward
        M[idx(t, x), idx(t, x + 1)] += 0.5 * eta1
        # backward
        M[idx(t, x), idx(t, x - 1)] += -0.5 * eta1

# ----------------------------------------------------------------------------
# Full determinant
# ----------------------------------------------------------------------------
det_full = det(M)

# ----------------------------------------------------------------------------
# Schur split on temporal blocking: E = first Nt_E*Nx rows/cols, rest = M
# ----------------------------------------------------------------------------
NE = Nt_E * Nx
NM = (Nt - Nt_E) * Nx

M_EE = M[:NE, :NE]
M_EM = M[:NE, NE:]
M_ME = M[NE:, :NE]
M_MM = M[NE:, NE:]

# Sigma = M_ME @ M_EE^{-1} @ M_EM
Sigma = M_ME @ solve(M_EE, M_EM)
det_E = det(M_EE)
det_MmSigma = det(M_MM - Sigma)

ratio = det_full / (det_E * det_MmSigma)

print(f"Nx, Nt, Nt_E = {Nx}, {Nt}, {Nt_E}")
print(f"m = {m}")
print()
print(f"det(M_full)          = {det_full}")
print(f"det(M_E)             = {det_E}")
print(f"det(M_M - Sigma)     = {det_MmSigma}")
print(f"det(M_E) * det(M-Σ)  = {det_E * det_MmSigma}")
print()
print(f"Ratio (should be 1)  = {ratio}")
print(f"|Ratio - 1|          = {abs(ratio - 1):.3e}")

# ----------------------------------------------------------------------------
# Diagnostics: structure of the off-diagonal blocks
# ----------------------------------------------------------------------------
nnz_EM = np.count_nonzero(np.abs(M_EM) > 1e-12)
nnz_ME = np.count_nonzero(np.abs(M_ME) > 1e-12)
print()
print(f"Nonzeros in M_EM = {nnz_EM} (each = a single temporal-hop coupling)")
print(f"Nonzeros in M_ME = {nnz_ME}")

# Show which sites couple across the junction
print()
print("M_EM nonzero pattern (rows in E, cols in M-block):")
for i in range(NE):
    for j in range(NM):
        v = M_EM[i, j]
        if abs(v) > 1e-12:
            t_E, x_E = divmod(i, Nx)
            t_M_offset, x_M = divmod(j, Nx)
            t_M = Nt_E + t_M_offset
            print(f"  ({t_E},{x_E}) <-> ({t_M},{x_M}): {v}")

print()
print("M_ME nonzero pattern (rows in M-block, cols in E):")
for i in range(NM):
    for j in range(NE):
        v = M_ME[i, j]
        if abs(v) > 1e-12:
            t_M_offset, x_M = divmod(i, Nx)
            t_M = Nt_E + t_M_offset
            t_E, x_E = divmod(j, Nx)
            print(f"  ({t_M},{x_M}) <-> ({t_E},{x_E}): {v}")
