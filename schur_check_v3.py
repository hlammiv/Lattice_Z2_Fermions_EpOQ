"""
M1 sandbox v3: matrix-element SK convention — open boundary conditions (OBC)
at the Euclidean-Minkowski seams.

Per 2001.11490 (Harmalkar/Lamm/Lawrence, NuQS): the SK contour is decomposed
into open Euclidean + open Minkowski legs, with matching via a sum over corner
states Ψ_i, Ψ_j. With OBC, the fermion matrix block-decouples (no implicit
Σ-matching), and the corner-state sum reintroduces the matching explicitly.

Tests:
  (a) With OBC at seams, M block-decouples and det M_OBC = det(M_E) × det(M_M).
  (b) The difference det(M_APBC) - det(M_OBC) is the "missing" contribution that
      the corner-state sum must reproduce.
"""

import numpy as np
from numpy.linalg import det

Nx = 4
Nt = 8
Nt_E = 4
m = 0.5

def idx(t, x):
    return (t % Nt) * Nx + (x % Nx)

def build_M(BC):
    """BC = 'APBC' (closed thermal trace) or 'OBC' (decoupled at seams)."""
    N = Nx * Nt
    M = np.zeros((N, N), dtype=complex)

    # Mass + spatial hops are unchanged
    for t in range(Nt):
        for x in range(Nx):
            M[idx(t, x), idx(t, x)] = m
    for t in range(Nt):
        eta1 = (-1) ** t
        for x in range(Nx):
            M[idx(t, x), idx(t, x + 1)] += 0.5 * eta1
            M[idx(t, x), idx(t, x - 1)] += -0.5 * eta1

    # Temporal hops differ between APBC and OBC
    # The two seams in our split are:
    #   seam 1 (low):  t=Nt_E-1 (E side, =3)  <-> t=Nt_E (M side, =4)
    #   seam 2 (high): t=Nt-1 (M side, =7)    <-> t=0 (E side, via APBC wrap)
    for t in range(Nt):
        for x in range(Nx):
            # forward hop: (t,x) -> (t+1 mod Nt, x)
            sfwd = 1
            if t == Nt - 1:
                sfwd = -1 if BC == "APBC" else 0      # APBC sign-flip, OBC drop
            if t == Nt_E - 1 and BC == "OBC":
                sfwd = 0
            M[idx(t, x), idx(t + 1, x)] += 0.5 * sfwd

            # backward hop: (t,x) -> (t-1 mod Nt, x)
            sbwd = 1
            if t == 0:
                sbwd = -1 if BC == "APBC" else 0
            if t == Nt_E and BC == "OBC":
                sbwd = 0
            M[idx(t, x), idx(t - 1, x)] += -0.5 * sbwd

    return M

M_APBC = build_M("APBC")
M_OBC = build_M("OBC")

NE = Nt_E * Nx

# --- (a) verify OBC decoupling --------------------------------------------
M_OBC_EE = M_OBC[:NE, :NE]
M_OBC_MM = M_OBC[NE:, NE:]
M_OBC_EM = M_OBC[:NE, NE:]
M_OBC_ME = M_OBC[NE:, :NE]

print("=" * 60)
print("(a) OBC decoupling check")
print("=" * 60)
print(f"||M_OBC_EM|| = {np.linalg.norm(M_OBC_EM):.3e}  (should be 0)")
print(f"||M_OBC_ME|| = {np.linalg.norm(M_OBC_ME):.3e}  (should be 0)")

det_OBC_full = det(M_OBC)
det_OBC_E = det(M_OBC_EE)
det_OBC_M = det(M_OBC_MM)
print()
print(f"det M_OBC (full)     = {det_OBC_full}")
print(f"det M_OBC_E          = {det_OBC_E}")
print(f"det M_OBC_M          = {det_OBC_M}")
print(f"det_E × det_M        = {det_OBC_E * det_OBC_M}")
print(f"Block-product check  = {det_OBC_full / (det_OBC_E * det_OBC_M)}  (should be 1)")

# --- (b) APBC vs OBC difference -------------------------------------------
print()
print("=" * 60)
print("(b) APBC vs OBC: the corner-state contribution")
print("=" * 60)

det_APBC = det(M_APBC)
print(f"det M_APBC = {det_APBC}")
print(f"det M_OBC  = {det_OBC_full}")
print(f"det M_APBC / det M_OBC = {det_APBC / det_OBC_full}")
print()
print("Interpretation: det M_APBC = det M_OBC + (corner-state sum contributions).")
print("The ratio above is what the explicit corner-state sum must reproduce")
print("when summed with proper signs over fermion boundary states.")

# Recover via direct identity: det M_APBC = det M_OBC × det(I + M_OBC^{-1} B)
# where B is the "removed" hop block.
print()
print("Cross-check: det M_APBC = det M_OBC × det(I + M_OBC^{-1} B) with B = M_APBC - M_OBC")
B = M_APBC - M_OBC
N = Nx * Nt
identity = np.eye(N, dtype=complex)
correction = det(identity + np.linalg.solve(M_OBC, B))
print(f"  det M_OBC × correction = {det_OBC_full * correction}")
print(f"  matches det M_APBC?    {np.isclose(det_OBC_full * correction, det_APBC)}")
print(f"  ratio = {(det_OBC_full * correction) / det_APBC}")
