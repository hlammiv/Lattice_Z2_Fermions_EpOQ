"""
c.2 continuation: Derive the Lagrangian↔Hamiltonian normalization for
free 1+1d staggered fermions.

Strategy:
  (i)   Compute all 32 eigenvalues of M_L numerically; identify (k, ω) labels.
  (ii)  Fit a closed-form formula λ(k, ω) and verify.
  (iii) From the formula, derive the relation
            det M_L = (something) × det(1 + e^{-βh})
        and identify the (something).
"""

import numpy as np
from numpy.linalg import det, eigvals
from scipy.linalg import expm

Nx, Nt, m_mass = 4, 8, 0.5

def idx(t, x):
    return (t % Nt) * Nx + (x % Nx)

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

M_L = build_M_APBC()
det_L = det(M_L)

eigs = eigvals(M_L)
print(f"det M_L = {det_L.real:.6f}")
print(f"Number of eigenvalues: {len(eigs)}")
print(f"Eigenvalue |λ| histogram:")
abs_eigs = np.sort(np.abs(eigs))
unique_abs = np.unique(abs_eigs.round(4))
print(f"  Unique |λ| values: {unique_abs}")
for u in unique_abs:
    count = np.sum(np.abs(abs_eigs - u) < 1e-3)
    print(f"    |λ| = {u:.4f}: {count} eigenvalues")

# --- candidate analytic formula -------------------------------------------
# For free 1+1d Lagrangian staggered with the (-1)^t spatial-phase structure,
# the natural Fourier decomposition pairs (ω, ω+π).  Let's compute the
# 2x2 block determinant analytically and compare.
print()
print("Analytic candidate: at each (k, ω-pair), the 2×2 Fourier block has")
print("determinant = (m² + sin²ω)(m² + sin²ω) + sin²k · ... — need to work out.")
print()
print("Test formula λ_pair² = (m² + sin²ω)² + 2(m² + sin²ω)sin²k + sin⁴k for each pair:")

# k modes: 2πn/Nx for n=0..Nx-1; sin²k: 0,1,0,1 for Nx=4
# ω modes (APBC): (2n+1)π/Nt for n=0..Nt-1; we pair (ω, ω+π) → 4 unique pairs
ks = [2 * np.pi * n / Nx for n in range(Nx)]
sin2_ks = [np.sin(k) ** 2 for k in ks]

# ω pairs: (ω_n, ω_n + π) for n=0..Nt/2-1
omega_pairs = []
for n in range(Nt // 2):
    omega = (2 * n + 1) * np.pi / Nt
    omega_pairs.append((omega, omega + np.pi))

print(f"\nk values: {ks}")
print(f"sin²k:    {sin2_ks}")
print(f"ω pairs:  {[(round(a,3), round(b,3)) for a, b in omega_pairs]}")

# Try: 2×2 block with i*sin(ω) on diagonal and m as off-diagonal mass-mix
# plus spatial k contribution. Specifically, the M_L block at (k, ω, ω+π) is:
#   diagonal: m + (-1)^t terms → after Fourier mix → m mixes ω with ω+π
#   temporal hop → i sin(ω) on diagonal
#   spatial hop with (-1)^t → mixes ω with ω+π too
#
# So the 2×2 block at (k, ω-pair) is:
#   | i sin(ω) + m_diag(ω,ω)  m_offdiag(ω,ω+π) |
#   | m_offdiag(ω+π,ω)        i sin(ω+π) + ... |
#
# Specifically: kinetic temporal → i sin(ω) on each ω, no mix.
# Mass m on diagonal — but with no (-1)^t alternation it stays diagonal.
# Spatial hop: η_1(t) = (-1)^t × (i sin(k)/2) × 2 = (-1)^t × i sin(k), where
# the factor of 2 came from the difference of fwd/bwd hops — wait actually
# for our convention η_1(t) (1/2)(e^{ik} - e^{-ik}) = η_1(t) × i sin(k).
# This (-1)^t mixes ω with ω+π in Fourier.
#
# So 2×2 block at (k, ω-pair):
#   | i sin(ω) + m              i sin(k) |
#   | i sin(k)                  i sin(ω+π) + m |
#   = | i sin(ω) + m            i sin(k) |
#     | i sin(k)                -i sin(ω) + m |
#
# det = (i sin ω + m)(-i sin ω + m) - (i sin k)²
#     = (m² + sin²ω) + sin²k
#     = m² + sin²ω + sin²k

# So per (k, ω-pair) the 2×2 determinant is m² + sin²ω + sin²k.
# Each ω-pair appears ONCE (not 2×), and there are Nt/2 pairs.

print()
print("Predicted formula: det M_L = ∏_{k=1..Nx} ∏_{ω-pair} (m² + sin²ω + sin²k)")
print(f"  Nt/2 ω-pairs × Nx k-values = {(Nt//2) * Nx} pair-contributions")

prediction = 1.0
for k_idx, sin2k in enumerate(sin2_ks):
    for omega_pair_idx, (omega, omega_pi) in enumerate(omega_pairs):
        # sin²(ω) = sin²(ω+π), so we can just use sin²(ω)
        sin2_omega = np.sin(omega) ** 2
        block_det = m_mass ** 2 + sin2_omega + sin2k
        prediction *= block_det
        print(f"  k={ks[k_idx]:.3f}, ω={omega:.3f}: block det = {block_det:.4f}")

print(f"\nPredicted det M_L = {prediction:.6f}")
print(f"Actual det M_L    = {det_L.real:.6f}")
print(f"Ratio             = {det_L.real / prediction:.6f}")

# --- now the Hamiltonian side ---------------------------------------------
print()
print("=" * 60)
print("Hamiltonian-side partition function")
print("=" * 60)

# Susskind staggered Hamiltonian in 1+1d:
#   H_F = m Σ_x (-1)^x χ†(x)χ(x) + (i/2) Σ_x [χ†(x+1)χ(x) - χ†(x)χ(x+1)]
h_stag = np.zeros((Nx, Nx), dtype=complex)
for x in range(Nx):
    h_stag[x, x] = m_mass * ((-1) ** x)
for x in range(Nx):
    h_stag[(x + 1) % Nx, x] += +1j / 2
    h_stag[x, (x + 1) % Nx] += -1j / 2

# Eigenvalues of h
eig_h = np.sort(np.linalg.eigvalsh(h_stag).real)
print(f"h_stag eigenvalues: {eig_h}")

# Positive eigenvalues squared: ε²(k) = m² + sin²(k_doubled)
# where k_doubled = 2π m / (Nx/2) for m = 0,...,Nx/2-1 is in the doubled BZ.
# For Nx=4: k_doubled = 0, π/2. So ε² = m²+0 = 0.25, m²+1 = 1.25.
# → ε = ±0.5, ±1.118.
print(f"Predicted h eigenvalues: ±{np.sqrt(m_mass**2):.4f}, ±{np.sqrt(m_mass**2 + 1):.4f}")

# Operator partition function
beta = Nt   # in lattice units (a=1)
Tr_exp_minus_betaH = np.prod([1 + np.exp(-beta * eps) for eps in eig_h])
print(f"\nβ = Nt = {beta}")
print(f"Tr e^{{-βH}} = det(1 + e^{{-βh}}) = ∏(1 + e^{{-βε}}) = {Tr_exp_minus_betaH:.6f}")

# --- the conjecture --------------------------------------------------------
# Naively we'd write det M_L = Tr e^{-βH_F}. But:
# - The Lagrangian counts MODES per (k,ω-pair), so total is Nx * Nt/2 = 16 modes
# - The Hamiltonian has only Nx = 4 single-particle modes
# - The det M_L = ∏(m² + sin²ω + sin²k) factor has more structure than (1+e^{-βε})
#
# Key insight: at each spatial k, the temporal product ∏_ω (m²+sin²ω+sin²k)
# is related to the SPATIAL transfer matrix.
#
# For free fermions:
#   ∏_{ω APBC} (a² + sin²ω) = 2 cosh(Nt · arcsinh(a)) = (e^{Nt·E(a)} + e^{-Nt·E(a)})/...
# where E(a) = arcsinh(a) is the lattice dispersion.

# Let me verify:
print()
print("Check: ∏_{ω-pair, APBC} (a² + sin²ω) = 2 cosh(Nt arcsinh(a))?")
a_test = np.sqrt(m_mass ** 2 + 1)  # for k=π/2, this is ε(k)
prod_over_omega = 1.0
for omega, _ in omega_pairs:
    prod_over_omega *= a_test ** 2 + np.sin(omega) ** 2
predicted = 2 * np.cosh(Nt * np.arcsinh(a_test))
print(f"  a² = m² + sin²k = {a_test**2:.4f}  (for k=π/2)")
print(f"  ∏_ω-pair (a² + sin²ω) = {prod_over_omega:.4f}")
print(f"  2·cosh(Nt·arcsinh(a))  = {predicted:.4f}")
print(f"  Ratio                  = {prod_over_omega / predicted:.4f}")

# If this formula works, then
#   det M_L = ∏_k 2 cosh(Nt arcsinh(ε(k)))
# and Tr e^{-βH} = ∏_k (1 + e^{-Nt ε(k)})(1 + e^{Nt ε(k)})  [for ±ε per k]
#                = ∏_k [2 + 2 cosh(Nt ε(k))] = ∏_k 4 cosh²(Nt ε(k)/2)
#                = ∏_k 2(1 + cosh(Nt ε(k)))
# The ratio:
# det M_L / Tr e^{-βH} = ∏_k [2 cosh(Nt arcsinh(ε)) / (4 cosh²(Nt ε/2))]
#                      = ∏_k [cosh(Nt arcsinh(ε)) / (2 cosh²(Nt ε/2))]
#
# In the continuum limit Nt → ∞ with β = Nt·a fixed (a → 0):
# arcsinh(ε·a)/a → ε (for small ε·a), so cosh(Nt·arcsinh(ε·a)) → cosh(β·ε)
# Then ratio → cosh(βε) / (2 cosh²(βε/2)) = 2cosh²(βε/2) / (2 cosh²(βε/2)) = 1
# so they agree in continuum limit but differ by O(a) on the lattice.

# Verify numerically at finite Nt:
print()
print("Continuum-limit prediction:")
print("  det M_L (Lagrangian, lattice)     ≈ det M_L (continuum) for a→0")
print("  Tr e^{-βH_F} (Hamiltonian)         differs by lattice artifacts")
print()
print("Final det M_L / Tr e^{-βH}:")
ratio = det_L.real / Tr_exp_minus_betaH
print(f"  = {det_L.real:.6f} / {Tr_exp_minus_betaH:.6f}")
print(f"  = {ratio:.6e}")
print()
print("This is the lattice O(a) artifact. The continuum-limit identification")
print("is det M_L ↔ Tr e^{-βH}; at finite a there's a relation through")
print("cosh(Nt·arcsinh(ε)) vs cosh(Nt·ε), with corrections suppressed as 1/Nt.")
