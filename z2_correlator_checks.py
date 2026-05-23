"""
Independent correctness checks for the 2+1d Z₂ + staggered thermal correlator.

The three-method agreement in z2_end_to_end.py only proves internal consistency
of the Z-C transformation. It does NOT verify:
  - The Hamiltonian construction is correct
  - The Gauss-law projection is the right physical subspace
  - The matter operator n_0 has the right form

Independent checks here:
  (1) Gauge invariance: [H, G(site)] = 0 for all sites
  (2) Sum rule: C(0) = ⟨n_0⟩_β  (since n_0² = n_0 for occupation)
  (3) Spectral decomposition: C(t) computed via H eigenbasis as cross-check
  (4) High-T limit: C(0) → ⟨n_0⟩_∞ (maximally mixed value)
  (5) Low-T limit: C(t) → ⟨0|n_0(t) n_0|0⟩ (ground-state expectation)
  (6) Symmetry: Re C(t) = Re C(-t) (KMS / time-reversal of real part)
"""

import numpy as np
from numpy import kron
from numpy.linalg import eigh, eigvalsh
from scipy.linalg import expm

# Build everything (same as z2_end_to_end.py)
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
N_op = np.diag([0, 1]).astype(complex)
ann = np.array([[0, 1], [0, 0]], dtype=complex)

NQ = 8
dim = 2 ** NQ

def single(op, q):
    r = np.eye(1, dtype=complex)
    for i in range(NQ):
        r = kron(r, op if i == q else I2)
    return r

X_q = [single(X, q) for q in range(NQ)]
Z_q = [single(Z, q) for q in range(NQ)]
N_q = [single(N_op, q) for q in range(NQ)]

def fermion(i):
    op = np.eye(dim, dtype=complex)
    for b in range(i):
        op = op @ Z_q[4 + b]
    return op @ single(ann, 4 + i)

psi = [fermion(a) for a in range(4)]
psi_dag = [p.conj().T for p in psi]

links_at = {(0,0):[0,2], (0,1):[0,3], (1,0):[1,2], (1,1):[1,3]}
parity = {(0,0):+1, (0,1):-1, (1,0):-1, (1,1):+1}
m_q = {(0,0):4, (0,1):5, (1,0):6, (1,1):7}
I_full = np.eye(dim, dtype=complex)

def G_op(s):
    G = I_full.copy()
    for ql in links_at[s]:
        G = G @ X_q[ql]
    sign = I_full - 2 * N_q[m_q[s]]
    if parity[s] == -1:
        sign = -sign
    return G @ sign

G_sites = {s: G_op(s) for s in [(0,0),(0,1),(1,0),(1,1)]}

g_e, g_m, g_hop, m_mass = 1.0, 0.5, 0.5, 0.5
H_E = -g_e * sum(X_q[q] for q in range(4))
H_M = -g_m * (Z_q[0] @ Z_q[3] @ Z_q[1] @ Z_q[2])
H_hop = np.zeros((dim, dim), dtype=complex)
for q_link, a, b in [(0,0,1),(1,2,3),(2,0,2),(3,1,3)]:
    t = psi_dag[a] @ Z_q[q_link] @ psi[b]
    H_hop += g_hop * (t + t.conj().T)
H_mass = sum(m_mass * parity[s] * N_q[m_q[s]] for s in m_q)

H = (H_E + H_M + H_hop + H_mass)
H = (H + H.conj().T) / 2

P_phys = I_full.copy()
for s in [(0,0),(0,1),(1,0),(1,1)]:
    P_phys = P_phys @ (I_full + G_sites[s]) / 2
P_phys = (P_phys + P_phys.conj().T) / 2

evals_P, evecs_P = eigh(P_phys)
phys_basis = evecs_P[:, evals_P > 0.5]
H_p = phys_basis.conj().T @ H @ phys_basis
H_p = (H_p + H_p.conj().T) / 2
n0_p = phys_basis.conj().T @ N_q[4] @ phys_basis
n0_p = (n0_p + n0_p.conj().T) / 2

# ----------------------------------------------------------------------------
# (1) Gauge invariance: [H, G(site)] = 0
# ----------------------------------------------------------------------------
print("=" * 70)
print("(1) Gauge invariance — [H, G(site)] should be 0 for all sites")
print("=" * 70)
for site in [(0,0),(0,1),(1,0),(1,1)]:
    comm = H @ G_sites[site] - G_sites[site] @ H
    err = np.linalg.norm(comm)
    print(f"  || [H, G{site}] || = {err:.3e}")

# ----------------------------------------------------------------------------
# (2) C(0) = ⟨n_0⟩_β  (since n_0² = n_0 for occupation operator)
# ----------------------------------------------------------------------------
print()
print("=" * 70)
print("(2) Sum rule: C(0) = ⟨n_0⟩_β  (n_0² = n_0 for occupation)")
print("=" * 70)
# Verify n_0² = n_0 first
n0_sq = N_q[4] @ N_q[4]
print(f"  || n_0² - n_0 || = {np.linalg.norm(n0_sq - N_q[4]):.3e}")

for beta in [0.1, 0.5, 1.0, 2.0, 5.0]:
    rho_p = expm(-beta * H_p)
    Z_p = np.trace(rho_p).real
    C0 = np.trace(rho_p @ n0_p @ n0_p).real / Z_p
    avg_n0 = np.trace(rho_p @ n0_p).real / Z_p
    print(f"  β={beta:>4.1f}: C(0) = {C0:.6f}, ⟨n_0⟩_β = {avg_n0:.6f}, "
          f"|diff| = {abs(C0 - avg_n0):.2e}")

# ----------------------------------------------------------------------------
# (3) Spectral decomposition cross-check
# ----------------------------------------------------------------------------
print()
print("=" * 70)
print("(3) Spectral decomposition: C(t) = (1/Z) Σ_{nm} e^{-βE_n} |⟨n|n_0|m⟩|² e^{i(E_m-E_n)t}")
print("=" * 70)
beta = 0.5
energies, eigvecs = eigh(H_p)
rho_p = expm(-beta * H_p)
Z_p = np.trace(rho_p).real

# Matrix elements ⟨n|n_0|m⟩ in energy basis
n0_eig = eigvecs.conj().T @ n0_p @ eigvecs

def C_spectral(t):
    """C(t) via spectral decomposition: ⟨n_0(t) n_0⟩."""
    boltzmann = np.exp(-beta * energies)
    # C(t) = (1/Z) Σ_nm boltzmann[n] |n0_nm|^2 e^{i(E_m-E_n)t}
    # = (1/Z) Σ_nm boltzmann[n] n0_nm conj(n0_nm) e^{i(E_m-E_n)t}
    phase = np.exp(1j * np.outer(-energies, np.ones_like(energies)) +
                   1j * np.outer(np.ones_like(energies), energies)) * t
    # Wait that's exp(i(E_m - E_n)t). Let's redo:
    E_diff = energies[None, :] - energies[:, None]
    phase = np.exp(1j * E_diff * t)
    weights = boltzmann[:, None]
    return np.sum(weights * np.abs(n0_eig) ** 2 * phase).real / Z_p

def C_direct(t):
    Ut = expm(-1j * H_p * t)
    n0_t = Ut.conj().T @ n0_p @ Ut
    return np.trace(rho_p @ n0_t @ n0_p).real / Z_p

print(f"  {'t':>5} | {'C(t) direct':>13} | {'C(t) spectral':>15} | diff")
for t in [0.0, 0.5, 1.0, 2.0, 3.0]:
    cD = C_direct(t)
    cS = C_spectral(t)
    print(f"  {t:>5.2f} | {cD:>13.6f} | {cS:>15.6f} | {abs(cD-cS):.2e}")

# ----------------------------------------------------------------------------
# (4) High-T limit: ρ_β → I/D, C(0) → ⟨n_0²⟩_{I/D}
# ----------------------------------------------------------------------------
print()
print("=" * 70)
print("(4) High-T limit β → 0: C(0) should → Tr[n_0]/D_phys")
print("=" * 70)
D_phys = H_p.shape[0]
C0_inf_T = np.trace(n0_p).real / D_phys
print(f"  Tr[n_0]/D_phys = {C0_inf_T:.6f}  (predicted high-T limit)")
for beta_small in [0.5, 0.1, 0.01, 0.001]:
    rho_b = expm(-beta_small * H_p)
    Z_b = np.trace(rho_b).real
    C0_b = np.trace(rho_b @ n0_p @ n0_p).real / Z_b
    print(f"  β={beta_small:>6.3f}: C(0) = {C0_b:.6f}, |C(0) - Tr/D| = {abs(C0_b - C0_inf_T):.4e}")

# ----------------------------------------------------------------------------
# (5) Low-T limit: C(t) → ⟨0|n_0(t) n_0|0⟩
# ----------------------------------------------------------------------------
print()
print("=" * 70)
print("(5) Low-T limit β → ∞: C(t) should → ground-state expectation")
print("=" * 70)
ground = eigvecs[:, 0]
E_gs = energies[0]
print(f"  Ground state energy: {E_gs:.4f}")
print(f"  Energy gap: ΔE = {energies[1] - energies[0]:.4f}")
print()
print(f"  {'t':>5} | {'C(t) low-T':>13} | {'⟨0|n_0(t)n_0|0⟩':>17}")
beta_large = 10.0
rho_b = expm(-beta_large * H_p)
Z_b = np.trace(rho_b).real
for t in [0.0, 0.5, 1.0, 2.0]:
    Ut = expm(-1j * H_p * t)
    cT = np.trace(rho_b @ Ut.conj().T @ n0_p @ Ut @ n0_p).real / Z_b
    n0_t = Ut.conj().T @ n0_p @ Ut
    cGS = (ground.conj() @ n0_t @ n0_p @ ground).real
    print(f"  {t:>5.2f} | {cT:>13.6f} | {cGS:>17.6f}")

# ----------------------------------------------------------------------------
# (6) Re C(t) should equal Re C(-t) by hermiticity of the thermal trace
# ----------------------------------------------------------------------------
print()
print("=" * 70)
print("(6) Time-reflection: Re C(t) = Re C(-t)?")
print("=" * 70)
beta = 0.5
rho_p = expm(-beta * H_p)
Z_p = np.trace(rho_p).real
for t in [0.5, 1.0, 2.0]:
    Ut = expm(-1j * H_p * t)
    Umt = expm(+1j * H_p * t)
    cT = np.trace(rho_p @ Ut.conj().T @ n0_p @ Ut @ n0_p) / Z_p
    cMT = np.trace(rho_p @ Umt.conj().T @ n0_p @ Umt @ n0_p) / Z_p
    print(f"  t={t}:  C(t) = {cT}, C(-t) = {cMT}")
    print(f"          Re[C(t) - C(-t)] = {(cT - cMT).real:.3e}")
    print(f"          Im[C(t) + C(-t)] = {(cT + cMT).imag:.3e}")
