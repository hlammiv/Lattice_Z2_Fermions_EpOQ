"""
2+1d Z₂ + staggered fermion: end-to-end thermal correlator demonstration
(the M3.3 analog for the M4 target system).

Computes C(t) = ⟨n_0(t) n_0(0)⟩_β three ways on the 16-dim physical
subspace, and verifies all three agree:

  (I)   Direct in original gauge×matter basis (reference)
  (II)  Matrix-element SK corner-state sum
  (III) Z-C basis: H, ρ, n_0 all rotated; corner-state sum
"""

import numpy as np
from numpy import kron
from numpy.linalg import eigh, eigvalsh
from scipy.linalg import expm

# Setup (copy from z2_zc_v2.py)
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
    return op @ single_qubit_op(ann, q)

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

# Hamiltonian
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
H = (H_E + H_M + H_hop + H_mass)
H = (H + H.conj().T) / 2

# Physical projector
P_phys = I_full.copy()
for s in [(0,0), (0,1), (1,0), (1,1)]:
    P_phys = P_phys @ (I_full + G_sites[s]) / 2
P_phys = (P_phys + P_phys.conj().T) / 2

evals_P, evecs_P = eigh(P_phys)
phys_basis = evecs_P[:, evals_P > 0.5]

# Build Z-C unitary
R = I_full.copy()
for q_link in range(4):
    R = R @ single_qubit_op(H_pauli, q_link)

def cnot_full(c_q, t_q):
    P0 = (I2 + Z) / 2
    P1 = (I2 - Z) / 2
    op_I = np.eye(1, dtype=complex)
    op_X = np.eye(1, dtype=complex)
    for q in range(NQ):
        if q == c_q:
            op_I = kron(op_I, P0); op_X = kron(op_X, P1)
        elif q == t_q:
            op_I = kron(op_I, I2);  op_X = kron(op_X, X)
        else:
            op_I = kron(op_I, I2);  op_X = kron(op_X, I2)
    return op_I + op_X

U_ZC_x = I_full.copy()
for site in [(0,0), (0,1), (1,0), (1,1)]:
    target_q = matter_qubit[site]
    for control_q in links_at_site[site]:
        U_ZC_x = cnot_full(control_q, target_q) @ U_ZC_x

U_offset = I_full.copy()
for site, q_m in matter_qubit.items():
    if parity[site] == -1:
        U_offset = U_offset @ X_q[q_m]
U_ZC = U_offset @ R @ U_ZC_x @ R.conj().T

# --- Thermal density and observable ----------------------------------------
beta = 0.5
n0_op = N_q[4]   # number operator at matter site (0,0)

# Restrict everything to physical subspace
H_phys = phys_basis.conj().T @ H @ phys_basis
H_phys = (H_phys + H_phys.conj().T) / 2
rho_phys = expm(-beta * H_phys)
Z_phys = np.trace(rho_phys).real
n0_phys = phys_basis.conj().T @ n0_op @ phys_basis

# Same in Z-C basis
H_ZC = U_ZC.conj().T @ H @ U_ZC
H_ZC = (H_ZC + H_ZC.conj().T) / 2
n0_ZC = U_ZC.conj().T @ n0_op @ U_ZC

# Project H_ZC, n0_ZC onto matter=|0000⟩ sector (where Z-C-rotated physical lives)
P_m0 = I_full.copy()
for q_m in range(4, 8):
    P_m0 = P_m0 @ (I_full - N_q[q_m])
P_m0 = (P_m0 + P_m0.conj().T) / 2
evals_Pm, evecs_Pm = eigh(P_m0)
m0_basis = evecs_Pm[:, evals_Pm > 0.5]

H_ZC_m0 = m0_basis.conj().T @ H_ZC @ m0_basis
H_ZC_m0 = (H_ZC_m0 + H_ZC_m0.conj().T) / 2
rho_ZC_m0 = expm(-beta * H_ZC_m0)
Z_ZC_m0 = np.trace(rho_ZC_m0).real
n0_ZC_m0 = m0_basis.conj().T @ n0_ZC @ m0_basis

# --- Three methods for C(t) ------------------------------------------------
def method_I_direct(t):
    Ut = expm(-1j * H_phys * t)
    n0_t = Ut.conj().T @ n0_phys @ Ut
    return np.trace(rho_phys @ n0_t @ n0_phys).real / Z_phys

def method_II_corner_sum(t):
    """⟨n_0(t) n_0⟩ = (1/Z) Σ_{i,j,k} ρ_{ki} n_0(t)_{ij} n_0_{jk}."""
    Ut = expm(-1j * H_phys * t)
    n0_t = Ut.conj().T @ n0_phys @ Ut
    D = H_phys.shape[0]
    total = 0j
    for i in range(D):
        for j in range(D):
            for k in range(D):
                total += rho_phys[k, i] * n0_t[i, j] * n0_phys[j, k]
    return (total / Z_phys).real

def method_III_via_ZC(t):
    Ut = expm(-1j * H_ZC_m0 * t)
    n0_t = Ut.conj().T @ n0_ZC_m0 @ Ut
    D = H_ZC_m0.shape[0]
    total = 0j
    for i in range(D):
        for j in range(D):
            for k in range(D):
                total += rho_ZC_m0[k, i] * n0_t[i, j] * n0_ZC_m0[j, k]
    return (total / Z_ZC_m0).real

times = [0.0, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0]

print(f"2+1d Z₂ + staggered, 2×2 OBC, β={beta}")
print(f"Observable: C(t) = ⟨n_0(t) n_0(0)⟩_β  (site (0,0))")
print()
print(f"    t  |  (I) direct  |  (II) corner sum  |  (III) Z-C basis  |   max diff")
print("  -----+--------------+-------------------+-------------------+-----------")
max_d = 0
for t in times:
    cI = method_I_direct(t)
    cII = method_II_corner_sum(t)
    cIII = method_III_via_ZC(t)
    d = max(abs(cI-cII), abs(cI-cIII), abs(cII-cIII))
    max_d = max(max_d, d)
    print(f"  {t:>4.2f} | {cI:>12.6f} | {cII:>17.6f} | {cIII:>17.6f} | {d:.2e}")

print(f"\nMax disagreement across all times: {max_d:.3e}")
if max_d < 1e-10:
    print("✓ End-to-end thermal correlator in 2+1d Z₂: three methods agree.")
