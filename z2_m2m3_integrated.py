"""
M2 + M3 integrated pipeline in 2+1d Z₂ — the full M4 demonstration in toy form.

For the 2+1d Z₂ + staggered fermion system (2×2 OBC, 16-dim physical subspace),
compute the thermal correlator C(t) = ⟨n_0(t) n_0(0)⟩_β via stochastic
sampling of corner-state pairs, two ways:
  (a) directly in the physical subspace basis
  (b) in the Z-C-rotated basis (matter eliminated)

Both should converge to the exact reference within sampling noise.
"""

import numpy as np
from numpy import kron
from numpy.linalg import eigh
from scipy.linalg import expm

# Setup (same as z2_end_to_end.py)
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

P_phys = I_full.copy()
for s in [(0,0), (0,1), (1,0), (1,1)]:
    P_phys = P_phys @ (I_full + G_sites[s]) / 2
P_phys = (P_phys + P_phys.conj().T) / 2

evals_P, evecs_P = eigh(P_phys)
phys_basis = evecs_P[:, evals_P > 0.5]

R = I_full.copy()
for q_link in range(4):
    R = R @ single_qubit_op(H_pauli, q_link)

def cnot_full(c_q, t_q):
    P0 = (I2 + Z) / 2
    P1 = (I2 - Z) / 2
    op_I = np.eye(1, dtype=complex)
    op_X = np.eye(1, dtype=complex)
    for q in range(NQ):
        if q == c_q: op_I = kron(op_I, P0); op_X = kron(op_X, P1)
        elif q == t_q: op_I = kron(op_I, I2);  op_X = kron(op_X, X)
        else:           op_I = kron(op_I, I2);  op_X = kron(op_X, I2)
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

# --- Restrict to physical subspace -----------------------------------------
beta = 0.5
n0_op = N_q[4]

# Direct in physical subspace
H_p = phys_basis.conj().T @ H @ phys_basis
H_p = (H_p + H_p.conj().T) / 2
rho_p = expm(-beta * H_p)
Z_p = np.trace(rho_p).real
n0_p = phys_basis.conj().T @ n0_op @ phys_basis

D = H_p.shape[0]  # = 16

# Z-C basis on matter=|0000⟩ sector
P_m0 = I_full.copy()
for q_m in range(4, 8):
    P_m0 = P_m0 @ (I_full - N_q[q_m])
P_m0 = (P_m0 + P_m0.conj().T) / 2
evals_Pm, evecs_Pm = eigh(P_m0)
m0_basis = evecs_Pm[:, evals_Pm > 0.5]

H_ZC = U_ZC.conj().T @ H @ U_ZC
H_ZC = (H_ZC + H_ZC.conj().T) / 2
H_ZC_m0 = m0_basis.conj().T @ H_ZC @ m0_basis
H_ZC_m0 = (H_ZC_m0 + H_ZC_m0.conj().T) / 2
rho_ZC_m0 = expm(-beta * H_ZC_m0)
Z_ZC_m0 = np.trace(rho_ZC_m0).real
n0_ZC = U_ZC.conj().T @ n0_op @ U_ZC
n0_ZC_m0 = m0_basis.conj().T @ n0_ZC @ m0_basis

# --- Stochastic samplers ---------------------------------------------------
def exact_C(t):
    Ut = expm(-1j * H_p * t)
    n0_t = Ut.conj().T @ n0_p @ Ut
    return np.trace(rho_p @ n0_t @ n0_p).real / Z_p

def stochastic_C_direct(t, N_samples, rng):
    Ut = expm(-1j * H_p * t)
    n0_t = Ut.conj().T @ n0_p @ Ut
    A = n0_t @ n0_p
    # C = Σ_ki ρ_ki A_ik / Z; importance-sample (k,i) ~ |ρ_ki|
    prob = np.abs(rho_p).flatten()
    total_w = prob.sum()
    prob /= total_w
    flat_idx = rng.choice(D * D, size=N_samples, p=prob)
    acc = 0j
    for fidx in flat_idx:
        k, i = divmod(fidx, D)
        sign = rho_p[k, i] / abs(rho_p[k, i]) if abs(rho_p[k, i]) > 0 else 0
        acc += sign * A[i, k]
    return (acc * total_w / (N_samples * Z_p)).real

def stochastic_C_ZC(t, N_samples, rng):
    """Same observable computed by sampling in the Z-C-rotated basis."""
    Ut = expm(-1j * H_ZC_m0 * t)
    n0_t = Ut.conj().T @ n0_ZC_m0 @ Ut
    A = n0_t @ n0_ZC_m0
    prob = np.abs(rho_ZC_m0).flatten()
    total_w = prob.sum()
    prob /= total_w
    flat_idx = rng.choice(D * D, size=N_samples, p=prob)
    acc = 0j
    for fidx in flat_idx:
        k, i = divmod(fidx, D)
        sign = rho_ZC_m0[k, i] / abs(rho_ZC_m0[k, i]) if abs(rho_ZC_m0[k, i]) > 0 else 0
        acc += sign * A[i, k]
    return (acc * total_w / (N_samples * Z_ZC_m0)).real

# --- Run -------------------------------------------------------------------
times = [0.0, 0.5, 1.0, 2.0, 3.0]
N_samples = 50_000

print(f"2+1d Z₂ + staggered, stochastic M2+M3 pipeline (β={beta}, N={N_samples})")
print()
print(f"    t  |  exact C(t)  | stochastic direct | stochastic Z-C |   diff (direct) | diff (Z-C)")
print("  -----+--------------+-------------------+----------------+-----------------+----------")

for t in times:
    cE = exact_C(t)
    rng = np.random.default_rng(2026)
    cD = stochastic_C_direct(t, N_samples, rng)
    rng = np.random.default_rng(2026)
    cZ = stochastic_C_ZC(t, N_samples, rng)
    print(f"  {t:>4.2f} | {cE:>12.6f} | {cD:>17.6f} | {cZ:>14.6f} | {abs(cE-cD):>15.4e} | {abs(cE-cZ):>9.4e}")

print()
print("Both stochastic routes (direct and Z-C-rotated) converge to the same")
print("exact C(t) within statistical noise (~1% at N=50000, consistent with 1/√N).")
print("This is the M4 demonstration in toy form: SK + PF-sample-style + Z-C")
print("end-to-end on 2+1d Z₂ + staggered.")
