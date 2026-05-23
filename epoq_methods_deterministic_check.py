"""
Deterministic cross-check of Method 2 and Method 3 estimators, removing the
stochastic sampling noise so we can verify the algorithms agree with the
physical-sector ED reference in EXPECTATION.

For each method, computes
    C(t) = (1/Z) Σ_{ij} O_j × G[j,i] × ⟨i|U†(t) O U(t) |j⟩
directly via dense matrix products on the propagator G that the sampler
would have used.  No (i,j) sampling, no bootstrap.  The remaining sources
of error are:
  - imaginary-time Trotter (controlled by n_trotter_imag)
  - real-time Trotter (controlled by n_trotter_real)
"""

import sys
import numpy as np
from scipy.linalg import expm

sys.path.insert(0, '/home/hlamm/Desktop/QC/logdet/m1_toy')
import epoq_classical_sampler as cs

# Parameters (match epoq_methods_compare.py)
BETA = 0.5
N_TROTTER_IMAG = 50
N_TROTTER_REAL = 200
TIMES = [0.0, 0.5, 1.0, 2.0]
LAMBDA_PEN = 10.0
NQ = 8; DIM = 256

# --- Pauli term -> dense matrix utility (for validation) ----
P2 = {'I': np.eye(2, dtype=complex), 'X': np.array([[0,1],[1,0]], dtype=complex),
      'Y': np.array([[0,-1j],[1j,0]], dtype=complex), 'Z': np.array([[1,0],[0,-1]], dtype=complex)}

def pauli_term_dense(factors, nq=NQ):
    by_q = {q: P2['I'] for q in range(nq)}
    for q, ax in factors:
        by_q[q] = P2[ax]
    result = by_q[nq-1]
    for q in range(nq-2, -1, -1):
        result = np.kron(result, by_q[q])
    return result

# --- Build dense H, n_0, U_ZC, projectors (for derivation only) ---
H_terms = cs.build_pauli_terms()
H_dense = np.zeros((DIM, DIM), dtype=complex)
for c, fac in H_terms:
    H_dense += c * pauli_term_dense(fac)
H_dense = (H_dense + H_dense.conj().T) / 2

n0_dense = 0.5 * np.eye(DIM, dtype=complex) - 0.5 * pauli_term_dense([(4, 'Z')])

# Gauss operators and physical projector
links_at = {(0,0):[0,2],(0,1):[0,3],(1,0):[1,2],(1,1):[1,3]}
parity = {(0,0):+1,(0,1):-1,(1,0):-1,(1,1):+1}
matter_q = {(0,0):4,(0,1):5,(1,0):6,(1,1):7}

def G_dense(site):
    G = np.eye(DIM, dtype=complex)
    for ql in links_at[site]:
        G = G @ pauli_term_dense([(ql, 'X')])
    sign = pauli_term_dense([(matter_q[site], 'Z')])
    if parity[site] == -1:
        sign = -sign
    return G @ sign

G_ops = {s: G_dense(s) for s in [(0,0),(0,1),(1,0),(1,1)]}

# Physical projector
P_phys = np.eye(DIM, dtype=complex)
for s in G_ops:
    P_phys = P_phys @ (np.eye(DIM, dtype=complex) + G_ops[s]) / 2
P_phys = (P_phys + P_phys.conj().T) / 2

# Reference: physical-sector C(t)
evals_P, evecs_P = np.linalg.eigh(P_phys)
phys_basis = evecs_P[:, evals_P > 0.5]
H_phys = phys_basis.conj().T @ H_dense @ phys_basis
n0_phys = phys_basis.conj().T @ n0_dense @ phys_basis
rho_phys = expm(-BETA * H_phys)
Z_phys = np.trace(rho_phys).real

def C_phys_ref(t):
    Ut = expm(-1j * H_phys * t)
    n0_t = Ut.conj().T @ n0_phys @ Ut
    return np.trace(rho_phys @ n0_t @ n0_phys).real / Z_phys

# --- Method 2: H + penalty, evaluate C(t) deterministically via dense G_M2 ---
def penalty_terms(lam):
    out = []
    for site in [(0,0),(0,1),(1,0),(1,1)]:
        coef = -lam * parity[site]
        factors = [(ql,'X') for ql in links_at[site]] + [(matter_q[site],'Z')]
        out.append((coef + 0j, factors))
    return out

H_eff_terms = H_terms + penalty_terms(LAMBDA_PEN)
print("Building Method 2 propagator (penalty, Trotter)...")
G_M2 = np.zeros((DIM, DIM), dtype=complex)
for i in range(DIM):
    e_i = np.zeros(DIM, dtype=complex); e_i[i] = 1.0
    G_M2[:, i] = cs.evolve_imag(e_i, H_eff_terms, BETA, N_TROTTER_IMAG)
Z_M2 = G_M2.diagonal().real.sum()

# C_M2(t) = (1/Z_M2) Σ_ij O_j G_M2[j,i] ⟨i|U†OU|j⟩
# = Tr[O ρ_eff U†OU] / Z_M2 (where ρ_eff = G_M2 as a matrix)
def C_method2_dense(t):
    Ut = expm(-1j * H_dense * t)
    n0_t = Ut.conj().T @ n0_dense @ Ut
    # Tr[O G_M2 U†OU] = Tr[n0 G_M2 n0_t]   (matrix Tr, cyclic)
    return np.trace(n0_dense @ G_M2 @ n0_t).real / Z_M2

# --- Method 3: 4-qubit gauge-only via Z-C ---
H_pauli_mat = np.array([[1,1],[1,-1]], dtype=complex)/np.sqrt(2)
R_full = np.eye(DIM, dtype=complex)
for ql in range(4):
    R_full = R_full @ np.kron(np.eye(2**(NQ-1-ql), dtype=complex),
                              np.kron(H_pauli_mat, np.eye(2**ql, dtype=complex)))
# Actually use existing pauli_term_dense conventions: single_qubit_op via kron
def sql(op, q):
    """q0 = LSB convention (matches epoq_classical_sampler module)."""
    r = np.array([[1.+0j]])
    for i in range(NQ-1, -1, -1):
        r = np.kron(r, op if i == q else P2['I'])
    return r
R_full = np.eye(DIM, dtype=complex)
for ql in range(4):
    R_full = R_full @ sql(H_pauli_mat, ql)

def cnot_full(c, t):
    """q0 = LSB convention."""
    P0 = (P2['I'] + P2['Z'])/2
    P1 = (P2['I'] - P2['Z'])/2
    op_I = np.array([[1.+0j]]); op_X = np.array([[1.+0j]])
    for q in range(NQ-1, -1, -1):
        if q==c: op_I=np.kron(op_I,P0); op_X=np.kron(op_X,P1)
        elif q==t: op_I=np.kron(op_I,P2['I']); op_X=np.kron(op_X,P2['X'])
        else: op_I=np.kron(op_I,P2['I']); op_X=np.kron(op_X,P2['I'])
    return op_I + op_X

U_ZC_x = np.eye(DIM, dtype=complex)
for site in [(0,0),(0,1),(1,0),(1,1)]:
    for ql in links_at[site]:
        U_ZC_x = cnot_full(ql, matter_q[site]) @ U_ZC_x

U_offset = np.eye(DIM, dtype=complex)
for site, qm in matter_q.items():
    if parity[site] == -1:
        U_offset = U_offset @ pauli_term_dense([(qm, 'X')])

U_ZC = U_offset @ R_full @ U_ZC_x @ R_full.conj().T

# Restrict H', n0' to matter=|0000⟩ → 16x16
# Convention from z2_zc_v2.py: U_ZC† maps phys → target; so rotated operator
# in target basis is U_ZC† X U_ZC.
matter0_idx = [i for i in range(DIM) if (i >> 4) == 0]
H_prime = (U_ZC.conj().T @ H_dense @ U_ZC)[np.ix_(matter0_idx, matter0_idx)]
H_prime = (H_prime + H_prime.conj().T) / 2
n0_prime = (U_ZC.conj().T @ n0_dense @ U_ZC)[np.ix_(matter0_idx, matter0_idx)]
n0_prime = (n0_prime + n0_prime.conj().T) / 2

# Now build G_M3 = e^{-β H'} on 4-qubit space — use dense expm here as proxy
# for what 4-qubit Trotter would give (Trotter convergence at 4q is excellent).
G_M3 = expm(-BETA * H_prime)
Z_M3 = G_M3.diagonal().real.sum()

def C_method3_dense(t):
    Ut = expm(-1j * H_prime * t)
    n0_t = Ut.conj().T @ n0_prime @ Ut
    return np.trace(n0_prime @ G_M3 @ n0_t).real / Z_M3

# --- Report ---
print("\nDeterministic cross-check (no sampling, no bootstrap):")
print(f"  {'t':>5} | {'C_phys_ref':>11} | {'C_M2':>10} | {'C_M3':>10} | {'M2 err':>10} | {'M3 err':>10}")
for t in TIMES:
    cR = C_phys_ref(t)
    c2 = C_method2_dense(t)
    c3 = C_method3_dense(t)
    print(f"  {t:>5.2f} | {cR:>11.6f} | {c2:>10.6f} | {c3:>10.6f} | "
          f"{abs(c2-cR):>10.2e} | {abs(c3-cR):>10.2e}")

print(f"\nZ_phys (ED ref) = {Z_phys:.4f}")
print(f"Z_M2            = {Z_M2:.4e}  (= e^(βλN_sites=20) × Z_phys × (1+O(e^-10)))")
print(f"Z_M2 / e^(20)   = {Z_M2 / np.exp(20):.4f}  (should be ≈ Z_phys)")
print(f"Z_M3            = {Z_M3:.4f}  (should be ≈ Z_phys)")
