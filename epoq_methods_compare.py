"""
Compare Method 2 (Lagrange-multiplier gauge penalty) vs Method 3 (Z-C-rotated,
4-qubit gauge-only) on the 2×2 OBC Z₂ + staggered toy.

Both should reproduce the physical-sector C(t) (≈ 0.31 at β=0.5, t=0, matching
z2_end_to_end.py's reference). The comparison records a range of metrics for
each method beyond just C(t).
"""

import sys
import time
import numpy as np
from scipy.linalg import expm

sys.path.insert(0, '/home/hlamm/Desktop/QC/logdet/m1_toy')
import epoq_classical_sampler as cs

# Shared parameters
BETA = 0.5
N_TROTTER_IMAG = 50
N_TROTTER_REAL = 200
N_SAMPLES = 200
N_BOOTSTRAP = 50
TIMES = [0.0, 0.5, 1.0, 2.0]
LAMBDA_PEN = 10.0

# ----------------------------------------------------------------------------
# Common: build H Pauli terms (8-qubit) and reference values from ED
# ----------------------------------------------------------------------------
H_terms_8q = cs.build_pauli_terms()
print(f"Original H: {len(H_terms_8q)} Pauli terms on 8 qubits")

# Build dense H, U_ZC, n_0 once for derivations and ED reference (validation only)
NQ = 8
DIM = 256

P2 = {
    'I': np.eye(2, dtype=np.complex128),
    'X': np.array([[0, 1], [1, 0]], dtype=np.complex128),
    'Y': np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    'Z': np.array([[1, 0], [0, -1]], dtype=np.complex128),
}

def pauli_term_dense(factors, n_qubits=NQ):
    by_q = {q: P2['I'] for q in range(n_qubits)}
    for q, ax in factors:
        by_q[q] = P2[ax]
    result = by_q[n_qubits - 1]
    for q in range(n_qubits - 2, -1, -1):
        result = np.kron(result, by_q[q])
    return result

H_dense = np.zeros((DIM, DIM), dtype=np.complex128)
for coef, factors in H_terms_8q:
    H_dense += coef * pauli_term_dense(factors)
assert np.allclose(H_dense, H_dense.conj().T)

# n_0 = (I - Z_4)/2
n0_dense = 0.5 * pauli_term_dense([]) - 0.5 * pauli_term_dense([(4, 'Z')])

# ----------------------------------------------------------------------------
# Reference C(t) from physical-sector ED (matches z2_end_to_end.py)
# ----------------------------------------------------------------------------
# Build Gauss operators G(site) and project to physical subspace
links_at = {(0, 0): [0, 2], (0, 1): [0, 3], (1, 0): [1, 2], (1, 1): [1, 3]}
parity = {(0, 0): +1, (0, 1): -1, (1, 0): -1, (1, 1): +1}
matter_q = {(0, 0): 4, (0, 1): 5, (1, 0): 6, (1, 1): 7}

I_full = np.eye(DIM, dtype=np.complex128)
def G_dense(site):
    G = I_full.copy()
    for ql in links_at[site]:
        G = G @ pauli_term_dense([(ql, 'X')])
    sign = I_full - 2 * pauli_term_dense([(matter_q[site], 'Z')]) @ ((I_full - pauli_term_dense([(matter_q[site], 'Z')]))/2 + pauli_term_dense([(matter_q[site], 'Z')]) @ (I_full + pauli_term_dense([(matter_q[site], 'Z')]))/2 - I_full)
    # Simpler: (I - 2N) = Z_matter
    sign = pauli_term_dense([(matter_q[site], 'Z')])
    if parity[site] == -1:
        sign = -sign
    return G @ sign

G_ops = {s: G_dense(s) for s in [(0,0),(0,1),(1,0),(1,1)]}
P_phys = I_full.copy()
for s in G_ops:
    P_phys = P_phys @ (I_full + G_ops[s]) / 2
P_phys = (P_phys + P_phys.conj().T) / 2
evals_P, evecs_P = np.linalg.eigh(P_phys)
phys_basis = evecs_P[:, evals_P > 0.5]
H_phys = phys_basis.conj().T @ H_dense @ phys_basis
H_phys = (H_phys + H_phys.conj().T) / 2
n0_phys = phys_basis.conj().T @ n0_dense @ phys_basis
n0_phys = (n0_phys + n0_phys.conj().T) / 2
rho_phys = expm(-BETA * H_phys)
Z_phys_ref = np.trace(rho_phys).real

def C_phys_ref(t):
    Ut = expm(-1j * H_phys * t)
    n0_t = Ut.conj().T @ n0_phys @ Ut
    return np.trace(rho_phys @ n0_t @ n0_phys).real / Z_phys_ref

reference_C = {t: C_phys_ref(t) for t in TIMES}
print(f"Reference C(t) from physical-sector ED:")
for t in TIMES:
    print(f"  t={t}: C_phys_ref = {reference_C[t]:.6f}")

# ----------------------------------------------------------------------------
# Method 2: H + λ Σ_site (1 - G(site)) gauge penalty
# ----------------------------------------------------------------------------
# G(site) as Pauli strings: ∏(X on links at site) × parity × Z on matter qubit
# Penalty: λΣ(1-G) = constant - λΣG. Drop constant. Add -λ × signed G terms.
def gauss_penalty_terms(lam):
    """Return list of Pauli terms for -λ Σ G(site), including parity signs."""
    out = []
    for site in [(0,0), (0,1), (1,0), (1,1)]:
        coef = -lam * parity[site]   # -λ × par(site)
        factors = [(ql, 'X') for ql in links_at[site]] + [(matter_q[site], 'Z')]
        out.append((coef + 0j, factors))
    return out

H_eff_terms_M2 = H_terms_8q + gauss_penalty_terms(LAMBDA_PEN)
print(f"\nMethod 2: H_eff with penalty = {len(H_eff_terms_M2)} Pauli terms")

# Build penalized propagator
print(f"Method 2: building penalized propagator (λ={LAMBDA_PEN}, Trotter n={N_TROTTER_IMAG})...")
t0 = time.time()
G_M2 = np.zeros((DIM, DIM), dtype=np.complex128)
for i in range(DIM):
    e_i = np.zeros(DIM, dtype=np.complex128)
    e_i[i] = 1.0
    psi = cs.evolve_imag(e_i, H_eff_terms_M2, BETA, N_TROTTER_IMAG)
    G_M2[:, i] = psi
t_M2_prop = time.time() - t0
Z_M2 = G_M2.diagonal().real.sum()
total_w_M2 = np.abs(G_M2).sum()
print(f"  Built in {t_M2_prop:.1f}s")
print(f"  Z(β, λ) = {Z_M2:.6f}; total |G| weight = {total_w_M2:.4f}")
print(f"  Z_ratio (penalized / physical-sector ED) = {Z_M2 / Z_phys_ref:.4f}")
print(f"  (Should be ~1 if penalty cleanly isolates physical sector)")

# Sample
rng = np.random.default_rng(2026)
abs_G = np.abs(G_M2)
flat_p = (abs_G / total_w_M2).ravel()
flat_p /= flat_p.sum()
idx = rng.choice(DIM * DIM, size=N_SAMPLES, p=flat_p)
j_M2 = idx // DIM
i_M2 = idx % DIM
signs_M2 = np.sign(G_M2.real[j_M2, i_M2]).astype(int)
signs_M2[signs_M2 == 0] = 1

# Diagnostic: what fraction of samples are in the physical sector?
# Check via the projection P_phys: ⟨bit_string i | P_phys | bit_string i⟩
def is_physical_state(idx):
    """Diagonal element of P_phys at bit-string index (probability mass on physical sector)."""
    return P_phys[idx, idx].real
phys_fraction_M2_i = np.mean([is_physical_state(int(i_M2[k])) for k in range(N_SAMPLES)])
print(f"  Avg ⟨i|P_phys|i⟩ over samples: {phys_fraction_M2_i:.4f}  (1 = fully physical)")

# Real-time matrix elements using the existing simulator
import epoq_quantum_simulator as qs
n0_obs = [(0.5 + 0j, []), (-0.5 + 0j, [(4, 'Z')])]

print(f"\nMethod 2: evaluating real-time matrix elements (Trotter n={N_TROTTER_REAL})...")
t0 = time.time()
contribs_M2 = {t: np.zeros(N_SAMPLES) for t in TIMES}
for k in range(N_SAMPLES):
    if k % 50 == 0:
        print(f"  sample {k}/{N_SAMPLES}...", flush=True)
    i, j = int(i_M2[k]), int(j_M2[k])
    Oj = (j >> 4) & 1
    if Oj == 0:
        continue
    for t in TIMES:
        mat_el = qs.observable_matrix_element(j, i, n0_obs, t, n_trotter=N_TROTTER_REAL)
        contribs_M2[t][k] = signs_M2[k] * Oj * mat_el.real
t_M2_real = time.time() - t0
print(f"  Real-time evaluation: {t_M2_real:.1f}s")

# Bootstrap
def bootstrap(contribs, total_w, Z, N_boot):
    N = len(contribs)
    rng_b = np.random.default_rng(31337)
    means = []
    for _ in range(N_boot):
        idx = rng_b.choice(N, N, replace=True)
        means.append(contribs[idx].mean() * total_w / Z)
    return np.mean(means), np.std(means)

results_M2 = {t: bootstrap(contribs_M2[t], total_w_M2, Z_M2, N_BOOTSTRAP) for t in TIMES}

# ----------------------------------------------------------------------------
# Method 3: Z-C-rotated, 4-qubit gauge-only
# ----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Method 3: Z-C rotation + 4-qubit gauge-only simulation")
print("=" * 60)

# Build U_ZC dense (same as z2_zc_v2.py)
H_pauli_mat = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=np.complex128)
X_mat = P2['X']; Z_mat = P2['Z']; I_2 = P2['I']

def single_op(op, q, nq):
    r = np.array([[1.0+0j]])
    for i in range(nq):
        r = np.kron(r, op if i == q else I_2)
    return r

R_full = I_full.copy()
for ql in range(4):
    R_full = R_full @ single_op(H_pauli_mat, ql, NQ)

def cnot_full(c, t):
    P0 = (I_2 + Z_mat) / 2
    P1 = (I_2 - Z_mat) / 2
    op_I = np.array([[1.0+0j]])
    op_X = np.array([[1.0+0j]])
    for q in range(NQ):
        if q == c:
            op_I = np.kron(op_I, P0); op_X = np.kron(op_X, P1)
        elif q == t:
            op_I = np.kron(op_I, I_2); op_X = np.kron(op_X, X_mat)
        else:
            op_I = np.kron(op_I, I_2); op_X = np.kron(op_X, I_2)
    return op_I + op_X

U_ZC_x = I_full.copy()
for site in [(0,0), (0,1), (1,0), (1,1)]:
    for ql in links_at[site]:
        U_ZC_x = cnot_full(ql, matter_q[site]) @ U_ZC_x

U_offset = I_full.copy()
for site, qm in matter_q.items():
    if parity[site] == -1:
        U_offset = U_offset @ pauli_term_dense([(qm, 'X')])

U_ZC = U_offset @ R_full @ U_ZC_x @ R_full.conj().T

# Build H' = U_ZC H U_ZC† and restrict to matter=|0000⟩
H_prime = U_ZC @ H_dense @ U_ZC.conj().T
n0_prime = U_ZC @ n0_dense @ U_ZC.conj().T

# Find indices of computational basis states with matter qubits = 0
# Matter qubits are q4..q7; matter=0 means bits 4..7 are all 0
matter0_indices = [i for i in range(DIM) if (i >> 4) == 0]
assert len(matter0_indices) == 16

H_prime_gauge = H_prime[np.ix_(matter0_indices, matter0_indices)]
n0_prime_gauge = n0_prime[np.ix_(matter0_indices, matter0_indices)]
H_prime_gauge = (H_prime_gauge + H_prime_gauge.conj().T) / 2

print(f"H' restricted to matter=|0000⟩ sector: 16×16 matrix")
print(f"Verify spectrum matches H_phys: max eig diff = "
      f"{np.max(np.abs(np.sort(np.linalg.eigvalsh(H_prime_gauge)) - np.sort(np.linalg.eigvalsh(H_phys)))):.3e}")

# Pauli-decompose H_prime_gauge into 4-qubit Pauli terms
def pauli_decompose_4q(M):
    """Decompose a 16x16 matrix into 4-qubit Pauli terms."""
    NQ4 = 4
    terms = []
    for k in range(4 ** NQ4):
        labels = []
        for q in range(NQ4):
            label_idx = (k // 4**q) % 4
            labels.append((q, ['I', 'X', 'Y', 'Z'][label_idx]))
        P = pauli_term_dense([(q, l) for q, l in labels if l != 'I'], n_qubits=NQ4)
        coef = np.trace(P.conj().T @ M) / (2 ** NQ4)
        if abs(coef) > 1e-10:
            non_id = [(q, l) for q, l in labels if l != 'I']
            terms.append((complex(coef), non_id))
    return terms

H_prime_terms_4q = pauli_decompose_4q(H_prime_gauge)
n0_prime_terms_4q = pauli_decompose_4q(n0_prime_gauge)
print(f"H'_gauge has {len(H_prime_terms_4q)} 4-qubit Pauli terms")
print(f"n_0'_gauge has {len(n0_prime_terms_4q)} 4-qubit Pauli terms")

# 4-qubit Pauli term application — inline (clone of agent 1's helper, DIM=16)
DIM4 = 16
ALL4 = np.arange(DIM4, dtype=np.int64)

def popcount4(arr):
    return np.array([bin(int(x)).count('1') for x in arr], dtype=np.int64)

def apply_pauli_term_4q(state, term):
    coef, factors = term
    if not factors:
        return coef * state
    X_mask = Z_mask = 0
    n_y = 0
    for q, ax in factors:
        if ax == 'X':
            X_mask |= 1 << q
        elif ax == 'Z':
            Z_mask |= 1 << q
        elif ax == 'Y':
            X_mask |= 1 << q
            Z_mask |= 1 << q
            n_y += 1
    phase_y = (-1j) ** n_y
    targets = ALL4 ^ X_mask
    z_par = (popcount4(ALL4 & Z_mask) % 2)
    z_phase = (1 - 2 * z_par).astype(np.complex128)
    out = np.zeros(DIM4, dtype=np.complex128)
    out[targets] = coef * phase_y * z_phase * state[ALL4]
    return out

def apply_exp_term_4q(state, term, tau, imag):
    coef, factors = term
    if not factors:
        return state * np.exp(-tau * coef)
    # e^{-tau coef P} = cosh(tau coef) I - sinh(tau coef) P  (imag)
    # or = cos(tau coef) I - i sin(tau coef) P  (real)
    if imag:
        alpha = -tau * coef
        cosh_a = np.cosh(alpha)
        sinh_a = np.sinh(alpha)
        Pstate = apply_pauli_term_4q(state, (1.0+0j, factors))
        return cosh_a * state + sinh_a * Pstate
    else:
        # e^{-i tau coef P}
        alpha = tau * coef.real   # if coef is purely real (which it is for Hermitian H)
        cos_a = np.cos(alpha)
        sin_a = np.sin(alpha)
        Pstate = apply_pauli_term_4q(state, (1.0+0j, factors))
        return cos_a * state - 1j * sin_a * Pstate

def trotter_4q(state, terms, dt, n_steps, imag):
    out = state
    step = dt / n_steps
    for _ in range(n_steps):
        for t in terms:
            out = apply_exp_term_4q(out, t, step, imag)
    return out

# Build 4-qubit propagator G_M3
print(f"Method 3: building 4-qubit propagator (Trotter n={N_TROTTER_IMAG})...")
t0 = time.time()
G_M3 = np.zeros((DIM4, DIM4), dtype=np.complex128)
for i in range(DIM4):
    e_i = np.zeros(DIM4, dtype=np.complex128)
    e_i[i] = 1.0
    psi = trotter_4q(e_i, H_prime_terms_4q, BETA, N_TROTTER_IMAG, imag=True)
    G_M3[:, i] = psi
t_M3_prop = time.time() - t0
Z_M3 = G_M3.diagonal().real.sum()
total_w_M3 = np.abs(G_M3).sum()
print(f"  Built in {t_M3_prop:.1f}s")
print(f"  Z(β) = {Z_M3:.6f}")
print(f"  Z_ratio (M3 / physical-sector ED) = {Z_M3 / Z_phys_ref:.4f}")

# Sample on 4-qubit space
rng = np.random.default_rng(2026)
abs_G3 = np.abs(G_M3)
flat_p3 = (abs_G3 / total_w_M3).ravel()
flat_p3 /= flat_p3.sum()
idx3 = rng.choice(DIM4 * DIM4, size=N_SAMPLES, p=flat_p3)
j_M3 = idx3 // DIM4
i_M3 = idx3 % DIM4
signs_M3 = np.sign(G_M3.real[j_M3, i_M3]).astype(int)
signs_M3[signs_M3 == 0] = 1

# Real-time matrix elements on 4-qubit space (custom Trotter)
def matrix_element_4q(i_bit, j_bit, t, n_trotter):
    """⟨j| U†(t) n_0' U(t) |i⟩ on 4-qubit space."""
    e_i = np.zeros(DIM4, dtype=np.complex128)
    e_i[i_bit] = 1.0
    psi = trotter_4q(e_i, H_prime_terms_4q, t, n_trotter, imag=False)
    # apply n_0'
    out = np.zeros(DIM4, dtype=np.complex128)
    for term in n0_prime_terms_4q:
        out = out + apply_pauli_term_4q(psi, term)
    # evolve back
    out = trotter_4q(out, H_prime_terms_4q, -t, n_trotter, imag=False)
    return out[j_bit]

print(f"Method 3: evaluating 4-qubit matrix elements (Trotter n={N_TROTTER_REAL})...")
t0 = time.time()
contribs_M3 = {t: np.zeros(N_SAMPLES) for t in TIMES}
# observable n_0 in original basis acts only on bit 4. After Z-C and restriction,
# n_0' applied to states |j⟩_gauge gives ⟨j|n_0'|j⟩|j⟩ × something — but in 4-qubit
# representation, n_0' isn't generally diagonal.  We need full ⟨i|U†OU O|j⟩, so:
# C contribution = sign × ⟨i|U†OU|j'⟩ where |j'⟩ = O|j⟩.
# But for non-diagonal O we can't just multiply by Oj scalar.
# Replace approach: compute ⟨i|U†OU O|j⟩ directly via two Trotter evolutions.
def matrix_element_full_4q(i_bit, j_bit, t, n_trotter):
    """Compute ⟨i| U†(t) n_0' U(t) n_0' |j⟩."""
    e_j = np.zeros(DIM4, dtype=np.complex128)
    e_j[j_bit] = 1.0
    # Apply O = n_0'_terms to |j⟩
    Oj = np.zeros(DIM4, dtype=np.complex128)
    for term in n0_prime_terms_4q:
        Oj = Oj + apply_pauli_term_4q(e_j, term)
    # Evolve forward
    psi = trotter_4q(Oj, H_prime_terms_4q, t, n_trotter, imag=False)
    # Apply O again
    psi2 = np.zeros(DIM4, dtype=np.complex128)
    for term in n0_prime_terms_4q:
        psi2 = psi2 + apply_pauli_term_4q(psi, term)
    # Evolve back
    psi3 = trotter_4q(psi2, H_prime_terms_4q, -t, n_trotter, imag=False)
    return psi3[i_bit]

# For C(t) = Σ ρ_{ji} ⟨i|U†OUO|j⟩ / Z
for k in range(N_SAMPLES):
    if k % 50 == 0:
        print(f"  sample {k}/{N_SAMPLES}...", flush=True)
    i, j = int(i_M3[k]), int(j_M3[k])
    for t in TIMES:
        mat_el = matrix_element_full_4q(i, j, t, N_TROTTER_REAL)
        contribs_M3[t][k] = signs_M3[k] * mat_el.real
t_M3_real = time.time() - t0
print(f"  Real-time evaluation: {t_M3_real:.1f}s")

results_M3 = {t: bootstrap(contribs_M3[t], total_w_M3, Z_M3, N_BOOTSTRAP) for t in TIMES}

# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------
print("\n" + "=" * 78)
print("METHOD COMPARISON: C(t) values")
print("=" * 78)
print(f"  {'t':>5} | {'M2 mean ± std':>20} | {'M3 mean ± std':>20} | {'Ref (phys ED)':>15}")
print("  ------+----------------------+----------------------+----------------")
for t in TIMES:
    m2, s2 = results_M2[t]
    m3, s3 = results_M3[t]
    rf = reference_C[t]
    print(f"  {t:>5.2f} | {m2:>9.4f} ± {s2:.4f}     | {m3:>9.4f} ± {s3:.4f}     | {rf:>15.6f}")

print()
print("=" * 78)
print("OTHER METRICS")
print("=" * 78)
print(f"{'Metric':<45} | {'Method 2':>12} | {'Method 3':>12}")
print("-" * 78)
print(f"{'Qubits in simulation':<45} | {'8':>12} | {'4':>12}")
print(f"{'Hilbert-space dim':<45} | {256:>12} | {16:>12}")
print(f"{'H Pauli terms (sampling)':<45} | {len(H_eff_terms_M2):>12} | {len(H_prime_terms_4q):>12}")
print(f"{'Observable Pauli terms':<45} | {len(n0_obs):>12} | {len(n0_prime_terms_4q):>12}")
print(f"{'Propagator-build time (s)':<45} | {t_M2_prop:>12.2f} | {t_M3_prop:>12.2f}")
print(f"{'Real-time eval time (s)':<45} | {t_M2_real:>12.2f} | {t_M3_real:>12.2f}")
print(f"{'Total wall-clock (s)':<45} | {t_M2_prop + t_M2_real:>12.2f} | {t_M3_prop + t_M3_real:>12.2f}")
print(f"{'Z(β,λ) / Z_phys_ref':<45} | {Z_M2 / Z_phys_ref:>12.4f} | {Z_M3 / Z_phys_ref:>12.4f}")
print(f"{'Negative-sign sample fraction':<45} | {np.mean(signs_M2 < 0):>12.4f} | {np.mean(signs_M3 < 0):>12.4f}")
print(f"{'Avg ⟨i|P_phys|i⟩ over samples (M2)':<45} | {phys_fraction_M2_i:>12.4f} | {'n/a':>12}")

print("\nINTERPRETATION:")
print("• Both methods should reproduce the physical-sector reference C(t) within")
print("  combined Trotter + stochastic uncertainty.")
print("• Method 2's Z ratio close to 1 means the penalty effectively isolated the")
print("  physical sector.")
print("• Method 3 uses 4 qubits instead of 8 — at scale this is the dominant win.")
print("• Method 3's Pauli-term count for H' may be larger (Z-C induces extended")
print("  interactions); for our toy the term count is essentially the same.")
