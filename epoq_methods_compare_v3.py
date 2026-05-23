"""
Apples-to-apples Method 2 vs Method 3 comparison: BOTH use Trotter for time
evolution (no expm shortcuts), with the same n_trotter step counts.

The previous v2 gave M3 an unfair advantage by using scipy.expm at 4 qubits.
This v3 fixes that by writing efficient 4-qubit Trotter that uses the same
algorithmic primitive as M2's 8-qubit Trotter (per-Pauli-term cosh/sinh
or cos/sin exponentiation).
"""

import sys
import time
import numpy as np
from scipy.linalg import expm

sys.path.insert(0, '/home/hlamm/Desktop/QC/logdet/m1_toy')
import epoq_classical_sampler as cs
import epoq_quantum_simulator as qs

BETA = 0.5
N_TROTTER_IMAG = 50
N_TROTTER_REAL = 200
N_SAMPLES = 1000
N_BOOTSTRAP = 100
TIMES = [0.0, 0.5, 1.0, 2.0]
LAMBDA_PEN = 10.0
NQ, DIM = 8, 256

# ---- helpers (q0=LSB) ----
P2 = {'I':np.eye(2,dtype=complex),'X':np.array([[0,1],[1,0]],dtype=complex),
      'Y':np.array([[0,-1j],[1j,0]],dtype=complex),'Z':np.diag([1,-1]).astype(complex)}
H_pauli_mat = (np.array([[1,1],[1,-1]],dtype=complex)/np.sqrt(2))

def pauli_term_dense(factors, nq):
    by_q = {q: P2['I'] for q in range(nq)}
    for q, ax in factors:
        by_q[q] = P2[ax]
    result = by_q[nq-1]
    for q in range(nq-2, -1, -1):
        result = np.kron(result, by_q[q])
    return result

def single_q_op(op, q, nq):
    r = np.array([[1+0j]])
    for i in range(nq-1, -1, -1):
        r = np.kron(r, op if i == q else P2['I'])
    return r

links_at = {(0,0):[0,2],(0,1):[0,3],(1,0):[1,2],(1,1):[1,3]}
parity = {(0,0):+1,(0,1):-1,(1,0):-1,(1,1):+1}
matter_q = {(0,0):4,(0,1):5,(1,0):6,(1,1):7}

# ---- dense H, n0, U_ZC (for derivation and ED ref only) ----
H_terms_8q = cs.build_pauli_terms()
H_dense = sum(c * pauli_term_dense(fac, NQ) for c, fac in H_terms_8q)
H_dense = (H_dense + H_dense.conj().T) / 2
n0_dense = 0.5*np.eye(DIM,dtype=complex) - 0.5*pauli_term_dense([(4,'Z')], NQ)

def G_dense(site):
    G = np.eye(DIM, dtype=complex)
    for ql in links_at[site]:
        G = G @ pauli_term_dense([(ql,'X')], NQ)
    sign = pauli_term_dense([(matter_q[site],'Z')], NQ)
    if parity[site] == -1: sign = -sign
    return G @ sign

P_phys = np.eye(DIM,dtype=complex)
for s in [(0,0),(0,1),(1,0),(1,1)]:
    P_phys = P_phys @ (np.eye(DIM,dtype=complex) + G_dense(s))/2
P_phys = (P_phys + P_phys.conj().T)/2
e_p, v_p = np.linalg.eigh(P_phys)
phys_basis = v_p[:, e_p > 0.5]
H_phys = phys_basis.conj().T @ H_dense @ phys_basis
H_phys = (H_phys + H_phys.conj().T)/2
n0_phys = phys_basis.conj().T @ n0_dense @ phys_basis
n0_phys = (n0_phys + n0_phys.conj().T)/2
rho_phys = expm(-BETA * H_phys)
Z_phys_ref = np.trace(rho_phys).real

def C_phys_ref(t):
    Ut = expm(-1j * H_phys * t)
    n0_t = Ut.conj().T @ n0_phys @ Ut
    return np.trace(rho_phys @ n0_t @ n0_phys).real / Z_phys_ref

ref_C = {t: C_phys_ref(t) for t in TIMES}

# ---- Build U_ZC, restrict H', n0' to 4-qubit gauge space ----
R_full = np.eye(DIM, dtype=complex)
for ql in range(4):
    R_full = R_full @ single_q_op(H_pauli_mat, ql, NQ)

def cnot_full(c, t):
    P0 = (P2['I'] + P2['Z'])/2; P1 = (P2['I'] - P2['Z'])/2
    op_I = np.array([[1+0j]]); op_X = np.array([[1+0j]])
    for q in range(NQ-1, -1, -1):
        if q == c: op_I = np.kron(op_I, P0); op_X = np.kron(op_X, P1)
        elif q == t: op_I = np.kron(op_I, P2['I']); op_X = np.kron(op_X, P2['X'])
        else: op_I = np.kron(op_I, P2['I']); op_X = np.kron(op_X, P2['I'])
    return op_I + op_X

U_ZC_x = np.eye(DIM, dtype=complex)
for site in [(0,0),(0,1),(1,0),(1,1)]:
    for ql in links_at[site]:
        U_ZC_x = cnot_full(ql, matter_q[site]) @ U_ZC_x
U_offset = np.eye(DIM, dtype=complex)
for site, qm in matter_q.items():
    if parity[site] == -1:
        U_offset = U_offset @ pauli_term_dense([(qm,'X')], NQ)
U_ZC = U_offset @ R_full @ U_ZC_x @ R_full.conj().T

matter0_idx = [i for i in range(DIM) if (i >> 4) == 0]
H_prime_dense = (U_ZC.conj().T @ H_dense @ U_ZC)[np.ix_(matter0_idx, matter0_idx)]
H_prime_dense = (H_prime_dense + H_prime_dense.conj().T)/2
n0_prime_dense = (U_ZC.conj().T @ n0_dense @ U_ZC)[np.ix_(matter0_idx, matter0_idx)]
n0_prime_dense = (n0_prime_dense + n0_prime_dense.conj().T)/2

# Verify spectrum match
eig_diff = np.max(np.abs(np.sort(np.linalg.eigvalsh(H_prime_dense)) - np.sort(np.linalg.eigvalsh(H_phys))))
assert eig_diff < 1e-10, f"Spectrum mismatch: {eig_diff}"

# ---- Pauli-decompose H_prime, n0_prime into 4-qubit Pauli terms ----
NQ4 = 4; DIM4 = 16
ALL4 = np.arange(DIM4, dtype=np.int64)
POP4 = np.array([bin(i).count('1') for i in range(DIM4)], dtype=np.int64)

def pauli_decompose_4q(M):
    terms = []
    for k in range(4 ** NQ4):
        labels = []
        for q in range(NQ4):
            ax_idx = (k // 4**q) % 4
            ax = ['I','X','Y','Z'][ax_idx]
            labels.append((q, ax))
        non_id = [(q, l) for q, l in labels if l != 'I']
        P = pauli_term_dense(non_id, NQ4)
        coef = np.trace(P.conj().T @ M) / DIM4
        if abs(coef) > 1e-10:
            terms.append((complex(coef), non_id))
    return terms

H_prime_terms = pauli_decompose_4q(H_prime_dense)
n0_prime_terms = pauli_decompose_4q(n0_prime_dense)
print(f"H_prime: {len(H_prime_terms)} 4-qubit Pauli terms")
print(f"n0_prime: {len(n0_prime_terms)} 4-qubit Pauli terms")

# ---- Efficient 4-qubit Trotter primitives ----
def apply_pauli_4q(state, term):
    coef, factors = term
    if not factors:
        return coef * state
    X_mask = Z_mask = 0
    n_y = 0
    for q, ax in factors:
        if ax == 'X': X_mask |= 1 << q
        elif ax == 'Z': Z_mask |= 1 << q
        elif ax == 'Y': X_mask |= 1 << q; Z_mask |= 1 << q; n_y += 1
    phase_y = (-1j) ** n_y
    targets = ALL4 ^ X_mask
    z_phase = (1 - 2 * (POP4[ALL4 & Z_mask] % 2)).astype(complex)
    out = np.zeros(DIM4, dtype=complex)
    out[targets] = coef * phase_y * z_phase * state
    return out

def apply_exp_pauli_4q(state, term, tau, imag):
    coef, factors = term
    if not factors:
        return state * (np.exp(-tau * coef.real) if imag else np.exp(-1j * tau * coef.real))
    a = tau * coef.real
    Pstate = apply_pauli_4q(state, (1.0+0j, factors))
    if imag:
        return np.cosh(a) * state - np.sinh(a) * Pstate
    else:
        return np.cos(a) * state - 1j * np.sin(a) * Pstate

def trotter_4q(state, terms, total_time, n_steps, imag):
    out = state
    dt = total_time / n_steps
    for _ in range(n_steps):
        for term in terms:
            out = apply_exp_pauli_4q(out, term, dt, imag)
    return out

# ---- Reference values ----
print("\nReference C(t) (physical-sector ED):")
for t in TIMES:
    print(f"  t={t}: {ref_C[t]:.6f}")

# ============================================================
# Method 2
# ============================================================
def penalty_terms(lam):
    out = []
    for site in [(0,0),(0,1),(1,0),(1,1)]:
        coef = -lam * parity[site]
        factors = [(ql,'X') for ql in links_at[site]] + [(matter_q[site],'Z')]
        out.append((coef + 0j, factors))
    return out

H_eff_terms = H_terms_8q + penalty_terms(LAMBDA_PEN)
print(f"\nMethod 2: {len(H_eff_terms)} Pauli terms; Trotter throughout")
print(f"Building Method 2 propagator (Trotter n={N_TROTTER_IMAG})...")
t0 = time.time()
G_M2 = np.zeros((DIM, DIM), dtype=complex)
for i in range(DIM):
    e_i = np.zeros(DIM, dtype=complex); e_i[i] = 1.0
    G_M2[:, i] = cs.evolve_imag(e_i, H_eff_terms, BETA, N_TROTTER_IMAG)
t_M2_prop = time.time() - t0
Z_M2 = G_M2.diagonal().real.sum()
total_w_M2 = np.abs(G_M2).sum()
print(f"  {t_M2_prop:.1f}s")

rng = np.random.default_rng(2026)
flat_p = (np.abs(G_M2) / total_w_M2).ravel()
flat_p /= flat_p.sum()
idx_M2 = rng.choice(DIM*DIM, size=N_SAMPLES, p=flat_p)
j_M2 = idx_M2 // DIM
i_M2 = idx_M2 % DIM
signs_M2 = np.sign(G_M2.real[j_M2, i_M2]).astype(int); signs_M2[signs_M2==0] = 1
neg_M2 = np.mean(signs_M2 < 0)
print(f"  Negative-sign fraction: {neg_M2:.4f}")

n0_obs_8q = [(0.5+0j, []), (-0.5+0j, [(4,'Z')])]
print(f"Evaluating Method 2 matrix elements (Trotter n={N_TROTTER_REAL})...")
t0 = time.time()
contribs_M2 = {t: np.zeros(N_SAMPLES) for t in TIMES}
for k in range(N_SAMPLES):
    if k % 200 == 0:
        print(f"  sample {k}/{N_SAMPLES}...", flush=True)
    i, j = int(i_M2[k]), int(j_M2[k])
    Oj = (j >> 4) & 1
    if Oj == 0: continue
    for t in TIMES:
        mat_el = qs.observable_matrix_element(j, i, n0_obs_8q, t, n_trotter=N_TROTTER_REAL)
        contribs_M2[t][k] = signs_M2[k] * Oj * mat_el.real
t_M2_real = time.time() - t0
print(f"  Real-time eval: {t_M2_real:.1f}s")

# ============================================================
# Method 3 (Trotter, 4-qubit)
# ============================================================
print(f"\nMethod 3: {len(H_prime_terms)} 4-qubit Pauli terms; Trotter throughout")
print(f"Building Method 3 propagator (Trotter n={N_TROTTER_IMAG})...")
t0 = time.time()
G_M3 = np.zeros((DIM4, DIM4), dtype=complex)
for i in range(DIM4):
    e_i = np.zeros(DIM4, dtype=complex); e_i[i] = 1.0
    G_M3[:, i] = trotter_4q(e_i, H_prime_terms, BETA, N_TROTTER_IMAG, imag=True)
t_M3_prop = time.time() - t0
Z_M3 = G_M3.diagonal().real.sum()
total_w_M3 = np.abs(G_M3).sum()
print(f"  {t_M3_prop:.1f}s")

rng = np.random.default_rng(2026)
flat_p3 = (np.abs(G_M3) / total_w_M3).ravel()
flat_p3 /= flat_p3.sum()
idx_M3 = rng.choice(DIM4*DIM4, size=N_SAMPLES, p=flat_p3)
j_M3 = idx_M3 // DIM4
i_M3 = idx_M3 % DIM4
signs_M3 = np.sign(G_M3.real[j_M3, i_M3]).astype(int); signs_M3[signs_M3==0] = 1
neg_M3 = np.mean(signs_M3 < 0)
print(f"  Negative-sign fraction: {neg_M3:.4f}")

print(f"Evaluating Method 3 matrix elements (Trotter n={N_TROTTER_REAL}, 4-qubit)...")
t0 = time.time()
contribs_M3 = {t: np.zeros(N_SAMPLES) for t in TIMES}
for k in range(N_SAMPLES):
    if k % 200 == 0:
        print(f"  sample {k}/{N_SAMPLES}...", flush=True)
    i, j = int(i_M3[k]), int(j_M3[k])
    for t in TIMES:
        # ⟨i|U†OUO|j⟩ via Trotter:
        # 1. start with O|j⟩
        e_j = np.zeros(DIM4, dtype=complex); e_j[j] = 1.0
        Oj_state = np.zeros(DIM4, dtype=complex)
        for term in n0_prime_terms:
            Oj_state = Oj_state + apply_pauli_4q(e_j, term)
        # 2. evolve forward by t
        psi = trotter_4q(Oj_state, H_prime_terms, t, N_TROTTER_REAL, imag=False)
        # 3. apply O
        psi2 = np.zeros(DIM4, dtype=complex)
        for term in n0_prime_terms:
            psi2 = psi2 + apply_pauli_4q(psi, term)
        # 4. evolve backward by t
        psi3 = trotter_4q(psi2, H_prime_terms, -t, N_TROTTER_REAL, imag=False)
        mat_el = psi3[i]
        contribs_M3[t][k] = signs_M3[k] * mat_el.real
t_M3_real = time.time() - t0
print(f"  Real-time eval: {t_M3_real:.1f}s")

# ---- Bootstrap and report ----
def bootstrap_stats(contribs, total_w, Z, N_boot):
    N = len(contribs)
    rng_b = np.random.default_rng(31337)
    means = []
    for _ in range(N_boot):
        idx = rng_b.choice(N, N, replace=True)
        means.append(contribs[idx].mean() * total_w / Z)
    return np.mean(means), np.std(means)

results_M2 = {t: bootstrap_stats(contribs_M2[t], total_w_M2, Z_M2, N_BOOTSTRAP) for t in TIMES}
results_M3 = {t: bootstrap_stats(contribs_M3[t], total_w_M3, Z_M3, N_BOOTSTRAP) for t in TIMES}

print("\n" + "=" * 90)
print("APPLES-TO-APPLES: BOTH METHODS USE TROTTER WITH SAME n_imag, n_real")
print("=" * 90)
print(f"  {'t':>5} | {'reference':>11} | {'Method 2 (mean ± std)':>26} | {'Method 3 (mean ± std)':>26}")
print("  ------+-------------+----------------------------+--------------------------")
for t in TIMES:
    m2, s2 = results_M2[t]
    m3, s3 = results_M3[t]
    rf = ref_C[t]
    z2 = (m2 - rf) / s2 if s2 > 0 else 0
    z3 = (m3 - rf) / s3 if s3 > 0 else 0
    print(f"  {t:>5.2f} | {rf:>11.6f} | {m2:>10.6f} ± {s2:.4f}  (z={z2:+.2f}) | "
          f"{m3:>10.6f} ± {s3:.4f}  (z={z3:+.2f})")

print("\n" + "=" * 90)
print("METRICS — TROTTER ON BOTH SIDES")
print("=" * 90)
print(f"  {'Metric':<45} | {'Method 2':>18} | {'Method 3':>18}")
print("  " + "-" * 80)
print(f"  {'Simulation qubits':<45} | {'8':>18} | {'4':>18}")
print(f"  {'Hilbert dim':<45} | {DIM:>18} | {DIM4:>18}")
print(f"  {'Pauli terms in H/H_prime':<45} | {len(H_eff_terms):>18} | {len(H_prime_terms):>18}")
print(f"  {'Negative-sign sample fraction':<45} | {neg_M2:>18.4f} | {neg_M3:>18.4f}")
print(f"  {'Propagator-build time (s)':<45} | {t_M2_prop:>18.2f} | {t_M3_prop:>18.4f}")
print(f"  {'Real-time eval time (s) [N=' + str(N_SAMPLES) + ']':<45} | {t_M2_real:>18.2f} | {t_M3_real:>18.2f}")
print(f"  {'Total wall-clock (s)':<45} | {t_M2_prop+t_M2_real:>18.2f} | {t_M3_prop+t_M3_real:>18.2f}")
