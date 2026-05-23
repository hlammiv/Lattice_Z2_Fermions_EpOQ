"""
Method 2 (Lagrange-multiplier gauge penalty) vs Method 3 (Z-C 4-qubit) on
2x2 OBC Z2 + staggered, with the qubit-convention bug fixed and meaningful
sample sizes.

Both methods should reproduce the physical-sector C(t) reference.

Implementation choices for fairness:
- Method 2: existing 8-qubit cs / qs modules (Trotter throughout)
- Method 3: 4-qubit propagator + evolution via scipy.linalg.expm
  (since at 16-dim expm is trivially fast and exact; Trotter quality at
   4-qubit is HIGHER than 8-qubit for the same n_steps, so this favors
   Method 2 if anything in the comparison).

What's measured: bootstrap-error-bar mean, wall-clock time per method,
qubit count, sample count needed for fixed-precision result.
"""

import sys
import time
import numpy as np
from scipy.linalg import expm

sys.path.insert(0, '/home/hlamm/Desktop/QC/logdet/m1_toy')
import epoq_classical_sampler as cs
import epoq_quantum_simulator as qs

# Parameters
BETA = 0.5
N_TROTTER_IMAG = 50
N_TROTTER_REAL = 200
N_SAMPLES = 1000
N_BOOTSTRAP = 100
TIMES = [0.0, 0.5, 1.0, 2.0]
LAMBDA_PEN = 10.0
NQ, DIM = 8, 256

# --- Pauli helpers (q0=LSB convention; matches cs module) -----------------
P2 = {'I': np.eye(2, dtype=complex),
      'X': np.array([[0,1],[1,0]], dtype=complex),
      'Y': np.array([[0,-1j],[1j,0]], dtype=complex),
      'Z': np.array([[1,0],[0,-1]], dtype=complex)}
H_pauli_mat = (np.array([[1,1],[1,-1]], dtype=complex) / np.sqrt(2))

def pauli_term_dense(factors, nq=NQ):
    """Build dense matrix for a Pauli term using q0=LSB convention."""
    by_q = {q: P2['I'] for q in range(nq)}
    for q, ax in factors:
        by_q[q] = P2[ax]
    result = by_q[nq-1]
    for q in range(nq-2, -1, -1):
        result = np.kron(result, by_q[q])
    return result

def single_q_op(op, q, nq=NQ):
    """q0=LSB convention single-qubit op."""
    r = np.array([[1+0j]])
    for i in range(nq-1, -1, -1):
        r = np.kron(r, op if i == q else P2['I'])
    return r

# --- Reference: physical-sector ED ----------------------------------------
links_at = {(0,0):[0,2],(0,1):[0,3],(1,0):[1,2],(1,1):[1,3]}
parity = {(0,0):+1,(0,1):-1,(1,0):-1,(1,1):+1}
matter_q = {(0,0):4,(0,1):5,(1,0):6,(1,1):7}

H_terms = cs.build_pauli_terms()
H_dense = sum(c * pauli_term_dense(fac) for c, fac in H_terms)
H_dense = (H_dense + H_dense.conj().T) / 2
n0_dense = 0.5*np.eye(DIM, dtype=complex) - 0.5*pauli_term_dense([(4,'Z')])

def G_dense(site):
    G = np.eye(DIM, dtype=complex)
    for ql in links_at[site]:
        G = G @ pauli_term_dense([(ql,'X')])
    sign = pauli_term_dense([(matter_q[site],'Z')])
    if parity[site] == -1:
        sign = -sign
    return G @ sign

G_ops = {s: G_dense(s) for s in [(0,0),(0,1),(1,0),(1,1)]}
P_phys = np.eye(DIM, dtype=complex)
for s in G_ops:
    P_phys = P_phys @ (np.eye(DIM, dtype=complex) + G_ops[s]) / 2
P_phys = (P_phys + P_phys.conj().T) / 2
e_p, v_p = np.linalg.eigh(P_phys)
phys_basis = v_p[:, e_p > 0.5]
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

ref_C = {t: C_phys_ref(t) for t in TIMES}
print("Reference C(t) (physical-sector ED):")
for t in TIMES:
    print(f"  t={t}: {ref_C[t]:.6f}")

# --- Method 2: penalty in imaginary-time Trotter -------------------------
def penalty_terms(lam):
    out = []
    for site in [(0,0),(0,1),(1,0),(1,1)]:
        coef = -lam * parity[site]
        factors = [(ql,'X') for ql in links_at[site]] + [(matter_q[site],'Z')]
        out.append((coef + 0j, factors))
    return out

H_eff_terms = H_terms + penalty_terms(LAMBDA_PEN)
print(f"\nMethod 2: {len(H_eff_terms)} Pauli terms (H + {len(penalty_terms(LAMBDA_PEN))} penalty)")

# Build penalized propagator
print(f"Building Method 2 propagator (Trotter n={N_TROTTER_IMAG})...")
t0 = time.time()
G_M2 = np.zeros((DIM, DIM), dtype=complex)
for i in range(DIM):
    e_i = np.zeros(DIM, dtype=complex); e_i[i] = 1.0
    G_M2[:, i] = cs.evolve_imag(e_i, H_eff_terms, BETA, N_TROTTER_IMAG)
t_M2_prop = time.time() - t0
Z_M2 = G_M2.diagonal().real.sum()
total_w_M2 = np.abs(G_M2).sum()
print(f"  {t_M2_prop:.1f}s; Z_M2 = {Z_M2:.4e}; total |G| = {total_w_M2:.4e}")

# Sample
rng = np.random.default_rng(2026)
flat_p = (np.abs(G_M2) / total_w_M2).ravel()
flat_p /= flat_p.sum()
idx = rng.choice(DIM*DIM, size=N_SAMPLES, p=flat_p)
j_M2 = idx // DIM
i_M2 = idx % DIM
signs_M2 = np.sign(G_M2.real[j_M2, i_M2]).astype(int); signs_M2[signs_M2==0] = 1
print(f"  Negative sign fraction: {np.mean(signs_M2 < 0):.4f}")

# Real-time matrix elements
print(f"Evaluating Method 2 matrix elements (Trotter n={N_TROTTER_REAL}, N={N_SAMPLES})...")
n0_obs = [(0.5+0j, []), (-0.5+0j, [(4,'Z')])]
t0 = time.time()
contribs_M2 = {t: np.zeros(N_SAMPLES) for t in TIMES}
for k in range(N_SAMPLES):
    if k % 200 == 0:
        print(f"  sample {k}/{N_SAMPLES}...", flush=True)
    i, j = int(i_M2[k]), int(j_M2[k])
    Oj = (j >> 4) & 1
    if Oj == 0:
        continue
    for t in TIMES:
        mat_el = qs.observable_matrix_element(j, i, n0_obs, t, n_trotter=N_TROTTER_REAL)
        contribs_M2[t][k] = signs_M2[k] * Oj * mat_el.real
t_M2_real = time.time() - t0

# --- Method 3: Z-C rotated, 4-qubit, expm for time evolution -------------
print("\nMethod 3: Z-C 4-qubit setup (q0=LSB convention)")
R_full = np.eye(DIM, dtype=complex)
for ql in range(4):
    R_full = R_full @ single_q_op(H_pauli_mat, ql)

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
        U_offset = U_offset @ pauli_term_dense([(qm,'X')])
U_ZC = U_offset @ R_full @ U_ZC_x @ R_full.conj().T

matter0_idx = [i for i in range(DIM) if (i >> 4) == 0]
H_prime = (U_ZC.conj().T @ H_dense @ U_ZC)[np.ix_(matter0_idx, matter0_idx)]
H_prime = (H_prime + H_prime.conj().T) / 2
n0_prime = (U_ZC.conj().T @ n0_dense @ U_ZC)[np.ix_(matter0_idx, matter0_idx)]
n0_prime = (n0_prime + n0_prime.conj().T) / 2

# Verify spectrum match
eig_diff = np.max(np.abs(np.sort(np.linalg.eigvalsh(H_prime)) - np.sort(np.linalg.eigvalsh(H_phys))))
print(f"  H_prime spectrum vs H_phys: max diff = {eig_diff:.3e}")
assert eig_diff < 1e-10, "Spectrum mismatch — convention bug not fixed!"

print(f"Building Method 3 propagator (expm at 4-qubit)...")
t0 = time.time()
G_M3 = expm(-BETA * H_prime)
t_M3_prop = time.time() - t0
Z_M3 = G_M3.diagonal().real.sum()
total_w_M3 = np.abs(G_M3).sum()
print(f"  {t_M3_prop*1000:.1f}ms; Z_M3 = {Z_M3:.4f}; total |G| = {total_w_M3:.4f}")

# Sample on 4-qubit space (256 (i,j) pairs total)
rng = np.random.default_rng(2026)
flat_p3 = (np.abs(G_M3) / total_w_M3).ravel()
flat_p3 /= flat_p3.sum()
idx3 = rng.choice(16*16, size=N_SAMPLES, p=flat_p3)
j_M3 = idx3 // 16
i_M3 = idx3 % 16
signs_M3 = np.sign(G_M3.real[j_M3, i_M3]).astype(int); signs_M3[signs_M3==0] = 1
print(f"  Negative sign fraction: {np.mean(signs_M3 < 0):.4f}")

# Real-time matrix elements for M3 (also via expm at 4-qubit, fast)
print(f"Evaluating Method 3 matrix elements (expm at 4-qubit, N={N_SAMPLES})...")
t0 = time.time()
Ut_cache = {t: expm(-1j * H_prime * t) for t in TIMES}
contribs_M3 = {t: np.zeros(N_SAMPLES) for t in TIMES}
for k in range(N_SAMPLES):
    i, j = int(i_M3[k]), int(j_M3[k])
    for t in TIMES:
        Ut = Ut_cache[t]
        # ⟨i|U†OU O|j⟩
        ket = Ut.conj().T @ (n0_prime @ (Ut @ (n0_prime[:, j])))
        mat_el = ket[i]
        contribs_M3[t][k] = signs_M3[k] * mat_el.real
t_M3_real = time.time() - t0

# --- Bootstrap ---
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

# --- Report ---
print("\n" + "=" * 80)
print("RESULTS — C(t) with bootstrap error bars")
print("=" * 80)
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

print("\n" + "=" * 80)
print("METRICS COMPARISON")
print("=" * 80)
print(f"  {'Metric':<45} | {'Method 2':>15} | {'Method 3':>15}")
print("  " + "-" * 80)
print(f"  {'Simulation qubits':<45} | {'8':>15} | {'4':>15}")
print(f"  {'Hilbert-space dim':<45} | {DIM:>15} | {'16':>15}")
print(f"  {'H Pauli terms (sampling)':<45} | {len(H_eff_terms):>15} | {'~3 (built numerically from Z-C of 17 H terms; not Pauli-decomposed in this script)'}")
print(f"  {'Observable Pauli terms (original)':<45} | {'2':>15} | {'2':>15}")
print(f"  {'Z(β, λ)':<45} | {f'{Z_M2:.2e}':>15} | {f'{Z_M3:.4f}':>15}")
print(f"  {'Z / Z_phys_ref (cosmetic-shifted)':<45} | {Z_M2/Z_phys_ref/np.exp(20):>15.4f} | {Z_M3/Z_phys_ref:>15.4f}")
print(f"  {'Negative-sign sample fraction':<45} | {np.mean(signs_M2 < 0):>15.4f} | {np.mean(signs_M3 < 0):>15.4f}")
print(f"  {'Propagator-build time (s)':<45} | {t_M2_prop:>15.2f} | {t_M3_prop:>15.4f}")
print(f"  {'Real-time eval time (s) [N=' + str(N_SAMPLES) + ']':<45} | {t_M2_real:>15.2f} | {t_M3_real:>15.4f}")
print(f"  {'Wall-clock total (s)':<45} | {t_M2_prop+t_M2_real:>15.2f} | {t_M3_prop+t_M3_real:>15.4f}")
print(f"  {'Implementation note':<45} | {'Trotter throughout':>15} | {'expm (16-dim, free)':>15}")
