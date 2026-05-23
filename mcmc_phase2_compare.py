"""
Phase 2 of MCMC integration: run heat-bath MCMC on Method 2 (H + penalty,
8 qubits) and Method 3 (Z-C-reduced H', 4 qubits), compute C(t), validate
against enumeration baseline and physical-sector ED reference.

Ratio estimator for C(t) (avoiding the Z_abs unknown):
    C(t) = (Σ_k sign_k · O_{j_k} · ⟨i_k|U†OU|j_k⟩) / N_diagonal
where N_diagonal = #{k : i_k = j_k}.
"""
import sys
import time
import numpy as np
from scipy.linalg import expm

sys.path.insert(0, '/home/hlamm/Desktop/QC/logdet/m1_toy')
import epoq_classical_sampler as cs
from mcmc_sampler import run_mcmc, tau_int_sokal, split_rhat, make_pauli_machinery, apply_pauli_term, evolve_state

# Common params
BETA = 0.5
N_TROTTER_IMAG = 50
N_TROTTER_REAL = 200
N_OUTER = 2000    # per chain (each outer is NQ Trotter runs) — 4× previous statistics
N_WARMUP = 200    # also bumped to retain mixing
N_CHAINS = 4
TIMES = [0.0, 0.5, 1.0, 2.0]
LAMBDA_PEN = 10.0

# ---- helpers ----
P2 = {'I':np.eye(2,dtype=complex),'X':np.array([[0,1],[1,0]],dtype=complex),
      'Y':np.array([[0,-1j],[1j,0]],dtype=complex),'Z':np.diag([1,-1]).astype(complex)}
def pauli_term_dense(factors, nq):
    by_q = {q: P2['I'] for q in range(nq)}
    for q, ax in factors: by_q[q] = P2[ax]
    result = by_q[nq-1]
    for q in range(nq-2,-1,-1): result = np.kron(result, by_q[q])
    return result
def single_q_op(op, q, nq):
    r = np.array([[1+0j]])
    for i in range(nq-1, -1, -1):
        r = np.kron(r, op if i==q else P2['I'])
    return r

links_at = {(0,0):[0,2],(0,1):[0,3],(1,0):[1,2],(1,1):[1,3]}
parity = {(0,0):+1,(0,1):-1,(1,0):-1,(1,1):+1}
matter_q = {(0,0):4,(0,1):5,(1,0):6,(1,1):7}
NQ, DIM = 8, 256
NQ4, DIM4 = 4, 16
H_pauli = np.array([[1,1],[1,-1]], dtype=complex)/np.sqrt(2)

# ---- reference: physical-sector ED ----
H_terms_8q = cs.build_pauli_terms()
H_dense = sum(c * pauli_term_dense(fac, NQ) for c, fac in H_terms_8q)
H_dense = (H_dense + H_dense.conj().T)/2
n0_dense = 0.5*np.eye(DIM,dtype=complex) - 0.5*pauli_term_dense([(4,'Z')], NQ)

def G_dense(site):
    G = np.eye(DIM,dtype=complex)
    for ql in links_at[site]: G = G @ pauli_term_dense([(ql,'X')], NQ)
    sign = pauli_term_dense([(matter_q[site],'Z')], NQ)
    if parity[site]==-1: sign = -sign
    return G @ sign

P_phys = np.eye(DIM,dtype=complex)
for s in [(0,0),(0,1),(1,0),(1,1)]:
    P_phys = P_phys @ (np.eye(DIM,dtype=complex) + G_dense(s))/2
P_phys = (P_phys + P_phys.conj().T)/2
e_p, v_p = np.linalg.eigh(P_phys); phys_basis = v_p[:, e_p > 0.5]
H_phys = phys_basis.conj().T @ H_dense @ phys_basis; H_phys = (H_phys + H_phys.conj().T)/2
n0_phys = phys_basis.conj().T @ n0_dense @ phys_basis; n0_phys = (n0_phys + n0_phys.conj().T)/2
rho_phys = expm(-BETA * H_phys); Z_phys = np.trace(rho_phys).real
def C_phys_ref(t):
    Ut = expm(-1j * H_phys * t)
    n0_t = Ut.conj().T @ n0_phys @ Ut
    return np.trace(rho_phys @ n0_t @ n0_phys).real / Z_phys
ref_C = {t: C_phys_ref(t) for t in TIMES}

# ---- Method 2 setup: H + penalty (8 qubits) ----
def penalty_terms(lam):
    out = []
    for site in [(0,0),(0,1),(1,0),(1,1)]:
        coef = -lam * parity[site]
        factors = [(ql,'X') for ql in links_at[site]] + [(matter_q[site],'Z')]
        out.append((coef + 0j, factors))
    return out
H_eff_terms = H_terms_8q + penalty_terms(LAMBDA_PEN)

# ---- Method 3 setup: Z-C-reduced H' (4 qubits) ----
R_full = np.eye(DIM, dtype=complex)
for ql in range(4): R_full = R_full @ single_q_op(H_pauli, ql, NQ)
def cnot_full(c, t):
    P0 = (P2['I']+P2['Z'])/2; P1 = (P2['I']-P2['Z'])/2
    op_I = np.array([[1+0j]]); op_X = np.array([[1+0j]])
    for q in range(NQ-1, -1, -1):
        if q==c: op_I = np.kron(op_I, P0); op_X = np.kron(op_X, P1)
        elif q==t: op_I = np.kron(op_I, P2['I']); op_X = np.kron(op_X, P2['X'])
        else: op_I = np.kron(op_I, P2['I']); op_X = np.kron(op_X, P2['I'])
    return op_I + op_X
U_ZC_x = np.eye(DIM, dtype=complex)
for site in [(0,0),(0,1),(1,0),(1,1)]:
    for ql in links_at[site]: U_ZC_x = cnot_full(ql, matter_q[site]) @ U_ZC_x
U_offset = np.eye(DIM, dtype=complex)
for site, qm in matter_q.items():
    if parity[site]==-1: U_offset = U_offset @ pauli_term_dense([(qm,'X')], NQ)
U_ZC = U_offset @ R_full @ U_ZC_x @ R_full.conj().T

matter0_idx = [i for i in range(DIM) if (i>>4)==0]
H_prime_dense = (U_ZC.conj().T @ H_dense @ U_ZC)[np.ix_(matter0_idx, matter0_idx)]
H_prime_dense = (H_prime_dense + H_prime_dense.conj().T)/2
n0_prime_dense = (U_ZC.conj().T @ n0_dense @ U_ZC)[np.ix_(matter0_idx, matter0_idx)]
n0_prime_dense = (n0_prime_dense + n0_prime_dense.conj().T)/2

def pauli_decompose_4q(M):
    terms = []
    for k in range(4 ** NQ4):
        labels = [(q, ['I','X','Y','Z'][(k // 4**q) % 4]) for q in range(NQ4)]
        non_id = [(q, l) for q, l in labels if l != 'I']
        P = pauli_term_dense(non_id, NQ4)
        coef = np.trace(P.conj().T @ M) / DIM4
        if abs(coef) > 1e-10: terms.append((complex(coef), non_id))
    return terms
H_prime_terms = pauli_decompose_4q(H_prime_dense)
n0_prime_terms = pauli_decompose_4q(n0_prime_dense)

# ---- Run MCMC for both methods ----
print(f"Phase 2 — MCMC heat-bath on both methods")
print(f"β={BETA}, N_chains={N_CHAINS}, N_outer={N_OUTER}, N_warmup={N_WARMUP}")
print()

print("Running Method 2 MCMC (H + penalty, 8 qubits)...")
t0 = time.time()
mcmc_M2 = run_mcmc(H_eff_terms, NQ=NQ, beta=BETA, n_trotter_imag=N_TROTTER_IMAG,
                   n_outer=N_OUTER, n_warmup=N_WARMUP, k_bits=1, n_chains=N_CHAINS,
                   base_seed=2026)
t_M2_mcmc = time.time() - t0
print(f"  wall-clock: {t_M2_mcmc:.1f}s, switch_rate={mcmc_M2.accept_rate:.3f}")

print("Running Method 3 MCMC (Z-C H', 4 qubits)...")
t0 = time.time()
mcmc_M3 = run_mcmc(H_prime_terms, NQ=NQ4, beta=BETA, n_trotter_imag=N_TROTTER_IMAG,
                   n_outer=N_OUTER, n_warmup=N_WARMUP, k_bits=1, n_chains=N_CHAINS,
                   base_seed=2026)
t_M3_mcmc = time.time() - t0
print(f"  wall-clock: {t_M3_mcmc:.1f}s, switch_rate={mcmc_M3.accept_rate:.3f}")

# ---- Compute C(t) via real-time evolution + ratio estimator ----
def compute_C_t_mcmc_M2(pairs, signs, times):
    """For Method 2 (8 qubits), use qs.observable_matrix_element with Trotter."""
    import epoq_quantum_simulator as qs
    n0_obs = [(0.5+0j, []), (-0.5+0j, [(4,'Z')])]
    N = len(pairs)
    contribs = {t: np.zeros(N) for t in times}
    is_diag = np.zeros(N, dtype=bool)
    for k in range(N):
        i, j = int(pairs[k,0]), int(pairs[k,1])
        is_diag[k] = (i == j)
        Oj = (j >> 4) & 1
        if Oj == 0: continue
        for t in times:
            mat_el = qs.observable_matrix_element(j, i, n0_obs, t, n_trotter=N_TROTTER_REAL)
            contribs[t][k] = signs[k] * Oj * mat_el.real
    return contribs, is_diag

def compute_C_t_mcmc_M3(pairs, signs, times):
    """For Method 3 (4 qubits), use the 4q Trotter primitives in mcmc_sampler."""
    DIM4_, ALL4, POP4 = make_pauli_machinery(NQ4)
    N = len(pairs)
    contribs = {t: np.zeros(N) for t in times}
    is_diag = np.zeros(N, dtype=bool)
    # Precompute U(t) action via Trotter — we'll do per-sample evolution
    for k in range(N):
        i, j = int(pairs[k,0]), int(pairs[k,1])
        is_diag[k] = (i == j)
        for t in times:
            # ⟨i|U†OU O|j⟩ via Trotter (forward + backward + observable application)
            e_j = np.zeros(DIM4_, dtype=complex); e_j[j] = 1.0
            Oj_state = np.zeros(DIM4_, dtype=complex)
            for term in n0_prime_terms:
                Oj_state += apply_pauli_term(e_j, term, ALL4, POP4)
            psi = evolve_state(Oj_state, H_prime_terms, t, N_TROTTER_REAL, imag=False,
                              ALL=ALL4, POP=POP4)
            psi2 = np.zeros(DIM4_, dtype=complex)
            for term in n0_prime_terms:
                psi2 += apply_pauli_term(psi, term, ALL4, POP4)
            psi3 = evolve_state(psi2, H_prime_terms, -t, N_TROTTER_REAL, imag=False,
                               ALL=ALL4, POP=POP4)
            contribs[t][k] = signs[k] * psi3[i].real
    return contribs, is_diag

print("\nEvaluating matrix elements for Method 2...")
t0 = time.time()
contribs_M2, is_diag_M2 = compute_C_t_mcmc_M2(mcmc_M2.pairs, mcmc_M2.signs, TIMES)
t_M2_obs = time.time() - t0
print(f"  wall-clock: {t_M2_obs:.1f}s")

print("Evaluating matrix elements for Method 3...")
t0 = time.time()
contribs_M3, is_diag_M3 = compute_C_t_mcmc_M3(mcmc_M3.pairs, mcmc_M3.signs, TIMES)
t_M3_obs = time.time() - t0
print(f"  wall-clock: {t_M3_obs:.1f}s")

# ---- Ratio estimator with bootstrap ----
def ratio_estimator_bootstrap(contribs, signs, is_diag, N_boot=100):
    N = len(contribs)
    rng_b = np.random.default_rng(31337)
    boot_C = []
    for _ in range(N_boot):
        idx = rng_b.choice(N, N, replace=True)
        num = contribs[idx].sum()
        denom = (signs[idx] * is_diag[idx]).sum()
        if abs(denom) < 1e-12:
            continue
        boot_C.append(num / denom)
    return np.mean(boot_C), np.std(boot_C)

print("\n" + "=" * 88)
print("RESULTS — MCMC ratio estimator C(t)")
print("=" * 88)
print(f"  {'t':>5} | {'reference':>11} | {'Method 2 MCMC (8q)':>26} | {'Method 3 MCMC (4q)':>26}")
print("  ------+-------------+----------------------------+--------------------------")
for t in TIMES:
    m2, s2 = ratio_estimator_bootstrap(contribs_M2[t], mcmc_M2.signs, is_diag_M2)
    m3, s3 = ratio_estimator_bootstrap(contribs_M3[t], mcmc_M3.signs, is_diag_M3)
    rf = ref_C[t]
    z2 = (m2 - rf) / s2 if s2 > 0 else 0
    z3 = (m3 - rf) / s3 if s3 > 0 else 0
    print(f"  {t:>5.2f} | {rf:>11.6f} | {m2:>10.6f} ± {s2:.4f}  (z={z2:+.2f}) | "
          f"{m3:>10.6f} ± {s3:.4f}  (z={z3:+.2f})")

print("\n" + "=" * 88)
print("METRICS")
print("=" * 88)
print(f"  {'Metric':<40} | {'Method 2':>20} | {'Method 3':>20}")
print(f"  {'Simulation qubits':<40} | {'8':>20} | {'4':>20}")
print(f"  {'Pauli terms':<40} | {len(H_eff_terms):>20} | {len(H_prime_terms):>20}")
print(f"  {'MCMC chains':<40} | {N_CHAINS:>20} | {N_CHAINS:>20}")
print(f"  {'N samples per chain':<40} | {N_OUTER:>20} | {N_OUTER:>20}")
print(f"  {'MCMC wall-clock (s)':<40} | {t_M2_mcmc:>20.1f} | {t_M3_mcmc:>20.1f}")
print(f"  {'Switch rate':<40} | {mcmc_M2.accept_rate:>20.3f} | {mcmc_M3.accept_rate:>20.3f}")
print(f"  {'Negative-sign sample fraction':<40} | {np.mean(mcmc_M2.signs < 0):>20.3f} | {np.mean(mcmc_M3.signs < 0):>20.3f}")
print(f"  {'Diagonal sample fraction':<40} | {np.mean(is_diag_M2):>20.3f} | {np.mean(is_diag_M3):>20.3f}")
print(f"  {'Matrix-element eval wall-clock (s)':<40} | {t_M2_obs:>20.1f} | {t_M3_obs:>20.1f}")
print(f"  {'Total wall-clock (s)':<40} | {t_M2_mcmc+t_M2_obs:>20.1f} | {t_M3_mcmc+t_M3_obs:>20.1f}")
