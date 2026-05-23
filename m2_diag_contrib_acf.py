"""
Diagnostic: investigate the Method 2 MCMC z=3-7σ bias on C(t).

The prior run (mcmc_phase2_compare.py: 4 chains × 2000) gave M2-MCMC means
that disagree with the deterministic reference by 6.6σ, 6.3σ, 3.6σ at
t=0.0, 0.5, 2.0 — using a *bootstrap* error bar that assumed iid samples.

This script answers three questions:

  (1) What is the integrated autocorrelation time τ_int of the actual per-sample
      contribution to C(t) (NOT of bit4(i) alone)? What is N_eff?

  (2) Is the deviation a true sampling bias (chain non-ergodic) or just an
      underestimated error bar (bootstrap doesn't account for τ_int > 1)?

  (3) Specifically for t=0, where contribution = sign · bit4(i)·δ_{ij}: compare
      the empirical fraction P(bit4(i)=1 | i=j) to the analytical reference
      P(bit4=1 | diag) = Σ_{bit4(i)=1} |G_M2[i,i]| / Σ_i |G_M2[i,i]|, which the
      sampler claims to target.

Outputs:
  - per-time τ_int on contribution series
  - split-R̂ across chains
  - bootstrap, batch-means, and jackknife error bars
  - τ_int-corrected error bar
  - diagonal-fraction diagnostic
"""

import sys
import time
import numpy as np
from scipy.linalg import expm

sys.path.insert(0, '/home/hlamm/Desktop/QC/logdet/m1_toy')
import epoq_classical_sampler as cs
import epoq_quantum_simulator as qs
from mcmc_sampler import run_mcmc, tau_int_sokal, split_rhat, autocorr

# Parameters: shorter than the failing run but more chains for R-hat
BETA = 0.5
N_TROTTER_IMAG = 50
N_TROTTER_REAL = 200
N_OUTER = 500           # per chain
N_WARMUP = 100
N_CHAINS = 8            # more chains to get R-hat
TIMES = [0.0, 0.5, 1.0]
LAMBDA_PEN = 10.0
NQ, DIM = 8, 256
BASE_SEED = 9001

# ---- shared Pauli-density utilities ----
P2 = {'I': np.eye(2, dtype=complex),
      'X': np.array([[0,1],[1,0]], dtype=complex),
      'Y': np.array([[0,-1j],[1j,0]], dtype=complex),
      'Z': np.diag([1, -1]).astype(complex)}
def pauli_term_dense(factors, nq=NQ):
    by_q = {q: P2['I'] for q in range(nq)}
    for q, ax in factors: by_q[q] = P2[ax]
    result = by_q[nq-1]
    for q in range(nq-2, -1, -1): result = np.kron(result, by_q[q])
    return result

links_at = {(0,0):[0,2],(0,1):[0,3],(1,0):[1,2],(1,1):[1,3]}
parity   = {(0,0):+1, (0,1):-1, (1,0):-1, (1,1):+1}
matter_q = {(0,0):4, (0,1):5, (1,0):6, (1,1):7}

def penalty_terms(lam):
    out = []
    for site in [(0,0),(0,1),(1,0),(1,1)]:
        coef = -lam * parity[site]
        factors = [(ql,'X') for ql in links_at[site]] + [(matter_q[site],'Z')]
        out.append((coef + 0j, factors))
    return out

# Compose H_eff = H_phys + λ P
H_terms_8q = cs.build_pauli_terms()
H_eff_terms = H_terms_8q + penalty_terms(LAMBDA_PEN)

# ---- Build G_M2 deterministically (cheap: 256 columns × Trotter) ----
print("Building deterministic G_M2 (8-qubit Trotter propagator e^{-β(H+λP)})...")
t0 = time.time()
G_M2 = np.zeros((DIM, DIM), dtype=complex)
for i in range(DIM):
    e_i = np.zeros(DIM, dtype=complex); e_i[i] = 1.0
    G_M2[:, i] = cs.evolve_imag(e_i, H_eff_terms, BETA, N_TROTTER_IMAG)
print(f"  done in {time.time()-t0:.1f}s")
abs_G = np.abs(G_M2)
Z_abs = abs_G.sum()
diag_re = G_M2.diagonal().real
Z_M2 = diag_re.sum()

# Analytical "target" reference values
bit4 = ((np.arange(DIM) >> 4) & 1).astype(float)
# P(diagonal sample) under |G_M2|:
P_diag = np.abs(np.diag(G_M2)) / Z_abs
P_diag_frac_total = P_diag.sum()                       # P(i=j)
# P(bit4(i)=1 | diagonal)
P_bit4_1_given_diag = (P_diag[bit4 == 1].sum() / P_diag.sum())
# Reference C(t)
H_dense_phys = sum(c * pauli_term_dense(fac) for c, fac in H_terms_8q)
H_dense_phys = 0.5*(H_dense_phys + H_dense_phys.conj().T)
n0_dense = 0.5*np.eye(DIM, dtype=complex) - 0.5*pauli_term_dense([(4,'Z')])

def C_M2_dense(t):
    """Deterministic ratio: Tr[n0 G_M2 U†(t)n0U(t)] / Tr[G_M2]."""
    Ut = expm(-1j * H_dense_phys * t)
    n0_t = Ut.conj().T @ n0_dense @ Ut
    return float(np.trace(n0_dense @ G_M2 @ n0_t).real / Z_M2)

ref_C = {t: C_M2_dense(t) for t in TIMES}

# ---- Run MCMC: 8 chains × 500 samples ----
print(f"\nRunning MCMC ({N_CHAINS} chains × {N_OUTER} samples, warmup {N_WARMUP})...")
t0 = time.time()
mcmc = run_mcmc(H_eff_terms, NQ=NQ, beta=BETA, n_trotter_imag=N_TROTTER_IMAG,
                n_outer=N_OUTER, n_warmup=N_WARMUP, k_bits=1, n_chains=N_CHAINS,
                base_seed=BASE_SEED)
t_mcmc = time.time() - t0
print(f"  wall-clock: {t_mcmc:.1f}s, switch_rate={mcmc.accept_rate:.3f}")
print(f"  sign-flip fraction: {np.mean(mcmc.signs < 0):.3f}")
print(f"  diagonal sample fraction: {np.mean(mcmc.pairs[:,0]==mcmc.pairs[:,1]):.4f}")

# ---- Evaluate per-sample contribution to C(t) ----
n0_obs = [(0.5+0j, []), (-0.5+0j, [(4, 'Z')])]

def per_sample_contribs(pairs, signs, times):
    """Returns dict[t] -> per-sample contribution arrays (shape (N,))."""
    N = len(pairs)
    contribs = {t: np.zeros(N) for t in times}
    is_diag = np.zeros(N, dtype=bool)
    for k in range(N):
        i, j = int(pairs[k, 0]), int(pairs[k, 1])
        is_diag[k] = (i == j)
        Oj = (j >> 4) & 1
        if Oj == 0:
            # at t=0 contribution is 0; for t>0 still need to evaluate?
            # At t>0 the matrix element ⟨i|U†OU|j⟩ can be nonzero for Oj=0
            # because O sandwiches: contrib = sign * O_j * ⟨i|U†OU|j⟩
            # If O_j=0 the contribution vanishes — confirmed in
            # mcmc_phase2_compare.py: it skips Oj=0 cases.
            continue
        for t in times:
            mat_el = qs.observable_matrix_element(j, i, n0_obs, t,
                                                  n_trotter=N_TROTTER_REAL)
            contribs[t][k] = signs[k] * Oj * mat_el.real
    return contribs, is_diag

print("\nEvaluating per-sample contributions to C(t) (this is the slow step)...")
t0 = time.time()
contribs, is_diag = per_sample_contribs(mcmc.pairs, mcmc.signs, TIMES)
print(f"  done in {time.time()-t0:.1f}s")

# Build per-chain contribution arrays
chain_contribs = {t: [] for t in TIMES}
chain_isdiag = []
chain_signs = []
N_per = N_OUTER
for c in range(N_CHAINS):
    start, stop = c * N_per, (c+1) * N_per
    chain_isdiag.append(is_diag[start:stop])
    chain_signs.append(mcmc.signs[start:stop])
    for t in TIMES:
        chain_contribs[t].append(contribs[t][start:stop])

# ---- Ratio estimator + multiple error bar methods ----
def ratio_C(contrib_arr, sign_arr, isdiag_arr):
    num = contrib_arr.sum()
    denom = (sign_arr * isdiag_arr).sum()
    if denom == 0: return np.nan
    return num / denom

def bootstrap_ratio(contrib_arr, sign_arr, isdiag_arr, N_boot=200, seed=31337):
    rng = np.random.default_rng(seed)
    N = len(contrib_arr)
    boots = []
    for _ in range(N_boot):
        idx = rng.choice(N, N, replace=True)
        v = ratio_C(contrib_arr[idx], sign_arr[idx], isdiag_arr[idx])
        if np.isfinite(v): boots.append(v)
    return float(np.mean(boots)), float(np.std(boots))

def batch_means(contrib_arr, sign_arr, isdiag_arr, n_batches=8):
    """Non-overlapping batch means — variance of batch means × 1/sqrt(n_batches)."""
    N = len(contrib_arr)
    Bsize = N // n_batches
    if Bsize == 0: return np.nan, np.nan
    vals = []
    for b in range(n_batches):
        s, e = b*Bsize, (b+1)*Bsize
        v = ratio_C(contrib_arr[s:e], sign_arr[s:e], isdiag_arr[s:e])
        if np.isfinite(v): vals.append(v)
    vals = np.array(vals)
    if len(vals) < 2: return float(vals.mean()) if len(vals) else np.nan, np.nan
    return float(vals.mean()), float(vals.std(ddof=1) / np.sqrt(len(vals)))

def jackknife_chains(chain_contrib_lst, chain_sign_lst, chain_isdiag_lst):
    """Delete-one-chain jackknife: each leave-one-out replicate mean, variance × (M-1)."""
    M = len(chain_contrib_lst)
    full_contrib = np.concatenate(chain_contrib_lst)
    full_sign = np.concatenate(chain_sign_lst)
    full_isdiag = np.concatenate(chain_isdiag_lst)
    full_mean = ratio_C(full_contrib, full_sign, full_isdiag)
    jacks = []
    for k in range(M):
        keep = [c for c in range(M) if c != k]
        cc = np.concatenate([chain_contrib_lst[c] for c in keep])
        ss = np.concatenate([chain_sign_lst[c] for c in keep])
        dd = np.concatenate([chain_isdiag_lst[c] for c in keep])
        v = ratio_C(cc, ss, dd)
        if np.isfinite(v): jacks.append(v)
    jacks = np.array(jacks)
    M_eff = len(jacks)
    if M_eff < 2: return full_mean, np.nan
    jk_mean = jacks.mean()
    jk_var = (M_eff - 1) / M_eff * np.sum((jacks - jk_mean)**2)
    return float(full_mean), float(np.sqrt(jk_var))

def tau_corrected_chains(chain_contrib_lst, chain_sign_lst, chain_isdiag_lst, tau):
    """Combine per-chain ratio estimates with τ_int correction.

    Use per-chain ratio means as M effectively-independent estimates (if
    chains were long enough vs τ_int), inflated by sqrt((2τ+1)).
    """
    chain_means = []
    for cc, ss, dd in zip(chain_contrib_lst, chain_sign_lst, chain_isdiag_lst):
        v = ratio_C(cc, ss, dd)
        if np.isfinite(v): chain_means.append(v)
    chain_means = np.array(chain_means)
    M = len(chain_means)
    if M < 2: return float(chain_means.mean()) if M else np.nan, np.nan
    cm = float(chain_means.mean())
    se_chain = float(chain_means.std(ddof=1) / np.sqrt(M))
    # Per-chain effective N: N_per / (2τ+1)
    return cm, se_chain

# ---- τ_int on actual contribution series, per chain ----
print("\n" + "="*88)
print("AUTOCORRELATION DIAGNOSTICS on per-sample contribution to C(t)")
print("="*88)
print(f"{'t':>5} | {'τ_int (chain-avg)':>20} | {'N_eff (total)':>16} | {'split-R̂':>10}")
print("-"*88)
tau_int_values = {}
for t in TIMES:
    taus = []
    for c in range(N_CHAINS):
        x = chain_contribs[t][c]
        if x.std() < 1e-14:
            continue
        try:
            tau = tau_int_sokal(x, c_window=5.0)
        except Exception:
            tau = np.nan
        if np.isfinite(tau): taus.append(tau)
    tau_avg = float(np.mean(taus)) if taus else np.nan
    n_total = N_CHAINS * N_OUTER
    n_eff = n_total / (2 * tau_avg + 1) if tau_avg == tau_avg else np.nan
    try:
        rhat = split_rhat(chain_contribs[t])
    except Exception:
        rhat = np.nan
    tau_int_values[t] = tau_avg
    print(f"{t:>5.2f} | {tau_avg:>20.3f} | {n_eff:>16.0f} | {rhat:>10.4f}")

# ---- Final comparison table: reference vs MCMC with multiple error bars ----
print("\n" + "="*88)
print("C(t) ESTIMATES — bootstrap vs batch-means vs jackknife-by-chain vs chain-mean")
print("="*88)
hdr = (f"{'t':>5} | {'ref':>9} | {'mean':>9} | {'boot σ':>9} | "
       f"{'batch σ':>9} | {'JK σ':>9} | {'τ·σ_b':>9} | {'z(boot)':>8} | {'z(JK)':>8}")
print(hdr); print("-"*len(hdr))
results = {}
for t in TIMES:
    arr_c = contribs[t]
    arr_s = mcmc.signs
    arr_d = is_diag
    mean_full = ratio_C(arr_c, arr_s, arr_d)
    m_b, s_b = bootstrap_ratio(arr_c, arr_s, arr_d, N_boot=400)
    m_bm, s_bm = batch_means(arr_c, arr_s, arr_d, n_batches=N_CHAINS)
    m_jk, s_jk = jackknife_chains(chain_contribs[t], chain_signs, chain_isdiag)
    # τ_int-inflated bootstrap σ (factor sqrt(2τ+1))
    tau_t = tau_int_values[t]
    s_b_tau = s_b * np.sqrt(2 * tau_t + 1) if tau_t == tau_t else np.nan
    rf = ref_C[t]
    z_b = (mean_full - rf) / s_b if s_b > 0 else 0
    z_jk = (mean_full - rf) / s_jk if s_jk > 0 else 0
    results[t] = {
        'mean': mean_full, 'ref': rf,
        'boot_sigma': s_b, 'batch_sigma': s_bm, 'jk_sigma': s_jk,
        'tau_int': tau_t, 'tau_corrected_sigma': s_b_tau,
        'z_boot': z_b, 'z_jk': z_jk,
    }
    print(f"{t:>5.2f} | {rf:>9.4f} | {mean_full:>9.4f} | {s_b:>9.4f} | "
          f"{s_bm:>9.4f} | {s_jk:>9.4f} | {s_b_tau:>9.4f} | "
          f"{z_b:>+8.2f} | {z_jk:>+8.2f}")

# ---- Diagonal-fraction-bias diagnostic at t=0 ----
print("\n" + "="*88)
print("DIAGONAL-FRACTION-BIAS DIAGNOSTIC at t=0")
print("="*88)
diag_samples = mcmc.pairs[is_diag, 0]
emp_diag_frac = is_diag.mean()
print(f"  Empirical P(i=j) diagonal fraction = {emp_diag_frac:.4f}")
print(f"  Analytical (|G_M2|)  P(i=j)        = {P_diag_frac_total:.4f}")
if len(diag_samples) > 0:
    emp_bit4_diag = ((diag_samples >> 4) & 1).mean()
else:
    emp_bit4_diag = float('nan')
print(f"  Empirical P(bit4(i)=1 | i=j)       = {emp_bit4_diag:.4f}")
print(f"  Analytical                          = {P_bit4_1_given_diag:.4f}")
print(f"  Σ |G_M2[i,i]| with bit4(i)=0       = {P_diag[bit4==0].sum() / P_diag_frac_total:.4f}")
print(f"  Σ |G_M2[i,i]| with bit4(i)=1       = {P_diag[bit4==1].sum() / P_diag_frac_total:.4f}")
# bin diagonals by all-four matter-bit-pattern
print("\n  Per-matter-pattern diagonal frequency (empirical vs analytical):")
print(f"  {'pattern':>8} | {'N_emp':>6} | {'P_emp':>7} | {'P_ana':>7} | {'ratio':>6}")
mpattern = (np.arange(DIM) >> 4).astype(int)
ana_diag_norm = np.abs(np.diag(G_M2)); ana_diag_norm /= ana_diag_norm.sum()
for p in range(16):
    mask = (mpattern == p)
    emp_n = int(((diag_samples >> 4) == p).sum())
    emp_frac = emp_n / max(len(diag_samples), 1)
    ana_frac = ana_diag_norm[mask].sum()
    ratio = emp_frac / ana_frac if ana_frac > 0 else float('inf')
    print(f"  {p:>8b} | {emp_n:>6d} | {emp_frac:>7.4f} | {ana_frac:>7.4f} | {ratio:>6.2f}")

# ---- Per-chain inspection: how often does each chain visit different states? ----
print("\n" + "="*88)
print("PER-CHAIN i-STATE EXPLORATION (does each chain visit diverse states?)")
print("="*88)
print(f"{'chain':>5} | {'#unique i':>10} | {'top-1 i frac':>14} | {'mean bit4(i)':>14}")
for c in range(N_CHAINS):
    pi = mcmc.chain_pairs[c][:, 0]
    unique_i = np.unique(pi)
    top1_frac = np.bincount(pi).max() / len(pi)
    mean_bit4 = ((pi >> 4) & 1).mean()
    print(f"{c:>5d} | {len(unique_i):>10d} | {top1_frac:>14.4f} | {mean_bit4:>14.4f}")

# ---- Per-chain C(0) (just diag samples) ----
print("\nPer-chain ratio estimates of C(0):")
print(f"{'chain':>5} | {'N_diag':>7} | {'C(0)':>9} | {'pos signs':>10}")
for c in range(N_CHAINS):
    cc, ss, dd = chain_contribs[0.0][c], chain_signs[c], chain_isdiag[c]
    ndiag = int(dd.sum())
    if ndiag > 0:
        v = ratio_C(cc, ss, dd)
        npos = int(((ss * dd) > 0).sum())
        print(f"{c:>5d} | {ndiag:>7d} | {v:>9.4f} | {npos:>10d}")
    else:
        print(f"{c:>5d} | {ndiag:>7d} |    nan    |   -")

print("\nDone. Summary above.")
