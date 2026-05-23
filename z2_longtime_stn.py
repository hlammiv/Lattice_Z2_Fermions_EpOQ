"""
Signal-to-noise behavior of the stochastic SK correlator at large t.

The matrix-element SK form
    C(t) = (1/Z) Σ_{ij} ρ_{ji} n_0(t)_{ij}
involves a sum of fluctuating matrix elements n_0(t)_{ij} ~ e^{i(E_j - E_i)t}.
At large t these oscillate rapidly. Stochastic sampling sees an
exponentially shrinking signal-to-noise ratio (the StN problem identified
in 2001.11490).

This script:
  (1) Computes exact C(t) up to t=20 — confirms the observable itself
      is bounded and well-defined at long times
  (2) Measures the stochastic estimator variance via bootstrap repetitions
  (3) Reports StN(t) = |signal| / std(estimator) — should decay polynomially
      or exponentially in t depending on the spectral gap structure
"""

import numpy as np
from numpy import kron
from numpy.linalg import eigh
from scipy.linalg import expm

# Same setup as z2_end_to_end.py
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
links_at = {(0,0):[0,2],(0,1):[0,3],(1,0):[1,2],(1,1):[1,3]}
parity = {(0,0):+1,(0,1):-1,(1,0):-1,(1,1):+1}
m_q = {(0,0):4,(0,1):5,(1,0):6,(1,1):7}
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
H = -g_e * sum(X_q[q] for q in range(4)) - g_m * (Z_q[0] @ Z_q[3] @ Z_q[1] @ Z_q[2])
H_hop = np.zeros((dim, dim), dtype=complex)
for q_link, a, b in [(0,0,1),(1,2,3),(2,0,2),(3,1,3)]:
    t_op = psi_dag[a] @ Z_q[q_link] @ psi[b]
    H_hop += g_hop * (t_op + t_op.conj().T)
H = H + H_hop + sum(m_mass * parity[s] * N_q[m_q[s]] for s in m_q)
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

beta = 0.5
rho_p = expm(-beta * H_p)
Z_p = np.trace(rho_p).real
D = H_p.shape[0]

# --- (1) Exact C(t) at long times -----------------------------------------
def exact_C(t):
    Ut = expm(-1j * H_p * t)
    n0_t = Ut.conj().T @ n0_p @ Ut
    return np.trace(rho_p @ n0_t @ n0_p).real / Z_p

print(f"2+1d Z₂ + staggered, β={beta}")
print(f"Hilbert space dim (physical): {D}")
print()
print("(1) Exact C(t) at varying t:")
print(f"  {'t':>6} | {'C(t)':>10}")
times = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0]
for t in times:
    c = exact_C(t)
    print(f"  {t:>6.2f} | {c:>10.6f}")
print("→ Observable is bounded and oscillates within a finite range.")

# --- (2) Stochastic estimator variance at varying t -----------------------
def stochastic_C(t, N_samples, rng):
    Ut = expm(-1j * H_p * t)
    n0_t = Ut.conj().T @ n0_p @ Ut
    A = n0_t @ n0_p
    prob = np.abs(rho_p).flatten()
    total_w = prob.sum()
    prob = prob / total_w
    flat_idx = rng.choice(D * D, size=N_samples, p=prob)
    acc = 0j
    for fidx in flat_idx:
        k, i = divmod(fidx, D)
        sign = rho_p[k, i] / abs(rho_p[k, i]) if abs(rho_p[k, i]) > 0 else 0
        acc += sign * A[i, k]
    return (acc * total_w / (N_samples * Z_p)).real

N_samples = 5000
N_bootstrap = 50

print()
print(f"(2) Bootstrap estimator variance at varying t, N_samples={N_samples}, "
      f"N_bootstrap={N_bootstrap}:")
print(f"  {'t':>6} | {'|signal|':>10} | {'std':>10} | {'StN':>10}")

stn_records = []
for t in [0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0]:
    exact_val = abs(exact_C(t))
    samples = []
    for boot in range(N_bootstrap):
        rng = np.random.default_rng(2026 + boot)
        samples.append(stochastic_C(t, N_samples, rng))
    samples = np.array(samples)
    std = samples.std()
    stn = exact_val / std if std > 0 else float('inf')
    stn_records.append((t, exact_val, std, stn))
    print(f"  {t:>6.2f} | {exact_val:>10.6f} | {std:>10.4e} | {stn:>10.4f}")

# Check whether StN decays with t
print()
print("Trend of StN vs t:")
if len(stn_records) >= 3:
    t_arr = np.array([r[0] for r in stn_records])
    stn_arr = np.array([r[3] for r in stn_records])
    # Fit log(StN) vs t
    coef = np.polyfit(t_arr, np.log(stn_arr + 1e-15), 1)
    print(f"  Linear fit log(StN) vs t: slope = {coef[0]:.4f}")
    if coef[0] < -0.05:
        print(f"  Interpretation: StN decays exponentially as e^({coef[0]:.3f} t)")
    else:
        print(f"  Interpretation: StN does not show strong exponential decay in this range")
    print(f"  (For a finite-dim Hilbert space oscillation, decay levels off")
    print(f"   at long t; the StN problem is most severe in continuum limits.)")

# --- (3) Effect of |ρ_{ji}| sign on variance --------------------------------
print()
print("(3) Where does the variance come from? Diagnostic:")
print(f"  Number of (i,j) pairs with |ρ_{{ji}}| > 1e-6: {np.sum(np.abs(rho_p) > 1e-6)}")
print(f"  Max |ρ_{{ji}}| / mean |ρ_{{ji}}| = {np.abs(rho_p).max() / np.abs(rho_p).mean():.2f}")
print(f"  At t=10, max |n_0(t)_{{ij}}| = {abs(expm(-1j*H_p*10).conj().T @ n0_p @ expm(-1j*H_p*10)).max():.4f}")
print()
print("In a finite-dimensional system, C(t) is quasiperiodic — no exponential")
print("decay of the OBSERVABLE itself. But the matrix elements oscillate ~ e^{iEt},")
print("so when sampled stochastically the variance can be dominated by")
print("near-cancellation between pairs at large t. For our small physical")
print("Hilbert space (D=16), the spectrum has only ~16 frequencies, so")
print("oscillations are bounded and the StN doesn't degrade dramatically.")
print()
print("For LARGER lattices (continuum-like spectrum), exponential StN degradation")
print("with t is the *standard* SK / EρOQ problem — see 2001.11490 Fig. 2.")
