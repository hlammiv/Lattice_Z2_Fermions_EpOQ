"""
M2 sandbox: pseudofermion stochastic estimator for the junction kernel Σ.

We want Σ = B_ME · M_E^{-1} · B_EM stochastically estimated from Gaussian
sources η. Estimator: for each sample, compute
    v = M_E^{-1} (B_EM η)
    Σ̂ = (B_ME v) ⊗ η†
Average over N samples gives an unbiased estimate of Σ.

Cost per sample: one Dirac solve on the Euclidean leg.

This script verifies (a) unbiased convergence of Σ̂ to the exact Σ, and (b)
the variance behavior as N grows.
"""

import numpy as np
from numpy.linalg import det, solve, norm
import sys

# Same lattice as v1/v2
Nx, Nt, Nt_E, m = 4, 8, 4, 0.5

def idx(t, x):
    return (t % Nt) * Nx + (x % Nx)

def build_M_APBC():
    N = Nx * Nt
    M = np.zeros((N, N), dtype=complex)
    for t in range(Nt):
        for x in range(Nx):
            M[idx(t, x), idx(t, x)] = m
    for t in range(Nt):
        for x in range(Nx):
            sfwd = -1 if (t == Nt - 1) else 1
            sbwd = -1 if (t == 0) else 1
            M[idx(t, x), idx(t + 1, x)] += 0.5 * sfwd
            M[idx(t, x), idx(t - 1, x)] += -0.5 * sbwd
    for t in range(Nt):
        eta1 = (-1) ** t
        for x in range(Nx):
            M[idx(t, x), idx(t, x + 1)] += 0.5 * eta1
            M[idx(t, x), idx(t, x - 1)] += -0.5 * eta1
    return M

M = build_M_APBC()
NE = Nt_E * Nx
NM = (Nt - Nt_E) * Nx

M_EE = M[:NE, :NE]
M_EM = M[:NE, NE:]
M_ME = M[NE:, :NE]
M_MM = M[NE:, NE:]

# Exact reference Σ
Sigma_exact = M_ME @ solve(M_EE, M_EM)

# --- PF stochastic estimator -----------------------------------------------
# η is a complex Gaussian on NM dimensions: η ∼ CN(0, I_NM)
# i.e., real and imaginary parts are independent N(0, 1/2). Then ⟨η η†⟩ = I_NM.

rng = np.random.default_rng(seed=42)

def pf_sample():
    """One stochastic sample of Σ via PF estimator."""
    eta = (rng.standard_normal(NM) + 1j * rng.standard_normal(NM)) / np.sqrt(2)
    u = M_EM @ eta              # NE-dim
    v = solve(M_EE, u)          # NE-dim, costs 1 Dirac solve
    Bv = M_ME @ v               # NM-dim
    return np.outer(Bv, eta.conj())  # NM × NM stochastic Σ

# Convergence study
N_samples_list = [10, 100, 1000, 10000, 100000]

print("PF estimator convergence to exact Σ:")
print(f"  Exact Σ Frobenius norm = {norm(Sigma_exact):.6e}")
print()
print(f"  {'N':>8} | {'||Σ_PF - Σ_exact||_F':>22} | {'rel error':>12} | {'std/mean(diag)':>14}")
print("  " + "-" * 70)

# Accumulate samples incrementally to share random samples across N values
Sigma_running = np.zeros((NM, NM), dtype=complex)
diag_running = []

n_done = 0
for N in N_samples_list:
    while n_done < N:
        Sigma_running += pf_sample()
        n_done += 1

    Sigma_avg = Sigma_running / N
    err = norm(Sigma_avg - Sigma_exact)
    rel = err / norm(Sigma_exact)

    # Estimate variance: re-run a small block to get std of diagonal entries
    diag_samples = []
    for _ in range(100):
        diag_samples.append(pf_sample().diagonal())
    diag_samples = np.array(diag_samples)
    n_done += 100
    std_diag = np.std(diag_samples, axis=0).mean()
    mean_diag = np.abs(Sigma_exact.diagonal()).mean()

    print(f"  {N:>8} | {err:>22.4e} | {rel:>12.4e} | {std_diag/mean_diag:>14.4e}")

# --- Detailed check at largest N -------------------------------------------
print()
print("Per-entry comparison on the junction-slice subblock (8×8):")
junction_M_offsets = [0, Nt - Nt_E - 1]
J = np.array([off * Nx + x for off in junction_M_offsets for x in range(Nx)])

Sigma_exact_J = Sigma_exact[np.ix_(J, J)]
Sigma_PF_J = (Sigma_running / n_done)[np.ix_(J, J)]

print(f"Largest entries of exact Σ on junction × junction:")
print(np.real_if_close(Sigma_exact_J, tol=1e-10).round(4))
print()
print(f"PF estimator (N≈{n_done}) on same block:")
print(np.real_if_close(Sigma_PF_J, tol=1e-10).round(4))
print()
print(f"Max abs deviation on this block: "
      f"{np.max(np.abs(Sigma_exact_J - Sigma_PF_J)):.4e}")
