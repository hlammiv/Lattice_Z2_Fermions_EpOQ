"""
b.2: M2 + M3 integrated pipeline test.

Compute the thermal two-point correlator C(t) = ⟨n_0(t) n_0(0)⟩_β in the 1+1d
Nx=2 Schwinger model via STOCHASTIC SAMPLING of corner-state pairs (M2-style)
combined with Z-C-rotated real-time evolution (M3-style).

The pipeline:
  1. Classical-side: sample basis states (i, j) ∼ |ρ_{ji}| / Z (importance sampling)
     For our toy, ρ_β is positive-definite so |ρ_{ji}| = Re ρ_{ji}.
  2. For each sample: compute the matrix element n_0(t)_{ij} two ways:
       (a) directly: ⟨i| e^{iHt} n_0 e^{-iHt} |j⟩
       (b) via Z-C: rotate to gauge-only basis, evolve under H'=U_ZC H U_ZC†,
           recover the matrix element
  3. Average with the sampling weights to estimate C(t).
  4. Compare to exact direct calculation (the reference).

If methods (a) and (b) agree under stochastic sampling, the M2+M3 pipeline
is internally consistent. If they both converge to the exact C(t), the
whole thing works end-to-end.
"""

import numpy as np
from scipy.linalg import expm

# Setup as before
g_sq, M_mass, epsilon, beta = 1.0, 0.5, 0.5, 1.0
dim_E = 5; E_max = 2
E_vals = np.arange(-E_max, E_max + 1)
dim_total = dim_E * 2 * 2

def site_op(O_E, O_n0, O_n1):
    return np.kron(np.kron(O_E, O_n0), O_n1)

E_op = np.diag(E_vals).astype(complex)
U_op = np.zeros((dim_E, dim_E), dtype=complex)
for k in range(dim_E - 1):
    U_op[k + 1, k] = 1.0
n_op = np.diag([0, 1]).astype(complex)
psi = np.array([[0, 1], [0, 0]], dtype=complex)
psi_dag = psi.conj().T
I_E = np.eye(dim_E, dtype=complex)
I_F = np.eye(2, dtype=complex)

E_full = site_op(E_op, I_F, I_F)
n0_full = site_op(I_E, n_op, I_F)
n1_full = site_op(I_E, I_F, n_op)

psi_dag0_U_psi1 = site_op(U_op, psi_dag, psi)
H = (
    (g_sq / 2) * (E_full @ E_full)
    + epsilon * (psi_dag0_U_psi1 + psi_dag0_U_psi1.conj().T)
    + M_mass * (n0_full - n1_full)
)
H = (H + H.conj().T) / 2

# Z-C unitary
U_ZC = np.eye(dim_total, dtype=complex)
def swap_rows(U, i, j):
    U[[i, j], :] = U[[j, i], :]
swap_rows(U_ZC, 9, 8)
swap_rows(U_ZC, 14, 12)

# Thermal density and rotation
rho_beta = expm(-beta * H)
H_ZC = U_ZC @ H @ U_ZC.conj().T
H_ZC = (H_ZC + H_ZC.conj().T) / 2
rho_ZC = expm(-beta * H_ZC)
n0_ZC = U_ZC @ n0_full @ U_ZC.conj().T

# --- Importance sampling distribution -------------------------------------
# Sample (i,j) ∼ |ρ_{ji}| / Σ |ρ_{ji}|
# (For our system ρ is positive definite so this just samples ρ_{ji}/Tr ρ approximately)
prob_pair = np.abs(rho_beta).T   # P(i,j) = |ρ_{ji}|
total_weight = prob_pair.sum()
prob_flat = (prob_pair / total_weight).flatten()
# Sanity check: rows of prob_pair correspond to i (column index of ρ), cols to j (row index of ρ)
# prob_pair[i, j] = |ρ_{j, i}|

# Use the same sampling in Z-C basis
prob_pair_ZC = np.abs(rho_ZC).T
total_weight_ZC = prob_pair_ZC.sum()
prob_flat_ZC = (prob_pair_ZC / total_weight_ZC).flatten()

rng = np.random.default_rng(seed=2026)

def stochastic_C(t, N_samples, use_ZC=False):
    """Stochastic estimator for C(t) via importance sampling of (i,j) pairs."""
    if use_ZC:
        rho_use = rho_ZC
        n0_use = n0_ZC
        Ut = expm(-1j * H_ZC * t)
        prob = prob_flat_ZC
        weight = total_weight_ZC
    else:
        rho_use = rho_beta
        n0_use = n0_full
        Ut = expm(-1j * H * t)
        prob = prob_flat
        weight = total_weight

    n0_t = Ut.conj().T @ n0_use @ Ut
    # C(t) = Σ_{ijk} ρ_{ki} n_0(t)_{ij} n_0_{jk} / Z
    # We have 3 indices; importance-sample (i,k) pairs from |ρ_{ki}|, sum k explicitly.
    # That's still a 2-index sample; the matrix multiplication over j is done exactly
    # at the cost of one matrix product per sample.

    # Actually simpler: rewrite as
    # C(t) = Σ_{ki} ρ_{ki} (n_0(t) n_0)_{ik} / Z
    n0t_n0 = n0_t @ n0_use
    # C(t) = Σ_{ki} ρ_{ki} n0t_n0_{ik} / Z = Σ_{ik} (n0t_n0)_{ik} ρ_{ki} / Z
    # = Σ_{(k,i)} A_{k,i} / Z, where A_{k,i} = ρ_{ki} n0t_n0_{ik}

    # Importance sampling: pick (k,i) with prob |ρ_{ki}| / Σ|ρ|, sum weight × A_{k,i}/|ρ_{ki}|
    # = sign(ρ_{ki}) × n0t_n0_{ik} × total_weight / N

    Z = np.trace(rho_use).real
    flat_indices = rng.choice(dim_total * dim_total, size=N_samples, p=prob)
    total = 0j
    for fidx in flat_indices:
        i = fidx // dim_total       # i corresponds to first index of prob_pair = column of ρ
        j_dummy = fidx % dim_total  # this is the j-index of prob_pair = row of ρ
        # so (k=j_dummy, i=i)
        k, ii = j_dummy, i
        sign = rho_use[k, ii] / abs(rho_use[k, ii]) if abs(rho_use[k, ii]) > 0 else 0
        total += sign * n0t_n0[ii, k]
    return (total * weight / (N_samples * Z))

# --- Exact reference for comparison ---------------------------------------
def exact_C(t):
    Ut = expm(-1j * H * t)
    n0_t = Ut.conj().T @ n0_full @ Ut
    return np.trace(rho_beta @ n0_t @ n0_full).real / np.trace(rho_beta).real

# --- Run --------------------------------------------------------------------
times = [0.0, 0.5, 1.0, 1.5, 2.0]
N_samples = 50_000

print(f"Stochastic C(t) via importance sampling of (i,j) pairs, N={N_samples}.")
print()
print(f"    t   |  exact C(t)  | stochastic direct | stochastic Z-C | (diff dir) | (diff ZC)")
print("  ------+--------------+-------------------+----------------+------------+----------")
for t in times:
    cE = exact_C(t)
    cS_dir = stochastic_C(t, N_samples, use_ZC=False).real
    cS_ZC = stochastic_C(t, N_samples, use_ZC=True).real
    print(f"  {t:>5.2f} | {cE:>12.6f} | {cS_dir:>17.6f} | {cS_ZC:>14.6f} | "
          f"{abs(cE-cS_dir):>10.2e} | {abs(cE-cS_ZC):>9.2e}")

print()
print("Expected: stochastic results agree with exact within statistical noise")
print("(~1/sqrt(N) scaling, ~0.5% at N=50000). Both 'direct' and 'Z-C' routes")
print("converge to the same exact value, demonstrating M2+M3 pipeline consistency.")
