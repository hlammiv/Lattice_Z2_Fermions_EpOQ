"""
M3.3: end-to-end SK observable in 1+1d Schwinger (Nx=2) — the
mini-flagship demonstration.

Computes the thermal SK observable
    ⟨n_0(t)⟩_β = Tr[ρ_β · e^{+iHt} · n_0 · e^{-iHt}] / Tr[ρ_β]
where ρ_β = e^{-βH_Schwinger}.

Computes it THREE ways:
  (I)   Direct in original gauge×matter basis (the "exact reference").
  (II)  Matrix-element SK: ⟨n_0(t)⟩ = (1/Z) Σ_{i,j} ρ_ji n_0(t)_ij,
        where {|i⟩} is the full product basis and we explicitly sum over
        corner states.  This is the 2001.11490 decomposition.
  (III) Z-C-translated: same matrix-element sum but with H, n_0, ρ_β all
        rotated to the Z-C basis where matter DOFs are trivialized.

The three methods must agree.  Result: agreement is exact at machine
precision.  This is the toy version of the SK + Z-C exact end-to-end
pipeline that's supposed to improve over 2603.22422's MPS truncation.

What's *not* in this toy:
  - PF stochastic sampling (here we sum over all (i,j) brute force —
    in the real pipeline classical PF samples a subset)
  - Gauge field on the Euclidean leg (here the SK contour is degenerate;
    in the real pipeline the classical Euclidean leg samples both
    fermion corner states and gauge configs)
  - Z-C-induced extended interactions (degenerate in 1+1d Nx=2;
    appear in higher dimensions)

But all the structural primitives — corner-state sum, Z-C unitary,
real-time evolution under H' — are exercised exactly.
"""

import numpy as np
from scipy.linalg import expm

# --- same Hilbert space and operators as previous M3 scripts ---------------
g_sq = 1.0
M_mass = 0.5
epsilon = 0.5
beta = 1.0
dim_E = 5
E_max = 2
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
I_full = np.eye(dim_total, dtype=complex)

psi_dag0_U_psi1 = site_op(U_op, psi_dag, psi)
H = (
    (g_sq / 2) * (E_full @ E_full)
    + epsilon * (psi_dag0_U_psi1 + psi_dag0_U_psi1.conj().T)
    + M_mass * (n0_full - n1_full)
)
H = (H + H.conj().T) / 2

# Z-C unitary as before
U_ZC = np.eye(dim_total, dtype=complex)
def swap_rows(U, i, j):
    U[[i, j], :] = U[[j, i], :]
swap_rows(U_ZC, 9, 8)
swap_rows(U_ZC, 14, 12)
assert np.allclose(U_ZC @ U_ZC.conj().T, np.eye(dim_total))

# --- thermal density matrix -----------------------------------------------
rho_beta = expm(-beta * H)
Z_beta = np.trace(rho_beta).real

print(f"Setup: Nx=2 Schwinger, β={beta}, g²={g_sq}, M={M_mass}, ε={epsilon}")
print(f"Z(β) = {Z_beta:.6f}")
print()

# --- helper: ⟨n_0(t)⟩ via three methods ------------------------------------
def method_I_direct(t):
    """Direct: Tr[ρ_β e^{iHt} n_0 e^{-iHt}] / Z."""
    Ut = expm(-1j * H * t)
    obs = Ut.conj().T @ n0_full @ Ut
    return np.trace(rho_beta @ obs).real / Z_beta

def method_II_corner_sum(t):
    """Matrix-element SK: ⟨n_0(t)⟩ = (1/Z) Σ_{i,j} ρ_ji · n_0(t)_ij."""
    Ut = expm(-1j * H * t)
    n0_t = Ut.conj().T @ n0_full @ Ut
    total = 0j
    for i in range(dim_total):
        for j in range(dim_total):
            total += rho_beta[j, i] * n0_t[i, j]
    return (total / Z_beta).real

def method_III_via_ZC(t):
    """Same corner-state sum in Z-C basis. ρ, n_0(t), all rotated."""
    H_ZC = U_ZC @ H @ U_ZC.conj().T
    H_ZC = (H_ZC + H_ZC.conj().T) / 2
    rho_ZC = expm(-beta * H_ZC)
    Ut_ZC = expm(-1j * H_ZC * t)
    n0_ZC = U_ZC @ n0_full @ U_ZC.conj().T
    n0_t_ZC = Ut_ZC.conj().T @ n0_ZC @ Ut_ZC
    total = 0j
    for i in range(dim_total):
        for j in range(dim_total):
            total += rho_ZC[j, i] * n0_t_ZC[i, j]
    Z_ZC = np.trace(rho_ZC).real
    return (total / Z_ZC).real

# --- run all three at a range of times -------------------------------------
times = np.linspace(0, 3.0, 16)

print("    t   |  (I) direct  |  (II) corner sum  |  (III) Z-C basis  |  max diff")
print("  ------+--------------+-------------------+-------------------+---------")
results = []
for t in times:
    vI = method_I_direct(t)
    vII = method_II_corner_sum(t)
    vIII = method_III_via_ZC(t)
    max_d = max(abs(vI - vII), abs(vI - vIII), abs(vII - vIII))
    results.append((t, vI, vII, vIII, max_d))
    print(f"  {t:>5.2f} | {vI:>12.6f} | {vII:>17.6f} | {vIII:>17.6f} | {max_d:.2e}")

max_overall = max(r[4] for r in results)
print(f"\nMax disagreement across all times: {max_overall:.3e}")

# --- check the matrix-element decomposition: a few sample (i,j) pairs ------
print()
print("=" * 70)
print("Diagnostic: which (i,j) corner pairs dominate the SK sum?")
print("=" * 70)

t_probe = 1.5
Ut = expm(-1j * H * t_probe)
n0_t = Ut.conj().T @ n0_full @ Ut

contributions = []
for i in range(dim_total):
    for j in range(dim_total):
        contrib = rho_beta[j, i] * n0_t[i, j] / Z_beta
        if abs(contrib) > 1e-3:
            contributions.append((i, j, contrib))

contributions.sort(key=lambda x: -abs(x[2]))
print(f"\nLargest |contribution| at t={t_probe}:")
print(f"  {'i':>4} | {'j':>4} | {'(E,n0,n1) i':>14} | {'(E,n0,n1) j':>14} | contribution")
for i, j, c in contributions[:8]:
    Ei = E_vals[i // 4]; n0i = (i//2) % 2; n1i = i % 2
    Ej = E_vals[j // 4]; n0j = (j//2) % 2; n1j = j % 2
    print(f"  {i:>4} | {j:>4} | ({Ei:>2},{n0i},{n1i})       | ({Ej:>2},{n0j},{n1j})       | {c:+.5f}")

print()
print("These dominant (i,j) pairs are what a PF + sampling pipeline would")
print("preferentially sample. Truncating the sum to these dominant terms is")
print("the EρOQ basis-truncation approximation. The Z-C-translated version")
print("samples the SAME pairs but in the gauge-only basis after translation.")
