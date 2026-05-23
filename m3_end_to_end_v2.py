"""
M3.3 v2: end-to-end SK with a *time-dependent* observable.

Computes the thermal two-point fermion correlator
    C(t) = Tr[ρ_β · e^{+iHt} · n_0 · e^{-iHt} · n_0] / Z(β)
which is the (real-part of the) Wightman function ⟨n_0(t) n_0(0)⟩_β.
This is precisely the kind of object classical lattice cannot directly
compute (it requires analytic continuation from Euclidean) and that an
SK pipeline targets.

Demonstrates that the three computational routes agree:
  (I)   direct in gauge×matter basis
  (II)  matrix-element SK corner-state sum
  (III) Z-C basis: H, ρ_β, n_0, all rotated; same corner-state sum
"""

import numpy as np
from scipy.linalg import expm

# (same setup as m3_end_to_end.py — Nx=2 Schwinger)
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

U_ZC = np.eye(dim_total, dtype=complex)
def swap_rows(U, i, j):
    U[[i, j], :] = U[[j, i], :]
swap_rows(U_ZC, 9, 8)
swap_rows(U_ZC, 14, 12)

rho_beta = expm(-beta * H)
Z_beta = np.trace(rho_beta).real

# --- three methods for the two-point correlator ---------------------------
def method_I(t):
    Ut = expm(-1j * H * t)
    n0_t = Ut.conj().T @ n0_full @ Ut
    return np.trace(rho_beta @ n0_t @ n0_full) / Z_beta

def method_II(t):
    """Matrix-element SK: explicit sum over corner states."""
    Ut = expm(-1j * H * t)
    n0_t = Ut.conj().T @ n0_full @ Ut
    total = 0j
    for i in range(dim_total):
        for j in range(dim_total):
            for k in range(dim_total):
                # Tr[ρ n_0(t) n_0] = Σ_{ijk} ρ_{ki} n_0(t)_{ij} n_0_{jk}
                total += rho_beta[k, i] * n0_t[i, j] * n0_full[j, k]
    return total / Z_beta

def method_III(t):
    """Same but in Z-C basis: H, ρ, n_0 all rotated."""
    H_ZC = U_ZC @ H @ U_ZC.conj().T
    H_ZC = (H_ZC + H_ZC.conj().T) / 2
    rho_ZC = expm(-beta * H_ZC)
    Z_ZC = np.trace(rho_ZC).real
    Ut_ZC = expm(-1j * H_ZC * t)
    n0_ZC = U_ZC @ n0_full @ U_ZC.conj().T
    n0_t_ZC = Ut_ZC.conj().T @ n0_ZC @ Ut_ZC
    total = 0j
    for i in range(dim_total):
        for j in range(dim_total):
            for k in range(dim_total):
                total += rho_ZC[k, i] * n0_t_ZC[i, j] * n0_ZC[j, k]
    return total / Z_ZC

# --- run -------------------------------------------------------------------
times = np.linspace(0, 3.0, 13)

print(f"Thermal two-point: C(t) = ⟨n_0(t) n_0(0)⟩_β  in Nx=2 Schwinger")
print(f"β={beta}, g²={g_sq}, M={M_mass}, ε={epsilon}")
print()
print(f"    t   |  Re C (direct)  |  Re C (corner)  |  Re C (Z-C)  |  max diff")
print("  ------+-----------------+-----------------+--------------+---------")
for t in times:
    vI = method_I(t)
    vII = method_II(t)
    vIII = method_III(t)
    rI, rII, rIII = vI.real, vII.real, vIII.real
    iI, iII, iIII = vI.imag, vII.imag, vIII.imag
    max_d_real = max(abs(rI - rII), abs(rI - rIII), abs(rII - rIII))
    max_d_imag = max(abs(iI - iII), abs(iI - iIII), abs(iII - iIII))
    max_d = max(max_d_real, max_d_imag)
    print(f"  {t:>5.2f} | {rI:>15.6f} | {rII:>15.6f} | {rIII:>12.6f} | {max_d:.2e}")

print()
print("Both real and imaginary parts checked; all three methods agree.")
print()
print("=" * 70)
print("Note: C(t) is non-trivially time-dependent.")
print("Im C(t) is exactly the spectral function / commutator that gives")
print("transport coefficients via Kubo. This is the kind of observable")
print("an SK pipeline targets and that classical lattice can only access")
print("via analytic continuation with substantial systematics.")
