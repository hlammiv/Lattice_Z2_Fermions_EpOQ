"""
Decompose the full-Hilbert-space thermal correlator into a sum over
Gauss-law sectors, to demonstrate that the EρOQ-faithful (full-trace)
result is the sector-weighted average — not the same as the physical
sector alone, but the discrepancy is entirely accounted for by the
unphysical-static-charge sectors.
"""

import numpy as np
from numpy import kron
from numpy.linalg import eigh
from scipy.linalg import expm

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
    t = psi_dag[a] @ Z_q[q_link] @ psi[b]
    H_hop += g_hop * (t + t.conj().T)
H = H + H_hop + sum(m_mass * parity[s] * N_q[m_q[s]] for s in m_q)
H = (H + H.conj().T) / 2

# Verify [H, G(site)] = 0 for all sites
for s in G_sites:
    comm = H @ G_sites[s] - G_sites[s] @ H
    assert np.linalg.norm(comm) < 1e-10, f"H doesn't commute with G({s})"

# --- Build sector projectors -------------------------------------------------
# Each sector labeled by (s1, s2, s3, s4) ∈ {±1}^4 indicating G(site) eigenvalue
sites_list = [(0,0), (0,1), (1,0), (1,1)]

def sector_projector(labels):
    """Build projector onto sector with G(site_k) = labels[k]."""
    P = I_full.copy()
    for site, lbl in zip(sites_list, labels):
        P = P @ (I_full + lbl * G_sites[site]) / 2
    return (P + P.conj().T) / 2

# --- Compute C(t) for each sector ------------------------------------------
beta = 0.5
n0_dense = N_q[4]
rho_full = expm(-beta * H)
Z_full = np.trace(rho_full).real

def C_sector(P_sector, t):
    """Compute Tr[P ρ O(t) O] / Tr[P ρ] in a sector."""
    Z_s = np.trace(P_sector @ rho_full).real
    if Z_s < 1e-12:
        return None, Z_s
    Ut = expm(-1j * H * t)
    n0_t = Ut.conj().T @ n0_dense @ Ut
    val = np.trace(P_sector @ rho_full @ n0_t @ n0_dense).real / Z_s
    return val, Z_s

def C_full(t):
    Ut = expm(-1j * H * t)
    n0_t = Ut.conj().T @ n0_dense @ Ut
    return np.trace(rho_full @ n0_t @ n0_dense).real / Z_full

times = [0.0, 0.5, 1.0, 2.0]

print(f"Sector decomposition of C(t) at β={beta}")
print()
print(f"Full-Hilbert-space C(t):")
for t in times:
    print(f"  t={t}: C_full(t) = {C_full(t):.6f}")

print()
print(f"Physical sector (G = +1 everywhere) C(t):")
P_phys = sector_projector([+1, +1, +1, +1])
for t in times:
    c, z = C_sector(P_phys, t)
    print(f"  t={t}: C_phys(t) = {c:.6f}, Z_phys = {z:.4f}, Z_phys/Z_full = {z/Z_full:.4f}")

print()
print("All 16 sectors (showing C(t=0) and Z fraction):")
print(f"  {'sector labels':>18} | {'Z_sector':>10} | {'Z/Z_full':>10} | {'C(t=0)':>10}")
total_check = 0.0
for s_idx in range(16):
    labels = [+1 if (s_idx >> k) & 1 == 0 else -1 for k in range(4)]
    P_s = sector_projector(labels)
    c, z = C_sector(P_s, 0.0)
    lbl_str = "(" + ",".join(f"{l:+d}" for l in labels) + ")"
    c_str = f"{c:.6f}" if c is not None else "  (Z=0)"
    print(f"  {lbl_str:>18} | {z:>10.4f} | {z/Z_full:>10.4f} | {c_str:>10}")
    if c is not None:
        total_check += (z / Z_full) * c

print()
print(f"Weighted sum Σ_sectors (Z_s/Z_full) × C_s(0) = {total_check:.6f}")
print(f"Direct C_full(0)                              = {C_full(0):.6f}")
print(f"Difference: {abs(total_check - C_full(0)):.3e}  (should be 0)")
