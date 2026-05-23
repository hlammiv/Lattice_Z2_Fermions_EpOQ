"""
staggered_ks_reference.py — proper staggered KS Hamiltonian reference.

Builds the staggered Kogut-Susskind Hamiltonian on 2x2 OBC with 4 matter +
4 gauge qubits, applies Gauss-law projection, and computes thermal
observables for direct comparison to our action MC + Trotter pipeline.

Why this exists: physical_sector_reference (in action_endtoend_pipeline.py)
uses NAIVE hopping (no staggered η phases) — different lattice fermion
theory from our staggered action.  Compare the action pipeline to THIS
reference instead.

Spatial staggered η conventions (2+1d KS, Hamiltonian form):
  η_x(x, y) = +1                  (direction 1: no preceding coords)
  η_y(x, y) = (-1)^x              (direction 2: preceded by x)

This is the K-S rotation of the action's spacetime η phases:
  Action: η_x = (-1)^t, η_y = (-1)^{t+x}
  K-S rotation absorbs the t-dependence into a field redefinition,
  yielding the time-independent spatial η used here.

Mass parity (matches both pipeline and z2_setup):
  m · (-1)^{x+y}  per site
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import expm


# 2x2 OBC site convention:  site_idx = x * Ly + y, Ly = 2
# (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
LX, LY = 2, 2
V3 = LX * LY                      # 4 spatial sites
N_LINKS = 4                       # 2 x-links + 2 y-links for 2x2 OBC
NQ = N_LINKS + V3                 # 4 gauge + 4 matter = 8 qubits
DIM = 1 << NQ                     # 256

# Pauli matrices and Kron helper -------------------------------------------
P2 = {
    'I': np.eye(2, dtype=complex),
    'X': np.array([[0, 1], [1, 0]], dtype=complex),
    'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
    'Z': np.diag([1, -1]).astype(complex),
}

def pauli_term_dense(factors, nq=NQ):
    """Build the dense 2^nq matrix for a Pauli string. factors = [(q, axis), ...]."""
    by_q = {q: P2['I'] for q in range(nq)}
    for q, ax in factors:
        by_q[q] = P2[ax]
    # qubit ordering matches z2_setup: q0 = LSB (rightmost) in z2_setup,
    # but build_pauli_terms uses leftmost = high qubit.  Mirror that here.
    result = by_q[nq - 1]
    for q in range(nq - 2, -1, -1):
        result = np.kron(result, by_q[q])
    return result


# Link qubit indices (matching z2_setup HOP_PAIRS layout) -------------------
# Z2_setup convention:
#   link 0 (q0): y-bond (0,0)-(0,1)   U_y[t=0, x=0, y=0]
#   link 1 (q1): y-bond (1,0)-(1,1)   U_y[t=0, x=1, y=0]
#   link 2 (q2): x-bond (0,0)-(1,0)   U_x[t=0, x=0, y=0]
#   link 3 (q3): x-bond (0,1)-(1,1)   U_x[t=0, x=0, y=1]
LINK_Q = {
    ('y', 0, 0): 0,  # y-bond connecting sites (0,0) and (0,1)
    ('y', 1, 0): 1,  # y-bond connecting sites (1,0) and (1,1)
    ('x', 0, 0): 2,  # x-bond connecting sites (0,0) and (1,0)
    ('x', 0, 1): 3,  # x-bond connecting sites (0,1) and (1,1)
}

def matter_q(site_idx):
    return 4 + site_idx


def build_pauli_terms_stagg(g_E=1.0, g_M=0.5, g_hop=0.5, m=0.5):
    """Build the staggered KS Hamiltonian as a list of (coef, [(q, axis), ...])
    Pauli-string terms.  Differences from z2_setup:
      - Hopping has staggered η phase: y-hop at x=1 gets a minus sign.

    H = - g_E Σ_l X_l                                       (electric)
        - g_M Z_0 Z_1 Z_2 Z_3                                (magnetic, 1 plaq)
        + Σ_l g_hop · η_l · (Z_l / 2) (X_a X_b + Y_a Y_b)    (staggered hop)
        + Σ_a m · (-1)^{x+y} · (I - Z_{matter,a}) / 2        (staggered mass)
    """
    terms = []

    # Electric: -g_E Σ X_l
    for ql in range(N_LINKS):
        terms.append((-g_E, [(ql, 'X')]))

    # Magnetic: -g_M Z_0 Z_1 Z_2 Z_3  (single OBC plaquette over all 4 links)
    terms.append((-g_M, [(0, 'Z'), (1, 'Z'), (2, 'Z'), (3, 'Z')]))

    # Hopping: g_hop · η · Z_l · (X_a X_b + Y_a Y_b) / 2 for each (a,b,l,η)
    # In Hamiltonian KS staggered, spatial η are:
    #   η_x(x, y) = +1
    #   η_y(x, y) = (-1)^x
    hop_bonds = [
        # (direction, x, y, site_a, site_b, eta)
        ('y', 0, 0, 0, 1, +1),    # (0,0)-(0,1): η_y at x=0 = (-1)^0 = +1
        ('y', 1, 0, 2, 3, -1),    # (1,0)-(1,1): η_y at x=1 = (-1)^1 = -1
        ('x', 0, 0, 0, 2, +1),    # (0,0)-(1,0): η_x = +1
        ('x', 0, 1, 1, 3, +1),    # (0,1)-(1,1): η_x = +1
    ]
    for dir_, x, y, a, b, eta in hop_bonds:
        ql = LINK_Q[(dir_, x, y)]
        qa = matter_q(a)
        qb = matter_q(b)
        coef = g_hop * eta * 0.5
        # X_a X_b · Z_l
        terms.append((coef, [(ql, 'Z'), (qa, 'X'), (qb, 'X')]))
        # Y_a Y_b · Z_l
        terms.append((coef, [(ql, 'Z'), (qa, 'Y'), (qb, 'Y')]))

    # Staggered mass: m·(-1)^{x+y}·n_a = m·(-1)^{x+y}·(I - Z_matter_a)/2
    site_parity = {0: +1, 1: -1, 2: -1, 3: +1}   # (x+y) mod 2 → parity
    for a, par in site_parity.items():
        qm = matter_q(a)
        # m·par·(1/2)·I  +  m·par·(-1/2)·Z
        terms.append((+m * par * 0.5, []))                  # identity term
        terms.append((-m * par * 0.5, [(qm, 'Z')]))
    return terms


def gauss_law_projector():
    """Build the Z_2 Gauss-law projector P_phys = Π_v (I + G_v)/2 where
    G_v = (Π_{l∋v} X_l) · (sign_v · Z_matter_v).
    Sign per site = (-1)^{x+y} (matches z2_setup convention).
    """
    # links touching each site:
    links_at = {0: [0, 2], 1: [0, 3], 2: [1, 2], 3: [1, 3]}
    parity = {0: +1, 1: -1, 2: -1, 3: +1}

    def G_v(site):
        # Π X_l for links touching this site
        ops = [(ql, 'X') for ql in links_at[site]]
        G = pauli_term_dense(ops)
        sign = pauli_term_dense([(matter_q(site), 'Z')])
        if parity[site] == -1:
            sign = -sign
        return G @ sign

    P = np.eye(DIM, dtype=complex)
    for s in [0, 1, 2, 3]:
        P = P @ (np.eye(DIM, dtype=complex) + G_v(s)) / 2.0
    P = (P + P.conj().T) / 2
    return P


def staggered_ks_thermal_n0(beta: float, g_E=1.0, g_M=0.5, g_hop=0.5, m=0.5):
    """Thermal ⟨n̂_0⟩ at site 0 in the Gauss-law-physical sector of the
    staggered KS Hamiltonian.
    """
    terms = build_pauli_terms_stagg(g_E=g_E, g_M=g_M, g_hop=g_hop, m=m)
    H = sum(c * pauli_term_dense(fac) for c, fac in terms)
    H = (H + H.conj().T) / 2

    P = gauss_law_projector()
    # Project to physical subspace (eigenvectors of P with eigenvalue 1)
    evals_P, evecs_P = np.linalg.eigh(P)
    phys_basis = evecs_P[:, evals_P > 0.5]
    print(f"  physical sector dim = {phys_basis.shape[1]}")

    H_phys = phys_basis.conj().T @ H @ phys_basis
    H_phys = (H_phys + H_phys.conj().T) / 2

    n0_full = 0.5 * np.eye(DIM, dtype=complex) - 0.5 * pauli_term_dense([(matter_q(0), 'Z')])
    n0_phys = phys_basis.conj().T @ n0_full @ phys_basis
    n0_phys = (n0_phys + n0_phys.conj().T) / 2

    rho = expm(-beta * H_phys)
    Z = np.trace(rho).real
    n0_thermal = (np.trace(rho @ n0_phys) / Z).real

    eigs = np.linalg.eigvalsh(H_phys)
    return {
        'n0_thermal': n0_thermal,
        'Z': Z,
        'ground_energy': float(eigs.min()),
        'first_excited': float(sorted(eigs)[1]),
        'phys_dim': phys_basis.shape[1],
    }


def main():
    import sys
    print("=" * 72)
    print("Staggered KS Hamiltonian reference (proper, with η phases)")
    print("=" * 72)
    print("Couplings: g_E=1.0, g_M=0.5, g_hop=0.5, m=0.5")
    print()

    for beta in [4.0, 6.0, 8.0, 10.0]:
        print(f"β = {beta}")
        r = staggered_ks_thermal_n0(beta=beta)
        print(f"  ⟨n̂_0⟩_thermal,phys = {r['n0_thermal']:+.6f}")
        print(f"  ground energy = {r['ground_energy']:+.4f}, "
              f"first excited = {r['first_excited']:+.4f}, "
              f"gap = {r['first_excited']-r['ground_energy']:.4f}")
        print()

    print("Comparison reference (naive hopping, z2_setup):")
    from action_endtoend_pipeline import physical_sector_reference
    for beta in [4.0, 6.0, 8.0]:
        ref_naive = physical_sector_reference(beta=beta, times=[0.0])[0.0]
        print(f"  β={beta}: physical_sector_reference (NAIVE)  = {ref_naive:+.6f}")

    print()
    print("Pipeline continuum-limit plateau: ~0.108 (a_τ ≤ 0.25 at β=4)")


if __name__ == "__main__":
    main()
