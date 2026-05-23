"""
verify_ks_vs_z2_setup.py — diagnostic: does z2_setup's Hamiltonian match
the standard Kogut-Susskind Hamiltonian on 2×2 with staggered mass?

Standard KS Hamiltonian (2 spatial dims):
    η_x(x,y) = 1
    η_y(x,y) = (-1)^x

For our 2×2 lattice with sites (0,0), (0,1), (1,0), (1,1) and HOP_PAIRS:
    pair (l=0, a=0, b=1): sites (0,0)-(0,1)  → y-hop at x=0 → η_y = +1
    pair (l=1, a=2, b=3): sites (1,0)-(1,1)  → y-hop at x=1 → η_y = -1
    pair (l=2, a=0, b=2): sites (0,0)-(1,0)  → x-hop at y=0 → η_x = +1
    pair (l=3, a=1, b=3): sites (0,1)-(1,1)  → x-hop at y=1 → η_x = +1

So pair (l=1, a=2, b=3) should have coefficient -g_hop in standard KS.
z2_setup uses +g_hop for ALL four pairs → not standard KS.

This script builds the PROPER KS Hamiltonian on 8 qubits with the η_y = (-1)^x
sign and compares its thermal expectation to z2_setup's physical_sector_reference.
"""
from __future__ import annotations
import sys
import numpy as np
from scipy.linalg import expm

sys.path.insert(0, '/home/hlamm/Desktop/QC/logdet/m1_toy')


def build_ks_pauli_terms(eta_signs: dict, g_e=1.0, g_m=0.5, g_hop=0.5, m=0.5):
    """Same Hamiltonian structure as z2_setup but with explicit eta signs per pair.

    eta_signs[pair_index] = ±1.  z2_setup uses {0:+1, 1:+1, 2:+1, 3:+1};
    proper KS uses {0:+1, 1:-1, 2:+1, 3:+1}.
    """
    HOP_PAIRS = [(0, 0, 1), (1, 2, 3), (2, 0, 2), (3, 1, 3)]
    MATTER_QUBIT = {(0, 0): 4, (0, 1): 5, (1, 0): 6, (1, 1): 7}
    PARITY = {(0, 0): +1, (0, 1): -1, (1, 0): -1, (1, 1): +1}

    terms = []
    for l in range(4):
        terms.append((complex(-g_e), [(l, "X")]))
    terms.append((complex(-g_m), [(0, "Z"), (1, "Z"), (2, "Z"), (3, "Z")]))
    for pi, (l, a, b) in enumerate(HOP_PAIRS):
        qa = 4 + a
        qb = 4 + b
        jw = [(4 + c, "Z") for c in range(a + 1, b)]
        common = [(l, "Z")] + jw
        sign = eta_signs.get(pi, 1)
        coef = complex(sign * g_hop * 0.5)
        terms.append((coef, common + [(qa, "X"), (qb, "X")]))
        terms.append((coef, common + [(qa, "Y"), (qb, "Y")]))
    const = 0.0
    for site, qm in MATTER_QUBIT.items():
        sgn = PARITY[site]
        const += 0.5 * m * sgn
        terms.append((complex(-0.5 * m * sgn), [(qm, "Z")]))
    if abs(const) > 0:
        terms.append((complex(const), []))
    return terms


def thermal_n0_at_beta(eta_signs: dict, beta: float = 4.0) -> float:
    """⟨n̂_0⟩ at β in the physical sector for given eta signs."""
    NQ = 8
    DIM = 256
    P2 = {
        'I': np.eye(2, dtype=complex),
        'X': np.array([[0,1],[1,0]], dtype=complex),
        'Y': np.array([[0,-1j],[1j,0]], dtype=complex),
        'Z': np.diag([1,-1]).astype(complex),
    }

    def pauli_term_dense(factors):
        by_q = {q: P2['I'] for q in range(NQ)}
        for q, ax in factors:
            by_q[q] = P2[ax]
        result = by_q[NQ-1]
        for q in range(NQ-2, -1, -1):
            result = np.kron(result, by_q[q])
        return result

    terms = build_ks_pauli_terms(eta_signs)
    H = sum(c * pauli_term_dense(fac) for c, fac in terms)
    H = (H + H.conj().T) / 2
    n0 = 0.5 * np.eye(DIM, dtype=complex) - 0.5 * pauli_term_dense([(4, 'Z')])

    # Physical sector projector (Gauss law)
    links_at = {(0,0):[0,2],(0,1):[0,3],(1,0):[1,2],(1,1):[1,3]}
    parity = {(0,0):+1,(0,1):-1,(1,0):-1,(1,1):+1}
    matter_q = {(0,0):4,(0,1):5,(1,0):6,(1,1):7}

    def G_dense(site):
        G = np.eye(DIM, dtype=complex)
        for ql in links_at[site]:
            G = G @ pauli_term_dense([(ql, 'X')])
        sign_op = pauli_term_dense([(matter_q[site], 'Z')])
        if parity[site] == -1:
            sign_op = -sign_op
        return G @ sign_op

    P_phys = np.eye(DIM, dtype=complex)
    for s in [(0,0),(0,1),(1,0),(1,1)]:
        P_phys = P_phys @ (np.eye(DIM, dtype=complex) + G_dense(s)) / 2
    P_phys = (P_phys + P_phys.conj().T) / 2
    evals_P, evecs_P = np.linalg.eigh(P_phys)
    phys_basis = evecs_P[:, evals_P > 0.5]
    H_phys = phys_basis.conj().T @ H @ phys_basis
    H_phys = (H_phys + H_phys.conj().T) / 2
    n0_phys = phys_basis.conj().T @ n0 @ phys_basis
    n0_phys = (n0_phys + n0_phys.conj().T) / 2
    rho = expm(-beta * H_phys)
    Z = np.trace(rho).real
    return np.trace(rho @ n0_phys).real / Z


def main():
    print("=" * 70)
    print("Compare KS-correct vs z2_setup-as-is Hamiltonian (2x2 staggered mass)")
    print("=" * 70)
    print(f"\n{'eta convention':<40} ⟨n̂_0⟩_β=4")
    print("-" * 70)
    # z2_setup (all +g_hop)
    z2 = thermal_n0_at_beta({0: +1, 1: +1, 2: +1, 3: +1}, beta=4.0)
    print(f"  z2_setup (naive, all +g_hop)         {z2:+.6f}")
    # Standard KS: η_y = (-1)^x → pair 1 (y-hop at x=1) gets -1
    ks_yx = thermal_n0_at_beta({0: +1, 1: -1, 2: +1, 3: +1}, beta=4.0)
    print(f"  KS (η_y=(-1)^x): pair 1 flipped      {ks_yx:+.6f}")
    # Alt KS: η_x = (-1)^y → pair 3 (x-hop at y=1) gets -1
    ks_xy = thermal_n0_at_beta({0: +1, 1: +1, 2: +1, 3: -1}, beta=4.0)
    print(f"  KS (η_x=(-1)^y): pair 3 flipped      {ks_xy:+.6f}")
    # Both: hmm not standard but for completeness
    both = thermal_n0_at_beta({0: +1, 1: -1, 2: +1, 3: -1}, beta=4.0)
    print(f"  Both signs flipped (pair 1, 3)        {both:+.6f}")

    print("\nFor reference:")
    print(f"  continuum 1/(1+e^{{βm}}) at β=4 m=0.5 = {1/(1+np.exp(2)):+.6f}")
    print("\nz2_setup (as is) = {0:+.6f}, M-K-from-action gave 0.072 at trivial U".format(z2))


if __name__ == "__main__":
    main()
