"""
EρOQ classical Euclidean sampler for the 2+1d Z2 + staggered toy system.

System
------
2+1d Z2 gauge + staggered fermion on a 2x2 OBC spatial lattice.
8 qubits, ordered as
    q0 = L_h0 [link (0,0)-(0,1)]
    q1 = L_h1 [link (1,0)-(1,1)]
    q2 = L_v0 [link (0,0)-(1,0)]
    q3 = L_v1 [link (0,1)-(1,1)]
    q4 = M(0,0), q5 = M(0,1), q6 = M(1,0), q7 = M(1,1).

Hamiltonian (built as a list of Pauli-string terms; never as a dense matrix
in the production path):

    H = - g_e  sum_{l in 0..3}  X_l
        - g_m  Z_0 Z_1 Z_2 Z_3
        + g_hop sum_l  ( psi_a^dag Z_l psi_b  +  h.c. )
        + m    sum_site  (-1)^(x+y) n_site

Hopping pairs (link_q, a, b) with a<b:
    (0, 0, 1), (1, 2, 3), (2, 0, 2), (3, 1, 3)

Jordan-Wigner with matter ordering q4 < q5 < q6 < q7:
    psi_a       = (prod_{c<a} Z_{4+c}) sigma_-_{4+a}
    psi_a^dag   = sigma_+_{4+a} (prod_{c<a} Z_{4+c})

For a hopping pair (l, a, b) with a < b, one finds after collapsing the
Jordan-Wigner strings:

    psi_a^dag Z_l psi_b + h.c.
        = (1/2) Z_l (prod_{c=a+1}^{b-1} Z_{4+c}) ( X_{4+a} X_{4+b} + Y_{4+a} Y_{4+b} ).

Explicit per-pair expansion:
    (l=0, a=0, b=1):   (1/2) Z_0 (X4X5 + Y4Y5)
    (l=1, a=2, b=3):   (1/2) Z_1 (X6X7 + Y6Y7)
    (l=2, a=0, b=2):   (1/2) Z_2 Z_5 (X4X6 + Y4Y6)
    (l=3, a=1, b=3):   (1/2) Z_3 Z_6 (X5X7 + Y5Y7)

Parameters: g_e=1.0, g_m=0.5, g_hop=0.5, m=0.5, beta=0.5.

Sampling target
---------------
P(i, j) = |<j| e^{-beta H} |i>| / Z_abs ,   i, j in {0, ..., 255},
with Z_abs = sum_{i,j} |<j| e^{-beta H} |i>|.

Matrix elements are evaluated by acting e^{-beta H} on the 256-dim state
|i> with a first-order Suzuki-Trotter product of Pauli-term factors.
"""

from __future__ import annotations

import math
import sys
from typing import List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Constants and parameters
# ---------------------------------------------------------------------------

NQ = 8
DIM = 1 << NQ          # 256
ALL = np.arange(DIM, dtype=np.int64)

G_E = 1.0
G_M = 0.5
G_HOP = 0.5
M_MASS = 0.5
BETA = 0.5

# Site -> matter qubit; site -> staggered sign (-1)^(x+y).
MATTER_QUBIT = {(0, 0): 4, (0, 1): 5, (1, 0): 6, (1, 1): 7}
PARITY = {(0, 0): +1, (0, 1): -1, (1, 0): -1, (1, 1): +1}

# Hopping pairs (link_qubit, a, b) with a < b.
HOP_PAIRS = [(0, 0, 1), (1, 2, 3), (2, 0, 2), (3, 1, 3)]

# Staggered KS η phases on hopping (Hamiltonian convention, time-fixed):
#   η_x(x,y) = +1   (direction 1)
#   η_y(x,y) = (-1)^x  (direction 2)
# site_idx = x*Ly + y with Ly=2: q_link=0,1 are y-bonds at x=0,1; q_link=2,3 are x-bonds.
STAGGERED_ETA_BY_LINK = {0: +1, 1: -1, 2: +1, 3: +1}


# A Pauli term is represented as (coefficient, [(qubit, 'X'|'Y'|'Z'), ...]).
# Identity factors are omitted from the list.
Term = Tuple[complex, List[Tuple[int, str]]]


# ---------------------------------------------------------------------------
# Build H as a list of Pauli-string terms
# ---------------------------------------------------------------------------

def build_pauli_terms() -> List[Term]:
    """Return H as a list of (coefficient, [(qubit, 'X'|'Y'|'Z'), ...]) terms.

    Identity factors are dropped.  The number operator is expanded as
    n = (I - Z)/2, so each staggered-mass site contributes an identity
    piece (a global energy shift) and a single-Z piece.
    """
    terms: List[Term] = []

    # --- Electric: -g_e sum_l X_l  (l = 0..3 are the link qubits) ---------
    for l in range(4):
        terms.append((complex(-G_E), [(l, "X")]))

    # --- Magnetic plaquette: -g_m  Z_0 Z_1 Z_2 Z_3 ------------------------
    terms.append((complex(-G_M), [(0, "Z"), (1, "Z"), (2, "Z"), (3, "Z")]))

    # --- Staggered hopping: (η_l * g_hop / 2) * Z_l * (JW string) * (X_a X_b + Y_a Y_b) ---
    # η_l from STAGGERED_ETA_BY_LINK (K-S spatial η).  a < b in matter-qubit ordering 4,5,6,7.
    for (l, a, b) in HOP_PAIRS:
        qa = 4 + a
        qb = 4 + b
        jw = [(4 + c, "Z") for c in range(a + 1, b)]
        common = [(l, "Z")] + jw
        eta = STAGGERED_ETA_BY_LINK[l]
        coef = complex(eta * G_HOP * 0.5)
        # XX piece
        terms.append((coef, common + [(qa, "X"), (qb, "X")]))
        # YY piece
        terms.append((coef, common + [(qa, "Y"), (qb, "Y")]))

    # --- Staggered mass: m * sum_site (-1)^(x+y) n_site -------------------
    # n = (I - Z)/2.  Collect the identity (constant) shift and a Z term
    # per site.  The constant shift becomes a single identity Pauli term.
    const_shift = 0.0
    for site, qm in MATTER_QUBIT.items():
        sgn = PARITY[site]
        # (m * sgn) * (I - Z)/2 = (m*sgn/2) * I  -  (m*sgn/2) * Z
        const_shift += 0.5 * M_MASS * sgn
        terms.append((complex(-0.5 * M_MASS * sgn), [(qm, "Z")]))
    if abs(const_shift) > 0.0:
        # Identity term: empty Pauli list.
        terms.append((complex(const_shift), []))

    # Drop any terms with zero coefficient.
    terms = [(c, ops) for (c, ops) in terms if abs(c) > 0.0]
    return terms


def build_pauli_terms_general(
    geom,
    g_e: float = G_E,
    g_m: float = G_M,
    g_hop: float = G_HOP,
    m_mass: float = None,
) -> List[Term]:
    """General H_QC builder for arbitrary L_x × L_y OBC.

    At L_x = L_y = 2 this returns a Pauli decomposition equivalent to
    build_pauli_terms() — same H, possibly different term ordering.  The
    qubit layout is the natural extension of the 2×2 convention
    (see action_minkowski_stitch.qc_layout_counts for the exact ordering):
        q_yl(x,y) = x·(Ly−1) + y                          for x∈[0,Lx), y∈[0,Ly−1)
        q_xl(x,y) = N_y + x·Ly + y                        for x∈[0,Lx−1), y∈[0,Ly)
        q_m(x,y)  = N_g + x·Ly + y                        for x∈[0,Lx), y∈[0,Ly)

    Hamiltonian (Kogut-Susskind staggered, time-fixed phases):
        H_E   = −g_e Σ_l  X_l                              (over all link qubits)
        H_M   = −g_m Σ_plaq  Z_{ℓ1} Z_{ℓ2} Z_{ℓ3} Z_{ℓ4}   (for each xy plaquette)
        H_hop = Σ_(link l = pair a−b)  (η_l · g_hop / 2) · Z_l · JW(qa,qb) · (X_qa X_qb + Y_qa Y_qb)
                with η_x = +1 on x-direction links, η_y = (−1)^x on y-direction links.
        H_m   = m · Σ_site  (−1)^(x+y) · n_site
              = m · Σ_site  (−1)^(x+y) · (I − Z_qm) / 2

    JW(qa,qb) is the product of Z over all matter qubits with index
    strictly between qa and qb (matter qubits are contiguous by construction).
    """
    if m_mass is None:
        m_mass = M_MASS
    Lx, Ly = geom.Lx, geom.Ly

    n_y = Lx * (Ly - 1)
    n_x = (Lx - 1) * Ly
    n_g = n_y + n_x

    def q_yl(x, y):
        return x * (Ly - 1) + y

    def q_xl(x, y):
        return n_y + x * Ly + y

    def q_m(x, y):
        return n_g + x * Ly + y

    terms: List[Term] = []

    # --- Electric: −g_e Σ_l X_l ---------------------------------------------
    for x in range(Lx):
        for y in range(Ly - 1):
            terms.append((complex(-g_e), [(q_yl(x, y), "X")]))
    for x in range(Lx - 1):
        for y in range(Ly):
            terms.append((complex(-g_e), [(q_xl(x, y), "X")]))

    # --- Magnetic plaquettes: −g_m Σ Z_{l1}Z_{l2}Z_{l3}Z_{l4} ----------------
    # Plaquette at (x,y) for x∈[0,Lx−1), y∈[0,Ly−1) is bounded by
    #   y-link (x,y), x-link (x,y+1), y-link (x+1,y), x-link (x,y).
    for x in range(Lx - 1):
        for y in range(Ly - 1):
            qs = sorted([q_yl(x, y), q_xl(x, y + 1),
                         q_yl(x + 1, y), q_xl(x, y)])
            terms.append((complex(-g_m), [(q, "Z") for q in qs]))

    def jw_string(qa, qb):
        """Z on every matter qubit strictly between qa and qb (qa < qb)."""
        return [(c, "Z") for c in range(qa + 1, qb)]

    # --- Hopping y-links: η_y = (−1)^x --------------------------------------
    for x in range(Lx):
        for y in range(Ly - 1):
            ql = q_yl(x, y)
            qa, qb = q_m(x, y), q_m(x, y + 1)
            if qa > qb:
                qa, qb = qb, qa
            eta = -1.0 if (x % 2) else 1.0
            jw = jw_string(qa, qb)
            coef = complex(eta * g_hop * 0.5)
            terms.append((coef, [(ql, "Z")] + jw + [(qa, "X"), (qb, "X")]))
            terms.append((coef, [(ql, "Z")] + jw + [(qa, "Y"), (qb, "Y")]))

    # --- Hopping x-links: η_x = +1 ------------------------------------------
    for x in range(Lx - 1):
        for y in range(Ly):
            ql = q_xl(x, y)
            qa, qb = q_m(x, y), q_m(x + 1, y)
            if qa > qb:
                qa, qb = qb, qa
            jw = jw_string(qa, qb)
            coef = complex(g_hop * 0.5)
            terms.append((coef, [(ql, "Z")] + jw + [(qa, "X"), (qb, "X")]))
            terms.append((coef, [(ql, "Z")] + jw + [(qa, "Y"), (qb, "Y")]))

    # --- Staggered mass: m · Σ (−1)^(x+y) n_site ----------------------------
    const_shift = 0.0
    for x in range(Lx):
        for y in range(Ly):
            qm = q_m(x, y)
            sgn = 1.0 if ((x + y) % 2 == 0) else -1.0
            const_shift += 0.5 * m_mass * sgn
            terms.append((complex(-0.5 * m_mass * sgn), [(qm, "Z")]))
    if abs(const_shift) > 1e-15:
        terms.append((complex(const_shift), []))

    # Drop zero-coefficient terms
    terms = [(c, ops) for (c, ops) in terms if abs(c) > 0.0]
    return terms


# ---------------------------------------------------------------------------
# Sparse Pauli-string application to a 256-element state vector
# ---------------------------------------------------------------------------

def apply_pauli_term(state: np.ndarray, term: Term) -> np.ndarray:
    """Apply a single Pauli-string term (coefficient * tensor product) to a
    state vector of length 2^NQ.

    No dense kron is used.  For each Pauli factor in the term we
    independently work out the bit flip and the phase as functions of the
    computational basis index, then build the output vector by indexing.

    The output is:   out[ index_with_x_flips ] = phase * coef * state[ index ]
    where x_flips is the XOR mask of all X- and Y- qubits, and phase comes
    from (Z and Y) factors.
    """
    coef, ops = term
    if not ops:
        return coef * state

    x_mask = 0      # bits to flip (X and Y)
    z_mask = 0      # bits that contribute Z-style phase (-1)^bit
    y_count = 0     # number of Y factors (each is iX*Z)

    for (q, p) in ops:
        bit = 1 << q
        if p == "X":
            x_mask ^= bit
        elif p == "Y":
            # Y = i * X * Z
            x_mask ^= bit
            z_mask ^= bit
            y_count += 1
        elif p == "Z":
            z_mask ^= bit
        elif p == "I":
            pass
        else:
            raise ValueError(f"Unknown Pauli '{p}'")

    # The Y-count contributes i**y_count overall.  i**k = 1, i, -1, -i.
    i_phase = (1j) ** y_count

    # Z-phase per basis index: (-1)^(popcount(z_mask & idx)).
    # For 8 qubits popcount fits in a small int; do it vectorized.
    if z_mask:
        # popcount via bit_count on int64
        parity = _popcount_parity(ALL & z_mask)
        z_phase = np.where(parity, -1.0, 1.0).astype(np.complex128)
    else:
        z_phase = 1.0  # broadcast

    if x_mask:
        # X bits flip indices: out[idx ^ x_mask] = phase[idx] * coef * state[idx]
        # Equivalently: out[idx] = phase[idx ^ x_mask] * coef * state[idx ^ x_mask]
        flipped = ALL ^ x_mask
        out = (coef * i_phase) * (z_phase if np.ndim(z_phase) == 0 else z_phase[flipped]) * state[flipped]
    else:
        out = (coef * i_phase) * z_phase * state

    return out


def _popcount_parity(arr: np.ndarray) -> np.ndarray:
    """Return parity (popcount mod 2) of each int in arr, vectorized for 8 bits."""
    # arr has values in [0, 256). Reduce by XOR-folding halves.
    x = arr.astype(np.int64)
    x = (x ^ (x >> 4)) & 0x0F
    x = (x ^ (x >> 2)) & 0x03
    x = (x ^ (x >> 1)) & 0x01
    return x.astype(bool)


# ---------------------------------------------------------------------------
# Trotterized imaginary-time evolution
# ---------------------------------------------------------------------------

def _apply_exp_pauli_term(state: np.ndarray, term: Term, tau: float) -> np.ndarray:
    """Apply exp(-tau * coef * P) to state, where P is a single Pauli string.

    For a Pauli string P with P^2 = I, we have
        exp(-tau * c * P) = cosh(tau * c) * I  -  sinh(tau * c) * P
    when c is real.  We handle the general (possibly complex) c by treating
    alpha = -tau * c and using exp(alpha P) = cosh(alpha) I + sinh(alpha) P.

    For the identity term, exp(alpha I) = exp(alpha) * I.
    """
    coef, ops = term
    alpha = -tau * coef  # exp(alpha * P)
    if not ops:
        # P = I, just a scalar.
        return np.exp(alpha) * state

    ch = np.cosh(alpha)
    sh = np.sinh(alpha)
    # P |state> = apply_pauli_term(state, (1, ops))
    p_state = apply_pauli_term(state, (1.0 + 0.0j, ops))
    return ch * state + sh * p_state


def trotter_step_imag(state: np.ndarray, terms: List[Term], dtau: float) -> np.ndarray:
    """One first-order Suzuki-Trotter step of exp(-dtau H) applied to state.

    H is assumed to be the sum of all (coef * Pauli) entries in `terms`,
    and each factor exp(-dtau * coef_k * P_k) is applied exactly using
    cosh/sinh because each Pauli string squares to the identity.
    """
    out = state
    for term in terms:
        out = _apply_exp_pauli_term(out, term, dtau)
    return out


def evolve_imag(state: np.ndarray, terms: List[Term], beta: float, n_trotter: int) -> np.ndarray:
    """Apply (exp(-dtau H))^{n_trotter} with dtau = beta / n_trotter."""
    dtau = beta / n_trotter
    out = state
    for _ in range(n_trotter):
        out = trotter_step_imag(out, terms, dtau)
    return out


# ---------------------------------------------------------------------------
# Matrix elements and sampling
# ---------------------------------------------------------------------------

# Module-level cache for the most recent (beta, n_trotter) propagator matrix.
_PROP_CACHE: dict = {}


def _build_propagator_columns(beta: float, n_trotter: int) -> np.ndarray:
    """Compute G[j, i] = <j| e^{-beta H} |i>  by Trotter-propagating each |i>.

    Returns a (DIM, DIM) complex array.  Cached.
    """
    key = (float(beta), int(n_trotter))
    if key in _PROP_CACHE:
        return _PROP_CACHE[key]
    terms = build_pauli_terms()
    G = np.empty((DIM, DIM), dtype=np.complex128)
    for i in range(DIM):
        e_i = np.zeros(DIM, dtype=np.complex128)
        e_i[i] = 1.0
        psi = evolve_imag(e_i, terms, beta, n_trotter)
        G[:, i] = psi
    _PROP_CACHE[key] = G
    return G


def matrix_element_e_minus_beta_H(i_bit: int, j_bit: int, beta: float, n_trotter: int) -> complex:
    """Compute <j| e^{-beta H} |i> by Trotter-propagating |i> and reading off j."""
    G = _build_propagator_columns(beta, n_trotter)
    return complex(G[j_bit, i_bit])


def sample_corner_pairs(beta: float, N_samples: int, rng: np.random.Generator,
                        n_trotter: int = 50) -> dict:
    """Sample (i, j) corner-state pairs proportional to |<j| e^{-beta H} |i>|.

    For the 2x2 toy lattice the state space has only 256^2 = 65536 (i, j)
    pairs, so we enumerate them and draw from the resulting categorical
    distribution.  At larger lattice size this enumeration would be
    replaced with an MCMC walker over (i, j) pairs.

    Returns
    -------
    dict with keys
        'pairs'        : (N_samples, 2) int array of (i, j) basis indices
        'signs'        : (N_samples,) int array (+1 / -1) of sign(Re G[j, i])
        'total_weight' : float scalar Z_abs = sum_{i,j} |G[j, i]|.
        'G'            : (256, 256) complex propagator matrix (cached).
    """
    G = _build_propagator_columns(beta, n_trotter)
    abs_G = np.abs(G)
    total_weight = float(abs_G.sum())
    if total_weight <= 0.0:
        raise RuntimeError("Total weight is zero; cannot sample.")

    flat_p = (abs_G / total_weight).ravel()
    # Numerical guard: renormalize so np.random sees probs that sum to 1.
    flat_p = flat_p / flat_p.sum()

    idx = rng.choice(DIM * DIM, size=N_samples, p=flat_p, replace=True)
    j_arr = (idx // DIM).astype(np.int64)
    i_arr = (idx % DIM).astype(np.int64)
    pairs = np.stack([i_arr, j_arr], axis=1)

    re_G = G.real
    signs = np.sign(re_G[j_arr, i_arr]).astype(np.int64)
    # Treat exact zeros as +1 (they have measure zero in the sampler anyway).
    signs[signs == 0] = 1

    return {
        "pairs": pairs,
        "signs": signs,
        "total_weight": total_weight,
        "G": G,
    }


# ---------------------------------------------------------------------------
# ED reference (validation only; not used by the sampler)
# ---------------------------------------------------------------------------

def _build_dense_H_for_validation() -> np.ndarray:
    """Build the 256x256 dense H from the Pauli-term list.

    Used only by the self-test as the reference.  The rest of the module
    does NOT use this construction.
    """
    I2 = np.eye(2, dtype=np.complex128)
    X2 = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y2 = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Z2 = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    PAULI = {"I": I2, "X": X2, "Y": Y2, "Z": Z2}

    terms = build_pauli_terms()
    H = np.zeros((DIM, DIM), dtype=np.complex128)
    for coef, ops in terms:
        op_per_q = {q: p for (q, p) in ops}
        mat = np.array([[1.0 + 0.0j]])
        # IMPORTANT: the sparse Pauli applier uses qubit q as bit (1<<q),
        # i.e. q=0 is the least-significant bit.  To match that ordering
        # in a kron-built dense H we must put q=NQ-1 LEFTMOST and q=0
        # RIGHTMOST in the tensor product.
        for q in reversed(range(NQ)):
            p = op_per_q.get(q, "I")
            mat = np.kron(mat, PAULI[p])
        H = H + coef * mat
    return H


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> int:
    """Run validation checks.  Returns 0 on success, nonzero on failure."""
    from scipy.linalg import expm  # validation only

    failures = 0
    print("=" * 70)
    print("EpOQ classical sampler self-test")
    print(f"  System: 2+1d Z2 + staggered, 2x2 OBC, {NQ} qubits, dim={DIM}")
    print(f"  Couplings: g_e={G_E}, g_m={G_M}, g_hop={G_HOP}, m={M_MASS}, "
          f"beta={BETA}")
    print("=" * 70)

    # --- 1. Build H as Pauli terms ---------------------------------------
    terms = build_pauli_terms()
    n_terms = len(terms)
    print(f"\n[1] H built as {n_terms} Pauli-string terms.")
    # Sanity: count by family
    n_X = sum(1 for c, ops in terms if len(ops) == 1 and ops[0][1] == "X")
    n_Z_single = sum(1 for c, ops in terms if len(ops) == 1 and ops[0][1] == "Z")
    n_ZZZZ = sum(1 for c, ops in terms
                 if len(ops) == 4 and all(p == "Z" for (_, p) in ops))
    n_id = sum(1 for c, ops in terms if not ops)
    print(f"    single-X terms : {n_X}  (expect 4)")
    print(f"    single-Z terms : {n_Z_single}  (expect 4 from mass)")
    print(f"    4-Z plaquette  : {n_ZZZZ}  (expect 1)")
    print(f"    identity terms : {n_id}  (expect 1 if mass shift != 0, else 0)")
    print(f"    remaining (hop): {n_terms - n_X - n_Z_single - n_ZZZZ - n_id}"
          f"  (expect 8 = 2 per hop pair * 4 pairs)")

    # --- 2. Dense ED reference ------------------------------------------
    print("\n[2] Building dense H (validation only) and exp(-beta H) via expm...")
    H_dense = _build_dense_H_for_validation()
    herm_err = np.linalg.norm(H_dense - H_dense.conj().T)
    print(f"    ||H - H^dag|| = {herm_err:.3e}")
    if herm_err > 1e-10:
        print("    FAIL: H not Hermitian.")
        failures += 1

    expmH = expm(-BETA * H_dense)
    Z_ed = float(np.trace(expmH).real)
    print(f"    Z_ED = tr(exp(-beta H))_dense = {Z_ed:.8f}")

    # --- 3. Trotter Z(beta) and convergence ------------------------------
    print("\n[3] Trotter evaluation of Z(beta) at several step counts...")
    n_trotter_choices = [10, 25, 50, 100]
    last_n = None
    for n_t in n_trotter_choices:
        # Clear cache so we rebuild for each step count.
        _PROP_CACHE.clear()
        G = _build_propagator_columns(BETA, n_t)
        Z_trotter = float(np.trace(G).real)
        rel_err = abs(Z_trotter - Z_ed) / abs(Z_ed)
        # Also check max element error vs ED.
        max_err = float(np.max(np.abs(G - expmH)))
        print(f"    n_trotter={n_t:4d} :  Z = {Z_trotter:.8f}   "
              f"|dZ|/Z = {rel_err:.3e}   max|G-expmH| = {max_err:.3e}")
        last_n = n_t
        last_rel_err = rel_err

    # Spec: agreement to ~0.1% or better with sufficient Trotter steps.
    if last_rel_err > 1e-3:
        print(f"    FAIL: Trotter at n={last_n} gives {last_rel_err:.3e} "
              f"relative error (>1e-3).")
        failures += 1
    else:
        print(f"    PASS: relative error {last_rel_err:.3e} at "
              f"n_trotter={last_n}.")

    # --- 4. Sampler test --------------------------------------------------
    print("\n[4] Sampling test: draw N=10000 corner pairs and compare "
          "marginal P(i) to ED.")
    rng = np.random.default_rng(20260515)
    N = 10_000
    out = sample_corner_pairs(BETA, N, rng, n_trotter=last_n)
    pairs = out["pairs"]
    Z_abs = out["total_weight"]
    G_used = out["G"]

    # Empirical marginal of i.
    i_arr = pairs[:, 0]
    emp_counts = np.bincount(i_arr, minlength=DIM)
    emp_marg = emp_counts / float(N)

    # Theoretical marginal of i from |G|: P(i) = sum_j |G[j,i]| / Z_abs.
    theo_marg_abs = np.abs(G_used).sum(axis=0) / Z_abs

    # Also compute the "diagonal" marginal P_diag(i) = G[i,i] / Z, the
    # Gibbs marginal that the user mentions in the docstring.  These are
    # different distributions: the sampler is over |G[j,i]|, and the
    # natural marginal there is sum_j |G[j,i]|/Z_abs.  Report both.
    diag = np.real(np.diag(G_used))
    Z_diag = diag.sum()
    p_diag = diag / Z_diag

    l1_err_abs = float(np.abs(emp_marg - theo_marg_abs).sum())
    print(f"    Z_abs = sum_ij |G[j,i]| = {Z_abs:.6f}")
    print(f"    L1 distance(empirical, |G|-marginal of i) "
          f"= {l1_err_abs:.4f}")
    # For N=10^4 over 256 bins, sqrt(256/N) ~ 0.16 -> expect L1 < ~0.4.
    if l1_err_abs > 0.5:
        print("    FAIL: empirical marginal too far from theoretical.")
        failures += 1
    else:
        print("    PASS: empirical marginal consistent with theoretical.")

    # For information: the diagonal (Gibbs) marginal sum and the diagonal
    # entries match between Trotter and ED.
    diag_ed = np.real(np.diag(expmH))
    diag_max_err = float(np.max(np.abs(diag - diag_ed)))
    print(f"    max |G_ii - expmH_ii|        = {diag_max_err:.3e}")
    if diag_max_err > 1e-3:
        print("    FAIL: diagonal mismatch with ED.")
        failures += 1

    # --- Sign / weight summary ------------------------------------------
    n_neg = int(np.sum(out["signs"] < 0))
    print(f"\n    sample signs: {N - n_neg} positive, {n_neg} negative "
          f"({n_neg / N * 100:.2f}% sign flips).")

    print("\n" + "=" * 70)
    if failures == 0:
        print("ALL SELF-TESTS PASSED.")
    else:
        print(f"{failures} SELF-TEST FAILURE(S).")
    print("=" * 70)
    return failures


if __name__ == "__main__":
    sys.exit(_self_test())
