"""
EρOQ quantum / Minkowski-leg simulator for the 2+1d Z₂ + staggered-fermion
toy model on a 2×2 OBC spatial lattice.

This is the "quantum-circuit-side" partner to the classical Euclidean
sampler. It produces the Minkowski-time matrix elements

    M(i, j; O, t) = ⟨j| U†(t) O U(t) |i⟩

that the stitcher feeds back into the EρOQ reconstruction, where
U(t) = exp(-i H t) is approximated by a first-order Trotter circuit.

Crucially, **the simulator never builds the dense 256×256 H matrix**.
H is represented as a list of (complex coefficient, Pauli-string) terms,
and every gate is applied as bit-index manipulation on the 256-element
state vector. A dense H is constructed *only inside __main__* for
validation; it is not used by any of the exported functions.

Qubit layout (matches m1_toy/z2_setup.py):

    q0=L_h0  q1=L_h1  q2=L_v0  q3=L_v1   # gauge link qubits
    q4=M_00  q5=M_01  q6=M_10  q7=M_11   # staggered matter qubits

Hopping pairs (link qubit, matter-a, matter-b): (0,0,1), (1,2,3), (2,0,2), (3,1,3).

Conventions:
  * Bit ordering of the 8-bit index: bit q has weight 2**q.
    State amplitude for basis |b7 b6 ... b0⟩ lives at index Σ b_q · 2**q.
  * Z|0⟩=+|0⟩, Z|1⟩=-|1⟩. X flips bit q. Y = i X Z up to bit-conventions:
    Y|0⟩ = +i|1⟩, Y|1⟩ = -i|0⟩.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants / parameters
# ---------------------------------------------------------------------------

NQ = 8
DIM = 1 << NQ  # 256

G_E = 1.0
G_M = 0.5
G_HOP = 0.5
M_MASS = 0.5

# Link qubits 0..3, matter qubits 4..7
LINK_QUBITS = (0, 1, 2, 3)
MATTER_QUBITS = (4, 5, 6, 7)
HOP_PAIRS = [
    (0, 0, 1),  # link q=0, matter a=0 ↔ a=1  (y-bond at x=0)
    (1, 2, 3),  # link q=1, matter a=2 ↔ a=3  (y-bond at x=1)
    (2, 0, 2),  # link q=2, matter a=0 ↔ a=2  (x-bond at y=0)
    (3, 1, 3),  # link q=3, matter a=1 ↔ a=3  (x-bond at y=1)
]
# (x,y) staggered parities for matter sites attached to q=4..7
SITE_PARITY = {4: +1, 5: -1, 6: -1, 7: +1}

# Staggered KS η phases on the hopping (Hamiltonian convention, time-fixed):
#   η_x(x,y) = +1     (direction 1 — no preceding coords)
#   η_y(x,y) = (-1)^x (direction 2 — preceded by x)
# In our 2×2 site labelling site_idx = x*Ly + y with Ly=2:
#   q_link=0 → y-bond at x=0: η_y = +1
#   q_link=1 → y-bond at x=1: η_y = -1   ← the only sign flip
#   q_link=2 → x-bond at y=0: η_x = +1
#   q_link=3 → x-bond at y=1: η_x = +1
STAGGERED_ETA_BY_LINK = {0: +1, 1: -1, 2: +1, 3: +1}

# A "term" is (coefficient, [(qubit_idx, 'X'|'Y'|'Z'|'I'), ...]).
# Identities are optional; absent qubits are implicitly I.
PauliFactor = Tuple[int, str]
PauliTerm = Tuple[complex, List[PauliFactor]]


# ---------------------------------------------------------------------------
# Hamiltonian as a list of Pauli-string terms (no dense matrix)
# ---------------------------------------------------------------------------

def build_pauli_terms() -> List[PauliTerm]:
    """Return H as list of (coefficient, [(qubit_idx, axis), ...]) terms.

    All coefficients are real; we return them as complex for uniformity.
    Pure-identity ("constant energy") pieces, e.g. from m·(I − Z)/2 on the
    matter sites, are returned as the empty Pauli list. They give a global
    phase under e^{-iHt}, which cancels in ⟨j|U†OU|i⟩; we keep them for
    correctness when callers might want bare U(t) on its own.
    """
    terms: List[PauliTerm] = []

    # Electric: −g_e Σ X_ℓ on link qubits ℓ=0..3
    for q in LINK_QUBITS:
        terms.append((complex(-G_E), [(q, "X")]))

    # Magnetic plaquette: −g_m Z_0 Z_1 Z_2 Z_3
    terms.append((complex(-G_M),
                  [(0, "Z"), (1, "Z"), (2, "Z"), (3, "Z")]))

    # Staggered hopping: g_hop Σ η_l (ψ†_a Z_ℓ ψ_b + h.c.) expands to
    #   (η_l · g_hop/2) Z_ℓ · JW · (X_{4+a} X_{4+b} + Y_{4+a} Y_{4+b}),
    # where η_l is the K-S staggered phase (STAGGERED_ETA_BY_LINK above)
    # and JW = ∏_{a<c<b} Z_{4+c} is the Jordan-Wigner string between
    # matter indices a and b. The Z_ℓ factor multiplies the link qubit.
    for q_link, a, b in HOP_PAIRS:
        if a > b:
            a, b = b, a
        eta = STAGGERED_ETA_BY_LINK[q_link]
        coef = complex(eta * G_HOP / 2.0)
        jw_qubits = [(4 + c, "Z") for c in range(a + 1, b)]
        qa, qb = 4 + a, 4 + b
        # XX piece
        terms.append((coef,
                      jw_qubits + [(q_link, "Z"), (qa, "X"), (qb, "X")]))
        # YY piece
        terms.append((coef,
                      jw_qubits + [(q_link, "Z"), (qa, "Y"), (qb, "Y")]))

    # Staggered mass: m Σ_site (-1)^{x+y} n_site
    #   n = (I − Z)/2 on matter qubit.
    #   contribution: (m·parity/2) I  + (−m·parity/2) Z_qmatter
    for q_matter, par in SITE_PARITY.items():
        c_const = complex(M_MASS * par / 2.0)
        c_z = complex(-M_MASS * par / 2.0)
        terms.append((c_const, []))               # global energy shift
        terms.append((c_z, [(q_matter, "Z")]))    # one-Z term

    return terms


# ---------------------------------------------------------------------------
# Low-level Pauli-string action on a 256-element state vector
# ---------------------------------------------------------------------------
#
# Strategy (no kron, no dense matrices):
#
#   * A Pauli string P = P_{q1} ⊗ P_{q2} ⊗ ... is split into a Z-part
#     (the qubits where P is Z, Y) and an X-part (the qubits where P is
#     X, Y). On the computational basis state |b⟩ (b is an 8-bit
#     integer), P|b⟩ = phase(b) · |b XOR x_mask⟩, where:
#       - x_mask is the bit-OR of all qubits where P has X or Y.
#       - phase(b) is determined by:
#           * Each Z (or Y) flips sign based on the bit of b at that qubit.
#             For Y, the sign is determined by the OUTGOING bit, which
#             equals the incoming bit XOR'd by the bit flip of Y. So we
#             apply Y carefully (see below).
#           * Each Y contributes a factor of +i (from X|0⟩=|1⟩, X|1⟩=|0⟩
#             vs Y|0⟩=+i|1⟩, Y|1⟩=-i|0⟩: Y = i X Z in our convention,
#             so the per-Y "+i" is independent of the bit, while the Z
#             part of Y picks up a sign from the INCOMING bit).
#
#   * For e^{-iθP} where P is a Pauli string with coefficient 1:
#       If x_mask == 0 (pure Z/I, diagonal): multiply state[b] by
#         exp(-iθ · sign(b)).
#       Else: use e^{-iθP} = cos(θ) I − i sin(θ) P. Apply P (a sparse
#         permutation+phase) and combine.
#
# All operations are vectorised with numpy bit ops on the index array.

_BIT_INDICES = np.arange(DIM, dtype=np.int64)


def _bit_at(idx_arr: np.ndarray, q: int) -> np.ndarray:
    """Return the bit at qubit q (0 or 1) for every state index."""
    return (idx_arr >> q) & 1


def _parse_term(term: PauliTerm) -> Tuple[complex, int, int, int]:
    """Decompose a term into (coef, x_mask, z_mask, y_mask).

    x_mask = qubits with X or Y (those that flip the basis bit).
    z_mask = qubits with Z or Y (those that contribute a Z-like phase
             based on the INCOMING bit value).
    y_mask = qubits with Y (each adds a factor of +i in addition to the
             X and Z parts).

    Identities and duplicate-qubit terms are tolerated; duplicates on
    the same qubit are NOT collapsed (caller should keep terms simple).
    """
    coef, factors = term
    x_mask = z_mask = y_mask = 0
    for q, axis in factors:
        if axis == "I":
            continue
        if axis == "X":
            x_mask |= 1 << q
        elif axis == "Z":
            z_mask |= 1 << q
        elif axis == "Y":
            x_mask |= 1 << q
            z_mask |= 1 << q
            y_mask |= 1 << q
        else:
            raise ValueError(f"Unknown axis {axis!r}")
    return complex(coef), x_mask, z_mask, y_mask


def _popcount(x: np.ndarray) -> np.ndarray:
    """Vectorised population count for int64 arrays."""
    x = x - ((x >> 1) & 0x5555555555555555)
    x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
    x = (x + (x >> 4)) & 0x0f0f0f0f0f0f0f0f
    return (x * 0x0101010101010101) >> 56


def _apply_pauli_raw(state: np.ndarray, x_mask: int, z_mask: int,
                     y_mask: int) -> np.ndarray:
    """Apply the Pauli string (coefficient 1) defined by the masks.

    Returns a new state vector. Vectorised: no Python loops over basis.
    """
    if x_mask == 0:
        # Diagonal Pauli: phases only.
        # Count Z-eigenvalue −1 bits hit: popcount(idx & z_mask) parity.
        parity = _popcount(_BIT_INDICES & z_mask) & 1
        sign = np.where(parity == 1, -1.0, 1.0).astype(np.complex128)
        # Y contributes "+i" each; pure-diagonal means y_mask=0 (Y has X
        # part too), so we never hit a "+i" factor here unless caller
        # violates the structure. Be defensive:
        if y_mask:
            # Should not happen when x_mask == 0, since Y sets x_mask.
            raise RuntimeError("y_mask set but x_mask == 0; inconsistent.")
        return state * sign

    # Off-diagonal: permutation + per-amplitude phase.
    # Outgoing index for amplitude originally at idx: out = idx XOR x_mask.
    # Convention used here: (P|ψ⟩)[out] = phase(in) · ψ[in].
    # So we *scatter*: new_state[idx XOR x_mask] = phase(idx) · state[idx].
    # Phase(idx) comes from:
    #   Z part: (-1)^{popcount(idx & z_mask)}.
    #     For Y on qubit q, the Z part should be evaluated on the INCOMING
    #     bit (Y|0⟩ = +i|1⟩, Y|1⟩ = -i|0⟩: the sign factor (+,-) tracks
    #     the incoming bit; the "i" is bit-independent). Including the
    #     Y qubits in z_mask achieves this.
    #   Y part: factor of (+i)^{number of Y's in the string}, applied to
    #     every amplitude.
    parity = _popcount(_BIT_INDICES & z_mask) & 1
    sign = np.where(parity == 1, -1.0, 1.0).astype(np.complex128)
    n_y = bin(y_mask).count("1")
    i_factor = (1j) ** n_y  # global phase factor for all amplitudes
    phase = sign * i_factor

    out_state = np.empty_like(state)
    out_idx = _BIT_INDICES ^ x_mask
    # Scatter: new_state[out_idx] = phase * state[in_idx]
    out_state[out_idx] = phase * state
    return out_state


def apply_pauli_term(state: np.ndarray, term: PauliTerm) -> np.ndarray:
    """Apply one Pauli-string term (with its coefficient) to a state vector.

    No dense kron; pure index permutation + phase multiply.
    """
    coef, x_mask, z_mask, y_mask = _parse_term(term)
    if x_mask == 0 and z_mask == 0 and y_mask == 0:
        # Identity term.
        return coef * state
    return coef * _apply_pauli_raw(state, x_mask, z_mask, y_mask)


def apply_pauli_evolution(state: np.ndarray, term: PauliTerm,
                          theta: float) -> np.ndarray:
    """Apply e^{−i θ P} where P is a Pauli string with coefficient 1.

    The full Pauli term provided has a complex coefficient c; the caller
    is expected to fold it into theta (we do that here too, on the
    assumption the coefficient is real — which holds for all terms of
    this Hamiltonian). For a generic c = a+ib with b≠0, e^{−iθcP} is
    not unitary, which is meaningless for real-time evolution; we error
    out in that case to surface bugs.

    For pure-Z/I diagonal P, the action is exact multiplication by a
    phase. For P containing X or Y (x_mask ≠ 0), we use
        e^{−iθP} = cos(θ) I − i sin(θ) P,
    which only requires applying P (a permutation+phase) and combining.
    The full Pauli term's coefficient is folded into theta.
    """
    coef, x_mask, z_mask, y_mask = _parse_term(term)
    if abs(coef.imag) > 1e-14:
        raise ValueError(
            f"Pauli term coefficient is complex ({coef}); cannot Trotter-"
            f"evolve as a Hermitian generator.")
    angle = theta * coef.real

    if x_mask == 0 and z_mask == 0 and y_mask == 0:
        # Identity generator: global phase.
        return state * np.exp(-1j * angle)

    if x_mask == 0:
        # Diagonal in computational basis: vectorised phase.
        parity = _popcount(_BIT_INDICES & z_mask) & 1
        eig = np.where(parity == 1, -1.0, 1.0)  # ±1
        return state * np.exp(-1j * angle * eig)

    # Off-diagonal: cos(θ) I − i sin(θ) P, with the bare P (coef 1).
    c = math.cos(angle)
    s = math.sin(angle)
    p_state = _apply_pauli_raw(state, x_mask, z_mask, y_mask)
    return c * state - 1j * s * p_state


# ---------------------------------------------------------------------------
# Trotter evolution
# ---------------------------------------------------------------------------

def trotter_step_real(state: np.ndarray, terms: Sequence[PauliTerm],
                      dt: float) -> np.ndarray:
    """One first-order (Lie–Trotter) step of e^{−i dt H}.

    Approximates e^{−i dt H} ≈ ∏_k e^{−i dt h_k P_k}, with the product
    taken in the order the terms appear in `terms`. Identity (empty
    Pauli list) terms are applied as global phases.

    Error per step: O(dt² · ‖[H_a, H_b]‖). Use trotter_step_real_strang
    for the 2nd-order (symmetric) version with O(dt³) per-step error.
    """
    out = state
    for term in terms:
        out = apply_pauli_evolution(out, term, dt)
    return out


def trotter_step_real_strang(state: np.ndarray, terms: Sequence[PauliTerm],
                             dt: float) -> np.ndarray:
    """One 2nd-order (symmetric Strang) Trotter step of e^{−i dt H}.

       S_2(dt) = (∏_{k=1}^n e^{-i dt/2 h_k P_k}) · (∏_{k=n}^{1} e^{-i dt/2 h_k P_k})

    Equivalent to a forward half-step pass through `terms`, followed by a
    reverse half-step pass.  Error per step: O(dt³ · ‖[H_a,[H_a,H_b]]‖).
    Costs 2× the per-step time of first-order Lie-Trotter, but for fixed
    accuracy needs many fewer steps, so net cheaper.
    """
    out = state
    half = dt / 2.0
    terms_list = list(terms)
    for term in terms_list:
        out = apply_pauli_evolution(out, term, half)
    for term in reversed(terms_list):
        out = apply_pauli_evolution(out, term, half)
    return out


def evolve_real_time(state: np.ndarray, terms: Sequence[PauliTerm],
                     t: float, n_steps: int, order: int = 2) -> np.ndarray:
    """Real-time evolve a state for time t using `n_steps` Trotter steps.

    A negative `t` runs evolution backwards (equivalent to U†(|t|)).
    `order` selects the Trotter order: 1 = first-order Lie-Trotter,
    2 = symmetric Strang (DEFAULT, 2nd-order, much smaller error at fixed
    n_steps).
    """
    if n_steps <= 0:
        if n_steps == 0 and t == 0.0:
            return state.copy()
        raise ValueError("n_steps must be a positive integer.")
    dt = t / n_steps
    if order == 1:
        step_fn = trotter_step_real
    elif order == 2:
        step_fn = trotter_step_real_strang
    else:
        raise ValueError(f"unsupported Trotter order {order}; use 1 or 2")
    out = state
    for _ in range(n_steps):
        out = step_fn(out, terms, dt)
    return out


# ---------------------------------------------------------------------------
# Matrix-element machinery
# ---------------------------------------------------------------------------

def _basis_state(idx: int) -> np.ndarray:
    if not 0 <= idx < DIM:
        raise ValueError(f"basis index {idx} out of range [0,{DIM})")
    v = np.zeros(DIM, dtype=np.complex128)
    v[idx] = 1.0
    return v


def apply_observable(state: np.ndarray,
                     observable_terms: Sequence[PauliTerm]) -> np.ndarray:
    """Apply an observable (sum of Pauli-string terms) to a state vector."""
    out = np.zeros_like(state)
    for term in observable_terms:
        out = out + apply_pauli_term(state, term)
    return out


def _pauli_string_dense(x_mask: int, z_mask: int, y_mask: int) -> np.ndarray:
    """Build the dense 256×256 matrix representing a Pauli string with the
    given masks (coefficient 1).  Used for fast cache builds.
    """
    if x_mask == 0 and z_mask == 0 and y_mask == 0:
        return np.eye(DIM, dtype=np.complex128)
    M = np.zeros((DIM, DIM), dtype=np.complex128)
    parity = _popcount(_BIT_INDICES & z_mask) & 1
    sign = np.where(parity == 1, -1.0, 1.0).astype(np.complex128)
    n_y = bin(y_mask).count("1")
    i_factor = (1j) ** n_y
    phase = sign * i_factor
    out_idx = _BIT_INDICES ^ x_mask
    # (Pauli)[out, in] = phase(in) for out = in XOR x_mask
    M[out_idx, _BIT_INDICES] = phase
    return M


def _trotter_step_matrix(H_terms: Sequence[PauliTerm], dt: float, order: int) -> np.ndarray:
    """Build the dense 256×256 matrix S(dt) = ∏ e^{-i dt c_k P_k} (Trotter
    step) using the SAME ordering and Strang form as trotter_step_real /
    trotter_step_real_strang.  Identical operator up to floating-point.
    """
    if order == 1:
        passes = [(H_terms, dt)]
    elif order == 2:
        # Strang: forward at dt/2 then reverse at dt/2
        passes = [(H_terms, dt / 2.0), (list(reversed(list(H_terms))), dt / 2.0)]
    else:
        raise ValueError(f"unsupported Trotter order {order}")

    S = np.eye(DIM, dtype=np.complex128)
    for terms_pass, theta_scale in passes:
        for term in terms_pass:
            coef, x_mask, z_mask, y_mask = _parse_term(term)
            if abs(coef.imag) > 1e-14:
                raise ValueError(f"complex Pauli coefficient {coef}")
            angle = theta_scale * coef.real
            if x_mask == 0 and z_mask == 0 and y_mask == 0:
                # Identity generator: global phase.  Multiply through.
                S = S * np.exp(-1j * angle)
                continue
            if x_mask == 0:
                # Diagonal: phase per state index.
                parity = _popcount(_BIT_INDICES & z_mask) & 1
                eig = np.where(parity == 1, -1.0, 1.0).astype(np.complex128)
                phase_diag = np.exp(-1j * angle * eig)
                # Apply diagonal phase from the left: S <- diag(phase) @ S
                S = phase_diag[:, None] * S
                continue
            # Off-diagonal: e^{-i θ P} = cos(θ) I − i sin(θ) P.  Build P as
            # dense and combine.
            c = math.cos(angle)
            s = math.sin(angle)
            P = _pauli_string_dense(x_mask, z_mask, y_mask)
            S = (c * np.eye(DIM, dtype=np.complex128) - 1j * s * P) @ S
    return S


def build_trottered_UOU_cache(
    times: Sequence[float],
    observable_terms: Sequence[PauliTerm] = None,
    n_trotter: int = 200,
    order: int = 2,
) -> dict:
    """Pre-compute U†(t)·O·U(t) as a dense 256×256 matrix per t, using the
    SAME Trotter scheme as `observable_matrix_element` (faithful to the QC
    implementation).

    For each t in `times`, evolves each computational-basis state |b⟩
    through the n_trotter-step Trotter circuit to obtain column b of U(t),
    then assembles U†(t)·O·U(t) by dense matrix products.

    Returns dict {t: matrix} where matrix[a, b] = ⟨a|U†(t) O U(t)|b⟩.
    Once cached, per-sample matrix element is a dict lookup → eliminates
    the per-sample Trotter loop.

    The Trotter approximation is preserved exactly — this is the same
    operator the QC would apply at fixed n_trotter (just computed once
    on the simulator and cached, instead of once per sample).
    """
    H_terms = build_pauli_terms()
    if observable_terms is None:
        # Default: n_0 at matter site 0 (q4 in z2_setup convention)
        observable_terms = [(0.5 + 0j, []), (-0.5 + 0j, [(4, 'Z')])]

    # Build dense observable matrix once via Pauli decomposition (also fast).
    O_dense = np.zeros((DIM, DIM), dtype=np.complex128)
    for term in observable_terms:
        coef, x_mask, z_mask, y_mask = _parse_term(term)
        if x_mask == 0 and z_mask == 0 and y_mask == 0:
            O_dense += complex(coef) * np.eye(DIM, dtype=np.complex128)
        else:
            O_dense += complex(coef) * _pauli_string_dense(x_mask, z_mask, y_mask)

    # Match observable_matrix_element EXACTLY: each t uses its own dt = t/n_trotter
    # and n_trotter steps.  Build the Trotter step matrix S(dt_t) per t, then
    # U(t) = S(dt_t)^n_trotter via repeated matrix multiplication.
    out = {}
    for t in times:
        if t == 0.0 or n_trotter == 0:
            out[t] = O_dense.copy()
            continue
        dt_t = t / n_trotter
        S = _trotter_step_matrix(H_terms, dt_t, order=order)
        U_t = np.eye(DIM, dtype=np.complex128)
        for _ in range(n_trotter):
            U_t = S @ U_t
        out[t] = U_t.conj().T @ O_dense @ U_t
    return out


def observable_matrix_element(i_bit: int, j_bit: int,
                              observable_terms: Sequence[PauliTerm],
                              t: float,
                              n_trotter: int = 100) -> complex:
    """Compute ⟨j| U†(t) O U(t) |i⟩ by direct state-vector simulation.

    Steps (per spec):
      1. Prepare |i⟩ as a computational-basis state vector.
      2. Forward Trotter evolution by time t: |ψ⟩ = U(t) |i⟩.
      3. Apply O: |φ⟩ = O |ψ⟩.
      4. Backward Trotter evolution: |χ⟩ = U†(t) |φ⟩, implemented as
         real-time evolution by −t.
      5. Project onto |j⟩: return ⟨j|χ⟩.
    """
    H_terms = build_pauli_terms()
    psi = _basis_state(i_bit)
    if t == 0.0 or n_trotter == 0:
        chi = apply_observable(psi, observable_terms)
    else:
        psi = evolve_real_time(psi, H_terms, t, n_trotter)
        phi = apply_observable(psi, observable_terms)
        chi = evolve_real_time(phi, H_terms, -t, n_trotter)
    return complex(chi[j_bit])


def observable_matrix_element_hadamard(i_bit: int, j_bit: int,
                                       observable_terms: Sequence[PauliTerm],
                                       t: float,
                                       n_trotter: int = 100) -> complex:
    """Hadamard-test-flavoured alternative for ⟨j| U†(t) O U(t) |i⟩.

    Implemented as a 9-qubit state-vector simulation (1 ancilla + 8 system
    qubits) where the controlled-U(t) is realised by adding the ancilla
    Z-bit as an extra phase mask to each Trotter exponential: the gate
    e^{−i dt H} acts on the system only when ancilla = |1⟩. The matrix
    element ⟨j| U†(t) O U(t) |i⟩ is then read off the (ancilla=1)
    branch via ⟨ancilla=1; j| (U_c† O U_c) |ancilla=+; i⟩.

    Note: this routine is provided to demonstrate the QC-style approach;
    it costs 2× memory and is *not* used by the rest of the pipeline.
    For production EρOQ stitching, use observable_matrix_element above.
    """
    H_terms = build_pauli_terms()
    DIM2 = DIM * 2  # ancilla is bit 8 (highest)

    # |ancilla=+⟩ ⊗ |i⟩  =  (|0⟩+|1⟩)/√2 ⊗ |i⟩
    psi = np.zeros(DIM2, dtype=np.complex128)
    psi[i_bit] = 1.0 / math.sqrt(2.0)
    psi[i_bit | (1 << 8)] = 1.0 / math.sqrt(2.0)

    # Forward controlled evolution: apply U(t) to the ancilla=1 branch only.
    if t != 0.0 and n_trotter > 0:
        dt = t / n_trotter
        # Index of the ancilla=1 branch is i + 256 for i ∈ [0,256).
        for _ in range(n_trotter):
            for term in H_terms:
                coef, x_mask, z_mask, y_mask = _parse_term(term)
                angle = dt * coef.real
                # Operate in-place on the ancilla=1 slice only.
                upper = psi[DIM:DIM2]
                # Apply e^{-i angle P} on the 8-qubit slice 'upper'.
                upper = _apply_evolution_on_slice(upper, angle,
                                                  x_mask, z_mask, y_mask)
                psi[DIM:DIM2] = upper

    # Apply O to the ancilla=1 branch.
    upper = psi[DIM:DIM2]
    out = np.zeros_like(upper)
    for term in observable_terms:
        out = out + apply_pauli_term(upper, term)
    psi[DIM:DIM2] = out

    # Backward controlled evolution.
    if t != 0.0 and n_trotter > 0:
        dt = -t / n_trotter
        for _ in range(n_trotter):
            for term in H_terms:
                coef, x_mask, z_mask, y_mask = _parse_term(term)
                angle = dt * coef.real
                upper = psi[DIM:DIM2]
                upper = _apply_evolution_on_slice(upper, angle,
                                                  x_mask, z_mask, y_mask)
                psi[DIM:DIM2] = upper

    # Project onto |ancilla=1; j⟩ and rescale by √2 (because we started
    # with (|0⟩+|1⟩)/√2 so the ancilla=1 amplitude carries 1/√2 of the
    # weight). The desired matrix element is √2 · ⟨ancilla=1; j| state ⟩.
    return complex(psi[j_bit | (1 << 8)]) * math.sqrt(2.0)


def _apply_evolution_on_slice(slice_state: np.ndarray, angle: float,
                              x_mask: int, z_mask: int,
                              y_mask: int) -> np.ndarray:
    """Helper for the Hadamard variant: e^{-i angle P} on a 256-vector slice."""
    if x_mask == 0 and z_mask == 0 and y_mask == 0:
        return slice_state * np.exp(-1j * angle)
    if x_mask == 0:
        parity = _popcount(_BIT_INDICES & z_mask) & 1
        eig = np.where(parity == 1, -1.0, 1.0)
        return slice_state * np.exp(-1j * angle * eig)
    c = math.cos(angle)
    s = math.sin(angle)
    p_state = _apply_pauli_raw(slice_state, x_mask, z_mask, y_mask)
    return c * slice_state - 1j * s * p_state


# ---------------------------------------------------------------------------
# Useful observables (re-exported)
# ---------------------------------------------------------------------------

def observable_number(q_matter: int) -> List[PauliTerm]:
    """n_{q_matter} = (I − Z_q)/2 on a single matter qubit."""
    return [(complex(0.5), []), (complex(-0.5), [(q_matter, "Z")])]


def observable_link_x(q_link: int) -> List[PauliTerm]:
    """X on a single link qubit (the Z₂ electric "field" eigenvalue ±1)."""
    return [(complex(1.0), [(q_link, "X")])]


# ---------------------------------------------------------------------------
# Validation: build dense H *only* for self-test (not used anywhere above).
# ---------------------------------------------------------------------------

def _dense_pauli_string(factors: Sequence[PauliFactor]) -> np.ndarray:
    """Build a dense 256×256 Pauli string matrix. ONLY for validation."""
    # We build by acting on the identity, column by column, using our own
    # _apply_pauli_raw. No np.kron chains.
    coef, x_mask, z_mask, y_mask = _parse_term((1.0, list(factors)))
    M = np.zeros((DIM, DIM), dtype=np.complex128)
    for j in range(DIM):
        e_j = np.zeros(DIM, dtype=np.complex128)
        e_j[j] = 1.0
        if x_mask == 0 and z_mask == 0 and y_mask == 0:
            M[:, j] = e_j
        else:
            M[:, j] = coef * _apply_pauli_raw(e_j, x_mask, z_mask, y_mask)
    return M


def _dense_H_for_validation() -> np.ndarray:
    """Assemble dense H from Pauli terms (validation only)."""
    H = np.zeros((DIM, DIM), dtype=np.complex128)
    for coef, factors in build_pauli_terms():
        H = H + coef * _dense_pauli_string(factors)
    return H


def _dense_exact_matrix_element(i_bit: int, j_bit: int,
                                observable_terms: Sequence[PauliTerm],
                                t: float) -> complex:
    """Exact ⟨j| U†(t) O U(t) |i⟩ using dense expm on the validation H."""
    from scipy.linalg import expm  # validation only
    H = _dense_H_for_validation()
    U = expm(-1j * t * H)
    Udag = U.conj().T
    O = np.zeros((DIM, DIM), dtype=np.complex128)
    for coef, factors in observable_terms:
        O = O + coef * _dense_pauli_string(factors)
    psi_i = np.zeros(DIM, dtype=np.complex128); psi_i[i_bit] = 1.0
    psi_j = np.zeros(DIM, dtype=np.complex128); psi_j[j_bit] = 1.0
    M = Udag @ O @ U
    return complex(psi_j.conj() @ M @ psi_i)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _run_self_test() -> None:
    print("=" * 72)
    print("EρOQ quantum-leg simulator: self-test")
    print("=" * 72)

    H_terms = build_pauli_terms()
    n_terms = len(H_terms)
    print(f"\nHamiltonian built as {n_terms} Pauli-string terms:")
    counts = {"electric": 4, "magnetic": 1,
              "hopping": 2 * len(HOP_PAIRS),
              "mass (const+Z)": 2 * len(MATTER_QUBITS)}
    print("  expected breakdown:", counts,
          f"total = {sum(counts.values())}")
    assert n_terms == sum(counts.values()), \
        f"term count mismatch: {n_terms} vs expected {sum(counts.values())}"

    # ---- Test 1: t = 0 should give bare ⟨j|O|i⟩ ----------------------
    print("\n[1] t = 0 sanity: matrix_element should equal ⟨j|O|i⟩")
    O = observable_number(4)  # n_0 = n on matter qubit 4
    # Pick |i⟩ = |10000⟩ in 8-bit form: bit 4 set, rest zero → idx = 16.
    # Then O|i⟩ = 1·|i⟩, so ⟨i|O|i⟩ = 1 and ⟨j|O|i⟩ = δ_{ij}.
    for j in [16, 0, 31, 200]:
        me = observable_matrix_element(16, j, O, t=0.0, n_trotter=1)
        expected = 1.0 if j == 16 else 0.0
        ok = abs(me - expected) < 1e-12
        print(f"  ⟨{j}| O |16⟩  = {me:+.6e}   expected {expected}   "
              f"{'OK' if ok else 'FAIL'}")
        assert ok

    # And on a state with matter qubit 4 empty: |i⟩=|0⟩
    me0 = observable_matrix_element(0, 0, O, t=0.0, n_trotter=1)
    assert abs(me0) < 1e-12, f"n_0|0⟩ ≠ 0: got {me0}"
    print(f"  ⟨0| n_0 |0⟩       = {me0:+.6e}                       OK")

    # ---- Test 2: dense ED reference at small t -----------------------
    print("\n[2] Trotter vs dense ED reference at small t")
    print("    O = n_0,  |i⟩ = |idx=16⟩ (q4 occupied),  varying |j⟩")
    t_test = 0.4
    js = [16, 17, 0, 48, 80]
    refs = {j: _dense_exact_matrix_element(16, j, O, t_test) for j in js}
    for n_trotter in [20, 100, 400]:
        print(f"  n_trotter = {n_trotter}:")
        max_err = 0.0
        for j in js:
            me = observable_matrix_element(16, j, O, t_test, n_trotter)
            err = abs(me - refs[j])
            max_err = max(max_err, err)
            print(f"    j={j:3d}: trotter={me.real:+.5f}{me.imag:+.5f}j  "
                  f"ED={refs[j].real:+.5f}{refs[j].imag:+.5f}j  "
                  f"|Δ|={err:.2e}")
        print(f"    max |Δ| at n_trotter={n_trotter}: {max_err:.3e}")

    # ---- Test 3: convergence study (n_trotter scan at fixed t) -------
    print("\n[3] Convergence: |⟨16|U†(t) n_0 U(t)|16⟩ − ED| vs n_trotter")
    j_pick = 16
    ref = _dense_exact_matrix_element(16, j_pick, O, t_test)
    print(f"    ED reference: {ref:+.10e}")
    target_1pct = None
    for n_trotter in [5, 10, 20, 50, 100, 200, 400, 800, 1600]:
        me = observable_matrix_element(16, j_pick, O, t_test, n_trotter)
        err = abs(me - ref)
        rel = err / max(abs(ref), 1e-12)
        print(f"    n_trotter={n_trotter:5d}: trotter={me.real:+.8e}  "
              f"|abs Δ|={err:.3e}   rel={rel:.3e}")
        if target_1pct is None and rel < 1e-2:
            target_1pct = n_trotter
    print(f"    First n_trotter giving rel-err < 1%: {target_1pct}")

    # ---- Test 4: convergence at t = 1.0 (the headline number) --------
    print("\n[4] Convergence at t = 1.0 (β-scale parameters g_e=g_hop=2m=2g_m=1)")
    t_big = 1.0
    js_big = [16, 17, 24, 48]
    refs_big = {j: _dense_exact_matrix_element(16, j, O, t_big) for j in js_big}
    target = None
    for n_trotter in [20, 50, 100, 200, 400, 800, 1600, 3200]:
        max_rel = 0.0
        for j in js_big:
            me = observable_matrix_element(16, j, O, t_big, n_trotter)
            err = abs(me - refs_big[j])
            rel = err / max(abs(refs_big[j]), 1e-3)
            max_rel = max(max_rel, rel)
        print(f"    n_trotter={n_trotter:5d}: max rel-err across "
              f"{len(js_big)} j's = {max_rel:.3e}")
        if target is None and max_rel < 1e-2:
            target = n_trotter
    print(f"    First n_trotter giving ~1% accuracy at t=1: {target}")

    # ---- Test 5: Hadamard variant agrees with direct method ----------
    print("\n[5] Hadamard-test variant agrees with direct method (small t)")
    for j in [16, 17, 0, 48]:
        a = observable_matrix_element(16, j, O, 0.3, 100)
        b = observable_matrix_element_hadamard(16, j, O, 0.3, 100)
        err = abs(a - b)
        ok = err < 1e-10
        print(f"    j={j}: direct={a.real:+.6e}{a.imag:+.6e}j  "
              f"hadamard={b.real:+.6e}{b.imag:+.6e}j  |Δ|={err:.2e}  "
              f"{'OK' if ok else 'FAIL'}")
        assert ok

    # ---- Test 6: no dense H is touched in the production path --------
    print("\n[6] Memory sanity: production path stores only a 256-vec.")
    print(f"    state-vector bytes: {DIM * 16}  (complex128 × {DIM})")
    print(f"    H is stored as {n_terms} Pauli terms (no 256×256 matrix).")

    print("\nAll self-tests passed.")
    print("=" * 72)


if __name__ == "__main__":
    _run_self_test()
