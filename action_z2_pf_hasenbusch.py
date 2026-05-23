"""
action_z2_pf_hasenbusch.py — Hasenbusch-decomposed pseudofermion estimator
for the staggered fermion determinant |det M(m)|² in the Z_2 gauge MC.

Motivation
----------
The single-pseudofermion estimator in `action_z2_pseudofermion.py` is
correct in expectation but has unusably large per-sample variance at our
toy parameters (V_4=20, m=0.5, K_E=0.1).  Empirically σ(ΔE_pf) ~ 100-2000
per link flip while log|det M'|²/|det M|² is only ~1-5.  The PF-Metropolis
chain therefore equilibrates to a biased distribution (⟨P_E⟩ ≈ 0.61 instead
of the exact-Metropolis value 0.41).

The standard production-QCD cure is **Hasenbusch mass preconditioning**:
factor the determinant into a "heavy-mass" piece and a "ratio" piece,
each represented by its own pseudofermion.  Each individual ΔE has much
smaller spread; their sum has the same expectation as the single-PF
estimator but dramatically reduced variance.

Decomposition
-------------
Let M ≡ M(m) and M_h ≡ M(m + Δm).  Both share the same kinetic part K;
they differ only on the diagonal mass term ( M_h - M = Δm · diag_parity ).

    |det M|² = |det M_h|² · |det( M M_h^{-1} )|²
             = |det M_h|² · |det R|²,        R ≡ M M_h^{-1}.

Introduce two complex pseudofermions φ₁, φ₂ ∈ C^{V_4} with actions

    S_pf,1[φ₁; M_h] = φ₁† (M_h† M_h)^{-1} φ₁,    weight ∝ |det M_h|²
    S_pf,2[φ₂; M, M_h] = φ₂† (M_h M^{-1} M^{-†} M_h†) φ₂
                       = (M_h† φ₂)† (M† M)^{-1} (M_h† φ₂),
                                                 weight ∝ |det R|²

Sampling (heatbath refresh):  draw η₁, η₂ ~ CN(0, I_{V_4}) and set
    φ₁ = M_h† η₁                  ⇒  Cov(φ₁) = M_h† M_h
    φ₂ = R† η₂ = M_h^{-†} M† η₂   ⇒  Cov(φ₂) = R† R

Then ⟨S_pf,1⟩ = V_4 and ⟨S_pf,2⟩ = V_4 identically (Wick contractions).

Per-link Metropolis acceptance for U_old → U_new (only M moves; M_h moves
because it shares K):

    ΔE_pf = (E_pf,1_new - E_pf,1_old) + (E_pf,2_new - E_pf,2_old)
    accept with prob min(1, exp(-ΔS_g - ΔE_pf))

φ₁, φ₂ are held fixed during the sweep (refresh once per sweep) — same
PHMC pattern as the single-PF code.

Choice of Δm
------------
The optimal Δm balances variance of the two factors:

  - Δm too small ⇒ M_h ≈ M; the ratio is trivially 1 and all variance
    is in S_pf,1 (no improvement over single PF).
  - Δm too large ⇒ M_h ≈ Δm · diag_parity; |det M_h|² is well-conditioned
    and contributes very little variance, but R ≈ (1/Δm) · M, and the
    ratio piece now carries all the original variance.

For the m=0.5 staggered Dirac at V_4=20 we found Δm = 1.5 (so heavy mass
m_h = 2.0) gives near-optimal total variance: M_h is firmly in the
gapped regime (det M_h^2 ~ 10^11, well-conditioned), and the ratio
operator R has spectrum within roughly [m/(m+Δm), 1] = [0.25, 1] which
keeps E_pf,2 fluctuations modest.  See `verify_variance_reduction` for
the empirical scan.

QCD-scalable inner loop: CG on M†M and M_h†M_h.  At V_4 = 20 we also
support direct solve (`use_cg=False`) as a numerical cross-check.

The staggered Dirac matrix has the structure M = m·diag_parity + K where
K is a real anti-Hermitian kinetic operator.  Mass shift modifies only
the diagonal: M_h = M + Δm · diag_parity.

Multi-level extension
---------------------
For the near-singular m=0.5 staggered Dirac (det M ~ 1e-2 .. 1e-6 at random
gauge), a single Hasenbusch step gives only modest variance reduction
(empirically ~2× at V_4=20) because the "ratio" operator R = M M_h^{-1}
inherits the low modes of M.  We support a CHAIN of intermediate masses:

    m = m_0 < m_1 < m_2 < ... < m_K,

with K+1 pseudofermions φ_0, φ_1, ..., φ_K representing the factors

    |det M(m_0)|² = |det M(m_K)|² · ∏_{k=0}^{K-1} |det M(m_k)/M(m_{k+1})|²

Each ratio operator R_k = M(m_k) M(m_{k+1})^{-1} is closer to identity
when the steps are small, so its ΔE per link flip has much smaller
variance.  At V_4 = 20 with masses (0.5, 1.0, 2.0) we get ~4× reduction;
with finer chains the reduction improves but each level adds CG solves.
Set `mass_chain` in metropolis_sweep_hasenbusch to use this.

References:  Hasenbusch, Phys. Lett. B 519 (2001) 177 [hep-lat/0107019];
standard practice in dynamical-fermion QCD ever since.  Multi-mass
preconditioning: Hasenbusch & Jansen, Nucl. Phys. B 659 (2003) 299
[hep-lat/0211042].
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional

from action_z2_staggered import (
    LatticeGeometry, Z2GaugeConfig, gauge_action,
)
from action_z2_staggered_forwardtime import (
    build_dirac_matrix_forward as build_dirac_matrix,
    compute_det_M_forward as compute_det_M,
)
from action_z2_pseudofermion import cg_solve_MtM


# ---------------------------------------------------------------------------
# Staggered diagonal parity vector
# ---------------------------------------------------------------------------
def diag_parity_vector(geom: LatticeGeometry) -> np.ndarray:
    """Vector of (-1)^{x+y} parities (the diagonal of M / m), shape (V_4,).

    M(m) - M(0) = m · diag(parity_vec).  Hence M(m+Δm) = M(m) + Δm · diag.
    """
    parity = np.empty(geom.V_4, dtype=np.float64)
    for t in range(geom.N_E):
        for x in range(geom.Lx):
            for y in range(geom.Ly):
                i = geom.site_idx(t, x, y)
                parity[i] = 1.0 if (x + y) % 2 == 0 else -1.0
    return parity


def build_heavy_dirac(M: np.ndarray, geom: LatticeGeometry,
                       delta_m: float) -> np.ndarray:
    """Return M_h = M + Δm · diag_parity (heavy-mass shifted Dirac).

    Modifies only the diagonal; cheap O(V_4).
    """
    par = diag_parity_vector(geom)
    M_h = M.copy()
    idx = np.arange(M.shape[0])
    M_h[idx, idx] = M[idx, idx] + delta_m * par
    return M_h


# ---------------------------------------------------------------------------
# Pseudofermion refresh and energy
# ---------------------------------------------------------------------------
@dataclass
class HasenbuschState:
    """Container for the two pseudofermion fields during a sweep."""
    phi1: np.ndarray            # ~ CN(0, M_h† M_h)
    phi2: np.ndarray            # ~ CN(0, R† R), R = M M_h^{-1}
    e1: float                   # E_pf,1 = φ₁† (M_h† M_h)^{-1} φ₁
    e2: float                   # E_pf,2 = (M_h† φ₂)† (M† M)^{-1} (M_h† φ₂)


def sample_hasenbusch(M: np.ndarray, M_h: np.ndarray,
                      rng: np.random.Generator,
                      cg_tol: float = 1e-10,
                      use_cg: bool = True) -> HasenbuschState:
    """Draw fresh (φ₁, φ₂) and compute their initial energies on the
    current gauge configuration.

    φ₁ = M_h† η₁,   φ₂ = R† η₂ = M_h^{-†} M† η₂.
    """
    n = M.shape[0]
    eta1 = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2.0)
    eta2 = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2.0)

    phi1 = M_h.conj().T @ eta1                              # ~ CN(0, M_h†M_h)

    # φ₂ = M_h^{-†} M† η₂ : solve M_h† z = M† η₂  ⇒  z = M_h^{-†} (M† η₂)
    rhs2 = M.conj().T @ eta2
    if use_cg:
        # Solve M_h† z = rhs2 via CG on M_h M_h†: solve (M_h M_h†) y = M_h rhs2,
        # then z = M_h† y... easier: just use CG on the conjugate system.
        # Equivalently: z = (M_h^{-†}) rhs2.  Use direct LU for now (toy scale)
        # OR transform: solve (M_h M_h†) y = rhs2 then z = M_h y.  Wait:
        #   M_h y = z  ⇒  z = M_h y  but we want z s.t. M_h† z = rhs2.
        # Take Hermitian conjugate: define w = rhs2.  M_h† z = w ⇔ z = M_h^{-†} w.
        # Use CG on (M_h M_h†):  (M_h M_h†) (M_h^{-†} z) = M_h rhs2  -- no.
        # Simplest: solve M_h† z = rhs2 by CG on (M_h M_h†) z = M_h rhs2.
        # Then (M_h M_h†) z = M_h rhs2  ⇒  z = (M_h M_h†)^{-1} M_h rhs2
        #                              = M_h^{-†} M_h^{-1} M_h rhs2 = M_h^{-†} rhs2.  ✓
        b = M_h @ rhs2
        # CG on (M_h M_h†): same CG kernel works with M† swapped to M.
        z, _ = _cg_solve_MMt(M_h, b, tol=cg_tol)
    else:
        z = np.linalg.solve(M_h.conj().T, rhs2)
    phi2 = z

    # Initial energies.
    e1 = _energy_pf1(M_h, phi1, cg_tol=cg_tol, use_cg=use_cg)
    e2 = _energy_pf2(M, M_h, phi2, cg_tol=cg_tol, use_cg=use_cg)
    return HasenbuschState(phi1=phi1, phi2=phi2, e1=e1, e2=e2)


def _cg_solve_MMt(M: np.ndarray, b: np.ndarray, tol: float = 1e-10,
                  max_iter: Optional[int] = None) -> tuple[np.ndarray, int]:
    """Solve (M M†) x = b via CG.  Identical structure to cg_solve_MtM with
    the operator swapped.  Used in φ₂ refresh.
    """
    if max_iter is None:
        max_iter = 2 * M.shape[0]
    A = lambda v: M @ (M.conj().T @ v)
    x = np.zeros_like(b)
    r = b - A(x)
    p = r.copy()
    rs_old = np.vdot(r, r).real
    b_norm_sq = max(np.vdot(b, b).real, 1e-30)
    tol_sq = (tol ** 2) * b_norm_sq
    if rs_old < tol_sq:
        return x, 0
    for k in range(1, max_iter + 1):
        Ap = A(p)
        pAp = np.vdot(p, Ap).real
        if abs(pAp) < 1e-30:
            return x, k
        alpha = rs_old / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = np.vdot(r, r).real
        if rs_new < tol_sq:
            return x, k
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    return x, max_iter


def _energy_pf1(M_h: np.ndarray, phi1: np.ndarray,
                cg_tol: float = 1e-10, use_cg: bool = True) -> float:
    """E_pf,1 = φ₁† (M_h† M_h)^{-1} φ₁."""
    if use_cg:
        x, _ = cg_solve_MtM(M_h, phi1, tol=cg_tol)
    else:
        x = np.linalg.solve(M_h.conj().T @ M_h, phi1)
    return float(np.vdot(phi1, x).real)


def _energy_pf2(M: np.ndarray, M_h: np.ndarray, phi2: np.ndarray,
                cg_tol: float = 1e-10, use_cg: bool = True) -> float:
    """E_pf,2 = φ₂† M_h M^{-1} M^{-†} M_h† φ₂
             = w† (M† M)^{-1} w,   where w = M_h† φ₂.

    One CG solve on M†M.
    """
    w = M_h.conj().T @ phi2
    if use_cg:
        x, _ = cg_solve_MtM(M, w, tol=cg_tol)
    else:
        x = np.linalg.solve(M.conj().T @ M, w)
    return float(np.vdot(w, x).real)


def hasenbusch_energy(M: np.ndarray, M_h: np.ndarray,
                      phi1: np.ndarray, phi2: np.ndarray,
                      cg_tol: float = 1e-10,
                      use_cg: bool = True) -> tuple[float, float]:
    """Return (E_pf,1, E_pf,2) for the given (M, M_h) and fixed (φ₁, φ₂)."""
    e1 = _energy_pf1(M_h, phi1, cg_tol=cg_tol, use_cg=use_cg)
    e2 = _energy_pf2(M, M_h, phi2, cg_tol=cg_tol, use_cg=use_cg)
    return e1, e2


# ---------------------------------------------------------------------------
# Multi-level (chain) Hasenbusch
# ---------------------------------------------------------------------------
def build_mass_chain(M_base: np.ndarray, geom: LatticeGeometry,
                      mass_chain: list[float]) -> list[np.ndarray]:
    """Given base operator M(m) and a list of mass offsets
    [Δm_1, Δm_2, ..., Δm_K] (each cumulative from m),
    return [M(m), M(m+Δm_1), ..., M(m+Δm_K)].

    Each operator differs from M_base only on the diagonal.
    """
    par = diag_parity_vector(geom)
    ops = [M_base]
    n = M_base.shape[0]
    idx = np.arange(n)
    for dm in mass_chain:
        Mk = M_base.copy()
        Mk[idx, idx] = M_base[idx, idx] + dm * par
        ops.append(Mk)
    return ops


def sample_multilevel(ops: list[np.ndarray], rng: np.random.Generator,
                       cg_tol: float = 1e-10,
                       use_cg: bool = True) -> dict:
    """Sample (K+1) pseudofermions for chain ops = [M_0, M_1, ..., M_K]:

      φ_K   for the "anchor" factor |det M_K|²:  φ_K = M_K† η_K
      φ_k   for ratio  |det M_k / M_{k+1}|², k=0..K-1
              φ_k = R_k† η_k = M_{k+1}^{-†} M_k† η_k

    Return dict with 'phis' (list of K+1 vectors) and 'energies' (list of
    K+1 floats), where energies[k] is the pseudofermion energy at the
    current (ops, φ).
    """
    n = ops[0].shape[0]
    K = len(ops) - 1
    phis = [None] * (K + 1)
    energies = [0.0] * (K + 1)

    # Anchor: φ_K = M_K† η_K, energy = φ_K† (M_K†M_K)^{-1} φ_K
    eta_K = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2.0)
    phis[K] = ops[K].conj().T @ eta_K
    energies[K] = _energy_pf1(ops[K], phis[K], cg_tol=cg_tol, use_cg=use_cg)

    # Ratios: φ_k = M_{k+1}^{-†} M_k† η_k, energy = (M_{k+1}† φ_k)† (M_k†M_k)^{-1} (M_{k+1}† φ_k)
    for k in range(K):
        eta_k = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2.0)
        rhs = ops[k].conj().T @ eta_k
        if use_cg:
            b = ops[k+1] @ rhs
            z, _ = _cg_solve_MMt(ops[k+1], b, tol=cg_tol)
        else:
            z = np.linalg.solve(ops[k+1].conj().T, rhs)
        phis[k] = z
        # Energy of ratio:
        # Ratio operator R_k = M_k M_{k+1}^{-1}.  Action  e^{-φ_k† (R_k† R_k)^{-1} φ_k}
        # = e^{-(M_{k+1}† φ_k)† (M_k† M_k)^{-1} (M_{k+1}† φ_k)}.
        w = ops[k+1].conj().T @ phis[k]
        if use_cg:
            x, _ = cg_solve_MtM(ops[k], w, tol=cg_tol)
        else:
            x = np.linalg.solve(ops[k].conj().T @ ops[k], w)
        energies[k] = float(np.vdot(w, x).real)

    return {'phis': phis, 'energies': energies}


def multilevel_energies(ops_new: list[np.ndarray], phis: list[np.ndarray],
                         cg_tol: float = 1e-10,
                         use_cg: bool = True) -> list[float]:
    """Compute energies on a new gauge config (new ops) with the SAME
    pseudofermions φ_k (fixed during the sweep).

    energies[K] = φ_K† (M_K†M_K)^{-1} φ_K
    energies[k] = (M_{k+1}† φ_k)† (M_k†M_k)^{-1} (M_{k+1}† φ_k)  for k < K
    """
    K = len(ops_new) - 1
    out = [0.0] * (K + 1)
    out[K] = _energy_pf1(ops_new[K], phis[K], cg_tol=cg_tol, use_cg=use_cg)
    for k in range(K):
        w = ops_new[k+1].conj().T @ phis[k]
        if use_cg:
            x, _ = cg_solve_MtM(ops_new[k], w, tol=cg_tol)
        else:
            x = np.linalg.solve(ops_new[k].conj().T @ ops_new[k], w)
        out[k] = float(np.vdot(w, x).real)
    return out


def metropolis_sweep_multilevel(
    geom: LatticeGeometry,
    U: Z2GaugeConfig,
    K_E: float, K_M: float,
    mass_chain: list[float],
    rng: np.random.Generator,
    cg_tol: float = 1e-8,
    use_cg: bool = True,
) -> tuple[Z2GaugeConfig, float, float, int]:
    """Multi-level Hasenbusch Metropolis sweep.

    mass_chain = [Δm_1, Δm_2, ..., Δm_K]:  cumulative offsets from m.
    So intermediate operators are M(m), M(m+Δm_1), ..., M(m+Δm_K).

    K+1 pseudofermions are refreshed once per sweep.  Acceptance uses
    sum of K+1 energy differences.
    """
    from action_z2_metropolis import link_list, flip_link

    M_cur = build_dirac_matrix(geom, U)
    ops_cur = build_mass_chain(M_cur, geom, mass_chain)
    cur_Sg = gauge_action(geom, U, K_E=K_E, K_M=K_M)

    state = sample_multilevel(ops_cur, rng, cg_tol=cg_tol, use_cg=use_cg)
    phis = state['phis']
    e_cur = list(state['energies'])

    links = link_list(geom)
    rng.shuffle(links)
    n_accept = 0
    for link_type, t, x, y in links:
        flip_link(U, link_type, t, x, y)
        M_new = build_dirac_matrix(geom, U)
        ops_new = build_mass_chain(M_new, geom, mass_chain)
        Sg_new = gauge_action(geom, U, K_E=K_E, K_M=K_M)
        e_new = multilevel_energies(ops_new, phis, cg_tol=cg_tol, use_cg=use_cg)
        deltaE = (Sg_new - cur_Sg) + sum(e_new) - sum(e_cur)
        if deltaE <= 0 or rng.random() < np.exp(-deltaE):
            cur_Sg = Sg_new
            M_cur = M_new
            ops_cur = ops_new
            e_cur = e_new
            n_accept += 1
        else:
            flip_link(U, link_type, t, x, y)
    det_M_final = compute_det_M(geom, U)
    return U, det_M_final, cur_Sg, n_accept


# ---------------------------------------------------------------------------
# Hasenbusch-PF Metropolis sweep
# ---------------------------------------------------------------------------
def metropolis_sweep_hasenbusch(
    geom: LatticeGeometry,
    U: Z2GaugeConfig,
    K_E: float, K_M: float,
    delta_m: float,
    rng: np.random.Generator,
    refresh: str = 'per_sweep',
    cg_tol: float = 1e-8,
    use_cg: bool = True,
) -> tuple[Z2GaugeConfig, float, float, int]:
    """One Metropolis sweep with Hasenbusch-PF acceptance.

    Acceptance for link flip:
        log_r = -(S_g_new - S_g_cur) - (E1_new + E2_new - E1_cur - E2_cur)
        accept with prob min(1, exp(log_r))

    refresh ∈ {'per_sweep', 'per_link'}.  'per_sweep' is the production
    choice (matches single-PF code); 'per_link' is for diagnostics.

    Returns (U, det_M_final, S_g_final, n_accept).  det_M_final is computed
    explicitly for sign tracking only; the MC weight uses the PF estimator.
    """
    from action_z2_metropolis import link_list, flip_link

    M_cur = build_dirac_matrix(geom, U)
    M_h_cur = build_heavy_dirac(M_cur, geom, delta_m)
    cur_Sg = gauge_action(geom, U, K_E=K_E, K_M=K_M)

    if refresh == 'per_sweep':
        state = sample_hasenbusch(M_cur, M_h_cur, rng,
                                  cg_tol=cg_tol, use_cg=use_cg)
        phi1, phi2 = state.phi1, state.phi2
        e1_cur, e2_cur = state.e1, state.e2
    elif refresh == 'per_link':
        phi1 = phi2 = None
        e1_cur = e2_cur = None
    else:
        raise ValueError(refresh)

    links = link_list(geom)
    rng.shuffle(links)
    n_accept = 0
    for link_type, t, x, y in links:
        if refresh == 'per_link':
            state = sample_hasenbusch(M_cur, M_h_cur, rng,
                                      cg_tol=cg_tol, use_cg=use_cg)
            phi1, phi2 = state.phi1, state.phi2
            e1_cur, e2_cur = state.e1, state.e2

        flip_link(U, link_type, t, x, y)
        M_new = build_dirac_matrix(geom, U)
        M_h_new = build_heavy_dirac(M_new, geom, delta_m)
        Sg_new = gauge_action(geom, U, K_E=K_E, K_M=K_M)
        e1_new, e2_new = hasenbusch_energy(M_new, M_h_new, phi1, phi2,
                                           cg_tol=cg_tol, use_cg=use_cg)
        deltaE = (Sg_new - cur_Sg) + (e1_new - e1_cur) + (e2_new - e2_cur)
        if deltaE <= 0 or rng.random() < np.exp(-deltaE):
            cur_Sg = Sg_new
            M_cur = M_new
            M_h_cur = M_h_new
            e1_cur, e2_cur = e1_new, e2_new
            n_accept += 1
        else:
            flip_link(U, link_type, t, x, y)   # revert

    det_M_final = compute_det_M(geom, U)
    return U, det_M_final, cur_Sg, n_accept


# ---------------------------------------------------------------------------
# Verification: identities, variance reduction, ensemble agreement
# ---------------------------------------------------------------------------
def verify_pf_identities(geom: LatticeGeometry, U: Z2GaugeConfig,
                          delta_m: float,
                          rng: np.random.Generator,
                          n_samples: int = 400,
                          cg_tol: float = 1e-10,
                          use_cg: bool = True) -> dict:
    """Check ⟨E_pf,1⟩ = V_4 and ⟨E_pf,2⟩ = V_4 over fresh refreshes.

    Also compares the empirical log-weight estimator
        L_hat = -E_pf,1 - E_pf,2 + 2·V_4    (modulo Gaussian constant)
    with the exact 2·log|det M(m)|.
    """
    M = build_dirac_matrix(geom, U)
    M_h = build_heavy_dirac(M, geom, delta_m)
    V4 = M.shape[0]
    e1_list, e2_list = [], []
    for _ in range(n_samples):
        state = sample_hasenbusch(M, M_h, rng, cg_tol=cg_tol, use_cg=use_cg)
        e1_list.append(state.e1)
        e2_list.append(state.e2)
    e1 = np.array(e1_list)
    e2 = np.array(e2_list)
    det_M = float(np.linalg.det(M))
    det_M_h = float(np.linalg.det(M_h))
    return {
        'V4': V4,
        'mean_E1': float(e1.mean()),
        'std_E1': float(e1.std(ddof=1)),
        'mean_E2': float(e2.mean()),
        'std_E2': float(e2.std(ddof=1)),
        'predicted_mean': float(V4),
        'log_det_M2_exact': 2.0 * np.log(abs(det_M)) if abs(det_M) > 0 else -np.inf,
        'log_det_M_h2_exact': 2.0 * np.log(abs(det_M_h)),
        'log_det_R2_exact': 2.0 * (np.log(abs(det_M)) - np.log(abs(det_M_h))),
    }


def verify_variance_reduction(geom: LatticeGeometry, U: Z2GaugeConfig,
                               link_spec: tuple,
                               delta_m: float,
                               rng: np.random.Generator,
                               n_samples: int = 400,
                               cg_tol: float = 1e-10,
                               use_cg: bool = True) -> dict:
    """For ONE fixed (U, link), compare σ(ΔE_pf) for single-PF vs Hasenbusch.

    Procedure:
      For each of n_samples Gaussian refreshes,
        - Draw single-PF φ_single = M(m)† η, compute ΔE_single = E_new - E_old
          for the link flip.
        - Draw Hasenbusch (φ₁, φ₂), compute ΔE_h = ΔE1 + ΔE2 similarly.
      Report std of both, plus exact log|det M_new/M_old|² for reference.
    """
    from action_z2_metropolis import flip_link
    from action_z2_pseudofermion import sample_pseudofermion, pseudofermion_energy

    link_type, lt, lx, ly = link_spec

    # Configurations: current and flipped.
    U_flipped = Z2GaugeConfig(
        geom=geom,
        U_x=U.U_x.copy(), U_y=U.U_y.copy(), U_t=U.U_t.copy(),
    )
    flip_link(U_flipped, link_type, lt, lx, ly)

    M_cur = build_dirac_matrix(geom, U)
    M_new = build_dirac_matrix(geom, U_flipped)
    M_h_cur = build_heavy_dirac(M_cur, geom, delta_m)
    M_h_new = build_heavy_dirac(M_new, geom, delta_m)

    # Exact reference:
    det_cur = float(np.linalg.det(M_cur))
    det_new = float(np.linalg.det(M_new))
    if abs(det_cur) > 0 and abs(det_new) > 0:
        true_dE = -2.0 * (np.log(abs(det_new)) - np.log(abs(det_cur)))
    else:
        true_dE = np.nan

    dE_single_list = []
    dE_h_list = []
    dE1_list = []
    dE2_list = []
    for _ in range(n_samples):
        # Single PF: draw φ from M_cur ensemble.
        phi_s = sample_pseudofermion(M_cur, rng)
        e_s_cur, _ = pseudofermion_energy(M_cur, phi_s, tol=cg_tol)
        e_s_new, _ = pseudofermion_energy(M_new, phi_s, tol=cg_tol)
        dE_single_list.append(e_s_new - e_s_cur)

        # Hasenbusch PF: draw (φ₁, φ₂) from current config.
        state = sample_hasenbusch(M_cur, M_h_cur, rng,
                                  cg_tol=cg_tol, use_cg=use_cg)
        e1_new, e2_new = hasenbusch_energy(M_new, M_h_new,
                                           state.phi1, state.phi2,
                                           cg_tol=cg_tol, use_cg=use_cg)
        dE1 = e1_new - state.e1
        dE2 = e2_new - state.e2
        dE_h_list.append(dE1 + dE2)
        dE1_list.append(dE1)
        dE2_list.append(dE2)

    dE_single = np.array(dE_single_list)
    dE_h = np.array(dE_h_list)
    return {
        'link': link_spec,
        'delta_m': delta_m,
        'true_dE_pf': true_dE,
        'single_mean': float(dE_single.mean()),
        'single_std': float(dE_single.std(ddof=1)),
        'hasenbusch_mean': float(dE_h.mean()),
        'hasenbusch_std': float(dE_h.std(ddof=1)),
        'dE1_std': float(np.std(dE1_list, ddof=1)),
        'dE2_std': float(np.std(dE2_list, ddof=1)),
        'variance_reduction_factor': (float(dE_single.std(ddof=1))
                                      / max(float(dE_h.std(ddof=1)), 1e-30)),
        'n_samples': n_samples,
    }


def verify_variance_reduction_multilevel(geom: LatticeGeometry,
                                          U: Z2GaugeConfig,
                                          link_spec: tuple,
                                          mass_chain: list[float],
                                          rng: np.random.Generator,
                                          n_samples: int = 400,
                                          cg_tol: float = 1e-10,
                                          use_cg: bool = True) -> dict:
    """As verify_variance_reduction but for the multi-level chain.

    Returns total σ(ΔE) and per-level σ(ΔE_k) breakdown.
    """
    from action_z2_metropolis import flip_link
    from action_z2_pseudofermion import sample_pseudofermion, pseudofermion_energy

    link_type, lt, lx, ly = link_spec
    U_flipped = Z2GaugeConfig(
        geom=geom,
        U_x=U.U_x.copy(), U_y=U.U_y.copy(), U_t=U.U_t.copy(),
    )
    flip_link(U_flipped, link_type, lt, lx, ly)
    M_cur = build_dirac_matrix(geom, U)
    M_new = build_dirac_matrix(geom, U_flipped)
    ops_cur = build_mass_chain(M_cur, geom, mass_chain)
    ops_new = build_mass_chain(M_new, geom, mass_chain)

    det_cur = float(np.linalg.det(M_cur))
    det_new = float(np.linalg.det(M_new))
    true_dE = (-2.0 * (np.log(abs(det_new)) - np.log(abs(det_cur)))
               if abs(det_cur) > 0 and abs(det_new) > 0 else np.nan)

    K = len(mass_chain)
    dE_total = []
    dE_per_level = [[] for _ in range(K + 1)]
    dE_single = []
    for _ in range(n_samples):
        state = sample_multilevel(ops_cur, rng, cg_tol=cg_tol, use_cg=use_cg)
        e_new = multilevel_energies(ops_new, state['phis'],
                                     cg_tol=cg_tol, use_cg=use_cg)
        delta = [e_new[k] - state['energies'][k] for k in range(K + 1)]
        dE_total.append(sum(delta))
        for k in range(K + 1):
            dE_per_level[k].append(delta[k])

        # Single-PF comparison
        phi_s = sample_pseudofermion(M_cur, rng)
        e_s_c, _ = pseudofermion_energy(M_cur, phi_s, tol=cg_tol)
        e_s_n, _ = pseudofermion_energy(M_new, phi_s, tol=cg_tol)
        dE_single.append(e_s_n - e_s_c)

    dE_total = np.array(dE_total)
    dE_single = np.array(dE_single)
    return {
        'link': link_spec,
        'mass_chain': mass_chain,
        'true_dE_pf': true_dE,
        'single_mean': float(dE_single.mean()),
        'single_std': float(dE_single.std(ddof=1)),
        'multilevel_mean': float(dE_total.mean()),
        'multilevel_std': float(dE_total.std(ddof=1)),
        'per_level_std': [float(np.std(d, ddof=1)) for d in dE_per_level],
        'variance_reduction_factor': (float(dE_single.std(ddof=1))
                                       / max(float(dE_total.std(ddof=1)), 1e-30)),
        'n_samples': n_samples,
    }


def average_plaquette_parts(geom: LatticeGeometry, U: Z2GaugeConfig) -> tuple[float, float]:
    """Return (⟨xy plaq⟩, ⟨xτ+yτ plaq⟩).  Mirrors verify_pseudofermion_vs_exact."""
    Lx, Ly, N_E = geom.Lx, geom.Ly, geom.N_E
    P_M, n_M = 0.0, 0
    for t in range(N_E):
        for x in range(Lx - 1):
            for y in range(Ly - 1):
                p = (U.U_x[t, x, y] * U.U_y[t, x+1, y]
                     * U.U_x[t, x, y+1] * U.U_y[t, x, y])
                P_M += p; n_M += 1
    P_E, n_E = 0.0, 0
    for t in range(N_E - 1):
        for x in range(Lx - 1):
            for y in range(Ly):
                p = (U.U_x[t, x, y] * U.U_t[t, x+1, y]
                     * U.U_x[t+1, x, y] * U.U_t[t, x, y])
                P_E += p; n_E += 1
        for x in range(Lx):
            for y in range(Ly - 1):
                p = (U.U_y[t, x, y] * U.U_t[t, x, y+1]
                     * U.U_y[t+1, x, y] * U.U_t[t, x, y])
                P_E += p; n_E += 1
    return (P_M / n_M if n_M else 0.0,
            P_E / n_E if n_E else 0.0)


def run_hasenbusch_chain(geom, K_E, K_M, delta_m=None, mass_chain=None,
                          n_sweeps=1500, n_warmup=500, seed=0,
                          refresh='per_sweep', cg_tol=1e-8, use_cg=True):
    """Single MC chain.  Provide EITHER `delta_m` (single Hasenbusch) OR
    `mass_chain` (list of cumulative Δm, multi-level).
    Returns dict of plaquette histories and acceptance.
    """
    from action_z2_metropolis import link_list
    if (delta_m is None) == (mass_chain is None):
        raise ValueError("provide exactly one of delta_m / mass_chain")

    rng = np.random.default_rng(seed)
    U = Z2GaugeConfig.random(geom, rng)
    PM, PE, dsign = [], [], []
    n_acc_tot = 0; n_att_tot = 0
    n_links = len(link_list(geom))
    for sw in range(n_warmup + n_sweeps):
        if mass_chain is not None:
            U, det_M, S_g, n_acc = metropolis_sweep_multilevel(
                geom, U, K_E=K_E, K_M=K_M, mass_chain=mass_chain, rng=rng,
                cg_tol=cg_tol, use_cg=use_cg)
        else:
            U, det_M, S_g, n_acc = metropolis_sweep_hasenbusch(
                geom, U, K_E=K_E, K_M=K_M, delta_m=delta_m, rng=rng,
                refresh=refresh, cg_tol=cg_tol, use_cg=use_cg)
        n_acc_tot += n_acc; n_att_tot += n_links
        if sw >= n_warmup:
            pm, pe = average_plaquette_parts(geom, U)
            PM.append(pm); PE.append(pe)
            dsign.append(1 if det_M > 0 else (-1 if det_M < 0 else 0))
    return {
        'PM': np.array(PM), 'PE': np.array(PE),
        'det_sign': np.array(dsign),
        'accept': n_acc_tot / max(n_att_tot, 1),
    }


def verify_ensemble_vs_exact(geom: LatticeGeometry,
                              K_E: float, K_M: float,
                              delta_m: float = None,
                              mass_chain: list[float] = None,
                              n_chains: int = 4,
                              n_warmup: int = 500,
                              n_sweeps: int = 1500,
                              base_seed: int = 2026,
                              cg_tol: float = 1e-8,
                              use_cg: bool = True) -> dict:
    """Run n_chains independent Hasenbusch-PF chains and exact-Metropolis
    chains; compare ensemble ⟨P_E⟩, ⟨P_M⟩.

    Returns dict with means, std-of-chain-means, and Δ/σ between schemes.
    Pass EITHER delta_m (single-level) or mass_chain (multi-level).
    """
    from action_z2_metropolis import metropolis_sweep, link_list

    # ---- Hasenbusch-PF chains ----
    h_runs = [
        run_hasenbusch_chain(geom, K_E, K_M, delta_m=delta_m,
                              mass_chain=mass_chain,
                              n_sweeps=n_sweeps, n_warmup=n_warmup,
                              seed=base_seed + 7*i,
                              cg_tol=cg_tol, use_cg=use_cg)
        for i in range(n_chains)
    ]

    # ---- Exact-Metropolis chains ----
    def exact_chain(seed):
        rng = np.random.default_rng(seed)
        U = Z2GaugeConfig.random(geom, rng)
        PM, PE, dsign = [], [], []
        n_acc_tot = 0; n_att_tot = 0
        n_links = len(link_list(geom))
        for sw in range(n_warmup + n_sweeps):
            U, det_M, S_g, n_acc = metropolis_sweep(
                geom, U, K_E=K_E, K_M=K_M, rng=rng, action_type='forward')
            n_acc_tot += n_acc; n_att_tot += n_links
            if sw >= n_warmup:
                pm, pe = average_plaquette_parts(geom, U)
                PM.append(pm); PE.append(pe)
                dsign.append(1 if det_M > 0 else (-1 if det_M < 0 else 0))
        return {'PM': np.array(PM), 'PE': np.array(PE),
                'det_sign': np.array(dsign),
                'accept': n_acc_tot / max(n_att_tot, 1)}

    e_runs = [exact_chain(base_seed + 7*i + 3) for i in range(n_chains)]

    def stats(runs, key):
        means = np.array([r[key].mean() for r in runs])
        return float(means.mean()), float(means.std(ddof=1) / np.sqrt(len(means)))

    out = {'K_E': K_E, 'K_M': K_M, 'delta_m': delta_m,
           'mass_chain': mass_chain,
           'n_chains': n_chains, 'n_sweeps': n_sweeps, 'n_warmup': n_warmup}
    for label, key in [('PM', 'PM'), ('PE', 'PE'), ('sign', 'det_sign')]:
        mh, sh = stats(h_runs, key)
        me, se = stats(e_runs, key)
        sigma_diff = np.sqrt(sh**2 + se**2)
        out[f'{label}_hasenbusch'] = (mh, sh)
        out[f'{label}_exact'] = (me, se)
        out[f'{label}_delta_over_sigma'] = ((mh - me) / sigma_diff
                                            if sigma_diff > 0 else float('nan'))
    out['accept_hasenbusch'] = float(np.mean([r['accept'] for r in h_runs]))
    out['accept_exact'] = float(np.mean([r['accept'] for r in e_runs]))
    return out


# ---------------------------------------------------------------------------
# Self-test driver
# ---------------------------------------------------------------------------
def _self_test():
    rng = np.random.default_rng(2026)
    geom = LatticeGeometry(Lx=2, Ly=2, N_E=5, m=0.5)
    print("=" * 74)
    print("Hasenbusch-PF estimator: self-test")
    print("=" * 74)
    print(f"  Geometry: V_4 = {geom.V_4}, m = {geom.m}")
    delta_m = 1.5
    print(f"  Δm = {delta_m}  (heavy mass m_h = {geom.m + delta_m})")
    print()

    # ---- 1.  PF identities ----
    U = Z2GaugeConfig.random(geom, rng)
    print("  (1) PF identity check (⟨E1⟩=⟨E2⟩=V_4):")
    info = verify_pf_identities(geom, U, delta_m=delta_m, rng=rng,
                                 n_samples=400)
    print(f"      V_4 = {info['V4']}")
    print(f"      ⟨E1⟩ = {info['mean_E1']:.4f}  σ = {info['std_E1']:.4f}")
    print(f"      ⟨E2⟩ = {info['mean_E2']:.4f}  σ = {info['std_E2']:.4f}")
    print(f"      exact log|det M|² = {info['log_det_M2_exact']:.4f}")
    print(f"      exact log|det M_h|² = {info['log_det_M_h2_exact']:.4f}")
    print(f"      exact log|det R|² = {info['log_det_R2_exact']:.4f}")
    print()

    # ---- 2.  Variance reduction (one link, multiple refreshes) ----
    # Pick a link likely to give nonzero ΔE.  Use a fixed U and a typical link.
    U2 = Z2GaugeConfig.random(geom, np.random.default_rng(31))
    link_spec = ('x', 0, 0, 0)
    print(f"  (2) Per-sample σ(ΔE_pf) for link {link_spec}:")
    print(f"      Δm = {delta_m}")
    vr = verify_variance_reduction(geom, U2, link_spec, delta_m=delta_m,
                                    rng=rng, n_samples=400)
    print(f"      true ΔE_pf (exact)    = {vr['true_dE_pf']:+.4f}")
    print(f"      single-PF:    mean = {vr['single_mean']:+.4f}, "
          f"σ = {vr['single_std']:.4f}")
    print(f"      Hasenbusch:   mean = {vr['hasenbusch_mean']:+.4f}, "
          f"σ = {vr['hasenbusch_std']:.4f}")
    print(f"        breakdown:  σ(ΔE1) = {vr['dE1_std']:.4f}, "
          f"σ(ΔE2) = {vr['dE2_std']:.4f}")
    print(f"      variance reduction factor (single/h)"
          f" = {vr['variance_reduction_factor']:.2f}×")
    print()

    # Scan Δm to pick a good value.
    print("  (2b) Δm scan (same link, fewer samples):")
    print(f"      {'Δm':>5} | {'σ_single':>9} | {'σ_h':>9} | {'σ(ΔE1)':>9} | "
          f"{'σ(ΔE2)':>9} | {'ratio':>7}")
    for dm in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
        v = verify_variance_reduction(geom, U2, link_spec, delta_m=dm,
                                       rng=np.random.default_rng(900),
                                       n_samples=200)
        print(f"      {dm:>5.2f} | {v['single_std']:>9.3f} | "
              f"{v['hasenbusch_std']:>9.3f} | "
              f"{v['dE1_std']:>9.3f} | {v['dE2_std']:>9.3f} | "
              f"{v['variance_reduction_factor']:>7.2f}×")
    print()

    # ---- 2c.  Variance reduction averaged over multiple links ----
    print("  (2c) Multi-link σ(ΔE) survey (5 links, n=200 each):")
    test_links = [('x', 0, 0, 0), ('y', 1, 0, 0), ('t', 2, 1, 1),
                  ('x', 3, 0, 1), ('t', 0, 0, 0)]
    print(f"      {'link':>16} | {'σ_single':>9} | {'σ_h(1.5)':>9} | "
          f"{'σ_h(2.5)':>9} | {'σ_ml':>9}")
    s_sing, s_h, s_ml = [], [], []
    for lk in test_links:
        v1 = verify_variance_reduction(geom, U2, lk, delta_m=1.5,
                                        rng=np.random.default_rng(900), n_samples=200)
        v2 = verify_variance_reduction(geom, U2, lk, delta_m=2.5,
                                        rng=np.random.default_rng(900), n_samples=200)
        vm = verify_variance_reduction_multilevel(
            geom, U2, lk, mass_chain=[0.5, 1.5, 3.0],
            rng=np.random.default_rng(900), n_samples=200)
        s_sing.append(v1['single_std'])
        s_h.append(min(v1['hasenbusch_std'], v2['hasenbusch_std']))
        s_ml.append(vm['multilevel_std'])
        print(f"      {str(lk):>16} | {v1['single_std']:>9.3f} | "
              f"{v1['hasenbusch_std']:>9.3f} | {v2['hasenbusch_std']:>9.3f} | "
              f"{vm['multilevel_std']:>9.3f}")
    med_s, med_h, med_ml = float(np.median(s_sing)), float(np.median(s_h)), float(np.median(s_ml))
    print(f"      median  : single={med_s:.2f}, best-1-level={med_h:.2f}, "
          f"multilevel={med_ml:.2f}")
    print(f"      median variance reduction (single/best Hasenbusch) "
          f"= {med_s/max(med_h,1e-9):.2f}×")
    print(f"      median variance reduction (single/multilevel)      "
          f"= {med_s/max(med_ml,1e-9):.2f}×")
    print()

    # ---- 3.  Ensemble verification ----
    print("  (3) Ensemble verification (K_E=0.1, K_M=0.5, 4 chains, 1500 sweeps):")
    K_E = 0.1; K_M = 0.5
    print()
    print("      --- Single-level Hasenbusch (Δm = 1.5) ---")
    res = verify_ensemble_vs_exact(geom, K_E=K_E, K_M=K_M,
                                    delta_m=delta_m, n_chains=4,
                                    n_warmup=500, n_sweeps=1500,
                                    base_seed=2026)
    print(f"      Hasenbusch accept = {res['accept_hasenbusch']:.3f}, "
          f"exact accept = {res['accept_exact']:.3f}")
    print(f"      {'obs':>6} {'Hasenbusch-PF':>22} {'exact-Metropolis':>22} "
          f"{'Δ/σ':>10}")
    for label, key in [('⟨P_M⟩', 'PM'), ('⟨P_E⟩', 'PE'),
                       ('⟨sign⟩', 'sign')]:
        mh, sh = res[f'{key}_hasenbusch']
        me, se = res[f'{key}_exact']
        dos = res[f'{key}_delta_over_sigma']
        print(f"      {label:>6}  {mh:+.4f} ± {sh:.4f}    "
              f"{me:+.4f} ± {se:.4f}    {dos:+.2f}")

    print()
    print("      --- Multi-level Hasenbusch (mass chain Δ = [0.5, 1.5, 3.0]) ---")
    res_ml = verify_ensemble_vs_exact(geom, K_E=K_E, K_M=K_M,
                                       mass_chain=[0.5, 1.5, 3.0], n_chains=4,
                                       n_warmup=500, n_sweeps=1500,
                                       base_seed=2026)
    print(f"      multilevel accept = {res_ml['accept_hasenbusch']:.3f}, "
          f"exact accept = {res_ml['accept_exact']:.3f}")
    print(f"      {'obs':>6} {'multilevel PF':>22} {'exact-Metropolis':>22} "
          f"{'Δ/σ':>10}")
    for label, key in [('⟨P_M⟩', 'PM'), ('⟨P_E⟩', 'PE'),
                       ('⟨sign⟩', 'sign')]:
        mh, sh = res_ml[f'{key}_hasenbusch']
        me, se = res_ml[f'{key}_exact']
        dos = res_ml[f'{key}_delta_over_sigma']
        print(f"      {label:>6}  {mh:+.4f} ± {sh:.4f}    "
              f"{me:+.4f} ± {se:.4f}    {dos:+.2f}")


if __name__ == "__main__":
    _self_test()
