"""
action_z2_pseudofermion.py — pseudofermion stochastic estimator for the
staggered fermion determinant |det M|² in the Z_2 gauge MC.

The QCD-scalability piece: replace explicit O(V_4³) det M evaluation with
O(V_4 · N_CG) pseudofermion accept-reject step.  Implements the standard
PHMC-style pseudofermion accept-reject:

    1. At pseudofermion REFRESH: sample η ~ N(0, I) of dim V_4 (complex),
       set φ = M(U) · η so that φ has Gaussian weight ∝ e^{-φ† (M†M)^{-1} φ}.
    2. For each Metropolis trial (link flip):
       ΔE_pf = φ† [(M_new† M_new)^{-1} − (M_old† M_old)^{-1}] φ
       Accept with probability min(1, exp(-ΔS_g − ΔE_pf)).
    3. Refresh φ periodically (e.g., once per sweep).

The two (M†M)^{-1} φ solves are done via conjugate gradient — the
QCD-scalable inner loop.

At toy V_4 ≈ 20-65, direct solve via np.linalg.solve is also fine and
serves as a cross-check; we use CG to demonstrate the scalable algorithm.

Sign tracking:  det M is REAL for our staggered action (anti-Hermitian
kinetic + Hermitian mass).  We still record sign(det M) separately for
the reweighting in the observable ratio estimator.  |det M|² is the MC
weight; the sign is observable-side.
"""
from __future__ import annotations
import numpy as np
from typing import Callable

from action_z2_staggered import LatticeGeometry, Z2GaugeConfig
from action_z2_staggered_forwardtime import (
    build_dirac_matrix_forward as build_dirac_matrix,
    compute_det_M_forward as compute_det_M,
)


# ---------------------------------------------------------------------------
# Conjugate gradient on M† M
# ---------------------------------------------------------------------------
def cg_solve_MtM(M: np.ndarray, b: np.ndarray, tol: float = 1e-10,
                  max_iter: int = None) -> tuple[np.ndarray, int]:
    """Solve (M† M) x = b via conjugate gradient.

    Returns (x, n_iter).  Suitable for sparse-ish M of moderate size.
    For our V_4 ≤ 65 toy this typically converges in ≤ 30 iterations.
    """
    if max_iter is None:
        max_iter = 2 * M.shape[0]
    Mt = M.conj().T
    A = lambda v: Mt @ (M @ v)        # operator M† M · v
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
            return x, k                  # stagnation / exact solution
        alpha = rs_old / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = np.vdot(r, r).real
        if rs_new < tol_sq:
            return x, k
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    return x, max_iter


# ---------------------------------------------------------------------------
# Pseudofermion sampling and energy
# ---------------------------------------------------------------------------
def sample_pseudofermion(M: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """φ = M† · η where η ~ CN(0, I).  Then Cov(φ) = M† M, matching the PHMC
    action  e^{-φ† (M†M)^{-1} φ}.  Averaging the Gaussian over φ ~ CN(0, M†M)
    reproduces |det M|² (the fermion determinant up to normalisation).

    Consistency check:  ⟨φ† (M†M)^{-1} φ⟩ = Tr((M†M)^{-1} · M†M) = V_4.
    """
    n = M.shape[0]
    # Complex normal: real and imaginary parts each ~ N(0, 1/√2) so that
    # ⟨|η_k|²⟩ = 1.
    eta = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2.0)
    return M.conj().T @ eta


def pseudofermion_energy(M: np.ndarray, phi: np.ndarray,
                          tol: float = 1e-10) -> tuple[float, int]:
    """E_pf = φ† (M† M)^{-1} φ.  Returns (E_pf, n_cg_iter)."""
    x, n_iter = cg_solve_MtM(M, phi, tol=tol)
    e = float(np.vdot(phi, x).real)
    return e, n_iter


# ---------------------------------------------------------------------------
# Sanity check vs explicit |det M|²
# ---------------------------------------------------------------------------
def verify_pseudofermion_estimator(geom: LatticeGeometry, U: Z2GaugeConfig,
                                    rng: np.random.Generator,
                                    n_samples: int = 200,
                                    tol: float = 1e-10):
    """Empirical check: average (V_4 · log π − E_pf − log|det M|²) over PF
    samples should be 0 (Gaussian normalisation cancels exactly).

    Reports the per-sample distribution of  E_pf vs ⟨E_pf⟩ = V_4.
    """
    M = build_dirac_matrix(geom, U)
    V4 = M.shape[0]
    det_M = float(np.linalg.det(M))
    log_det_M2 = 2.0 * np.log(abs(det_M))
    energies = []
    iters = []
    for _ in range(n_samples):
        phi = sample_pseudofermion(M, rng)
        e, n_it = pseudofermion_energy(M, phi, tol=tol)
        energies.append(e)
        iters.append(n_it)
    energies = np.array(energies)
    return {
        'V4': V4,
        'det_M_exact': det_M,
        'log_det_M2_exact': log_det_M2,
        'mean_E_pf': float(energies.mean()),
        'std_E_pf': float(energies.std(ddof=1)),
        'predicted_mean_E_pf': float(V4),  # ⟨φ†(M†M)^{-1}φ⟩ for φ=M·η, η~CN(0,I) is V_4
        'cg_iters_mean': float(np.mean(iters)),
        'cg_iters_max': int(max(iters)),
    }


# ---------------------------------------------------------------------------
# Pseudofermion Metropolis sweep
# ---------------------------------------------------------------------------
def heatbath_sweep_pf(
    geom: LatticeGeometry,
    U: Z2GaugeConfig,
    K_E: float, K_M: float,
    rng: np.random.Generator,
    refresh: str = 'per_sweep',
    cg_tol: float = 1e-8,
) -> tuple[Z2GaugeConfig, float, float, int]:
    """One Metropolis sweep over all links using pseudofermion-estimated
    fermion determinant ratio in the acceptance.

    refresh:
      'per_sweep' — sample φ once at start of sweep, reuse for all link tries.
      'per_link'  — sample fresh φ before each link try (most accurate,
                    slowest; useful as a cross-check).

    Acceptance for a link flip U_old → U_new:
       weight ratio = exp(−ΔS_g − ΔE_pf)
       ΔE_pf = φ† (M_new† M_new)^{-1} φ − φ† (M_old† M_old)^{-1} φ
    Each Δ is one CG solve on the trial config.

    Returns (U_after_sweep, det_M_after, S_g_after, n_accepted).  det_M is
    computed explicitly for sign-of-det tracking only; the MC weight uses
    the PF estimator.
    """
    from action_z2_staggered import gauge_action
    from action_z2_metropolis import link_list, flip_link

    cur_Sg = gauge_action(geom, U, K_E=K_E, K_M=K_M)
    M_cur = build_dirac_matrix(geom, U)
    if refresh == 'per_sweep':
        phi = sample_pseudofermion(M_cur, rng)
        e_pf_cur, _ = pseudofermion_energy(M_cur, phi, tol=cg_tol)
    elif refresh == 'per_link':
        phi = None    # will refresh inside loop
        e_pf_cur = None
    else:
        raise ValueError(refresh)

    links = link_list(geom)
    rng.shuffle(links)
    n_accept = 0
    for link_type, t, x, y in links:
        if refresh == 'per_link':
            phi = sample_pseudofermion(M_cur, rng)
            e_pf_cur, _ = pseudofermion_energy(M_cur, phi, tol=cg_tol)

        flip_link(U, link_type, t, x, y)
        M_new = build_dirac_matrix(geom, U)
        Sg_new = gauge_action(geom, U, K_E=K_E, K_M=K_M)
        e_pf_new, _ = pseudofermion_energy(M_new, phi, tol=cg_tol)

        deltaE = (Sg_new - cur_Sg) + (e_pf_new - e_pf_cur)
        if deltaE < 0 or rng.random() < np.exp(-deltaE):
            cur_Sg = Sg_new
            e_pf_cur = e_pf_new
            M_cur = M_new
            n_accept += 1
        else:
            flip_link(U, link_type, t, x, y)   # revert

    det_M_final = compute_det_M(geom, U)
    return U, det_M_final, cur_Sg, n_accept


def _self_test():
    rng = np.random.default_rng(2026)
    print("=" * 70)
    print("Pseudofermion estimator self-test")
    print("=" * 70)
    geom = LatticeGeometry(Lx=2, Ly=2, N_E=5, m=0.5)
    U = Z2GaugeConfig.random(geom, rng)
    print(f"  V_4 = {geom.V_4}, m = {geom.m}, gauge = random Z_2")
    print()

    # CG vs direct solve consistency
    M = build_dirac_matrix(geom, U)
    Mt = M.conj().T
    MtM = Mt @ M
    b = sample_pseudofermion(M, rng)
    x_cg, n_iter = cg_solve_MtM(M, b, tol=1e-12)
    x_direct = np.linalg.solve(MtM, b)
    err = float(np.linalg.norm(x_cg - x_direct) / np.linalg.norm(x_direct))
    print(f"  CG vs direct (M† M)⁻¹ b:  |Δ|/|x| = {err:.2e}  ({n_iter} CG iters)")

    # PF identity: ⟨E_pf⟩ = V_4 over PF Gaussian samples
    print("\n  PF identity check (⟨φ†(M†M)⁻¹φ⟩ should equal V_4):")
    info = verify_pseudofermion_estimator(geom, U, rng, n_samples=500)
    print(f"    V_4 = {info['V4']}")
    print(f"    measured ⟨E_pf⟩ = {info['mean_E_pf']:.4f} ± "
          f"{info['std_E_pf']/np.sqrt(500):.4f}")
    print(f"    expected ⟨E_pf⟩ = {info['predicted_mean_E_pf']:.1f}")
    print(f"    CG iters: mean {info['cg_iters_mean']:.1f}, max {info['cg_iters_max']}")
    print(f"    log|det M|² (exact)  = {info['log_det_M2_exact']:.4f}")

    # PF Metropolis vs exact-det Metropolis ensemble check
    print("\n  PF Metropolis sweep test:")
    rng_pf = np.random.default_rng(7)
    rng_ex = np.random.default_rng(7)
    U_pf = Z2GaugeConfig.random(geom, rng_pf)
    U_ex = Z2GaugeConfig.random(geom, rng_ex)
    from action_z2_metropolis import heatbath_sweep
    import time

    for sw in range(5):
        U_pf, det_pf, Sg_pf, acc_pf = heatbath_sweep_pf(
            geom, U_pf, K_E=1.0, K_M=0.5, rng=rng_pf, refresh='per_sweep')
        U_ex, det_ex, Sg_ex, acc_ex = heatbath_sweep(
            geom, U_ex, K_E=1.0, K_M=0.5, rng=rng_ex, action_type='forward')
        print(f"    sweep {sw}: PF accept = {acc_pf}, exact-det accept = {acc_ex} "
              f"(out of {len(U_pf.U_x.flatten()) + len(U_pf.U_y.flatten()) + len(U_pf.U_t.flatten())} links)")


if __name__ == "__main__":
    _self_test()
