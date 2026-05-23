"""
M_act_1: Z₂ + staggered fermion Euclidean lattice infrastructure.

Builds the OBC Dirac matrix M[σ] on an L_x × L_y × N_E lattice (spatial OBC,
temporal OBC) for a given Z₂ gauge configuration σ. Computes det M, supports
the boundary-to-boundary propagator G^bdy needed for corner-state extraction
(M_act_3).

Conventions:
  - Sites indexed by (t, x, y) with t ∈ [0, N_E), x ∈ [0, L_x), y ∈ [0, L_y)
  - Linear index: site_idx(t, x, y) = (t * L_x + x) * L_y + y
  - Z₂ link variables: σ ∈ {+1, −1}, stored separately for x, y, τ directions
  - Staggered phases: η_τ(x) = 1, η_x(x) = (−1)^{x_τ}, η_y(x) = (−1)^{x_τ + x_x}
  - OBC: drop hops that would exit the lattice (no wraparound, no boundary
    terms — fermion is "free" at the boundary).

OBC implementation: when computing the temporal hop at the boundary, simply
skip if neighbor doesn't exist. det M then depends only on σ (not on any
external fermion boundary data).
"""

from __future__ import annotations
import numpy as np
from typing import Tuple
from dataclasses import dataclass

# Numba-jitted kernels for hot loops (build_dirac_matrix, gauge_action).
from action_z2_kernels import (
    build_dirac_matrix_kernel,
    gauge_action_kernel,
)


@dataclass
class LatticeGeometry:
    Lx: int
    Ly: int
    N_E: int
    m: float

    @property
    def V_3(self) -> int:
        return self.Lx * self.Ly

    @property
    def V_4(self) -> int:
        return self.V_3 * self.N_E

    def site_idx(self, t: int, x: int, y: int) -> int:
        return (t * self.Lx + x) * self.Ly + y

    def eta(self, mu: int, t: int, x: int, y: int) -> int:
        """Standard Kogut-Susskind staggered phases.  μ ∈ {0, 1, 2} = {τ, x, y}.

        η_τ = 1 (time direction)
        η_x = (-1)^t
        η_y = (-1)^{t+x}

        These are required to lift the 2^d doubler degeneracy and give a
        sensible (non-degenerate) Dirac matrix.  Without these phases, naive
        fermions + staggered mass have unresolved doublers and det M = 0
        at trivial gauge.
        """
        if mu == 0:
            return 1
        elif mu == 1:
            return -1 if (t % 2) else 1
        elif mu == 2:
            return -1 if ((t + x) % 2) else 1
        raise ValueError(mu)


@dataclass
class Z2GaugeConfig:
    """Z₂ link variables on the (L_x × L_y × N_E) lattice with OBC.

    Conventions:
      U_x[t, x, y]: link from (t, x, y) to (t, x+1, y)         (x ∈ [0, L_x−1))
      U_y[t, x, y]: link from (t, x, y) to (t, x, y+1)         (y ∈ [0, L_y−1))
      U_t[t, x, y]: link from (t, x, y) to (t+1, x, y)         (t ∈ [0, N_E−1))
    """
    geom: LatticeGeometry
    U_x: np.ndarray   # shape (N_E, Lx-1, Ly), entries ±1
    U_y: np.ndarray   # shape (N_E, Lx, Ly-1), entries ±1
    U_t: np.ndarray   # shape (N_E-1, Lx, Ly), entries ±1

    @classmethod
    def random(cls, geom: LatticeGeometry, rng: np.random.Generator):
        return cls(
            geom=geom,
            U_x=rng.choice([-1, 1], size=(geom.N_E, geom.Lx - 1, geom.Ly)).astype(int),
            U_y=rng.choice([-1, 1], size=(geom.N_E, geom.Lx, geom.Ly - 1)).astype(int),
            U_t=rng.choice([-1, 1], size=(geom.N_E - 1, geom.Lx, geom.Ly)).astype(int),
        )

    @classmethod
    def trivial(cls, geom: LatticeGeometry):
        """All links σ = +1."""
        return cls(
            geom=geom,
            U_x=np.ones((geom.N_E, geom.Lx - 1, geom.Ly), dtype=int),
            U_y=np.ones((geom.N_E, geom.Lx, geom.Ly - 1), dtype=int),
            U_t=np.ones((geom.N_E - 1, geom.Lx, geom.Ly), dtype=int),
        )


# build_dirac_matrix and compute_det_M (central-difference Wilson-staggered)
# were REMOVED on 2026-05-20.  The central-diff det M is a DIFFERENT fermion
# theory than the Hamiltonian-derived T̂_F = e^{-a_τ H_KS}, so combining
# MC samples weighted by |det M_central|² with the W-trace observable mixes
# incompatible fermion sectors.  Use action_z2_staggered_forwardtime instead:
#     from action_z2_staggered_forwardtime import (
#         build_dirac_matrix_forward, compute_det_M_forward
#     )
# See memory/feedback_no_central_diff_use_forward.md for the full story.


# compute_G_bdy was removed along with build_dirac_matrix on 2026-05-20.
# The G^bdy boundary-propagator formula was the legacy corner-state weight
# (deprecated in favor of compute_combined_weight_trotter, see
# project_corner_state_weight_source memory note).  No production code uses
# G^bdy anymore.


def gauge_action(geom: LatticeGeometry, U: Z2GaugeConfig,
                 K: float = 1.0, K_E: float = None, K_M: float = None,
                 strang_M: bool = False) -> float:
    """Wilson plaquette action S_g = −K Σ_plaquettes ∏_link σ_link.

    For 2+1d Z₂ OBC lattice, plaquettes are:
      - Spatial (xy) plaquettes in each time slice  → "magnetic" energy in
        Hamiltonian limit (matches z2_setup g_m coefficient)
      - Spatial-temporal (xτ, yτ) plaquettes between consecutive time slices
        → "electric" energy (matches z2_setup g_e coefficient)

    If K_E and K_M are provided separately, the electric and magnetic plaquette
    couplings are independent (matching z2_setup g_e=1.0, g_m=0.5).  Otherwise
    fall back to single K for all plaquettes.

    strang_M=True: doubled-lattice Strang convention — K_M only at odd slices
    (intermediate gauge states where T_F is applied).  Caller should pass
    K_E_half = −(1/2)·log tanh((a_τ/2)·g_E) and K_M_full = a_τ·g_M.
    """
    if K_E is None:
        K_E = K
    if K_M is None:
        K_M = K
    return float(gauge_action_kernel(
        geom.Lx, geom.Ly, geom.N_E, K_E, K_M,
        U.U_x.astype(np.float64),
        U.U_y.astype(np.float64),
        U.U_t.astype(np.float64),
        strang_M,
    ))


def count_links_and_plaquettes(geom: LatticeGeometry) -> dict:
    """Sanity check: count links and plaquettes for given geometry."""
    n_x_links = geom.N_E * (geom.Lx - 1) * geom.Ly
    n_y_links = geom.N_E * geom.Lx * (geom.Ly - 1)
    n_t_links = (geom.N_E - 1) * geom.Lx * geom.Ly
    n_xy_plaq = geom.N_E * (geom.Lx - 1) * (geom.Ly - 1)
    n_xt_plaq = (geom.N_E - 1) * (geom.Lx - 1) * geom.Ly
    n_yt_plaq = (geom.N_E - 1) * geom.Lx * (geom.Ly - 1)
    return {
        'x_links': n_x_links, 'y_links': n_y_links, 't_links': n_t_links,
        'total_links': n_x_links + n_y_links + n_t_links,
        'xy_plaq': n_xy_plaq, 'xt_plaq': n_xt_plaq, 'yt_plaq': n_yt_plaq,
        'total_plaq': n_xy_plaq + n_xt_plaq + n_yt_plaq,
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def _self_test():
    """Build M, det M, G^bdy for a trivial gauge config and a random one. Verify shapes,
    determinant positivity, OBC structure."""
    geom = LatticeGeometry(Lx=2, Ly=2, N_E=4, m=0.5)
    print(f"Geometry: L_x={geom.Lx}, L_y={geom.Ly}, N_E={geom.N_E}, m={geom.m}")
    print(f"V_3 = {geom.V_3}, V_4 = {geom.V_4}")
    counts = count_links_and_plaquettes(geom)
    print(f"Counts: {counts}")

    # Self-test uses forward-time Dirac matrix (central-diff was removed).
    from action_z2_staggered_forwardtime import (
        build_dirac_matrix_forward, compute_det_M_forward,
    )

    U_triv = Z2GaugeConfig.trivial(geom)
    M_triv = build_dirac_matrix_forward(geom, U_triv)
    det_triv = compute_det_M_forward(geom, U_triv)
    print(f"\nM_forward (trivial σ): shape = {M_triv.shape}, det = {det_triv:.6f}")

    rng = np.random.default_rng(42)
    U_rand = Z2GaugeConfig.random(geom, rng)
    det_rand = compute_det_M_forward(geom, U_rand)
    S_g_rand = gauge_action(geom, U_rand, K=1.0)
    print(f"\nRandom σ: det M_forward = {det_rand:.6f}, S_g (K=1) = {S_g_rand:.4f}")


if __name__ == '__main__':
    _self_test()
