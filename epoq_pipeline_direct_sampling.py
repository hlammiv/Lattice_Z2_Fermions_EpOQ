"""
epoq_pipeline_direct_sampling.py

Direct accumulation of C(t) = Re Tr[ρ_H · O_t] / Re Tr[ρ_H] scalars per
MC config, without ever building the full 2^N_qubits dense ρ̃ matrix.

Mathematical setup
==================
The corner-state EρOQ pipeline (check_beta4_temporal_order1.py) assembles

   ρ̃ = Σ_U e^{-S_g[U]} · Σ_{pt, pb} |pt, g_top(U)⟩⟨pb, g_bot(U)| · W_psi[pt, pb]

on the full 8-qubit basis (V_3=4 case), then Hermitizes
   ρ_H = (ρ̃ + ρ̃†)/2
and computes
   C(t) = Re Tr[ρ_H · O_t] / Re Tr[ρ_H],   O_t = UOU(t) · n_0_op

where (matching the inline conventions):
  - W_psi(U)[pt, pb] = ⟨pt|T̂_F^{N_E-1}|pb⟩  in psi-bit Fock basis
  - g_top(U) = gauge_qc_bits_from_slice(U, N_E-1),  g_bot(U) = same at t=0
  - 8-qubit basis index = (pt << 4) | g  (gauge bits 0-3, fermion bits 4-7)

Per-config contributions
========================
  • Trace denominator (only when g_top == g_bot, since bit_a == bit_b
    requires pt == pb AND g_top == g_bot):
        Δ Tr[ρ̃]_U  =  trace(W_psi)
        Δ Tr[ρ_H]_U =  Re Δ Tr[ρ̃]_U   (Hermitization preserves diagonal real part)

  • Numerator at time t:
        Δ Tr[ρ̃ · O_t]_U      = Σ_{pt,pb} W_psi[pt,pb] · O_t[(pb<<4)|g_bot, (pt<<4)|g_top]
                               = trace(W_psi @ O_sub_bot_top)
        Δ Tr[ρ̃† · O_t]_U     = Σ_{pt,pb} W_psi[pt,pb]^* · O_t[(pt<<4)|g_top, (pb<<4)|g_bot]
                               = sum(W_psi.conj() * O_sub_top_bot)
    where the (F, F) = (16, 16) submatrices come from reshaping
    O_t (256, 256) → (F, G, F, G) and slicing fixed gauge indices.

Why this refactor
=================
Inline assembly builds a 256×256 dense ρ̃ per chain.  This scales as
2^(2·N_qubits) and breaks at V_3≥6 (RAM wall).  Direct sampling
accumulates a few scalars per config and one 16×16 submatrix per
(config, t).  No 2^(2·N_qubits) intermediate.  The submatrices stay
size F×F = 2^(2·V_3) regardless of gauge-qubit count.

Limitations
===========
Current implementation requires V_3 = 4 with dense O_t per time.
For V_3 ≥ 6 the dense O_t (= U(t)† n_0 U(t) n_0) grows past RAM
and we'd want a matrix-free apply_O_t(v) interface — see
project_3d_scaling_plan for Phase 40+ work.
"""
from __future__ import annotations
import numpy as np

from action_z2_staggered import LatticeGeometry
from action_corner_direct_v3 import _fock_index_to_psi_map
from action_minkowski_stitch import gauge_qc_bits_from_slice
from transfer_matrix_kbc_trotterized import compute_combined_weight_trotter


def accumulate_C_direct(
    geom: LatticeGeometry,
    configs,
    a_tau: float,
    times,
    O_total_dict: dict,
    m_obs: float,
    g_hop: float,
    w_order: int = 1,
    require_temporal_gauge: bool = True,
):
    """Accumulate C(t) numerator/denominator scalars over MC configs.

    Args:
        geom:               LatticeGeometry (V_3 must equal 4 in current impl).
        configs:            Iterable of Z2GaugeConfig (MC samples).
        a_tau:              Temporal lattice spacing.
        times:              Iterable of Minkowski times t for the observable.
        O_total_dict:       Dict {t: 256x256 complex} = {t: UOU(t) @ n_0_op}.
        m_obs:              Hamiltonian mass for T̂_F (NOT a_tau-scaled).
        g_hop:              Hop coupling for H_KS.
        w_order:            Trotter order for compute_combined_weight_trotter.
        require_temporal_gauge: assert U.U_t ≡ +1 on each config.

    Returns:
        C:        {t: real C(t)}    — final observable
        C_denom:  real Tr[ρ_H]       — accumulated denominator
        Trho_O:   {t: complex Tr[ρ_H · O_t]} — accumulated numerator
                  (real part is the load-bearing piece; imag is MC noise)
        n_configs: int               — number of configs processed
    """
    V3 = geom.V_3
    if V3 != 4:
        raise NotImplementedError(
            f"accumulate_C_direct currently supports V_3=4 only; got V_3={V3}. "
            f"For V_3>=6 see project_3d_scaling_plan Phase 40+ (matrix-free O_t)."
        )

    F = 1 << V3                  # fermion sector dim per slice = 16
    NQ_GAUGE = 8 - V3            # 4 gauge link qubits in V_3=4 2x2 case
    G = 1 << NQ_GAUGE            # gauge bit count = 16
    DIM = F * G                  # 256

    idx_to_psi = _fock_index_to_psi_map(V3)
    # Vectorize the psi-permutation: W_psi = P @ W_full @ P.T where
    # P is the F×F permutation matrix taking lex index -> psi bit index.
    P = np.zeros((F, F), dtype=complex)
    for li in range(F):
        P[idx_to_psi[li], li] = 1.0

    # Reshape each O_t into (F, G, F, G).
    # O_reshaped[pt, g_top, pb, g_bot] = O[(pt<<4)|g_top, (pb<<4)|g_bot]
    # (default C-order: last axis varies fastest)
    O_reshaped = {}
    for t in times:
        O = O_total_dict[t]
        if O.shape != (DIM, DIM):
            raise ValueError(
                f"O_total_dict[{t}] has shape {O.shape}, expected ({DIM},{DIM})")
        O_reshaped[t] = O.reshape(F, G, F, G)

    C_denom = 0.0
    Trho_O = {t: 0.0 + 0.0j for t in times}
    n_configs = 0

    for U in configs:
        if require_temporal_gauge:
            assert (U.U_t == 1).all(), (
                "U_t not pinned to +1 despite require_temporal_gauge=True. "
                "MC sweep was not in temporal gauge."
            )
        W_full, _ = compute_combined_weight_trotter(
            geom, U, a_tau=a_tau, K_E=0.0, K_M=0.0,
            m_obs=m_obs, g_hop=g_hop, order=w_order,
        )
        W_psi = P @ W_full @ P.T

        g_top = gauge_qc_bits_from_slice(U, geom.N_E - 1)
        g_bot = gauge_qc_bits_from_slice(U, 0)

        if g_top == g_bot:
            C_denom += np.trace(W_psi).real

        for t in times:
            O_r = O_reshaped[t]
            O_sub_bot_top = O_r[:, g_bot, :, g_top]   # shape (F, F)
            O_sub_top_bot = O_r[:, g_top, :, g_bot]   # shape (F, F)
            tr_rho_O = np.trace(W_psi @ O_sub_bot_top)
            tr_rho_dag_O = np.sum(W_psi.conj() * O_sub_top_bot)
            Trho_O[t] += 0.5 * (tr_rho_O + tr_rho_dag_O)

        n_configs += 1

    if C_denom == 0.0:
        raise RuntimeError(
            f"Tr[ρ_H] = 0 after {n_configs} configs (no config had g_top==g_bot). "
            f"In temporal gauge this should happen with non-zero probability."
        )

    C = {t: Trho_O[t].real / C_denom for t in times}
    return C, C_denom, Trho_O, n_configs


def assemble_rho_dense(
    geom: LatticeGeometry,
    configs,
    a_tau: float,
    m_obs: float,
    g_hop: float,
    w_order: int = 1,
):
    """Reference path: same as the inline ρ̃ assembly in check_beta4_*.py.

    Returned ρ̃ is NOT Hermitized — caller does (ρ̃ + ρ̃†)/2 if desired.
    Provided here so the identity check can compute C(t) from the dense
    ρ̃ side using exactly the same numerical conventions.
    """
    V3 = geom.V_3
    if V3 != 4:
        raise NotImplementedError("assemble_rho_dense: V_3=4 only.")
    F = 1 << V3
    G = 1 << (8 - V3)
    DIM = F * G
    idx_to_psi = _fock_index_to_psi_map(V3)

    rho = np.zeros((DIM, DIM), dtype=complex)
    for U in configs:
        W_full, _ = compute_combined_weight_trotter(
            geom, U, a_tau=a_tau, K_E=0.0, K_M=0.0,
            m_obs=m_obs, g_hop=g_hop, order=w_order,
        )
        W_psi = np.zeros((F, F), dtype=complex)
        for li in range(F):
            for lj in range(F):
                W_psi[idx_to_psi[li], idx_to_psi[lj]] = W_full[li, lj]
        g_top = gauge_qc_bits_from_slice(U, geom.N_E - 1)
        g_bot = gauge_qc_bits_from_slice(U, 0)
        for pt in range(F):
            for pb in range(F):
                bit_a = (pt << 4) | g_top
                bit_b = (pb << 4) | g_bot
                rho[bit_a, bit_b] += W_psi[pt, pb]
    return rho
