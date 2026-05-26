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
Accepts arbitrary Lx × Ly OBC (V_3 = Lx · Ly) with the qubit layout
defined by `build_pauli_terms_general` / `gauge_qc_bits_general`
(matter qubits at high bit positions, gauge link qubits at low).
Caller is responsible for providing dense O_total_dict[t] = U(t)†·n_0·U(t)·n_0
matrices of shape (DIM, DIM) where DIM = 2^(N_gauge + V_3).
For V_3 ≥ 8 the dense O_t grows past RAM and we'd want a matrix-free
apply_O_t(v) interface — see project_3d_scaling_plan for Phase 41+.
"""
from __future__ import annotations
import numpy as np

from action_z2_staggered import LatticeGeometry
from action_corner_direct_v3 import _fock_index_to_psi_map
from action_minkowski_stitch import (
    gauge_qc_bits_from_slice,
    gauge_qc_bits_general,
    qc_layout_counts,
)
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
    F = 1 << V3
    n_gauge_qubits, _ = qc_layout_counts(geom)
    G = 1 << n_gauge_qubits
    DIM = F * G

    # Pick the gauge-bit extractor that matches the qubit layout.
    # 2×2 keeps the legacy gauge_qc_bits_from_slice so V_3=4 scripts get
    # exact backward compatibility (and the legacy build_pauli_terms layout
    # is preserved bit-for-bit).
    if geom.Lx == 2 and geom.Ly == 2 and geom.Lz == 1:
        gauge_extractor = gauge_qc_bits_from_slice
    else:
        def gauge_extractor(U, t_slice):
            return gauge_qc_bits_general(U, t_slice, geom)

    idx_to_psi = _fock_index_to_psi_map(V3)
    # Vectorize the psi-permutation: W_psi = P @ W_full @ P.T where
    # P is the F×F permutation matrix taking lex index -> psi bit index.
    P = np.zeros((F, F), dtype=complex)
    for li in range(F):
        P[idx_to_psi[li], li] = 1.0

    # Reshape each O_t into (F, G, F, G).
    # Default C-order with index = (matter << n_gauge_qubits) | gauge gives
    # O_reshaped[pt, g_top, pb, g_bot] = O[(pt<<n_gauge)|g_top, (pb<<n_gauge)|g_bot]
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

        g_top = gauge_extractor(U, geom.N_E - 1)
        g_bot = gauge_extractor(U, 0)

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


def accumulate_C_direct_slab(
    geom: LatticeGeometry,
    configs,
    a_tau: float,
    times,
    H_sparse,
    m_obs: float,
    g_hop: float,
    w_order: int = 1,
    require_temporal_gauge: bool = True,
    progress: bool = True,
    progress_every: int = 32,
    expm_batch_size: int = None,
):
    """Slab-projected accumulator: never materializes the dense (DIM, DIM) O_t.

    Phase 41.5 — for V_3 ≥ 6 (or wherever dense O_t doesn't fit RAM).

    Algorithm
    =========
    Per unique gauge sector g ∈ {g_top, g_bot}(configs), build
        E[g, t] :=  O_t · (F/2 unit vectors at gauge sector g)   ∈ ℂ^(DIM × F/2)
    via two sparse `expm_multiply` calls:
        v_{g,pb}  =  |pb, g⟩  for pb ∈ surviving (= pb-bit-0 = 1)
        c_{g,pb}  =  U(t) · n_0 · v_{g,pb}   (right n_0 is folded into surviving)
        d_{g,pb}  =  n_0 · c_{g,pb}          (middle n_0, diagonal apply)
        E[g, t][:, k]  =  U(t)† · d_{g,pb(k)}  (left side of O_t)

    Then per config (g_top, g_bot, W_psi), the two submatrices needed
    by the half-and-half Hermitization sum (mirror of the dense path):
        O_sub_top_bot[pt, pb] = E[g_bot, t][g_top::G, :]   (re-embedded over dead pb)
        O_sub_bot_top[pb, pt] = E[g_top, t][g_bot::G, :].T  (same)
    The row-stride `g_left::G` exactly extracts indices (pt<<n_g) | g_left
    for pt ∈ [0, F), since G = 2^n_g.

    Cost
    ====
    Phase 1 (W_psi per config):  same as dense path.
    Phase 2 (E cache per (g, t)):  |unique g| × |times| sparse expm pairs.
        At V_3=6 (dim 8192, nnz ~92K, F/2=32): ~100ms per sparse expm pair.
        |unique g| ≤ G = 128 → ~100 s for 4 times.
    Phase 3 (per-config contraction):  F³ matmul per (config, t).  Negligible.

    Args:
        H_sparse: sparse CSR H_QC (epoq_classical_sampler.build_sparse_H_QC).
        Other args mirror accumulate_C_direct.

    Returns:
        (C, C_denom, Trho_O, n_configs)  — same shape as accumulate_C_direct.
    """
    import time as _time
    import scipy.sparse as sp
    from scipy.sparse.linalg import expm_multiply

    V3 = geom.V_3
    F = 1 << V3
    n_gauge_qubits, _ = qc_layout_counts(geom)
    G = 1 << n_gauge_qubits
    DIM = F * G
    if H_sparse.shape != (DIM, DIM):
        raise ValueError(
            f"H_sparse shape {H_sparse.shape} != ({DIM},{DIM}) expected "
            f"for geom (Lx={geom.Lx}, Ly={geom.Ly}).")

    q_n0 = n_gauge_qubits  # matter site (0,0) qubit position

    if geom.Lx == 2 and geom.Ly == 2 and geom.Lz == 1:
        gauge_extractor = gauge_qc_bits_from_slice
    else:
        def gauge_extractor(U, t_slice):
            return gauge_qc_bits_general(U, t_slice, geom)

    idx_to_psi = _fock_index_to_psi_map(V3)
    P = np.zeros((F, F), dtype=complex)
    for li in range(F):
        P[idx_to_psi[li], li] = 1.0

    if w_order != 1:
        raise NotImplementedError(
            f"accumulate_C_direct_slab currently only supports w_order=1 "
            f"(Lie time-ordered).  Got w_order={w_order}.")

    # -- Phase 1: build W_psi per config using a T_F slice cache.
    #
    # Per-slice T_F = expm(-a_τ · H_lat_slice[gauge_slice]) depends only on
    # the slice gauge config — 2^n_gauge possible values.  At V_3=6 only
    # 128 unique slice configs ⇒ ~15× Phase 1 speedup at large N_E.
    from transfer_matrix_kbc_trotterized import build_T_F_trotter

    Lx, Ly, Lz = geom.Lx, geom.Ly, geom.Lz
    n_y = Lx * (Ly - 1) * Lz
    n_x = (Lx - 1) * Ly * Lz

    if Lz == 1:
        def decode_slice_bits(slice_bits):
            """2D inverse of gauge_qc_bits_general at Lz=1.  Returns
            (U_x_slice, U_y_slice, None) (no z-links)."""
            U_y_slice = np.ones((Lx, Ly - 1), dtype=int)
            U_x_slice = np.ones((Lx - 1, Ly), dtype=int)
            for x in range(Lx):
                for y in range(Ly - 1):
                    if (slice_bits >> (x * (Ly - 1) + y)) & 1:
                        U_y_slice[x, y] = -1
            for x in range(Lx - 1):
                for y in range(Ly):
                    if (slice_bits >> (n_y + x * Ly + y)) & 1:
                        U_x_slice[x, y] = -1
            return U_x_slice, U_y_slice, None
    else:
        def decode_slice_bits(slice_bits):
            """3D inverse of gauge_qc_bits_general at Lz>1.  Returns
            (U_x_slice, U_y_slice, U_z_slice) matching the (Lx-1, Ly, Lz),
            (Lx, Ly-1, Lz), (Lx, Ly, Lz-1) shapes that build_H_lat_slice
            expects."""
            U_y_slice = np.ones((Lx, Ly - 1, Lz), dtype=int)
            U_x_slice = np.ones((Lx - 1, Ly, Lz), dtype=int)
            U_z_slice = np.ones((Lx, Ly, Lz - 1), dtype=int)
            for x in range(Lx):
                for y in range(Ly - 1):
                    for z in range(Lz):
                        pos = (x * (Ly - 1) + y) * Lz + z
                        if (slice_bits >> pos) & 1:
                            U_y_slice[x, y, z] = -1
            for x in range(Lx - 1):
                for y in range(Ly):
                    for z in range(Lz):
                        pos = n_y + (x * Ly + y) * Lz + z
                        if (slice_bits >> pos) & 1:
                            U_x_slice[x, y, z] = -1
            for x in range(Lx):
                for y in range(Ly):
                    for z in range(Lz - 1):
                        pos = n_y + n_x + (x * Ly + y) * (Lz - 1) + z
                        if (slice_bits >> pos) & 1:
                            U_z_slice[x, y, z] = -1
            return U_x_slice, U_y_slice, U_z_slice

    if progress:
        print("  Phase 1: scanning configs, collecting unique slice configs...",
              flush=True)
    t1 = _time.time()

    # 1a: scan configs, collect (g_top, g_bot) and unique slice gauge bits.
    config_slice_bits = []   # for each config, length-(N_E-1) list of bits
    config_g_top = []
    config_g_bot = []
    unique_g = set()
    unique_slice_bits = set()
    for U in configs:
        if require_temporal_gauge:
            assert (U.U_t == 1).all()
        slice_bits_list = []
        for t_slice in range(geom.N_E - 1):
            sb = gauge_extractor(U, t_slice)
            slice_bits_list.append(sb)
            unique_slice_bits.add(sb)
        config_slice_bits.append(slice_bits_list)
        g_top = gauge_extractor(U, geom.N_E - 1)
        g_bot = gauge_extractor(U, 0)
        config_g_top.append(g_top)
        config_g_bot.append(g_bot)
        unique_g.add(g_top)
        unique_g.add(g_bot)
    n_configs = len(config_slice_bits)
    if progress:
        print(f"    {n_configs} configs, {len(unique_slice_bits)} unique slice "
              f"gauge configs, |unique g_boundary|={len(unique_g)} "
              f"(of {G} possible), wall {_time.time()-t1:.1f}s", flush=True)

    # 1b: build T_F per unique slice config (cache).
    if progress:
        print(f"  Phase 1b: building T_F for {len(unique_slice_bits)} unique "
              f"slice configs...", flush=True)
    t1b = _time.time()
    T_F_cache = {}
    for sb in unique_slice_bits:
        U_x_slice, U_y_slice, U_z_slice = decode_slice_bits(sb)
        T_F_cache[sb] = build_T_F_trotter(
            geom, U_x_slice, U_y_slice, a_tau=a_tau, m=m_obs,
            K_E=0.0, K_M=0.0, g_hop=g_hop, U_z_slice=U_z_slice,
        )
    if progress:
        print(f"    {len(T_F_cache)} T_F's built in {_time.time()-t1b:.1f}s.",
              flush=True)

    # 1c: per config, multiply T_F's via cache and apply lex→psi permutation.
    if progress:
        print("  Phase 1c: contracting W per config via cache...", flush=True)
    t1c = _time.time()
    config_W_psi = []
    for slice_bits_list in config_slice_bits:
        W = np.eye(F, dtype=complex)
        for sb in slice_bits_list:
            W = T_F_cache[sb] @ W
        config_W_psi.append(P @ W @ P.T)
    if progress:
        print(f"    {n_configs} W_psi built in {_time.time()-t1c:.1f}s "
              f"(Phase 1 total {_time.time()-t1:.1f}s).", flush=True)
    unique_g_list = sorted(unique_g)

    # -- Phase 2+3 fused (streaming over g): build E[g, t] for one g at
    # a time, contract immediately with all configs that need it, then
    # release E.  RAM is bounded by O(DIM · F/2 · |times|) per g
    # instead of |unique_g| · DIM · F/2 · |times| in the all-at-once
    # cache scheme.  At 3D 2×2×2 dim=1M F/2=128: per-g E = 2 GB; the
    # all-at-once approach would need |unique_g| × |t| × 2 GB ≫ RAM.
    surviving_pb = np.array([pb for pb in range(F) if (pb & 1)],
                            dtype=np.int64)
    n_surv = len(surviving_pb)

    n0_diag = ((np.arange(DIM, dtype=np.int64) >> q_n0) & 1).astype(complex)

    # Group config indices by their g_top and g_bot.
    configs_by_g_top = {}   # g_value -> list of config indices i where g_top_i == g
    configs_by_g_bot = {}
    for i in range(n_configs):
        configs_by_g_top.setdefault(config_g_top[i], []).append(i)
        configs_by_g_bot.setdefault(config_g_bot[i], []).append(i)

    # Trace denominator: configs with g_top == g_bot.  One-time pass.
    C_denom = 0.0
    for i in range(n_configs):
        if config_g_top[i] == config_g_bot[i]:
            C_denom += np.trace(config_W_psi[i]).real

    if progress:
        print(f"  Phase 2+3 (streaming): {len(unique_g_list)} unique g × "
              f"{len(times)} t × 2 sparse expm calls each ...", flush=True)
    t23 = _time.time()

    # Choose batch size to keep scipy expm_multiply's hidden workspace
    # bounded.  scipy's Higham&Al-Mohy expm_multiply allocates several
    # (n, k) work matrices internally, and empirically the total memory
    # footprint at large n is ~30×16×n×k bytes (verified by OOM at
    # n=2^20 k=128 hitting 27 GB total-vm).  Pick batch so that the
    # transient peak stays under ~3 GB above the steady-state E array:
    #     k_max ≈ 3e9 / (30 × 16 × n) ≈ 6.25e6 / n.
    # At n=2^20=1M → k_max ≈ 6.  At n=2^18=262K → k_max ≈ 24.
    # At V_3=6 dim=8192 → k_max ≈ 760 (no batching needed).
    if expm_batch_size is None:
        max_k = max(1, int(3e9 / (30 * 16 * DIM)))
        expm_batch_size = min(n_surv, max(1, max_k))
    expm_batch_size = int(expm_batch_size)

    if progress:
        print(f"    column-batch size: {expm_batch_size} "
              f"(of n_surv={n_surv}, "
              f"{(n_surv + expm_batch_size - 1) // expm_batch_size} batches/g/t)",
              flush=True)

    Trho_O = {t: 0.0 + 0.0j for t in times}
    for g_idx, g in enumerate(unique_g_list):
        # Build (DIM, n_surv) RHS B for this g.
        B = np.zeros((DIM, n_surv), dtype=complex)
        for k, pb in enumerate(surviving_pb):
            B[(pb << n_gauge_qubits) | g, k] = 1.0

        for t in times:
            if t == 0.0:
                # U(0) = I → E = n_0 · B = n0_diag[:, None] * B
                E = n0_diag[:, None] * B
            else:
                # Build E in column batches to bound scipy's memory.
                E = np.empty((DIM, n_surv), dtype=complex)
                for b_start in range(0, n_surv, expm_batch_size):
                    b_end = min(b_start + expm_batch_size, n_surv)
                    B_batch = B[:, b_start:b_end]
                    C_batch = expm_multiply(-1j * t * H_sparse, B_batch)
                    C_batch *= n0_diag[:, None]
                    E[:, b_start:b_end] = expm_multiply(
                        +1j * t * H_sparse, C_batch)
                    del C_batch

            # Now contract E (= E[g, t], "right gauge" = g) with all configs
            # that use it.  Two cases:
            # (a) g_top_i == g: this E backs the O_sub_bot_top piece for config i,
            #     since O_sub_bot_top[pb, pt] = O_t[(pb<<n_g)|g_bot_i, (pt<<n_g)|g_top]
            #     and the right gauge is g_top == g.
            #     partial_bt[m, k] = E[(m<<n_g)|g_bot_i, k] = O_t[..., (pb(k)<<n_g)|g]
            #     → O_sub_bot_top: re-embed across pb.
            for i in configs_by_g_top.get(g, ()):
                W_psi = config_W_psi[i]
                g_bot_i = config_g_bot[i]
                partial_bt = E[g_bot_i::G, :]   # (F, F/2)
                O_sub_bot_top = np.zeros((F, F), dtype=complex)
                O_sub_bot_top[:, surviving_pb] = partial_bt
                tr_rho_O = np.trace(W_psi @ O_sub_bot_top)
                Trho_O[t] += 0.5 * tr_rho_O

            # (b) g_bot_i == g: this E backs the O_sub_top_bot piece.
            #     O_sub_top_bot[pt, pb] = O_t[(pt<<n_g)|g_top_i, (pb<<n_g)|g_bot_i=g]
            #     partial_tb[m, k] = E[(m<<n_g)|g_top_i, k] = O_t[..., (pb(k)<<n_g)|g]
            for i in configs_by_g_bot.get(g, ()):
                W_psi = config_W_psi[i]
                g_top_i = config_g_top[i]
                partial_tb = E[g_top_i::G, :]   # (F, F/2)
                O_sub_top_bot = np.zeros((F, F), dtype=complex)
                O_sub_top_bot[:, surviving_pb] = partial_tb
                tr_rho_dag_O = np.sum(W_psi.conj() * O_sub_top_bot)
                Trho_O[t] += 0.5 * tr_rho_dag_O

            # E and intermediate arrays go out of scope at end of t loop;
            # garbage collected before the next t (or next g).
            del E

        del B   # free per-g RHS
        if progress and ((g_idx + 1) % progress_every == 0
                         or g_idx == len(unique_g_list) - 1):
            elapsed = _time.time() - t23
            eta = elapsed / (g_idx + 1) * (len(unique_g_list) - g_idx - 1)
            print(f"    g {g_idx+1}/{len(unique_g_list)}: "
                  f"elapsed {elapsed:.1f}s, ETA {eta:.1f}s", flush=True)

    if progress:
        print(f"  Phase 2+3 done in {_time.time()-t23:.1f}s.", flush=True)

    if C_denom == 0.0:
        # No MC config landed on the diagonal (g_top == g_bot) — at
        # 3D 2×2×2 n_gauge=12 the per-config probability is 1/4096 so
        # this is the EXPECTED outcome at small chain length.
        # Return NaN for C(t); caller can renormalize using Trho_O[t]
        # ratios + a Hutchinson Tr[ρ_β] scale.  See
        # [[feedback_3d_denominator_rare_event]].
        import warnings as _w
        _w.warn(
            f"slab C_denom = 0 after {n_configs} configs "
            f"(no g_top == g_bot).  At n_gauge >= 10 this is rare-event-"
            f"dominated.  Returning NaN C(t); caller should renormalize "
            f"Trho_O[t] via Trho_O[t_ref]/C_ED(t_ref).")
        C = {t: float('nan') for t in times}
    else:
        C = {t: Trho_O[t].real / C_denom for t in times}
    return C, C_denom, Trho_O, n_configs


# ---------------------------------------------------------------------------
# Parallel Phase 2+3 via multiprocessing.Pool (fork) across unique g.
# Each worker processes one g end-to-end: build B, build E per t, contract,
# return per-t partial Trho_O complex sum.  H_sparse and other heavy
# read-only state are inherited via fork copy-on-write (no per-task pickle
# of the ~400 MB sparse H).
# ---------------------------------------------------------------------------

_PARALLEL_STATE = {}


def _set_parallel_state(**kwargs):
    """Stash read-only state in a module-level dict so workers can access
    it via fork() inheritance without per-task pickle overhead."""
    _PARALLEL_STATE.clear()
    _PARALLEL_STATE.update(kwargs)


def _worker_phase23_one_g(task):
    """Phase 2+3 work for a single unique g.

    task = (g, times, configs_g_top, configs_g_bot)
      configs_g_top: list of (g_bot_i, W_psi_i)  — configs where g_top_i == g
      configs_g_bot: list of (g_top_i, W_psi_i)  — configs where g_bot_i == g

    Returns: {t: complex partial Trho_O contribution from this g}
    """
    import numpy as _np
    from scipy.sparse.linalg import expm_multiply as _expm_multiply

    g, times, configs_g_top, configs_g_bot = task
    H_sparse = _PARALLEL_STATE['H_sparse']
    n0_diag = _PARALLEL_STATE['n0_diag']
    surviving_pb = _PARALLEL_STATE['surviving_pb']
    n_gauge_qubits = _PARALLEL_STATE['n_gauge_qubits']
    F = _PARALLEL_STATE['F']
    G = _PARALLEL_STATE['G']
    DIM = _PARALLEL_STATE['DIM']
    expm_batch_size = _PARALLEL_STATE['expm_batch_size']
    n_surv = len(surviving_pb)

    # Build (DIM, n_surv) RHS B for this g.
    B = _np.zeros((DIM, n_surv), dtype=complex)
    for k, pb in enumerate(surviving_pb):
        B[(pb << n_gauge_qubits) | g, k] = 1.0

    partial = {t: 0.0 + 0.0j for t in times}
    for t in times:
        if t == 0.0:
            E = n0_diag[:, None] * B
        else:
            E = _np.empty((DIM, n_surv), dtype=complex)
            for b_start in range(0, n_surv, expm_batch_size):
                b_end = min(b_start + expm_batch_size, n_surv)
                B_batch = B[:, b_start:b_end]
                C_batch = _expm_multiply(-1j * t * H_sparse, B_batch)
                C_batch *= n0_diag[:, None]
                E[:, b_start:b_end] = _expm_multiply(
                    +1j * t * H_sparse, C_batch)
                del C_batch

        for (g_bot_i, W_psi_i) in configs_g_top:
            partial_bt = E[g_bot_i::G, :]
            O_sub_bot_top = _np.zeros((F, F), dtype=complex)
            O_sub_bot_top[:, surviving_pb] = partial_bt
            partial[t] += 0.5 * _np.trace(W_psi_i @ O_sub_bot_top)

        for (g_top_i, W_psi_i) in configs_g_bot:
            partial_tb = E[g_top_i::G, :]
            O_sub_top_bot = _np.zeros((F, F), dtype=complex)
            O_sub_top_bot[:, surviving_pb] = partial_tb
            partial[t] += 0.5 * _np.sum(W_psi_i.conj() * O_sub_top_bot)

        del E

    return partial


def accumulate_C_direct_slab_parallel(
    geom: LatticeGeometry,
    configs,
    a_tau: float,
    times,
    H_sparse,
    m_obs: float,
    g_hop: float,
    w_order: int = 1,
    require_temporal_gauge: bool = True,
    progress: bool = True,
    n_workers: int = 4,
    expm_batch_size: int = None,
):
    """Multi-process Phase 2+3 variant of accumulate_C_direct_slab.

    Phase 1 (W_psi cache + slice config enumeration) is sequential and
    inexpensive.  Phase 2+3 — building E[g, t] via sparse expm_multiply
    and contracting against per-config W_psi — is embarrassingly parallel
    across unique g values.  We dispatch one (g, contractions) task per
    unique g to a multiprocessing.Pool with fork start method so workers
    inherit H_sparse / n0_diag / etc. via copy-on-write (no per-task
    pickling of the 400 MB sparse H).

    Args mirror accumulate_C_direct_slab plus:
        n_workers: number of parallel worker processes.  Each holds
                   ~30 GB peak scipy expm workspace at V_3=8 3D k=128;
                   tune to fit RAM.

    Returns: same shape as accumulate_C_direct_slab.
    """
    import multiprocessing as _mp
    import time as _time

    V3 = geom.V_3
    F = 1 << V3
    n_gauge_qubits, _ = qc_layout_counts(geom)
    G = 1 << n_gauge_qubits
    DIM = F * G
    if H_sparse.shape != (DIM, DIM):
        raise ValueError(
            f"H_sparse shape {H_sparse.shape} != ({DIM},{DIM}) expected.")
    if w_order != 1:
        raise NotImplementedError("parallel slab only supports w_order=1")

    q_n0 = n_gauge_qubits

    if geom.Lx == 2 and geom.Ly == 2 and geom.Lz == 1:
        gauge_extractor = gauge_qc_bits_from_slice
    else:
        def gauge_extractor(U, t_slice):
            return gauge_qc_bits_general(U, t_slice, geom)

    idx_to_psi = _fock_index_to_psi_map(V3)
    P = np.zeros((F, F), dtype=complex)
    for li in range(F):
        P[idx_to_psi[li], li] = 1.0

    from transfer_matrix_kbc_trotterized import build_T_F_trotter
    Lx, Ly, Lz = geom.Lx, geom.Ly, geom.Lz
    n_y = Lx * (Ly - 1) * Lz
    n_x = (Lx - 1) * Ly * Lz

    if Lz == 1:
        def decode_slice_bits(slice_bits):
            U_y_slice = np.ones((Lx, Ly - 1), dtype=int)
            U_x_slice = np.ones((Lx - 1, Ly), dtype=int)
            for x in range(Lx):
                for y in range(Ly - 1):
                    if (slice_bits >> (x * (Ly - 1) + y)) & 1:
                        U_y_slice[x, y] = -1
            for x in range(Lx - 1):
                for y in range(Ly):
                    if (slice_bits >> (n_y + x * Ly + y)) & 1:
                        U_x_slice[x, y] = -1
            return U_x_slice, U_y_slice, None
    else:
        def decode_slice_bits(slice_bits):
            U_y_slice = np.ones((Lx, Ly - 1, Lz), dtype=int)
            U_x_slice = np.ones((Lx - 1, Ly, Lz), dtype=int)
            U_z_slice = np.ones((Lx, Ly, Lz - 1), dtype=int)
            for x in range(Lx):
                for y in range(Ly - 1):
                    for z in range(Lz):
                        if (slice_bits >> ((x * (Ly - 1) + y) * Lz + z)) & 1:
                            U_y_slice[x, y, z] = -1
            for x in range(Lx - 1):
                for y in range(Ly):
                    for z in range(Lz):
                        if (slice_bits >> (n_y + (x * Ly + y) * Lz + z)) & 1:
                            U_x_slice[x, y, z] = -1
            for x in range(Lx):
                for y in range(Ly):
                    for z in range(Lz - 1):
                        pos = n_y + n_x + (x * Ly + y) * (Lz - 1) + z
                        if (slice_bits >> pos) & 1:
                            U_z_slice[x, y, z] = -1
            return U_x_slice, U_y_slice, U_z_slice

    # ----- Phase 1: T_F cache + W_psi list (serial; fast) -----
    if progress:
        print("  Phase 1: building W_psi via T_F slice cache...", flush=True)
    t1 = _time.time()
    config_slice_bits = []
    config_g_top = []
    config_g_bot = []
    unique_g = set()
    unique_slice_bits = set()
    for U in configs:
        if require_temporal_gauge:
            assert (U.U_t == 1).all()
        slice_bits_list = []
        for t_slice in range(geom.N_E - 1):
            sb = gauge_extractor(U, t_slice)
            slice_bits_list.append(sb)
            unique_slice_bits.add(sb)
        config_slice_bits.append(slice_bits_list)
        g_top = gauge_extractor(U, geom.N_E - 1)
        g_bot = gauge_extractor(U, 0)
        config_g_top.append(g_top)
        config_g_bot.append(g_bot)
        unique_g.add(g_top)
        unique_g.add(g_bot)
    n_configs = len(config_slice_bits)

    T_F_cache = {}
    for sb in unique_slice_bits:
        U_x_slice, U_y_slice, U_z_slice = decode_slice_bits(sb)
        T_F_cache[sb] = build_T_F_trotter(
            geom, U_x_slice, U_y_slice, a_tau=a_tau, m=m_obs,
            K_E=0.0, K_M=0.0, g_hop=g_hop, U_z_slice=U_z_slice)

    config_W_psi = []
    for slice_bits_list in config_slice_bits:
        W = np.eye(F, dtype=complex)
        for sb in slice_bits_list:
            W = T_F_cache[sb] @ W
        config_W_psi.append(P @ W @ P.T)
    unique_g_list = sorted(unique_g)
    if progress:
        print(f"    Phase 1 done in {_time.time()-t1:.1f}s "
              f"({n_configs} configs, {len(unique_g_list)} unique g, "
              f"{len(unique_slice_bits)} unique slices).", flush=True)

    # ----- Trace denominator (rare event at large n_gauge) -----
    C_denom = 0.0
    for i in range(n_configs):
        if config_g_top[i] == config_g_bot[i]:
            C_denom += np.trace(config_W_psi[i]).real

    # ----- Phase 2+3 parallel: dispatch tasks per unique g -----
    surviving_pb = np.array([pb for pb in range(F) if (pb & 1)],
                            dtype=np.int64)
    n_surv = len(surviving_pb)
    n0_diag = ((np.arange(DIM, dtype=np.int64) >> q_n0) & 1).astype(complex)

    if expm_batch_size is None:
        max_k = max(1, int(3e9 / (30 * 16 * DIM)))
        expm_batch_size = min(n_surv, max(1, max_k))
    expm_batch_size = int(expm_batch_size)

    # Group config refs by g (both as g_top and g_bot endpoints)
    configs_by_g_top = {}
    configs_by_g_bot = {}
    for i in range(n_configs):
        configs_by_g_top.setdefault(config_g_top[i], []).append(
            (config_g_bot[i], config_W_psi[i]))
        configs_by_g_bot.setdefault(config_g_bot[i], []).append(
            (config_g_top[i], config_W_psi[i]))

    tasks = [(g, list(times),
              configs_by_g_top.get(g, []),
              configs_by_g_bot.get(g, []))
             for g in unique_g_list]

    # Stash shared state in module global; workers inherit via fork.
    _set_parallel_state(
        H_sparse=H_sparse, n0_diag=n0_diag,
        surviving_pb=surviving_pb, n_gauge_qubits=n_gauge_qubits,
        F=F, G=G, DIM=DIM, expm_batch_size=expm_batch_size,
    )

    if progress:
        print(f"  Phase 2+3 parallel: {len(tasks)} g-tasks across "
              f"{n_workers} workers, batch={expm_batch_size}...", flush=True)
    t23 = _time.time()

    ctx = _mp.get_context('fork')
    with ctx.Pool(n_workers) as pool:
        partial_results = pool.map(_worker_phase23_one_g, tasks)

    if progress:
        print(f"    Phase 2+3 done in {_time.time()-t23:.1f}s "
              f"({(_time.time()-t23) / max(len(tasks),1):.1f}s/g).", flush=True)

    # Sum partial contributions
    Trho_O = {t: 0.0 + 0.0j for t in times}
    for partial in partial_results:
        for t in times:
            Trho_O[t] += partial[t]

    if C_denom == 0.0:
        import warnings as _w
        _w.warn(
            f"parallel slab C_denom = 0 after {n_configs} configs.  At "
            f"n_gauge >= 10 this is rare-event-dominated.  Returning NaN "
            f"C(t); use Trho_O[t]/Trho_O[t_ref] × C_Hutch(t_ref).")
        C = {t: float('nan') for t in times}
    else:
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
    Provided so the identity check can compute C(t) from the dense ρ̃
    using exactly the same numerical conventions.  At V_3 > 4 the
    DIM × DIM array can be huge (memory wall this refactor exists to avoid);
    expect to use it only as a V_3=4 reference.
    """
    V3 = geom.V_3
    F = 1 << V3
    n_gauge_qubits, _ = qc_layout_counts(geom)
    G = 1 << n_gauge_qubits
    DIM = F * G
    idx_to_psi = _fock_index_to_psi_map(V3)

    if geom.Lx == 2 and geom.Ly == 2 and geom.Lz == 1:
        gauge_extractor = gauge_qc_bits_from_slice
    else:
        def gauge_extractor(U, t_slice):
            return gauge_qc_bits_general(U, t_slice, geom)

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
        g_top = gauge_extractor(U, geom.N_E - 1)
        g_bot = gauge_extractor(U, 0)
        for pt in range(F):
            for pb in range(F):
                bit_a = (pt << n_gauge_qubits) | g_top
                bit_b = (pb << n_gauge_qubits) | g_bot
                rho[bit_a, bit_b] += W_psi[pt, pb]
    return rho
