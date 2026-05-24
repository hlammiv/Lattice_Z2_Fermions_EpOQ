"""Direct comparison: lattice ρ̃_det (path integral, deterministic enumeration)
vs Hamiltonian Trotter ρ_T = [e^{−a_τH_g}·e^{−a_τH_F[U_op]}]^N on the full
256-dim Hilbert space.

The path-integral identity claims these should be EQUAL (gauge-basis insertion
on H_F[U_op] is exact since it's diagonal in gauge basis).

If ρ̃_det ≈ ρ_T → construction matches Hamiltonian Trotter; gap to ED is "just"
Trotter error.

If ρ̃_det ≠ ρ_T → real implementation bug.  The discrepancy pattern shows
WHERE the bug is (which matrix elements differ).

Setup: β=2, m=0.5, a_τ=1, N_E=3 (matches diag_det_beta2.py).
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import sys, time
import numpy as np
from multiprocessing import Pool
from scipy.linalg import expm

sys.path.insert(0, '/home/hlamm/Desktop/QC/logdet/m1_toy')

import epoq_classical_sampler as cs
cs.M_MASS = 0.5
import epoq_quantum_simulator as qs
qs.M_MASS = 0.5

from action_z2_staggered import LatticeGeometry, Z2GaugeConfig
from action_z2_kernels import gauge_action_kernel, warm_up
from transfer_matrix_kbc_trotterized import compute_combined_weight_trotter
from action_minkowski_stitch import gauge_qc_bits_from_slice
from action_corner_direct_v3 import _fock_index_to_psi_map

M_HAM, G_E_HAM, G_M_HAM, G_HOP = 0.5, 1.0, 0.5, 0.5
BETA = 2.0
A_TAU = 1.0
N_E = int(round(BETA / A_TAU)) + 1   # 3
N_steps = N_E - 1                     # = 2 Trotter steps
W_ORDER = 2


def pauli_op(factors, nq=8):
    P2 = {'I': np.eye(2, dtype=complex),
          'X': np.array([[0, 1], [1, 0]], dtype=complex),
          'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
          'Z': np.diag([1, -1]).astype(complex)}
    by_q = {q: P2['I'] for q in range(nq)}
    for q, ax in factors:
        by_q[q] = P2[ax]
    r = by_q[nq - 1]
    for q in range(nq - 2, -1, -1):
        r = np.kron(r, by_q[q])
    return r


def build_H_QC_components():
    """Partition H_QC into H_g (gauge-only) and H_F (rest).  Returns (H, H_g, H_F)."""
    terms = cs.build_pauli_terms()
    H = np.zeros((256, 256), dtype=complex)
    H_g = np.zeros((256, 256), dtype=complex)
    H_F = np.zeros((256, 256), dtype=complex)
    for coef, ops in terms:
        op_mat = pauli_op(ops)
        H += coef * op_mat
        # gauge-only if all qubits in 0..3
        if all(q < 4 for (q, _) in ops):
            H_g += coef * op_mat
        else:
            H_F += coef * op_mat
    H = (H + H.conj().T) / 2
    H_g = (H_g + H_g.conj().T) / 2
    H_F = (H_F + H_F.conj().T) / 2
    # Sanity: H = H_g + H_F
    assert np.max(np.abs(H - H_g - H_F)) < 1e-12, "H_g + H_F != H_QC"
    return H, H_g, H_F


# ---- Deterministic ρ̃ (re-used from diag_det_beta2.py) ----
def n_links(N_E):
    return N_E * 1 * 2 + N_E * 2 * 1 + (N_E - 1) * 2 * 2


def decode_to_U(idx, N_E):
    geom = LatticeGeometry(Lx=2, Ly=2, N_E=N_E, m=A_TAU * M_HAM)
    U_x = np.empty((N_E, 1, 2), dtype=np.int8)
    U_y = np.empty((N_E, 2, 1), dtype=np.int8)
    U_t = np.empty((N_E - 1, 2, 2), dtype=np.int8)
    b = 0
    for t in range(N_E):
        for x in range(1):
            for y in range(2):
                U_x[t, x, y] = 1 if (idx >> b) & 1 == 0 else -1
                b += 1
    for t in range(N_E):
        for x in range(2):
            for y in range(1):
                U_y[t, x, y] = 1 if (idx >> b) & 1 == 0 else -1
                b += 1
    for t in range(N_E - 1):
        for x in range(2):
            for y in range(2):
                U_t[t, x, y] = 1 if (idx >> b) & 1 == 0 else -1
                b += 1
    return Z2GaugeConfig(geom=geom, U_x=U_x, U_y=U_y, U_t=U_t)


def per_config(args):
    idx, K_E, K_M = args
    Lx = Ly = 2
    geom = LatticeGeometry(Lx=Lx, Ly=Ly, N_E=N_E, m=A_TAU * M_HAM)
    U = decode_to_U(idx, N_E)
    U.geom = geom
    S_g = gauge_action_kernel(
        Lx, Ly, N_E, K_E, K_M,
        U.U_x.astype(np.float64),
        U.U_y.astype(np.float64),
        U.U_t.astype(np.float64),
    )
    boltz = np.exp(-S_g)
    W_full, _ = compute_combined_weight_trotter(
        geom, U, a_tau=A_TAU, K_E=0.0, K_M=0.0,
        m_obs=M_HAM, g_hop=G_HOP, order=W_ORDER)
    V3 = geom.V_3
    idx_to_psi = _fock_index_to_psi_map(V3)
    dim_F = 1 << V3
    W_psi = np.zeros((dim_F, dim_F), dtype=complex)
    for li in range(dim_F):
        for lj in range(dim_F):
            W_psi[idx_to_psi[li], idx_to_psi[lj]] = W_full[li, lj]
    g_top = gauge_qc_bits_from_slice(U, N_E - 1)
    g_bot = gauge_qc_bits_from_slice(U, 0)
    return idx, boltz, W_psi, g_top, g_bot


def build_rho_det():
    K_E = -0.5 * np.log(np.tanh(A_TAU * G_E_HAM))
    K_M = A_TAU * G_M_HAM
    total = n_links(N_E)
    n_configs = 1 << total
    print(f"  Enumerating 2^{total} = {n_configs} configs", flush=True)
    print(f"  K_E={K_E:.4f}, K_M={K_M:.4f}", flush=True)
    args = ((i, K_E, K_M) for i in range(n_configs))
    rho = np.zeros((256, 256), dtype=complex)
    t0 = time.time()
    completed = 0
    report_every = max(1, n_configs // 10)
    with Pool(6, initializer=warm_up) as pool:
        for idx, w, W_psi, g_top, g_bot in pool.imap_unordered(
                per_config, args, chunksize=max(1, n_configs // 192)):
            completed += 1
            if w == 0:
                continue
            for psi_top in range(16):
                for psi_bot in range(16):
                    bit_a = (psi_top << 4) | g_top
                    bit_b = (psi_bot << 4) | g_bot
                    rho[bit_a, bit_b] += w * W_psi[psi_top, psi_bot]
            if completed % report_every == 0:
                print(f"    {100*completed/n_configs:.0f}% ({time.time()-t0:.0f}s)", flush=True)
    print(f"  done in {time.time()-t0:.0f}s", flush=True)
    return (rho + rho.conj().T) / 2


def main():
    print("=" * 80)
    print(f"Lattice ρ̃_det vs Hamiltonian Trotter ρ_T")
    print(f"  β={BETA}, m={M_HAM}, a_τ={A_TAU}, N_E={N_E}, N_steps={N_steps}")
    print("=" * 80, flush=True)

    print(f"\n[1] Building dense H_QC, H_g, H_F (256×256)...", flush=True)
    H, H_g, H_F = build_H_QC_components()
    print(f"  H_g spectrum span: [{np.linalg.eigvalsh(H_g).real.min():.3f}, "
          f"{np.linalg.eigvalsh(H_g).real.max():.3f}]")
    print(f"  H_F spectrum span: [{np.linalg.eigvalsh(H_F).real.min():.3f}, "
          f"{np.linalg.eigvalsh(H_F).real.max():.3f}]")
    print(f"  ‖[H_g, H_F]‖ = {np.linalg.norm(H_g @ H_F - H_F @ H_g):.3f}",
          flush=True)

    # ED reference
    rho_ED = expm(-BETA * H)
    rho_ED_n = rho_ED / np.trace(rho_ED).real

    # Hamiltonian Trotter: [e^{-a_τH_g} · e^{-a_τH_F}]^N
    print(f"\n[2] Building Hamiltonian Trotter ρ_T = [e^{{-a_τH_g}}·e^{{-a_τH_F}}]^N", flush=True)
    print(f"    (Lie ordering: H_g acts AFTER H_F per right-to-left step)", flush=True)
    E_g = expm(-A_TAU * H_g)
    E_F = expm(-A_TAU * H_F)
    step = E_g @ E_F
    rho_T_OP = np.linalg.matrix_power(step, N_steps)
    # Also try other ordering
    step_alt = E_F @ E_g
    rho_T_alt = np.linalg.matrix_power(step_alt, N_steps)
    # And Strang within gauge-fermion
    E_g_half = expm(-(A_TAU/2) * H_g)
    step_strang = E_g_half @ E_F @ E_g_half
    rho_T_strang = np.linalg.matrix_power(step_strang, N_steps)

    # Lattice deterministic
    print(f"\n[3] Building lattice ρ̃_det (deterministic enumeration)...", flush=True)
    rho_det = build_rho_det()

    # Normalize all
    Z_ED = np.trace(rho_ED).real
    Z_T_OP = np.trace(rho_T_OP).real
    Z_T_alt = np.trace(rho_T_alt).real
    Z_T_strang = np.trace(rho_T_strang).real
    Z_det = np.trace(rho_det).real

    rho_T_OP_n = rho_T_OP / Z_T_OP
    rho_T_alt_n = rho_T_alt / Z_T_alt
    rho_T_strang_n = rho_T_strang / Z_T_strang
    rho_det_n = rho_det / Z_det

    print(f"\n[4] Normalizations and Frobenius gaps:")
    print(f"  Z_ED         = {Z_ED:.4e}")
    print(f"  Z_T (Lie HF→Hg) = {Z_T_OP:.4e}   ratio to ED: {Z_T_OP/Z_ED:.4f}")
    print(f"  Z_T (Lie Hg→HF) = {Z_T_alt:.4e}   ratio to ED: {Z_T_alt/Z_ED:.4f}")
    print(f"  Z_T (Strang)    = {Z_T_strang:.4e}   ratio to ED: {Z_T_strang/Z_ED:.4f}")
    print(f"  Z_det        = {Z_det:.4e}   ratio to ED: {Z_det/Z_ED:.4f}")

    print(f"\n  ‖ρ_T_OP_n     − ρ_ED_n‖_F = {np.linalg.norm(rho_T_OP_n - rho_ED_n):.5f}")
    print(f"  ‖ρ_T_alt_n    − ρ_ED_n‖_F = {np.linalg.norm(rho_T_alt_n - rho_ED_n):.5f}")
    print(f"  ‖ρ_T_strang_n − ρ_ED_n‖_F = {np.linalg.norm(rho_T_strang_n - rho_ED_n):.5f}")
    print(f"  ‖ρ_det_n      − ρ_ED_n‖_F = {np.linalg.norm(rho_det_n - rho_ED_n):.5f}")

    print(f"\n  Critical: does lattice match Hamiltonian Trotter?")
    print(f"  ‖ρ_det_n − ρ_T_OP_n‖_F     = {np.linalg.norm(rho_det_n - rho_T_OP_n):.5f}")
    print(f"  ‖ρ_det_n − ρ_T_alt_n‖_F    = {np.linalg.norm(rho_det_n - rho_T_alt_n):.5f}")
    print(f"  ‖ρ_det_n − ρ_T_strang_n‖_F = {np.linalg.norm(rho_det_n - rho_T_strang_n):.5f}")
    # also unnormalized
    print(f"\n  Unnormalized (test of raw weights):")
    print(f"  ‖ρ_det − ρ_T_OP·(Z_det/Z_T_OP)‖_F = {np.linalg.norm(rho_det - rho_T_OP*(Z_det/Z_T_OP)):.4e}")

    # C(t) comparison
    Z = np.diag([1.0, -1.0]).astype(complex)
    n0_minus = np.array([[1.0]], dtype=complex)
    for k in reversed(range(8)):
        m = Z if k == 4 else np.eye(2, dtype=complex)
        n0_minus = np.kron(n0_minus, m)
    n0_op = 0.5 * np.eye(256, dtype=complex) - 0.5 * n0_minus

    TIMES = [0.0, 0.5, 1.0]
    print(f"\n[5] C(t) comparison:")
    print(f"  {'t':>5} | {'C_ED':>10} | {'C_T_OP':>10} | {'C_T_alt':>10} | {'C_T_strang':>11} | {'C_det':>10}")
    for t in TIMES:
        Ut = expm(-1j * H * t)
        UOU = Ut.conj().T @ n0_op @ Ut
        C_ED = np.real(np.trace(rho_ED_n @ UOU @ n0_op))
        C_T_OP = np.real(np.trace(rho_T_OP_n @ UOU @ n0_op))
        C_T_alt = np.real(np.trace(rho_T_alt_n @ UOU @ n0_op))
        C_T_strang = np.real(np.trace(rho_T_strang_n @ UOU @ n0_op))
        C_det = np.real(np.trace(rho_det_n @ UOU @ n0_op))
        print(f"  {t:>5.2f} | {C_ED:>+10.5f} | {C_T_OP:>+10.5f} | {C_T_alt:>+10.5f} | "
              f"{C_T_strang:>+11.5f} | {C_det:>+10.5f}")


if __name__ == "__main__":
    main()
