"""Phase 38b: test option 2 — fix U_t = +1 only at BOUNDARY transitions
(transition 0 [slice 0→1] and transition N_E-2 [slice N_E-2 → N_E-1]),
let bulk U_t fluctuate.

Tested at N_E=4 (β=2, a_τ=2/3), which is the smallest N_E where option 2
is genuinely different from option 1 (full temporal gauge).

Three comparisons:
  (A) Option 1: full temporal gauge (U_t=+1 everywhere).  Enumerate 16
      spatial dof per slice × 4 slices = 16 dof. 2^16 = 65k configs.
  (B) Option 2: boundary-only U_t fix.  Enumerate 16 spatial + bulk U_t.
      For N_E=4: 1 bulk transition × 4 U_t links = 4 bulk U_t dof.
      Total 20 dof, 1M configs.
  (C) Matched Hamiltonian Trotter T_match for N=3 steps.

Expected: option 2 ≈ option 1 ≈ const · T_match (up to per-link Suzuki).
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
from action_z2_kernels import warm_up
from transfer_matrix_kbc_trotterized import compute_combined_weight_trotter
from action_minkowski_stitch import gauge_qc_bits_from_slice
from action_corner_direct_v3 import _fock_index_to_psi_map

M_HAM = 0.5
G_E_HAM, G_M_HAM, G_HOP = 1.0, 0.5, 0.5
BETA = 2.0
N_E = 4
A_TAU = BETA / (N_E - 1)   # 2/3
N = N_E - 1                 # 3 Trotter steps


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


def build_HE_HM_HF():
    terms = cs.build_pauli_terms()
    H = np.zeros((256, 256), dtype=complex)
    H_E = np.zeros((256, 256), dtype=complex)
    H_M = np.zeros((256, 256), dtype=complex)
    H_F = np.zeros((256, 256), dtype=complex)
    for coef, ops in terms:
        op_mat = pauli_op(ops)
        H += coef * op_mat
        if len(ops) == 1 and ops[0][0] < 4 and ops[0][1] == 'X':
            H_E += coef * op_mat
        elif len(ops) == 4 and all(q < 4 and p == 'Z' for (q, p) in ops):
            H_M += coef * op_mat
        else:
            H_F += coef * op_mat
    return ((H + H.conj().T) / 2, (H_E + H_E.conj().T) / 2,
            (H_M + H_M.conj().T) / 2, (H_F + H_F.conj().T) / 2)


def n_spatial_links_per_slice():
    return 1 * 2 + 2 * 1   # 4 spatial links

def n_temporal_links_per_transition():
    return 2 * 2   # 4 temporal links

def decode_config_option1(idx, N_E):
    """Option 1: full temporal gauge.  Decode 16 bits → spatial fields at all slices.
    U_t = +1 everywhere."""
    geom = LatticeGeometry(Lx=2, Ly=2, N_E=N_E, m=A_TAU * M_HAM)
    U_x = np.empty((N_E, 1, 2), dtype=np.int8)
    U_y = np.empty((N_E, 2, 1), dtype=np.int8)
    U_t = np.ones((N_E - 1, 2, 2), dtype=np.int8)
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
    return Z2GaugeConfig(geom=geom, U_x=U_x, U_y=U_y, U_t=U_t)


def decode_config_option2(idx, N_E):
    """Option 2: boundary-only U_t fix.
    First 16 bits: spatial fields at all slices.
    Remaining bits: BULK U_t fields at transitions 1, 2, ..., N_E-3.
    U_t fixed = +1 at transitions 0 (slice 0→1) and N_E-2 (slice N_E-2 → N_E-1).
    """
    geom = LatticeGeometry(Lx=2, Ly=2, N_E=N_E, m=A_TAU * M_HAM)
    U_x = np.empty((N_E, 1, 2), dtype=np.int8)
    U_y = np.empty((N_E, 2, 1), dtype=np.int8)
    U_t = np.ones((N_E - 1, 2, 2), dtype=np.int8)
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
    # Bulk U_t: transitions 1..N_E-3 (skip 0 and N_E-2)
    for t in range(1, N_E - 2):
        for x in range(2):
            for y in range(2):
                U_t[t, x, y] = 1 if (idx >> b) & 1 == 0 else -1
                b += 1
    return Z2GaugeConfig(geom=geom, U_x=U_x, U_y=U_y, U_t=U_t)


def gauge_action_full_wilson(N_E, K_E, K_M, U_x, U_y, U_t):
    """Standard 4-link Wilson plaquette action."""
    Lx = Ly = 2
    total_E = 0.0
    total_M = 0.0
    for t in range(N_E):
        for x in range(Lx - 1):
            for y in range(Ly - 1):
                u1 = U_x[t, x, y]; u2 = U_y[t, x + 1, y]
                u3 = U_x[t, x, y + 1]; u4 = U_y[t, x, y]
                total_M += u1 * u2 * u3 * u4
    for t in range(N_E - 1):
        for x in range(Lx - 1):
            for y in range(Ly):
                u1 = U_x[t, x, y]; u2 = U_t[t, x + 1, y]
                u3 = U_x[t + 1, x, y]; u4 = U_t[t, x, y]
                total_E += u1 * u2 * u3 * u4
        for x in range(Lx):
            for y in range(Ly - 1):
                u1 = U_y[t, x, y]; u2 = U_t[t, x, y + 1]
                u3 = U_y[t + 1, x, y]; u4 = U_t[t, x, y]
                total_E += u1 * u2 * u3 * u4
    return -K_E * total_E - K_M * total_M


def per_config(args):
    idx, decode_fn, K_E, K_M = args
    geom = LatticeGeometry(Lx=2, Ly=2, N_E=N_E, m=A_TAU * M_HAM)
    U = decode_fn(idx, N_E)
    U.geom = geom
    S_g = gauge_action_full_wilson(
        N_E, K_E, K_M,
        U.U_x.astype(np.float64), U.U_y.astype(np.float64),
        U.U_t.astype(np.float64),
    )
    boltz = np.exp(-S_g)
    W_full, _ = compute_combined_weight_trotter(
        geom, U, a_tau=A_TAU, K_E=0.0, K_M=0.0,
        m_obs=M_HAM, g_hop=G_HOP, order=1)
    idx_to_psi = _fock_index_to_psi_map(4)
    W_psi = np.zeros((16, 16), dtype=complex)
    for li in range(16):
        for lj in range(16):
            W_psi[idx_to_psi[li], idx_to_psi[lj]] = W_full[li, lj]
    g_top = gauge_qc_bits_from_slice(U, N_E - 1)
    g_bot = gauge_qc_bits_from_slice(U, 0)
    return boltz, W_psi, g_top, g_bot


def build_rho_det(decode_fn, n_dof, label):
    K_E = -0.5 * np.log(np.tanh(A_TAU * G_E_HAM))
    K_M = A_TAU * G_M_HAM
    n_configs = 1 << n_dof
    print(f"  [{label}] enumerating 2^{n_dof} = {n_configs} configs", flush=True)
    args = ((i, decode_fn, K_E, K_M) for i in range(n_configs))
    rho = np.zeros((256, 256), dtype=complex)
    t0 = time.time()
    with Pool(6, initializer=warm_up) as pool:
        chunksize = max(1, n_configs // 192)
        for w, W_psi, g_top, g_bot in pool.imap_unordered(per_config, args, chunksize=chunksize):
            if w == 0:
                continue
            for psi_top in range(16):
                for psi_bot in range(16):
                    bit_a = (psi_top << 4) | g_top
                    bit_b = (psi_bot << 4) | g_bot
                    rho[bit_a, bit_b] += w * W_psi[psi_top, psi_bot]
    print(f"    done in {time.time()-t0:.0f}s", flush=True)
    return (rho + rho.conj().T) / 2


def main():
    print("=" * 80)
    print(f"Phase 38b: boundary-only U_t fix vs full temporal gauge")
    print(f"  β={BETA}, m={M_HAM}, a_τ={A_TAU:.4f}, N_E={N_E}, N={N}")
    print("=" * 80, flush=True)

    H, H_E, H_M, H_F = build_HE_HM_HF()
    rho_ED = expm(-BETA * H)
    rho_ED_n = rho_ED / np.trace(rho_ED).real

    E_E = expm(-A_TAU * H_E)
    E_M = expm(-A_TAU * H_M)
    E_F = expm(-A_TAU * H_F)
    step = E_E @ E_F @ E_M
    rho_T_match = E_M @ np.linalg.matrix_power(step, N)
    rho_T_match_n = rho_T_match / np.trace(rho_T_match).real

    Z = np.diag([1.0, -1.0]).astype(complex)
    n0_minus = np.array([[1.0]], dtype=complex)
    for k in reversed(range(8)):
        m = Z if k == 4 else np.eye(2, dtype=complex)
        n0_minus = np.kron(n0_minus, m)
    n0_op = 0.5 * np.eye(256, dtype=complex) - 0.5 * n0_minus

    TIMES = [0.0, 0.5, 1.0]
    UOU = {t: expm(-1j * H * t).conj().T @ n0_op @ expm(-1j * H * t) for t in TIMES}
    C_ED = {t: np.real(np.trace(rho_ED_n @ UOU[t] @ n0_op)) for t in TIMES}
    C_T_match = {t: np.real(np.trace(rho_T_match_n @ UOU[t] @ n0_op)) for t in TIMES}
    print(f"\nED:      " + "  ".join(f"C({t})={C_ED[t]:+.5f}" for t in TIMES))
    print(f"T_match: " + "  ".join(f"C({t})={C_T_match[t]:+.5f}" for t in TIMES))

    print()
    # Option 1: full temporal gauge
    n_dof_1 = N_E * n_spatial_links_per_slice()   # = 16
    rho_opt1 = build_rho_det(decode_config_option1, n_dof_1,
                              label='Option 1 (U_t=+1 everywhere)')
    rho_opt1_n = rho_opt1 / np.trace(rho_opt1).real

    # Option 2: boundary-only U_t fix
    n_bulk_transitions = max(0, N_E - 3)
    n_dof_2 = N_E * n_spatial_links_per_slice() + n_bulk_transitions * n_temporal_links_per_transition()
    rho_opt2 = build_rho_det(decode_config_option2, n_dof_2,
                              label='Option 2 (boundary U_t=+1 only)')
    rho_opt2_n = rho_opt2 / np.trace(rho_opt2).real

    print(f"\n[Frobenius distances normalized]")
    print(f"  ‖ρ_opt1_n − ρ_ED_n‖_F = {np.linalg.norm(rho_opt1_n - rho_ED_n):.5f}")
    print(f"  ‖ρ_opt1_n − T_match_n‖_F = {np.linalg.norm(rho_opt1_n - rho_T_match_n):.4e}")
    print(f"  ‖ρ_opt2_n − ρ_ED_n‖_F = {np.linalg.norm(rho_opt2_n - rho_ED_n):.5f}")
    print(f"  ‖ρ_opt2_n − T_match_n‖_F = {np.linalg.norm(rho_opt2_n - rho_T_match_n):.4e}")
    print(f"  ‖ρ_opt1_n − ρ_opt2_n‖_F = {np.linalg.norm(rho_opt1_n - rho_opt2_n):.4e}")

    print(f"\n[C(t) comparison]")
    print(f"  {'t':>5} | {'C_ED':>10} | {'C_T_match':>10} | {'C_opt1':>10} | {'C_opt2':>10}")
    for t in TIMES:
        C_opt1 = np.real(np.trace(rho_opt1_n @ UOU[t] @ n0_op))
        C_opt2 = np.real(np.trace(rho_opt2_n @ UOU[t] @ n0_op))
        print(f"  {t:>5.2f} | {C_ED[t]:>+10.5f} | {C_T_match[t]:>+10.5f} | "
              f"{C_opt1:>+10.5f} | {C_opt2:>+10.5f}")


if __name__ == "__main__":
    main()
