"""Deterministic path-integral test (no MC) — Phase 28.

Strip the Monte Carlo and sum the EXACT Lagrangian path integral over all
2^(8 N_E − 4) Z_2 gauge configurations.  If the construction limits to
e^{-β H_QC} as a_τ → 0, then the pipeline is sound and the only blocker
is the MC sign problem.  If a STRUCTURAL gap remains at a_τ → 0,
there's a bug in W / det M / gauge action that's been hiding behind MC
noise for months.

Geometry: Lx=Ly=2, β=0.5.  Per-slice spatial links = 4 (2 U_x + 2 U_y),
per-slice-transition temporal links = 4.  Total = 8 N_E − 4.
    N_E=2 → 12 links, 4096 configs.
    N_E=3 → 20 links, ~1M configs.
"""
from __future__ import annotations
import os
# Pin BLAS to 1 thread BEFORE importing numpy — otherwise each worker spawns
# its own BLAS thread pool and we oversubscribe the cores.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import sys, time, itertools
import numpy as np
from multiprocessing import Pool
from scipy.linalg import expm

sys.path.insert(0, '/home/hlamm/Desktop/QC/logdet/m1_toy')

# Use the SAME Hamiltonian and couplings as the production pipeline.
import epoq_classical_sampler as cs
import epoq_quantum_simulator as qs

from action_z2_staggered import LatticeGeometry, Z2GaugeConfig
from action_z2_staggered_forwardtime import compute_det_M_forward
from action_z2_kernels import gauge_action_kernel, warm_up
from transfer_matrix_kbc_trotterized import compute_combined_weight_trotter
from action_minkowski_stitch import gauge_qc_bits_from_slice
from action_corner_direct_v3 import _fock_index_to_psi_map


# --- couplings: align with epoq_classical_sampler defaults ---------
M_HAM   = cs.M_MASS   # 0.5
G_E_HAM = cs.G_E      # 1.0
G_M_HAM = cs.G_M      # 0.5
G_HOP   = cs.G_HOP    # 0.5
BETA    = 0.5         # small enough to enumerate

W_ORDER = 2  # palindrome (Hermitian PD per-config weight)


# ------------------------------------------------------------------
# Bit-pack / unpack: encode (U_x, U_y, U_t) as a single 64-bit integer
# ------------------------------------------------------------------
def n_links(N_E: int) -> tuple[int, int, int, int]:
    """Returns (nx, ny, nt, total)."""
    Lx = Ly = 2
    nx = N_E * (Lx - 1) * Ly        # 2 * N_E
    ny = N_E * Lx * (Ly - 1)        # 2 * N_E
    nt = (N_E - 1) * Lx * Ly        # 4 * (N_E - 1)
    return nx, ny, nt, nx + ny + nt


def decode_to_U(idx: int, N_E: int) -> Z2GaugeConfig:
    """Bit-decode integer idx into a Z_2 gauge config.
    bit 0..nx-1   → U_x (slowest-varying axis: t, then x, then y)
    bit nx..nx+ny-1 → U_y
    bit nx+ny..total → U_t
    """
    Lx = Ly = 2
    nx, ny, nt, total = n_links(N_E)
    geom = LatticeGeometry(Lx=Lx, Ly=Ly, N_E=N_E, m=0.5)  # dummy m; det uses real m later
    U_x = np.empty((N_E, Lx - 1, Ly), dtype=np.int8)
    U_y = np.empty((N_E, Lx, Ly - 1), dtype=np.int8)
    U_t = np.empty((N_E - 1, Lx, Ly), dtype=np.int8)
    b = 0
    for t in range(N_E):
        for x in range(Lx - 1):
            for y in range(Ly):
                U_x[t, x, y] = 1 if (idx >> b) & 1 == 0 else -1
                b += 1
    for t in range(N_E):
        for x in range(Lx):
            for y in range(Ly - 1):
                U_y[t, x, y] = 1 if (idx >> b) & 1 == 0 else -1
                b += 1
    for t in range(N_E - 1):
        for x in range(Lx):
            for y in range(Ly):
                U_t[t, x, y] = 1 if (idx >> b) & 1 == 0 else -1
                b += 1
    return Z2GaugeConfig(geom=geom, U_x=U_x, U_y=U_y, U_t=U_t)


# ------------------------------------------------------------------
# Per-config contribution: det_M · e^{-S_g} · W_palindrome
# ------------------------------------------------------------------
def per_config_contribution(args):
    """Compute (w_total · W_psi block, g_top, g_bot) for one gauge config."""
    idx, N_E, a_tau, K_E, K_M, m_action = args
    Lx = Ly = 2
    geom = LatticeGeometry(Lx=Lx, Ly=Ly, N_E=N_E, m=m_action)
    U = decode_to_U(idx, N_E)
    U.geom = geom  # re-attach correct geom

    S_g = gauge_action_kernel(
        Lx, Ly, N_E, K_E, K_M,
        U.U_x.astype(np.float64),
        U.U_y.astype(np.float64),
        U.U_t.astype(np.float64),
    )
    boltz = np.exp(-S_g)
    # NO det M — W already covers the fermion sector for OPEN boundaries
    # (det M is for fermion-traced closed loops, which would double-count).
    w_total = boltz

    # Palindrome W = B† B with B = ∏_{t=0}^{N_E-2} T_F^{1/2}[U(t)]
    W_full, _ = compute_combined_weight_trotter(
        geom, U, a_tau=a_tau,
        K_E=0.0, K_M=0.0,            # gauge factors handled by e^{-S_g} above
        m_obs=M_HAM, g_hop=G_HOP, order=W_ORDER,
    )
    # Re-index from lex Fock basis → ψ-bit basis.
    V3 = geom.V_3
    idx_to_psi = _fock_index_to_psi_map(V3)
    dim_F = 2 ** V3
    W_psi = np.zeros((dim_F, dim_F), dtype=complex)
    for li in range(dim_F):
        for lj in range(dim_F):
            W_psi[idx_to_psi[li], idx_to_psi[lj]] = W_full[li, lj]

    g_top = gauge_qc_bits_from_slice(U, N_E - 1)
    g_bot = gauge_qc_bits_from_slice(U, 0)
    return idx, w_total, W_psi, g_top, g_bot


def build_rho_deterministic(N_E: int, a_tau: float, n_workers: int = 8) -> np.ndarray:
    """Enumerate ALL 2^total configs and accumulate ρ̃ exactly."""
    Lx = Ly = 2
    K_E = -0.5 * np.log(np.tanh(a_tau * G_E_HAM))
    K_M = a_tau * G_M_HAM
    m_action = a_tau * M_HAM   # action-form mass (a_τ · m_Ham, matches MC convention)

    _, _, _, total = n_links(N_E)
    n_configs = 1 << total
    print(f"  N_E={N_E}: enumerating 2^{total} = {n_configs} configs "
          f"with {n_workers} workers", flush=True)

    args = ((i, N_E, a_tau, K_E, K_M, m_action) for i in range(n_configs))
    rho = np.zeros((256, 256), dtype=complex)
    t0 = time.time()
    completed = 0
    report_every = max(1, n_configs // 20)

    with Pool(n_workers, initializer=warm_up) as pool:
        for idx, w_total, W_psi, g_top, g_bot in pool.imap_unordered(
                per_config_contribution, args, chunksize=max(1, n_configs // (n_workers * 32))):
            completed += 1
            if W_psi is None:
                continue
            for psi_top in range(16):
                for psi_bot in range(16):
                    bit_a = (psi_top << 4) | g_top
                    bit_b = (psi_bot << 4) | g_bot
                    rho[bit_a, bit_b] += w_total * W_psi[psi_top, psi_bot]
            if completed % report_every == 0:
                elapsed = time.time() - t0
                print(f"    {completed}/{n_configs} ({100*completed/n_configs:.1f}%), "
                      f"elapsed {elapsed:.0f}s", flush=True)

    print(f"  done in {time.time()-t0:.0f}s", flush=True)
    # Hermitize per EρOQ paper convention
    rho_herm = (rho + rho.conj().T) / 2
    return rho_herm


# ------------------------------------------------------------------
# Comparisons to ED
# ------------------------------------------------------------------
def build_H_QC_dense() -> np.ndarray:
    """Build dense 256×256 H_QC from epoq_classical_sampler Pauli terms."""
    P2 = {'I': np.eye(2, dtype=complex),
          'X': np.array([[0, 1], [1, 0]], dtype=complex),
          'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
          'Z': np.diag([1, -1]).astype(complex)}
    def pauli(factors, nq=8):
        by_q = {q: P2['I'] for q in range(nq)}
        for q, ax in factors:
            by_q[q] = P2[ax]
        r = by_q[nq - 1]
        for q in range(nq - 2, -1, -1):
            r = np.kron(r, by_q[q])
        return r
    H = sum(c * pauli(f) for c, f in cs.build_pauli_terms())
    return (H + H.conj().T) / 2


def report(rho_lat: np.ndarray, rho_ED: np.ndarray, H: np.ndarray, label: str,
           times=(0.0, 0.25, 0.5)):
    print(f"\n--- {label} ---")
    # Trace normalize
    Z_lat = np.trace(rho_lat).real
    Z_ED = np.trace(rho_ED).real
    rl = rho_lat / Z_lat
    re = rho_ED / Z_ED
    print(f"  Z_lat = {Z_lat:+.6e},   Z_ED = {Z_ED:+.6e}")
    print(f"  Im(Tr ρ̃) = {np.trace(rho_lat).imag:+.3e}  (should be ~0)")

    eigs = np.linalg.eigvalsh(rl).real
    eigs.sort()
    n_neg = int((eigs < -1e-9).sum())
    print(f"  min_eig(ρ̃_norm) = {eigs[0]:+.4e}, max_eig = {eigs[-1]:+.4e}, "
          f"# neg = {n_neg}/256")
    print(f"  ‖Δρ‖_F = {np.linalg.norm(rl - re):.5f}")
    print(f"  max |ρ̃_lat - ρ̃_ED| = {np.max(np.abs(rl - re)):.5f}")

    # C(t) = Tr[ρ̃ · U†(t) n₀ U(t) · n₀]
    n0 = 0.5 * np.eye(256, dtype=complex) - 0.5 * _z_on_q(4)
    print(f"  {'t':>5} | {'C_lat':>12} | {'C_ED':>12} | gap")
    for t in times:
        Ut = expm(-1j * H * t)
        UOU = Ut.conj().T @ n0 @ Ut
        C_lat = np.real(np.trace(rl @ UOU @ n0))
        C_ED  = np.real(np.trace(re @ UOU @ n0))
        print(f"  {t:>5.2f} | {C_lat:>+12.6f} | {C_ED:>+12.6f} | "
              f"{C_lat - C_ED:+.5f}")


def _z_on_q(q: int) -> np.ndarray:
    """Z on qubit q (q0=LSB), 256-dim."""
    out = np.array([[1.0]], dtype=complex)
    for k in reversed(range(8)):
        m = np.diag([1, -1]).astype(complex) if k == q else np.eye(2, dtype=complex)
        out = np.kron(out, m)
    return out


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    print("=" * 80, flush=True)
    print("DETERMINISTIC PATH-INTEGRAL LIMIT TEST (no MC) — Phase 28", flush=True)
    print(f"  Lx=Ly=2, β={BETA}, palindrome W (order={W_ORDER})", flush=True)
    print(f"  couplings: m={M_HAM}, g_E={G_E_HAM}, g_M={G_M_HAM}, g_hop={G_HOP}", flush=True)
    print("=" * 80, flush=True)

    print("\n[1] Building ED reference at β=0.5 ...", flush=True)
    H = build_H_QC_dense()
    rho_ED = expm(-BETA * H)
    print(f"    H Hermitian: ‖H - H†‖ = {np.linalg.norm(H - H.conj().T):.2e}", flush=True)
    print(f"    Tr e^{{-βH}} = {np.trace(rho_ED).real:.6f}", flush=True)

    # ---------- a_τ = 0.5, N_E = 2  (4096 configs) ----------
    a_tau = 0.5
    N_E = int(round(BETA / a_tau)) + 1
    assert N_E == 2
    print(f"\n[2] DETERMINISTIC sum @ a_τ={a_tau}, N_E={N_E}", flush=True)
    t0 = time.time()
    rho_a050 = build_rho_deterministic(N_E=N_E, a_tau=a_tau, n_workers=6)
    print(f"    walltime: {time.time()-t0:.0f}s", flush=True)
    report(rho_a050, rho_ED, H, label=f"a_τ={a_tau} (N_E={N_E})")

    # ---------- a_τ = 0.25, N_E = 3  (1M configs) -----------
    a_tau = 0.25
    N_E = int(round(BETA / a_tau)) + 1
    assert N_E == 3
    print(f"\n[3] DETERMINISTIC sum @ a_τ={a_tau}, N_E={N_E}", flush=True)
    t0 = time.time()
    rho_a025 = build_rho_deterministic(N_E=N_E, a_tau=a_tau, n_workers=6)
    print(f"    walltime: {time.time()-t0:.0f}s", flush=True)
    report(rho_a025, rho_ED, H, label=f"a_τ={a_tau} (N_E={N_E})")

    # ---------- Convergence summary ----------
    print("\n" + "=" * 80)
    print("CONVERGENCE CHECK")
    print("=" * 80)
    rl1 = rho_a050 / np.trace(rho_a050).real
    rl2 = rho_a025 / np.trace(rho_a025).real
    re  = rho_ED / np.trace(rho_ED).real
    d1 = np.linalg.norm(rl1 - re)
    d2 = np.linalg.norm(rl2 - re)
    print(f"  ‖ρ̃(a_τ=0.50) - ρ_ED‖_F = {d1:.5f}")
    print(f"  ‖ρ̃(a_τ=0.25) - ρ_ED‖_F = {d2:.5f}")
    print(f"  ratio (expect ~0.25 if O(a_τ²)): {d2/d1:.3f}")
    if d2 < d1 * 0.5:
        print("  → Pipeline appears to LIMIT correctly toward H_QC.")
        print("    The MC sign problem is the remaining obstacle.")
    elif d2 < d1 * 0.9:
        print("  → Slow convergence trend visible.  Need a_τ=0.125 to confirm.")
    else:
        print("  → NO convergence trend — possible bug in W/det M/S_g construction.")


if __name__ == "__main__":
    main()
