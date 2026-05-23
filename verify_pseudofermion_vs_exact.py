"""
verify_pseudofermion_vs_exact.py — sanity check: PF-MC ensemble should
agree with exact-det MC ensemble at LOW K_E (where mixing is not an
issue), to within statistical noise.

At K_E=0.1 (low) the heat-bath chain mixes well and reaches the true
equilibrium quickly.  Both MC schemes target the SAME Boltzmann weight
e^{-S_g} · |det M|² (PF estimates it stochastically; exact computes it).
So ensemble averages of observables (plaquettes, fermion-bilinear sums)
should agree.

If they DON'T agree at low K_E, the PF implementation has a bug.
If they DO agree, the PF code is correct and we can move on to the
mixing-fix (parallel tempering).
"""
from __future__ import annotations
import sys
import time
import numpy as np

sys.path.insert(0, '/home/hlamm/Desktop/QC/logdet/m1_toy')

from action_z2_staggered import (
    LatticeGeometry, Z2GaugeConfig, gauge_action, build_dirac_matrix,
    compute_det_M,
)
from action_z2_metropolis import heatbath_sweep, metropolis_sweep, link_list
from action_z2_pseudofermion import heatbath_sweep_pf


def average_plaquette_parts(geom: LatticeGeometry, U: Z2GaugeConfig) -> tuple[float, float]:
    """Return (⟨xy plaq⟩, ⟨xτ+yτ plaq⟩)."""
    Lx, Ly, N_E = geom.Lx, geom.Ly, geom.N_E
    # xy plaquettes (magnetic)
    P_M, n_M = 0.0, 0
    for t in range(N_E):
        for x in range(Lx - 1):
            for y in range(Ly - 1):
                p = (U.U_x[t, x, y] * U.U_y[t, x+1, y]
                     * U.U_x[t, x, y+1] * U.U_y[t, x, y])
                P_M += p; n_M += 1
    # xτ + yτ plaquettes (electric)
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


def run_chain(geom, K_E, K_M, n_sweeps, n_warmup, seed, mode):
    """mode = 'exact' (heatbath_sweep with action_type=central) or 'pf'
    (heatbath_sweep_pf with per-sweep refresh).
    """
    rng = np.random.default_rng(seed)
    U = Z2GaugeConfig.random(geom, rng)
    PM_hist, PE_hist, det_sign_hist = [], [], []
    n_acc_total = 0; n_attempt_total = 0
    n_links = len(link_list(geom))
    for sw in range(n_warmup + n_sweeps):
        if mode == 'exact':
            U, det_M, S_g, n_acc = heatbath_sweep(
                geom, U, K_E=K_E, K_M=K_M, rng=rng, action_type='central')
        elif mode == 'exact_metro':
            U, det_M, S_g, n_acc = metropolis_sweep(
                geom, U, K_E=K_E, K_M=K_M, rng=rng, action_type='central')
        elif mode == 'pf':
            U, det_M, S_g, n_acc = heatbath_sweep_pf(
                geom, U, K_E=K_E, K_M=K_M, rng=rng, refresh='per_link')
        else:
            raise ValueError(mode)
        n_acc_total += n_acc; n_attempt_total += n_links
        if sw >= n_warmup:
            P_M, P_E = average_plaquette_parts(geom, U)
            PM_hist.append(P_M)
            PE_hist.append(P_E)
            det_sign_hist.append(1 if det_M > 0 else -1)
    return {
        'mode': mode,
        'PM': np.array(PM_hist),
        'PE': np.array(PE_hist),
        'det_sign': np.array(det_sign_hist),
        'accept': n_acc_total / n_attempt_total,
    }


def main():
    geom = LatticeGeometry(Lx=2, Ly=2, N_E=5, m=0.5)
    K_E = 0.1   # LOW, well-mixing
    K_M = 0.5
    n_warmup = 500
    n_sweeps = 1500
    print(f"V_4 = {geom.V_4}, K_E = {K_E}, K_M = {K_M}, m = {geom.m}")
    print(f"  n_warmup = {n_warmup}, n_sweeps = {n_sweeps}")
    print()

    print("Running exact-det heat-bath MC (4 chains)...")
    t0 = time.time()
    exact_runs = [run_chain(geom, K_E, K_M, n_sweeps, n_warmup,
                            seed=10*i, mode='exact') for i in range(4)]
    print(f"  wall: {time.time()-t0:.1f}s")
    print("Running exact-det METROPOLIS MC (4 chains, same mixing class as PF)...")
    t0 = time.time()
    exact_metro_runs = [run_chain(geom, K_E, K_M, n_sweeps, n_warmup,
                                  seed=10*i+5, mode='exact_metro') for i in range(4)]
    print(f"  wall: {time.time()-t0:.1f}s")
    print("Running PF MC (4 chains)...")
    t0 = time.time()
    pf_runs = [run_chain(geom, K_E, K_M, n_sweeps, n_warmup,
                         seed=10*i+1, mode='pf') for i in range(4)]
    print(f"  wall: {time.time()-t0:.1f}s")

    def row(runs, key):
        means = np.array([r[key].mean() for r in runs])
        return means.mean(), means.std(ddof=1) / np.sqrt(len(means))

    print()
    header = f"{'obs':>10} {'heat-bath':>18} {'exact-Metro':>18} {'PF-Metro':>18}"
    print(header)
    for label, key in [('⟨P_M⟩', 'PM'), ('⟨P_E⟩', 'PE'), ('⟨sign(det)⟩', 'det_sign')]:
        m1, s1 = row(exact_runs, key)
        m2, s2 = row(exact_metro_runs, key)
        m3, s3 = row(pf_runs, key)
        print(f"{label:>10}  {m1:+.4f}±{s1:.4f}  {m2:+.4f}±{s2:.4f}  {m3:+.4f}±{s3:.4f}")
    print(f"{'accept':>10}  {np.mean([r['accept'] for r in exact_runs]):.3f}             "
          f"{np.mean([r['accept'] for r in exact_metro_runs]):.3f}             "
          f"{np.mean([r['accept'] for r in pf_runs]):.3f}")


if __name__ == "__main__":
    main()
