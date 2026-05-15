#!/usr/bin/env ccp4-python
# -*- coding: utf-8 -*-
"""
test_phase_transform.py - verify the phase-transformation rule for
alt-indexing a structure factor set:

    F'(h) = F(R^T . h) * exp(2 pi i h . t)

The test stays entirely in memory (no PDB I/O) to avoid the 1e-3 A
position rounding in PDB files, which otherwise pollutes the comparison
at high-resolution reflections.

Recipe:
  1. Random N atoms in the ASU of the target SG (fractional coords, NumPy).
  2. Expand by every SG op (incl. centring) to a P1-style set in memory.
  3. Apply the alt-indexing op (R, t) to all P1 atoms in memory.
  4. For a representative HKL grid, compute F_orig(h_old) and F_truth(h_new)
     by direct exp-sum (no form factors, no B-factors - the formula is
     independent of those, since they depend only on |h*| which is
     preserved by metric-preserving R).
  5. Compare F_pred(h_new) = F_orig(R^T h_new) * exp(2 pi i h_new . t)
     vs F_truth(h_new).

Test ops MUST satisfy R^T G R = G (i.e., be in the lattice holohedry of
the cell), or h_new and R^T h_new have different d-spacings and the test
question becomes ill-posed.
"""
import os
import sys
import numpy as np
import gemmi

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from xyz_to_hkl_op import parse_op  # noqa

PRESETS = {
    "P4": {
        "cell": (60.0, 60.0, 80.0, 90.0, 90.0, 90.0),
        "sg":   "P 4",
        # 2[110] is in P 4 2 2 (the lattice holohedry's chiral subgroup
        # along c) but not in P 4. Both metric-preserving and non-trivial.
        "ops":  [("2_110_symm",     "y, x, -z"),
                 ("2_110_nonsymm",  "y+1/2, x+1/2, -z")],
    },
    "H32": {
        "cell": (77.31, 77.31, 330.36, 90.0, 90.0, 120.0),
        "sg":   "H 3 2",
        # 6+ rotation along c: in 6/mmm (the H lattice holohedry) but not
        # in H 3 2, so a true alt-indexing AND metric-preserving in the
        # original cell. Non-symmorphic variant adds an R-centring shift.
        "ops":  [("6plus_symm",     "x-y, x, z"),
                 ("6plus_nonsymm",  "x-y+2/3, x+1/3, z+1/3")],
    },
}

NATOM = 200
SEED = 1
HMAX = 20   # generate test reflections up to this index in each direction
KMAX = 20
LMAX = 40
# Memory budget: |H|*|K|*|L|*N_atoms*16 bytes per chunk; chunk in F_bare.


def random_asu_atoms(sg_name, n_atoms):
    """Return Nx3 array of random fractional positions in the SG ASU."""
    rng = np.random.default_rng(SEED)
    return rng.random((n_atoms, 3))


def expand_to_p1(asu_atoms, sg_name):
    """Apply each SG op (incl. centring) to every ASU atom; return Mx3
    fractional positions, wrapped into [0,1)."""
    sg = gemmi.find_spacegroup_by_name(sg_name)
    out = []
    for op in sg.operations():
        R = np.array(op.rot, dtype=float) / op.DEN
        t = np.array(op.tran, dtype=float) / op.DEN
        out.append((asu_atoms @ R.T + t) % 1.0)
    return np.vstack(out)


def apply_op(atoms_frac, R, t):
    """Apply (R, t) in fractional coords; wrap mod 1."""
    return (atoms_frac @ R.T + t) % 1.0


def F_bare(atoms_frac, hkls, chunk=20000):
    """Bare exp-sum structure factor F(h) for an Nx3 atom array and an Mx3
    HKL array. Returns Mx1 complex array. Chunked over hkls to bound memory
    (chunk * N_atoms * 16 bytes per chunk)."""
    out = np.empty(len(hkls), dtype=complex)
    for i in range(0, len(hkls), chunk):
        H = hkls[i:i + chunk]
        phase = 2 * np.pi * (H @ atoms_frac.T)
        out[i:i + chunk] = np.exp(1j * phase).sum(axis=1)
    return out


def metric_tensor(cell_tuple):
    cell = gemmi.UnitCell(*cell_tuple)
    O = np.zeros((3, 3))
    for i, b in enumerate([(1, 0, 0), (0, 1, 0), (0, 0, 1)]):
        p = cell.orthogonalize(gemmi.Fractional(*b))
        O[:, i] = [p.x, p.y, p.z]
    return O.T @ O


def is_metric_preserving(R, G, atol=1e-6):
    return np.allclose(R.T @ G @ R, G, atol=atol * np.max(np.abs(G)))


def hkl_grid(hmax=HMAX, kmax=KMAX, lmax=LMAX):
    """Generate a 3D grid of HKLs (skipping origin)."""
    hh, kk, ll = np.meshgrid(
        np.arange(-hmax, hmax + 1),
        np.arange(-kmax, kmax + 1),
        np.arange(-lmax, lmax + 1),
        indexing="ij",
    )
    grid = np.stack([hh, kk, ll], axis=-1).reshape(-1, 3)
    return grid[~(grid == 0).all(axis=1)]   # drop (0,0,0)


def signed_angular_diff_deg(a, b):
    return ((a - b + 180.0) % 360.0) - 180.0


def run_test(label, op_str, A_orig, cell_tuple):
    R, t = parse_op(op_str)
    G = metric_tensor(cell_tuple)
    if not is_metric_preserving(R, G):
        return {"label": label, "op": op_str, "verdict": "SKIPPED (not metric-preserving)",
                "n": 0, "rel_amp_err": float("nan"), "max_phi_err_deg": float("nan")}

    A_new = apply_op(A_orig, R, t)
    H = hkl_grid().astype(float)

    # Truth: F at h_new on the transformed atoms
    F_truth = F_bare(A_new, H)
    # Predicted: F_orig at h_old = R^T h_new, then phase-shifted
    H_old = (H @ R)   # H @ R is (R^T H^T)^T row-by-row -- equivalent to H · R^T per-row
    # Wait: we want h_old per row = R^T h_new for that row. (h_new is a row, treat as col vector).
    # h_old_col = R^T @ h_new_col -> h_old_row = h_new_row @ R
    F_orig_at_old = F_bare(A_orig, H_old)
    phase_shift = np.exp(2j * np.pi * (H @ t))
    F_pred = F_orig_at_old * phase_shift

    # Compare
    mask = (np.abs(F_truth) > 1.0) & (np.abs(F_orig_at_old) > 1.0)
    if mask.sum() == 0:
        return {"label": label, "op": op_str, "verdict": "no usable refls", "n": 0}
    F_pred_m = F_pred[mask]
    F_truth_m = F_truth[mask]
    rel_amp_err = float(np.max(np.abs(F_pred_m - F_truth_m)) / np.mean(np.abs(F_truth_m)))
    phi_pred = np.degrees(np.angle(F_pred_m))
    phi_truth = np.degrees(np.angle(F_truth_m))
    max_phi_err = float(np.max(np.abs(signed_angular_diff_deg(phi_pred, phi_truth))))
    verdict = "PASS" if (rel_amp_err < 1e-10 and max_phi_err < 1e-8) else "FAIL"
    return {
        "label": label, "op": op_str, "n": int(mask.sum()),
        "rel_amp_err": rel_amp_err, "max_phi_err_deg": max_phi_err,
        "verdict": verdict,
    }


def main():
    preset = sys.argv[1] if len(sys.argv) > 1 else "P4"
    if preset not in PRESETS:
        sys.exit(f"unknown preset {preset!r}; choose from {list(PRESETS)}")
    cfg = PRESETS[preset]

    asu = random_asu_atoms(cfg["sg"], NATOM)
    A_orig = expand_to_p1(asu, cfg["sg"])
    sg = gemmi.find_spacegroup_by_name(cfg["sg"])
    n_sym = len(list(sg.operations()))

    print(f"preset = {preset}")
    print(f"cell   = {cfg['cell']}")
    print(f"sg     = {cfg['sg']} ({n_sym} ops incl. centring)")
    print(f"natom  = {NATOM} ASU x {n_sym} = {len(A_orig)} P1 atoms (in memory)")
    print(f"hkls   = +/-{HMAX} x +/-{KMAX} x +/-{LMAX} = {(2*HMAX+1)*(2*KMAX+1)*(2*LMAX+1)-1} test indices")
    print()

    results = [run_test(label, op, A_orig, cfg["cell"])
               for label, op in cfg["ops"]]

    hdr = (f"{'op':<24s}  {'N_used':>8s}  {'max|dF|/<F>':>12s}  "
           f"{'max dphi (deg)':>14s}  verdict")
    print(hdr); print("-" * len(hdr))
    for r in results:
        if r.get("rel_amp_err") is None or np.isnan(r.get("rel_amp_err", float("nan"))):
            print(f"{r['op']:<24s}  ----      ----          ----            {r['verdict']}")
        else:
            print(f"{r['op']:<24s}  {r['n']:>8d}  {r['rel_amp_err']:>12.2e}  "
                  f"{r['max_phi_err_deg']:>14.2e}  {r['verdict']}")


if __name__ == "__main__":
    main()
