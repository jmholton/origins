#!/usr/bin/env ccp4-python
# -*- coding: utf-8 -*-
"""
nearest_altindex.py - for two PDBs already in the same cell, find the
LSQ rigid-body fit between them, then rank crystallographically-allowed
(altindex op) x (symop) combinations by how well they reproduce that fit.

The point: an LSQ rotation is a continuous rigid-body answer; origins.com
only searches over a discrete set of altindex+symop combinations plus an
origin grid.  This script tells you which discrete combination is closest
to the LSQ truth, so you know what answer origins.com *could* have found
(if it explored the right basin).

Usage:
    ccp4-python nearest_altindex.py <moving.pdb> <reference.pdb> [altops.txt]

altops.txt defaults to ./origins_tempall_symops (the list origins.com
generates from othercell + symop.lib for this cell and lattice).
"""
import os
import sys
import numpy as np
import gemmi

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from xyz_to_hkl_op import parse_op   # noqa
from lsq_fit import collect_atoms, kabsch   # noqa
from gemmi_altindex import enumerate_alt_ops, fmt_op as fmt_alt_op   # noqa


def ortho_matrices(cell):
    O = np.zeros((3, 3))
    for i, basis in enumerate([(1, 0, 0), (0, 1, 0), (0, 0, 1)]):
        p = cell.orthogonalize(gemmi.Fractional(*basis))
        O[:, i] = [p.x, p.y, p.z]
    return O, np.linalg.inv(O)


def parse_ops_file(path):
    out = []
    for s in open(path):
        s = s.strip()
        if not s:
            continue
        R, t = parse_op(s)
        out.append((s, R, t))
    return out


def spacegroup_ops(structure):
    sg_name = structure.spacegroup_hm or "P 1"
    sg = gemmi.find_spacegroup_by_name(sg_name)
    if sg is None:
        sys.exit(f"unknown spacegroup {sg_name!r}")
    out = []
    for op in sg.operations():
        triplet = op.triplet().upper()
        R, t = parse_op(triplet)
        out.append((triplet, R, t))
    return out


def rot_deviation_deg(R_a, R_b):
    """Rotation angle (deg) of R_a · R_bᵀ.  Both arguments MUST be orthogonal
    (proper or improper); otherwise the formula returns garbage."""
    D = R_a @ R_b.T
    cos_t = max(-1.0, min(1.0, (np.trace(D) - 1.0) / 2.0))
    return float(np.degrees(np.arccos(cos_t)))


def is_metric_preserving(R_frac, G, atol=1e-3):
    """True iff R_frac (fractional rotation matrix) preserves the lattice
    metric tensor G.  Equivalently, O·R_frac·F is orthogonal in cartesian.
    Non-metric-preserving ops are alt-cell-derived; in the original cell
    they are shears, not rotations, so the rotation-deviation formula
    cannot be used directly on them."""
    return np.allclose(R_frac.T @ G @ R_frac, G, atol=atol * np.max(np.abs(G)))


def trans_residual_A(t_lsq_cart, t_cand_cart, O, F):
    """Residual translation after subtracting candidate's t and wrapping
    to nearest-image in fractional coords (since origin shifts are mod-1)."""
    t_diff_frac = F @ (t_lsq_cart - t_cand_cart)
    t_diff_frac = ((t_diff_frac + 0.5) % 1.0) - 0.5
    return float(np.linalg.norm(O @ t_diff_frac))


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    moving_pdb = sys.argv[1]
    ref_pdb = sys.argv[2]
    alt_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, "origins_tempall_symops")
    use_gemmi_fallback = not os.path.isfile(alt_path)
    if use_gemmi_fallback:
        print(f"NOTE: {alt_path} not found; generating altindex ops with gemmi_altindex.")

    moving = gemmi.read_structure(moving_pdb)
    ref = gemmi.read_structure(ref_pdb)

    # Cells need not match — Kabsch is purely cartesian.  The cell only
    # enters when we convert candidate altindex ops to cartesian rotations.
    # We use the *moving's* cell (the one whose altindex search space we're
    # asking about), matching origins.com's own convention.
    if not np.allclose(
        [ref.cell.a, ref.cell.b, ref.cell.c, ref.cell.alpha, ref.cell.beta, ref.cell.gamma],
        [moving.cell.a, moving.cell.b, moving.cell.c, moving.cell.alpha, moving.cell.beta, moving.cell.gamma],
        atol=1e-2,
    ):
        print(f"NOTE: cells differ; using moving's cell for altindex conversion.")
        print(f"  moving:    {moving.cell.parameters}")
        print(f"  reference: {ref.cell.parameters}")
        print()

    # 1. Kabsch fit
    moving_atoms = dict(collect_atoms(moving))
    ref_atoms = dict(collect_atoms(ref))
    common = sorted(set(moving_atoms) & set(ref_atoms))
    if not common:
        sys.exit("no atoms matched by (chain, resSeq, name)")
    A = np.array([moving_atoms[k] for k in common])
    B = np.array([ref_atoms[k] for k in common])
    R_lsq, t_lsq, rmsd_after = kabsch(A, B)
    rmsd_before = float(np.sqrt(np.mean(np.sum((A - B) ** 2, axis=1))))
    print(f"LSQ fit: {len(common)} matched atoms")
    print(f"  RMSD before = {rmsd_before:.2f} A    after = {rmsd_after:.2f} A")
    print(f"  R_lsq    rotation angle = {rot_deviation_deg(R_lsq, np.eye(3)):.2f} deg")
    print(f"  t_lsq    cartesian      = {np.round(t_lsq, 3)}")
    print()

    # 2. cell matrices — use moving's cell (its altindex search space)
    O, F = ortho_matrices(moving.cell)

    # 3. candidate ops
    if use_gemmi_fallback:
        gemmi_ops = enumerate_alt_ops(moving.cell, moving.spacegroup_hm)
        alt_ops = []
        for R_flat, t in sorted(gemmi_ops):
            R = np.array(R_flat).reshape(3, 3).astype(float)
            t_arr = np.array(t)
            alt_ops.append((fmt_alt_op(R, t_arr), R, t_arr))
        alt_src = "gemmi_altindex"
    else:
        alt_ops = parse_ops_file(alt_path)
        alt_src = os.path.basename(alt_path)
    sym_ops = spacegroup_ops(moving)
    print(f"  altindex ops: {len(alt_ops)} (from {alt_src})")
    print(f"  symmetry ops: {len(sym_ops)} (from {moving.spacegroup_hm})")
    print()

    # 4. rank — only metric-preserving (alt × sym) combinations are valid
    # rotations in this cell; alt-cell-derived ops are shears here and would
    # need lattCELL re-expression to be applied correctly.
    G = O.T @ O
    rows = []
    n_skipped = 0
    for alt_str, alt_R, alt_t in alt_ops:
        for sym_str, sym_R, sym_t in sym_ops:
            R_comb = sym_R @ alt_R
            if not is_metric_preserving(R_comb, G):
                n_skipped += 1
                continue
            t_comb = sym_R @ alt_t + sym_t
            R_cart = O @ R_comb @ F
            t_cart = O @ t_comb
            d_rot = rot_deviation_deg(R_lsq, R_cart)
            d_t = trans_residual_A(t_lsq, t_cart, O, F)
            rows.append((d_rot, d_t, alt_str, sym_str))
    rows.sort()
    print(f"  metric-preserving combinations: {len(rows)}    "
          f"skipped (alt-cell shears in this cell): {n_skipped}")
    print()

    print(f"top 15 (alt x sym) combinations ranked by rotational deviation:")
    print(f"{'rank':>4}  {'drot':>6}  {'dt(A)':>7}  {'altindex op':<28s}  symop")
    print("-" * 80)
    for i, (a, t, alts, syms) in enumerate(rows[:15], 1):
        print(f"{i:>4}  {a:>6.2f}  {t:>7.2f}  {alts:<28s}  {syms}")

    # Apply top-ranked op to moving and write PDB.
    if not rows:
        print("\nno metric-preserving combinations found; no PDB written")
        return
    drot, dt, top_alt, top_sym = rows[0]
    alt_R, alt_t = parse_op(top_alt)
    sym_R, sym_t = parse_op(top_sym)
    R_comb = sym_R @ alt_R
    R_cart = O @ R_comb @ F
    # Optimal cartesian translation given R_cart fixed: best COM alignment
    # of matched atoms (dropping the discrete t_comb in favor of LSQ-optimal t)
    A_rot = (R_cart @ A.T).T
    t_opt = B.mean(axis=0) - A_rot.mean(axis=0)
    out = moving.clone()
    for m in out:
        for c in m:
            for r in c:
                for atom in r:
                    p = np.array([atom.pos.x, atom.pos.y, atom.pos.z])
                    q = R_cart @ p + t_opt
                    atom.pos = gemmi.Position(float(q[0]), float(q[1]), float(q[2]))
    out.cell = moving.cell
    out.spacegroup_hm = moving.spacegroup_hm
    out_path = (sys.argv[4] if len(sys.argv) > 4
                else os.path.splitext(os.path.basename(moving_pdb))[0] + "_after_top_altindex.pdb")
    out.write_pdb(out_path)
    # report fit quality
    fit = A_rot + t_opt
    rmsd_top = float(np.sqrt(np.mean(np.sum((fit - B) ** 2, axis=1))))
    print(f"\ntop op applied to {os.path.basename(moving_pdb)} -> {out_path}")
    print(f"  rotation:  drot = {drot:.2f} deg from LSQ-optimal")
    print(f"  RMSD of fitted moving vs reference: {rmsd_top:.2f} A   "
          f"(LSQ baseline: {rmsd_after:.2f} A)")


if __name__ == "__main__":
    main()
