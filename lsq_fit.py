#!/usr/bin/env ccp4-python
# -*- coding: utf-8 -*-
"""
lsq_fit.py - least-squares superposition of two PDBs (Kabsch).

Matches atoms by (chain, resSeq, atom name) and finds the rigid rotation+
translation that minimizes RMSD of one onto the other.  Decomposes the
rotation into axis+angle, and reports whether it looks like a crystallographic
alt-indexing operation (180 deg about a low-index direction).

Usage:
    ccp4-python lsq_fit.py <moving.pdb> <reference.pdb> [<out.pdb>]

The moving structure is rotated/translated to fit the reference; if <out.pdb>
is given, the transformed moving structure is written there.
"""
import sys
from pathlib import Path

import gemmi
import numpy as np


def collect_atoms(structure):
    """Return list of (key, position) where key=(chain, resSeq, name)."""
    out = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    key = (chain.name, residue.seqid.num, atom.name.strip())
                    out.append((key, np.array([atom.pos.x, atom.pos.y, atom.pos.z])))
    return out


def kabsch(A, B):
    """Optimal rotation R and translation t such that  R @ A + t  best fits B.
    Returns (R, t, rmsd)."""
    cA = A.mean(axis=0)
    cB = B.mean(axis=0)
    Ac = A - cA
    Bc = B - cB
    H = Ac.T @ Bc                    # 3x3 cross-covariance
    U, _S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])           # avoid reflection
    R = Vt.T @ D @ U.T
    t = cB - R @ cA
    fitted = (A @ R.T) + t
    rmsd = float(np.sqrt(np.mean(np.sum((fitted - B) ** 2, axis=1))))
    return R, t, rmsd


def axis_angle(R):
    """Return (axis_unit_vector, angle_deg) for rotation matrix R."""
    cos_t = max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))
    angle = np.degrees(np.arccos(cos_t))
    if abs(angle) < 1e-3 or abs(angle - 180) < 1e-3:
        # eigen-decomposition for axis (R has eigenvalue +1 on its rotation axis)
        evals, evecs = np.linalg.eig(R)
        idx = np.argmin(np.abs(evals - 1.0))
        axis = np.real(evecs[:, idx])
    else:
        axis = np.array([R[2, 1] - R[1, 2],
                         R[0, 2] - R[2, 0],
                         R[1, 0] - R[0, 1]])
        axis /= (2 * np.sin(np.radians(angle)))
    n = np.linalg.norm(axis)
    if n > 0:
        axis = axis / n
    return axis, float(angle)


def apply_to_structure(structure, R, t):
    """Return a copy of structure with all atoms transformed by R x + t."""
    new = structure.clone()
    for model in new:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    p = np.array([atom.pos.x, atom.pos.y, atom.pos.z])
                    q = R @ p + t
                    atom.pos = gemmi.Position(float(q[0]), float(q[1]), float(q[2]))
    return new


def fractional_axis(axis, cell):
    """Convert orthogonal-frame axis vector to fractional indices, normalized
    to integers (best 3-int approximation)."""
    # build (de)orthogonalization matrix
    frac = np.array([cell.fractionalize(gemmi.Position(*axis)).x,
                     cell.fractionalize(gemmi.Position(*axis)).y,
                     cell.fractionalize(gemmi.Position(*axis)).z])
    # subtract the fractionalized origin (Position(0,0,0)) to get pure direction
    origin = np.array([cell.fractionalize(gemmi.Position(0, 0, 0)).x,
                       cell.fractionalize(gemmi.Position(0, 0, 0)).y,
                       cell.fractionalize(gemmi.Position(0, 0, 0)).z])
    direction = frac - origin
    # rescale so the largest |component| ~ 1, then round
    m = np.max(np.abs(direction))
    if m == 0:
        return direction
    return direction / m


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    moving_path = Path(sys.argv[1])
    ref_path    = Path(sys.argv[2])
    out_path    = Path(sys.argv[3]) if len(sys.argv) >= 4 else None

    moving = gemmi.read_structure(str(moving_path))
    ref    = gemmi.read_structure(str(ref_path))

    moving_atoms = dict(collect_atoms(moving))
    ref_atoms    = dict(collect_atoms(ref))
    common = sorted(set(moving_atoms) & set(ref_atoms))
    if not common:
        sys.exit("ERROR: no atoms matched by (chain, resSeq, name)")

    A = np.array([moving_atoms[k] for k in common])
    B = np.array([ref_atoms[k]    for k in common])

    rmsd_before = float(np.sqrt(np.mean(np.sum((A - B) ** 2, axis=1))))
    R, t, rmsd_after = kabsch(A, B)
    axis, angle = axis_angle(R)
    frac_axis = fractional_axis(axis, ref.cell)

    print(f"Moving:    {moving_path}  ({len(moving_atoms)} atoms)")
    print(f"Reference: {ref_path}  ({len(ref_atoms)} atoms)")
    print(f"Matched:   {len(common)} atoms (by chain+resSeq+name)")
    print()
    print(f"RMSD before fit: {rmsd_before:8.3f} A")
    print(f"RMSD after  fit: {rmsd_after :8.3f} A")
    print()
    print("Rotation matrix R (R x + t = fitted moving):")
    for row in R:
        print("    [{:9.5f} {:9.5f} {:9.5f}]".format(*row))
    print(f"Translation t:    [{t[0]:9.3f} {t[1]:9.3f} {t[2]:9.3f}]")
    print()
    print(f"Rotation angle: {angle:7.2f} deg")
    print(f"Rotation axis (orthogonal frame): "
          f"[{axis[0]:.4f} {axis[1]:.4f} {axis[2]:.4f}]")
    print(f"Rotation axis (cell-frac, scaled): "
          f"[{frac_axis[0]:.4f} {frac_axis[1]:.4f} {frac_axis[2]:.4f}]")
    if abs(angle - 180.0) < 5.0:
        print("  -> ~180 deg rotation: candidate alt-indexing 2-fold")

    if out_path is not None:
        new = apply_to_structure(moving, R, t)
        new.cell = ref.cell
        new.write_pdb(str(out_path))
        print(f"\nFitted moving structure written to {out_path}")


if __name__ == "__main__":
    main()
