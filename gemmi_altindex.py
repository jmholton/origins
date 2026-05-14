#!/usr/bin/env ccp4-python
# -*- coding: utf-8 -*-
"""
gemmi_altindex.py - generate the altindex operator list for a given cell+SG
using gemmi alone.  Replicates origins.com's `othercell + symop.lib` step.

Algorithm: enumerate integer basis-change matrices M (components in {-2..2},
|det(M)| in {1..4}); for each, compute the alt-cell metric and look up SGs
whose crystal-system constraints fit it; conjugate each match's symops back
to the original frame and keep the integer-rotation results.

Usage:
    ccp4-python gemmi_altindex.py "<cell>" "<sg>"
e.g.
    ccp4-python gemmi_altindex.py "85.71 85.71 332.94 90 90 120" "H 3 2"
"""
import sys
import itertools
import numpy as np
import gemmi


def ortho_matrix(cell):
    O = np.zeros((3, 3))
    for i, b in enumerate([(1, 0, 0), (0, 1, 0), (0, 0, 1)]):
        p = cell.orthogonalize(gemmi.Fractional(*b))
        O[:, i] = [p.x, p.y, p.z]
    return O


def cell_params_from_G(G):
    a = float(np.sqrt(G[0, 0])); b = float(np.sqrt(G[1, 1])); c = float(np.sqrt(G[2, 2]))
    al = float(np.degrees(np.arccos(np.clip(G[1, 2] / (b * c), -1, 1))))
    be = float(np.degrees(np.arccos(np.clip(G[0, 2] / (a * c), -1, 1))))
    ga = float(np.degrees(np.arccos(np.clip(G[0, 1] / (a * b), -1, 1))))
    return (a, b, c, al, be, ga)


def matches_crystal_system(params, sg, atol_a=1.0, deg=1.5):
    """Heuristic: do these cell parameters fit the SG's crystal system?"""
    a, b, c, al, be, ga = params
    cs = sg.crystal_system_str()
    is_90 = lambda x: abs(x - 90) < deg
    is_120 = lambda x: abs(x - 120) < deg
    if cs == 'triclinic':
        return True
    if cs == 'monoclinic':
        # exactly one (or zero) angle != 90
        n_non90 = sum(1 for x in (al, be, ga) if not is_90(x))
        return n_non90 <= 1
    if cs == 'orthorhombic':
        return is_90(al) and is_90(be) and is_90(ga)
    if cs == 'tetragonal':
        return is_90(al) and is_90(be) and is_90(ga) and abs(a - b) < atol_a
    if cs in ('trigonal', 'hexagonal'):
        return is_90(al) and is_90(be) and is_120(ga) and abs(a - b) < atol_a
    if cs == 'cubic':
        return is_90(al) and is_90(be) and is_90(ga) and abs(a - b) < atol_a and abs(b - c) < atol_a
    return False


def fmt_op(R, t):
    """Format integer R + rational t as 'X+1/2,Y,Z' style triplet."""
    sym = ['X', 'Y', 'Z']
    from fractions import Fraction
    rows = []
    for r in range(3):
        terms = []
        for c in range(3):
            v = int(round(R[r, c]))
            if v == 0:
                continue
            sign = '+' if v > 0 else '-'
            mag = abs(v)
            terms.append(f'{sign}{sym[c]}' if mag == 1 else f'{sign}{mag}*{sym[c]}')
        ti = Fraction(float(t[r])).limit_denominator(12)
        if ti != 0:
            sign = '+' if ti > 0 else '-'
            mag = abs(ti)
            mag_str = str(mag) if mag.denominator == 1 else f'{mag.numerator}/{mag.denominator}'
            terms.insert(0, f'{sign}{mag_str}')
        if not terms:
            rows.append('0')
        else:
            joined = ''.join(terms)
            if joined.startswith('+'):
                joined = joined[1:]
            rows.append(joined)
    return ','.join(rows)


def crystal_system_of_params(params, deg=1.5, atol_a=1.0):
    """Return one of: triclinic, monoclinic, orthorhombic, tetragonal,
    trig_hex, cubic — whichever the params satisfy (most restrictive first)."""
    a, b, c, al, be, ga = params
    is_90 = lambda x: abs(x - 90) < deg
    is_120 = lambda x: abs(x - 120) < deg
    eq = lambda x, y: abs(x - y) < atol_a
    if is_90(al) and is_90(be) and is_90(ga):
        if eq(a, b) and eq(b, c):
            return 'cubic'
        if eq(a, b) or eq(b, c) or eq(a, c):
            return 'tetragonal'
        return 'orthorhombic'
    if is_90(al) and is_90(be) and is_120(ga) and eq(a, b):
        return 'trig_hex'
    if sum(1 for x in (al, be, ga) if not is_90(x)) == 1:
        return 'monoclinic'
    return 'triclinic'


SYSTEM_GROUPS = {
    'triclinic':    ('triclinic',),
    'monoclinic':   ('triclinic', 'monoclinic'),
    'orthorhombic': ('triclinic', 'monoclinic', 'orthorhombic'),
    'tetragonal':   ('triclinic', 'monoclinic', 'orthorhombic', 'tetragonal'),
    'trig_hex':     ('triclinic', 'monoclinic', 'orthorhombic', 'trigonal', 'hexagonal'),
    'cubic':        ('triclinic', 'monoclinic', 'orthorhombic', 'tetragonal', 'cubic'),
}


def enumerate_alt_ops(cell, sg_name, max_M=1, det_max=1, verbose=False):
    """Return set of unique (R_tuple, t_tuple) altindex operators."""
    O = ortho_matrix(cell)
    G = O.T @ O
    orig_sg = gemmi.find_spacegroup_by_name(sg_name)
    if orig_sg is None:
        sys.exit(f"unknown SG: {sg_name}")

    # Bucket catalog SGs by crystal system for fast pruning
    by_cs = {}
    for sg in gemmi.spacegroup_table():
        by_cs.setdefault(sg.crystal_system_str(), []).append(sg)
    if verbose:
        for cs, sgs in by_cs.items():
            print(f"  catalog: {cs:12s} {len(sgs)} SGs")

    all_ops = set()
    n_M = 0
    n_pairs = 0
    for M_flat in itertools.product(range(-max_M, max_M + 1), repeat=9):
        M = np.array(M_flat, dtype=float).reshape(3, 3)
        d = int(round(np.linalg.det(M)))
        if d == 0 or abs(d) > det_max:
            continue
        n_M += 1
        G_alt = M @ G @ M.T
        try:
            params = cell_params_from_G(G_alt)
        except (ValueError, FloatingPointError):
            continue
        if any(p < 1.0 for p in params[:3]):
            continue
        try:
            M_inv_T = np.linalg.inv(M.T)
        except np.linalg.LinAlgError:
            continue
        cs = crystal_system_of_params(params)
        candidate_sgs = []
        for sub_cs in SYSTEM_GROUPS[cs]:
            candidate_sgs.extend(by_cs.get(sub_cs, []))
        for cand_sg in candidate_sgs:
            n_pairs += 1
            for op in cand_sg.operations():
                R_alt = np.array(op.rot, dtype=float) / op.DEN
                t_alt = np.array(op.tran, dtype=float) / op.DEN
                R_old = M.T @ R_alt @ M_inv_T
                t_old = M.T @ t_alt
                R_round = np.round(R_old)
                if not np.allclose(R_old, R_round, atol=1e-4):
                    continue
                R_int = R_round.astype(int)
                # Accept only "canonical" altindex form: R entries in {-1,0,1}.
                # This matches origins.com's convention and excludes shears
                # like X+2Y that come from non-physical M choices.
                if np.max(np.abs(R_int)) > 1:
                    continue
                # Accept only translations with small denominators (2 or 3) —
                # these are the centring fractions of standard SG settings;
                # 1/4, 1/6 etc. arise from screw axes, which aren't relevant
                # for altindex (origin shifts modulo lattice).
                t_mod = np.array([round(x % 1.0, 6) for x in t_old])
                from fractions import Fraction
                ok = True
                for x in t_mod:
                    f = Fraction(float(x)).limit_denominator(6)
                    if f.denominator not in (1, 2, 3):
                        ok = False; break
                if not ok:
                    continue
                R_tup = tuple(R_int.flatten())
                all_ops.add((R_tup, tuple(t_mod)))
    if verbose:
        print(f"  enumerated {n_M} M matrices, {n_pairs} (M, SG) pairs checked")
    return all_ops


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    cell_str, sg_name = sys.argv[1], sys.argv[2]
    cell = gemmi.UnitCell(*[float(x) for x in cell_str.split()])
    print(f"cell: {cell.parameters}")
    print(f"sg:   {sg_name}")
    ops = enumerate_alt_ops(cell, sg_name, verbose=True)
    print(f"\nunique (R, t) altindex ops: {len(ops)}")
    for R_flat, t in sorted(ops):
        R = np.array(R_flat).reshape(3, 3)
        print(fmt_op(R, np.array(t)))


if __name__ == '__main__':
    main()
