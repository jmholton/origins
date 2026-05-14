#!/usr/bin/env ccp4-python
# -*- coding: utf-8 -*-
"""
stretch_to_cell.py - re-orthogonalize a PDB's atoms into a target unit cell
while preserving each atom's fractional coordinates.

Use this before sfcalc when comparing two non-isomorphous crystals: putting
both structures in a common cell (with fractions preserved) absorbs the
isomorphous cell-distortion shift into a uniform elastic stretch, so the
remaining real-space difference is just the alt-indexing rotation +
translation we actually want to find.

Usage:
    ccp4-python stretch_to_cell.py <in.pdb> <out.pdb> a b c alpha beta gamma
"""
import sys
import gemmi

if len(sys.argv) != 9:
    sys.exit(__doc__)
in_pdb, out_pdb = sys.argv[1], sys.argv[2]
a, b, c, alpha, beta, gamma = (float(x) for x in sys.argv[3:9])
target_cell = gemmi.UnitCell(a, b, c, alpha, beta, gamma)

st = gemmi.read_structure(in_pdb)
# Snapshot the original cell parameters BEFORE changing st.cell — gemmi's
# UnitCell binding is mutable-in-place, so a bare `old = st.cell` would alias
# the same object and follow the assignment.
oc = st.cell
old_cell = gemmi.UnitCell(oc.a, oc.b, oc.c, oc.alpha, oc.beta, oc.gamma)
st.cell = target_cell
for model in st:
    for chain in model:
        for residue in chain:
            for atom in residue:
                frac = old_cell.fractionalize(atom.pos)
                pos  = target_cell.orthogonalize(frac)
                atom.pos = gemmi.Position(pos.x, pos.y, pos.z)
st.write_pdb(out_pdb)
