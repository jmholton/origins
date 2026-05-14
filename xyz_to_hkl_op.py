#!/usr/bin/env ccp4-python
# -*- coding: utf-8 -*-
"""
xyz_to_hkl_op.py - convert real-space (xyz) reindexing operators to their
reciprocal-space (hkl) reindexing operators.

For an xyz coordinate transformation  x_new = R · x_old + t
the corresponding HKL transformation is  h_new = (R^T)^-1 · h_old.

This is the *definitive* HKL relabel implied by an xyz alt-indexing operator,
independent of any data-merging heuristics (pointless gets distracted by
Friedel/Laue equivalence in trigonal/hexagonal SGs and may report h,k,l for
non-trivial xyz ops).

Usage:
    ccp4-python xyz_to_hkl_op.py "y,x,-z" "+1/3-X,+2/3+Y-X,+2/3-Z" ...
    -> prints one HKL triplet per input op, on its own line, in the same order
"""
import sys
import re
import numpy as np

# parse a single component string like "+1/3+X-Y" or "y" or "-z"
# into (3-vector of x/y/z coefficients, scalar translation)
_VAR_IDX = {'X': 0, 'Y': 1, 'Z': 2}
_TOKEN_RE = re.compile(r'([+-]?)(?:(\d+(?:/\d+)?)(?:\*)?)?([XYZ])?')

def _parse_term(term: str):
    """Parse one signed token like '+1/3', '-Y', '+X', '-1/2*Y'.  Returns
    (var_coef_vec_3, translation_scalar)."""
    s = term.strip()
    if not s:
        return np.zeros(3), 0.0
    m = _TOKEN_RE.fullmatch(s)
    if not m:
        raise ValueError(f"cannot parse term {s!r}")
    sign_str, num_str, var = m.groups()
    sign = -1.0 if sign_str == '-' else 1.0
    num = 1.0
    if num_str:
        if '/' in num_str:
            a, b = num_str.split('/')
            num = float(a) / float(b)
        else:
            num = float(num_str)
    if var:
        vec = np.zeros(3)
        vec[_VAR_IDX[var]] = sign * num
        return vec, 0.0
    return np.zeros(3), sign * num


def parse_op(s: str):
    """Parse a full xyz triplet like '+1/3+X-Y,+2/3-Y,+2/3-Z' into (3x3 matrix R, 3-vec translation t)."""
    s = s.replace(' ', '').upper()
    comps = s.split(',')
    if len(comps) != 3:
        raise ValueError(f"need 3 comma-separated components: {s!r}")
    R = np.zeros((3, 3))
    t = np.zeros(3)
    for row, comp in enumerate(comps):
        # split into signed terms; ensure leading + on first if no sign
        if comp[0] not in '+-':
            comp = '+' + comp
        # token boundaries: every '+' or '-' starts a new token
        terms = re.findall(r'[+-][^+-]+', comp)
        for term in terms:
            vec, tr = _parse_term(term)
            R[row] += vec
            t[row] += tr
    return R, t


def format_hkl_op(M):
    """Format a 3x3 integer matrix as an HKL triplet like 'k,h,-l' or '-h+k,k,-l'."""
    sym = ['h', 'k', 'l']
    rows = []
    for r in range(3):
        terms = []
        for c in range(3):
            v = int(round(M[r, c]))
            if v == 0:
                continue
            sign = '+' if v > 0 else '-'
            mag = abs(v)
            if mag == 1:
                terms.append(f'{sign}{sym[c]}')
            else:
                terms.append(f'{sign}{mag}*{sym[c]}')
        if not terms:
            rows.append('0')
        else:
            joined = ''.join(terms)
            if joined.startswith('+'):
                joined = joined[1:]
            rows.append(joined)
    return ','.join(rows)


def xyz_to_hkl(xyz_op: str) -> str:
    R, _t = parse_op(xyz_op)
    # HKL transformation = (R^T)^-1
    R_hkl = np.linalg.inv(R.T)
    return format_hkl_op(R_hkl)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for s in sys.argv[1:]:
        print(xyz_to_hkl(s))
