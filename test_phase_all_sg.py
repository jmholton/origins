#!/usr/bin/env ccp4-python
# -*- coding: utf-8 -*-
"""
test_phase_all_sg.py - run the F'(h) = F(R^T h) * exp(2 pi i h t) phase
transformation test across every SG in gemmi's catalog.

Strategy:
  1. For each SG, build a representative cell that satisfies its constraints.
  2. Find a metric-preserving alt-indexing op (proper rotation in the lattice
     holohedry but not in the SG itself). If none exists -- SG already has its
     full lattice holohedry as symmetry -- mark SKIPPED.
  3. Generate random ASU atoms in memory, expand by SG ops to a P1-style set,
     apply the alt op, compute F by exp-sum at a small HKL grid.
  4. Verify F'(h) = F(R^T h) * exp(2 pi i h t) at machine precision.
  5. Print summary: PASS / FAIL / SKIPPED counts with details.

Usage:
    ccp4-python test_phase_all_sg.py [--limit N]    # run first N SGs (debug)
    ccp4-python test_phase_all_sg.py                # full sweep (~few min)
"""
import os
import sys
import time
import itertools
import numpy as np
import gemmi

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from test_phase_transform import (
    F_bare, metric_tensor, is_metric_preserving,
    hkl_grid as _hkl_grid_full, signed_angular_diff_deg,
)

NATOM_ASU = 50          # smaller than the single-SG test to bound memory
HMAX = KMAX = LMAX = 12
SEED = 1
F_MIN = 1.0
PASS_AMP = 1e-10
PASS_PHI = 1e-7         # deg
CHUNK = 5000


def representative_cell(sg):
    """Pick a cell that satisfies the SG's crystal-system constraints."""
    cs = sg.crystal_system_str()
    hm = sg.hm
    if cs == 'triclinic':
        return (50.0, 60.0, 70.0, 80.0, 85.0, 95.0)
    if cs == 'monoclinic':
        # Decide unique axis from the position of "2" (or "21") in the HM symbol.
        # Standard settings: "P 1 2 1" b-unique, "P 1 1 2" c-unique, "P 2 1 1" a-unique.
        toks = hm.split()
        if len(toks) >= 4:
            # which token is the 2-fold?
            for i, tk in enumerate(toks[1:4]):
                if tk and tk != '1':
                    if i == 0:    # a-unique
                        return (50.0, 60.0, 70.0, 100.0, 90.0, 90.0)
                    if i == 1:    # b-unique
                        return (50.0, 60.0, 70.0, 90.0, 100.0, 90.0)
                    if i == 2:    # c-unique
                        return (50.0, 60.0, 70.0, 90.0, 90.0, 100.0)
        # fallback
        return (50.0, 60.0, 70.0, 90.0, 100.0, 90.0)
    if cs == 'orthorhombic':
        return (50.0, 60.0, 70.0, 90.0, 90.0, 90.0)
    if cs == 'tetragonal':
        return (50.0, 50.0, 70.0, 90.0, 90.0, 90.0)
    if cs in ('trigonal', 'hexagonal'):
        return (60.0, 60.0, 120.0, 90.0, 90.0, 120.0)
    if cs == 'cubic':
        return (60.0, 60.0, 60.0, 90.0, 90.0, 90.0)
    raise ValueError(f"unknown crystal system: {cs!r}")


def find_alt_index_op(cell_tuple, sg):
    """Search for a proper integer rotation R (entries in {-1,0,1}) that
    preserves the cell's metric tensor but is NOT in sg's rotational ops.
    Returns (R, t=zero) or None."""
    G = metric_tensor(cell_tuple)
    sg_R = set()
    for op in sg.operations():
        R = (np.array(op.rot, dtype=int) // op.DEN)
        sg_R.add(tuple(R.flatten()))
    for comps in itertools.product([-1, 0, 1], repeat=9):
        R = np.array(comps, dtype=int).reshape(3, 3)
        d = int(round(np.linalg.det(R)))
        if d not in (-1, 1):
            continue
        if tuple(R.flatten()) in sg_R:
            continue
        if not is_metric_preserving(R.astype(float), G, atol=1e-6):
            continue
        return R.astype(float), np.zeros(3)
    return None


def expand_to_p1(asu_atoms, sg):
    """Apply each SG op (incl. centring) to every ASU atom; return Mx3
    fractional positions wrapped into [0,1). Uses an SG object directly."""
    out = []
    for op in sg.operations():
        R = np.array(op.rot, dtype=float) / op.DEN
        t = np.array(op.tran, dtype=float) / op.DEN
        out.append((asu_atoms @ R.T + t) % 1.0)
    return np.vstack(out)


def hkl_grid():
    return _hkl_grid_full(HMAX, KMAX, LMAX).astype(float)


def run_one_sg(sg):
    cell_tuple = representative_cell(sg)
    res = {"sg_number": sg.number, "sg_hm": sg.hm,
           "cs": sg.crystal_system_str(), "n_sym": len(list(sg.operations()))}
    found = find_alt_index_op(cell_tuple, sg)
    if found is None:
        res["verdict"] = "SKIPPED"
        res["reason"] = "no non-trivial metric-preserving op outside SG"
        res["alt_op"] = ""
        return res
    R, t = found
    res["alt_op"] = (R, t)

    rng = np.random.default_rng(SEED)
    asu = rng.random((NATOM_ASU, 3))
    A_orig = expand_to_p1(asu, sg)
    A_new = (A_orig @ R.T + t) % 1.0

    H = hkl_grid()
    F_truth = F_bare(A_new, H, chunk=CHUNK)
    F_orig_at_old = F_bare(A_orig, H @ R, chunk=CHUNK)
    F_pred = F_orig_at_old * np.exp(2j * np.pi * (H @ t))

    mask = (np.abs(F_truth) > F_MIN) & (np.abs(F_orig_at_old) > F_MIN)
    if mask.sum() == 0:
        res["verdict"] = "no usable refls"
        res["n_used"] = 0
        return res
    F_pred_m = F_pred[mask]
    F_truth_m = F_truth[mask]
    rel_amp_err = float(np.max(np.abs(F_pred_m - F_truth_m)) / np.mean(np.abs(F_truth_m)))
    phi_pred = np.degrees(np.angle(F_pred_m))
    phi_truth = np.degrees(np.angle(F_truth_m))
    max_phi_err = float(np.max(np.abs(signed_angular_diff_deg(phi_pred, phi_truth))))
    res.update({
        "n_used": int(mask.sum()),
        "rel_amp_err": rel_amp_err,
        "max_phi_err_deg": max_phi_err,
        "verdict": "PASS" if (rel_amp_err < PASS_AMP and max_phi_err < PASS_PHI) else "FAIL",
    })
    return res


def fmt_R(R):
    sym = ['x', 'y', 'z']
    rows = []
    for r in range(3):
        terms = []
        for c in range(3):
            v = int(round(R[r, c]))
            if v == 0:
                continue
            sign = '+' if v > 0 else '-'
            mag = abs(v)
            terms.append(f'{sign}{sym[c]}' if mag == 1 else f'{sign}{mag}{sym[c]}')
        joined = ''.join(terms)
        if joined.startswith('+'):
            joined = joined[1:]
        rows.append(joined or '0')
    return ','.join(rows)


def main():
    limit = None
    if "--limit" in sys.argv:
        i = sys.argv.index("--limit")
        limit = int(sys.argv[i + 1])

    sgs = list(gemmi.spacegroup_table())
    if limit:
        sgs = sgs[:limit]
    print(f"Sweeping {len(sgs)} SGs from gemmi catalog...")
    print(f"params: NATOM_ASU={NATOM_ASU}, HKL grid +/-{HMAX}, F_MIN={F_MIN}")
    print()

    results = []
    n_pass = n_fail = n_skip = 0
    t0 = time.time()
    for i, sg in enumerate(sgs, 1):
        try:
            r = run_one_sg(sg)
        except Exception as e:
            r = {"sg_number": sg.number, "sg_hm": sg.hm, "cs": sg.crystal_system_str(),
                 "n_sym": len(list(sg.operations())), "verdict": "ERROR", "reason": str(e)}
        results.append(r)
        v = r["verdict"]
        if v == "PASS":  n_pass += 1
        elif v == "FAIL": n_fail += 1
        elif v == "SKIPPED": n_skip += 1

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s.  PASS={n_pass}  FAIL={n_fail}  SKIPPED={n_skip}  "
          f"OTHER={len(results) - n_pass - n_fail - n_skip}")
    print()

    # Show all FAILs and ERRORs first
    print("=== FAILs / ERRORs ===")
    nshown = 0
    for r in results:
        if r["verdict"] in ("PASS", "SKIPPED"):
            continue
        nshown += 1
        op = fmt_R(r["alt_op"][0]) if isinstance(r.get("alt_op"), tuple) else "-"
        amp = r.get("rel_amp_err", float("nan"))
        phi = r.get("max_phi_err_deg", float("nan"))
        print(f"  #{r['sg_number']:>3} {r['sg_hm']:<14}  cs={r['cs']:<12}  "
              f"alt={op:<14}  amp={amp:.2e}  phi={phi:.2e}  -- {r['verdict']}  "
              f"{r.get('reason', '')}")
    if nshown == 0:
        print("  (none)")

    # SKIPPED summary
    print()
    print("=== SKIPPED (no non-trivial alt-index op) ===")
    nshow = 0
    for r in results:
        if r["verdict"] != "SKIPPED": continue
        nshow += 1
        if nshow <= 20:
            print(f"  #{r['sg_number']:>3} {r['sg_hm']:<14} cs={r['cs']}")
    if nshow > 20:
        print(f"  ... and {nshow - 20} more")

    # PASS summary by crystal system
    print()
    print("=== PASS counts by crystal system ===")
    from collections import Counter
    by_cs = Counter()
    by_cs_total = Counter()
    for r in results:
        by_cs_total[r["cs"]] += 1
        if r["verdict"] == "PASS":
            by_cs[r["cs"]] += 1
    for cs in sorted(by_cs_total):
        print(f"  {cs:<14}  PASS {by_cs[cs]} / {by_cs_total[cs]}")


if __name__ == "__main__":
    main()
