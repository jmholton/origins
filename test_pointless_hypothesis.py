#!/usr/bin/env ccp4-python
"""
test_pointless_hypothesis.py — empirical test of whether pointless's
"reindexing operator: h,k,l" answer for the alt-indexing ops origins.com
generates for an H 3 2 cell really collapses identical merged data, or
whether pointless is dropping real distinguishing info.

Method (cad-sorted):
  1. Pack random atoms into the target cell.
  2. Compute Fcalc in the target space group via gemmi sfcalc -> orig.mtz.
  3. For each xyz alt-indexing op origins.com generates (read from
     ./origins_tempreindexings.txt, which is what origins.com built from
     othercell + symop.lib for this exact cell):
       - convert via xyz_to_hkl_op  -> hkl operator string
       - run CCP4 `reindex` with that hkl op
       - run CCP4 `cad` to fold back to the conventional ASU
       - compare F values to orig.mtz, hkl-by-hkl
     R == 0  =>  the relabel is just a symmetry permutation; pointless's
     "h,k,l" answer is then a valid (non-unique) representative.
     R >  0  =>  the relabel is genuinely distinct; pointless was wrong
     to collapse it to "h,k,l" (and xyz_to_hkl_op gave the right answer).

Run:
    ccp4-python test_pointless_hypothesis.py
"""
import os
import sys
import subprocess
import numpy as np
import gemmi

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from xyz_to_hkl_op import parse_op, format_hkl_op  # noqa

CELL  = (77.31, 77.31, 330.36, 90.0, 90.0, 120.0)
SG    = "H 3 2"
DMIN  = 3.0
NATOM = 200
SEED  = 1
REIDX = os.path.join(HERE, "origins_tempreindexings.txt")
WORK  = "/tmp/ptl_hyp"


def make_random_pdb(path):
    rng = np.random.default_rng(SEED)
    cell = gemmi.UnitCell(*CELL)
    with open(path, "w") as f:
        f.write(
            f"CRYST1{cell.a:9.3f}{cell.b:9.3f}{cell.c:9.3f}"
            f"{cell.alpha:7.2f}{cell.beta:7.2f}{cell.gamma:7.2f} {SG:<11}1\n"
        )
        for i in range(NATOM):
            frac = rng.random(3)
            pos  = cell.orthogonalize(gemmi.Fractional(*frac))
            f.write(
                f"HETATM{i+1:5d}  C   RND A{i+1:4d}    "
                f"{pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}  1.00 20.00           C\n"
            )
        f.write("END\n")


def run(cmd, stdin=None, log=None):
    res = subprocess.run(cmd, input=stdin, capture_output=True, text=True)
    if log:
        with open(log, "w") as f:
            f.write(res.stdout); f.write(res.stderr)
    if res.returncode != 0:
        raise RuntimeError(
            f"{cmd[0]} failed (rc={res.returncode})\n"
            f"  cmd: {' '.join(cmd)}\n  stdin: {stdin!r}\n"
            f"  see log: {log}"
        )


def sfcalc(pdb_path, mtz_path):
    subprocess.run(
        ["gemmi", "sfcalc", f"--dmin={DMIN}", f"--to-mtz={mtz_path}", pdb_path],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def reindex_mtz(in_mtz, out_mtz, hkl_op_str, log=None):
    run(
        ["reindex", "hklin", in_mtz, "hklout", out_mtz],
        stdin=f"reindex {hkl_op_str}\nend\n", log=log,
    )
    # reindex can flip the cell into a non-standard hex setting (gamma=60
    # instead of 120), which makes cad refuse the SG-vs-cell consistency
    # check.  We are comparing hkl-by-hkl in the *original* H 3 2 frame,
    # so overwrite cell+SG back to the conventional cell before cad.
    mtz = gemmi.read_mtz_file(out_mtz)
    mtz.set_cell_for_all(gemmi.UnitCell(*CELL))
    mtz.spacegroup = gemmi.find_spacegroup_by_name(SG)
    mtz.write_to_file(out_mtz)


def cad_sort(in_mtz, out_mtz, log=None):
    run(
        ["cad", "hklin1", in_mtz, "hklout", out_mtz],
        stdin="labin file 1 ALL\nend\n", log=log,
    )


def load_F(mtz_path):
    mtz = gemmi.read_mtz_file(mtz_path)
    F_col = next((c for c in mtz.columns if c.type == "F"), None)
    if F_col is None:
        sys.exit(f"no F column in {mtz_path}")
    F = F_col.array
    H = mtz.column_with_label("H").array.astype(int)
    K = mtz.column_with_label("K").array.astype(int)
    L = mtz.column_with_label("L").array.astype(int)
    return dict(zip(zip(H, K, L), F))


def compare(F1, F2):
    keys = set(F1) & set(F2)
    if not keys:
        return float("nan"), float("nan"), 0, len(F1), len(F2)
    a = np.array([F1[k] for k in keys])
    b = np.array([F2[k] for k in keys])
    R = float(np.sum(np.abs(a - b)) / np.sum(np.abs(a))) if np.sum(np.abs(a)) > 0 else float("nan")
    cc = float(np.corrcoef(a, b)[0, 1]) if len(a) > 1 else float("nan")
    return R, cc, len(keys), len(set(F1) - keys), len(set(F2) - keys)


def hkl_op_for(xyz):
    R, _ = parse_op(xyz)
    M = np.linalg.inv(R.T)
    M_int = np.round(M).astype(int)
    if not np.allclose(M, M_int, atol=1e-6):
        return None
    return format_hkl_op(M_int)


def main():
    if not os.path.isfile(REIDX):
        sys.exit(f"need reindexings list at {REIDX}\n"
                 f"(run origins.com altindex with `debug` first)")
    ops = [s.strip() for s in open(REIDX) if s.strip()]
    seen = set(); uniq = []
    for o in ops:
        if o in seen: continue
        seen.add(o); uniq.append(o)
    ops = uniq

    os.makedirs(WORK, exist_ok=True)
    pdb  = os.path.join(WORK, "rand.pdb")
    orig = os.path.join(WORK, "orig.mtz")
    make_random_pdb(pdb)
    sfcalc(pdb, orig)
    F_orig = load_F(orig)

    print(f"cell  = {CELL}")
    print(f"sg    = {SG}")
    print(f"dmin  = {DMIN} A")
    print(f"natom = {NATOM},   nrefl(orig) = {len(F_orig)}")
    print(f"workdir = {WORK}/")
    print(f"testing {len(ops)} alt-indexing ops from {os.path.basename(REIDX)}")
    print()
    hdr = (f"{'#':>3s}  {'xyz_op':<28s}  {'hkl_op':<14s}  "
           f"{'R':>8s}  {'CC':>7s}  {'common':>7s}")
    print(hdr); print("-" * len(hdr))

    for i, xyz in enumerate(ops, 1):
        hkl = hkl_op_for(xyz)
        if hkl is None:
            print(f"{i:>3d}  {xyz:<28s}  non-integer M_hkl")
            continue
        tag = (xyz.replace(',','_').replace('+','p').replace('-','m')
                  .replace('/','d').replace('*','x'))
        rdx = os.path.join(WORK, f"reindex_{tag}.mtz")
        srt = os.path.join(WORK, f"sorted_{tag}.mtz")
        try:
            reindex_mtz(orig, rdx, hkl, log=os.path.join(WORK, f"reindex_{tag}.log"))
            cad_sort(rdx, srt, log=os.path.join(WORK, f"cad_{tag}.log"))
        except RuntimeError as e:
            print(f"{i:>3d}  {xyz:<28s}  {hkl:<14s}  FAIL ({e.args[0].splitlines()[0]})")
            continue
        F_test = load_F(srt)
        R, cc, n, _o, _t = compare(F_orig, F_test)
        flag = "" if R < 1e-4 else "  <-- not Laue-equivalent"
        print(f"{i:>3d}  {xyz:<28s}  {hkl:<14s}  {R:8.5f}  {cc:7.4f}  {n:7d}{flag}")


if __name__ == "__main__":
    main()
