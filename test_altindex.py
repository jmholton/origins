#!/usr/bin/env ccp4-python
"""
test_altindex.py — parallel test bed for origins.com altindex mode in
R3/H3 and R32/H32 space groups.

Pipeline per test case:
  1. Use gemmi to build a Structure with random ASU atoms expanded by the
     full crystallographic symmetry of the target space group.
  2. Apply an alt-indexing operator in real space (coordinate transformation)
     to make a "wrong" PDB.
  3. Run origins.com altindex in its own working directory (parallel-safe:
     unique cwd -> unique neworigin.pdb; tcsh tempfiles use $CCP4_SCR/$$).
  4. Parse origins.com's reported RMSD AND compute an independent RMSD
     between right.pdb and neworigin.pdb using gemmi (with symmetry-aware
     nearest-image matching).  PASS if either is below threshold.

Output: /tmp/altindex_report.txt + per-test logs in /tmp/altindex_runs/.
"""
import math
import os
import random
import re
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import gemmi

# ---- config -------------------------------------------------------
SCRIPT_DIR    = Path(__file__).resolve().parent
ORIGINS       = SCRIPT_DIR / "origins.com"
ROOT          = Path("/tmp/altindex_runs")
REPORT        = Path("/tmp/altindex_report.txt")
RMSD_PASS     = 1.0    # angstroms (RMSD pass threshold)
CC_PASS       = 0.9    # correlation coefficient pass threshold
NRES          = 20
SEED          = 42
# Optional resolution override for the gemmi-sfcalc maps in origins.com.
# Higher reso -> larger BADD (=79*(reso/3)^2) -> more blur.  Set via env var
# RESO=4 etc. to sweep blur levels.  None = origins.com default (3 A).
RESO          = float(os.environ["RESO"]) if "RESO" in os.environ else None
# right.pdb cell
A, B, C       = 77.310, 77.310, 330.360
# wrong.pdb cell — different unit cell (mimics non-isomorphous crystals)
WRONG_A, WRONG_B, WRONG_C = 85.710, 85.710, 332.941   # ~10.9% in a/b
MAX_PARALLEL  = 8

# (sg, label, real-space op) — op is the inverse-transpose of the HKL op.
# Chirality-preserving alt-indexings (det=+1 of HKL matrix) are the meaningful
# tests; det=-1 ops change hand and need otherhand mode (not tested here).
#
# For R3 (hex setting) the chiral alt-indexings are cosets of point group 3
# in lattice point group 622:
#   identity           h,k,l    coord  x,y,z
#   2-fold along c    -h,-k,l   coord -x,-y,z
#   2-fold along [110] k,h,-l   coord  y,x,-z
#   product (6-fold)  -k,-h,-l  coord -y,-x,-z   (= 2c * 2[110])
#
# For R32 the structure point group is 32 (already includes the basal 2-folds),
# so most of these become symmetry ops; only the 6-fold (-y,-x,-z) is a
# non-trivial chiral alt-indexing.
TESTS = [
    ("H 3",   "identity",  "x,y,z"),
    ("H 3",   "kh-l",      "y,x,-z"),
    ("H 3",   "-h-kl",     "-x,-y,z"),
    ("H 3",   "-k-h-l",    "-y,-x,-z"),
    ("H 3 2", "identity",  "x,y,z"),
    ("H 3 2", "kh-l",      "y,x,-z"),
    ("H 3 2", "-h-kl",     "-x,-y,z"),
    ("H 3 2", "-k-h-l",    "-y,-x,-z"),
]


# ---- structure generation ----------------------------------------
def build_right_structure(sg_hm: str) -> gemmi.Structure:
    """Random ASU atoms expanded by full SG symmetry."""
    cell = gemmi.UnitCell(A, B, C, 90, 90, 120)
    sg = gemmi.SpaceGroup(sg_hm)

    st = gemmi.Structure()
    st.cell = cell
    st.spacegroup_hm = sg_hm
    model = gemmi.Model("1")
    chain = gemmi.Chain("A")

    rng = random.Random(SEED)
    resnum = 0
    for i in range(NRES):
        x = rng.uniform(0.10, 0.55)
        y = rng.uniform(0.10, 0.55)
        z = rng.uniform(0.05, 0.28)
        seen = set()
        for op in sg.operations():
            sx, sy, sz = op.apply_to_xyz([x, y, z])
            sx %= 1.0; sy %= 1.0; sz %= 1.0
            key = (round(sx, 4), round(sy, 4), round(sz, 4))
            if key in seen:
                continue
            seen.add(key)
            pos = cell.orthogonalize(gemmi.Fractional(sx, sy, sz))
            # one CA atom per residue, each with a unique seqid so origins.com
            # can match atoms between right and wrong by residue number.
            resnum += 1
            res = gemmi.Residue()
            res.name = "ALA"
            res.seqid = gemmi.SeqId(resnum, " ")
            atom = gemmi.Atom()
            atom.name = "CA"
            atom.element = gemmi.Element("C")
            atom.pos = gemmi.Position(pos.x, pos.y, pos.z)
            atom.b_iso = 20.0
            atom.occ = 1.0
            res.add_atom(atom)
            chain.add_residue(res)
    model.add_chain(chain)
    st.add_model(model)
    return st


def apply_alt_indexing(st: gemmi.Structure, op_str: str,
                       wrong_cell: gemmi.UnitCell = None) -> gemmi.Structure:
    """Apply real-space coord op (e.g. 'y,x,-z') in fractional space.

    The atoms are fractionalized using the input structure's cell, transformed
    by the alt-indexing op, and then re-orthogonalized using `wrong_cell`
    (defaults to the input cell).  This lets the wrong PDB live in a slightly
    different unit cell, as is common between related real crystals.
    """
    components = tuple(c.strip() for c in op_str.split(","))
    right_cell = st.cell
    if wrong_cell is None:
        wrong_cell = right_cell
    new_st = st.clone()
    new_st.cell = wrong_cell
    for model in new_st:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    frac = right_cell.fractionalize(atom.pos)
                    env = {"x": frac.x, "y": frac.y, "z": frac.z}
                    xp = eval(components[0], {"__builtins__": {}}, env) % 1.0
                    yp = eval(components[1], {"__builtins__": {}}, env) % 1.0
                    zp = eval(components[2], {"__builtins__": {}}, env) % 1.0
                    pos = wrong_cell.orthogonalize(gemmi.Fractional(xp, yp, zp))
                    atom.pos = gemmi.Position(pos.x, pos.y, pos.z)
    return new_st


# ---- independent RMSD verification -------------------------------
def independent_rmsd(right_pdb, new_pdb):
    """Symmetry-aware nearest-image RMSD between right and neworigin atoms."""
    if not new_pdb.is_file():
        return None
    right = gemmi.read_structure(str(right_pdb))
    new = gemmi.read_structure(str(new_pdb))
    cell = right.cell
    sg = gemmi.SpaceGroup(right.spacegroup_hm)
    ops = list(sg.operations())

    right_atoms = [a for m in right for c in m for r in c for a in r]
    new_atoms = [a for m in new for c in m for r in c for a in r]
    if len(new_atoms) == 0 or len(right_atoms) == 0:
        return None

    # For each new atom, find closest right atom under any symmetry op + lattice
    # translation.  Skip atom-count mismatch by using min(len) — origins.com
    # may add/strip dummy atoms.
    sum_sq = 0.0
    n = 0
    # Pre-compute fractional positions of right atoms under all SG ops.
    right_frac_orbit = []
    for ra in right_atoms:
        f0 = cell.fractionalize(ra.pos)
        orbit = []
        for op in ops:
            fx, fy, fz = op.apply_to_xyz([f0.x, f0.y, f0.z])
            orbit.append((fx % 1.0, fy % 1.0, fz % 1.0))
        right_frac_orbit.append(orbit)

    for na in new_atoms:
        nf = cell.fractionalize(na.pos)
        nx, ny, nz = nf.x % 1.0, nf.y % 1.0, nf.z % 1.0
        best_d2 = float("inf")
        for orbit in right_frac_orbit:
            for (rx, ry, rz) in orbit:
                # nearest image (lattice translation in frac space)
                dx = (nx - rx) - round(nx - rx)
                dy = (ny - ry) - round(ny - ry)
                dz = (nz - rz) - round(nz - rz)
                # convert to orth distance
                op_pos = cell.orthogonalize(gemmi.Fractional(dx, dy, dz))
                d2 = op_pos.x ** 2 + op_pos.y ** 2 + op_pos.z ** 2
                if d2 < best_d2:
                    best_d2 = d2
        sum_sq += best_d2
        n += 1
    return math.sqrt(sum_sq / n) if n else None


# ---- runner ------------------------------------------------------
def run_one(test):
    sg_hm, label, op = test
    safe_sg = sg_hm.replace(" ", "")
    workdir = ROOT / f"{safe_sg}_{label}"
    workdir.mkdir(parents=True, exist_ok=True)
    right_path = workdir / "right.pdb"
    wrong_path = workdir / "wrong.pdb"
    log_path = workdir / "run.log"
    new_path = workdir / "neworigin.pdb"

    right_st = build_right_structure(sg_hm)
    right_st.write_pdb(str(right_path))
    wrong_cell = gemmi.UnitCell(WRONG_A, WRONG_B, WRONG_C, 90, 90, 120)
    wrong_st = apply_alt_indexing(right_st, op, wrong_cell)
    wrong_st.write_pdb(str(wrong_path))

    natoms = sum(1 for m in right_st for c in m for r in c for _ in r)

    cmd = ["tcsh", str(ORIGINS), "nochains",
           "right.pdb", "wrong.pdb", "altindex", "correlate"]
    if RESO is not None:
        cmd.append(f"reso={RESO}")
    with open(log_path, "wb") as f:
        proc = subprocess.run(cmd, cwd=workdir, stdout=f,
                              stderr=subprocess.STDOUT)

    indep = independent_rmsd(right_path, new_path)
    return sg_hm, label, op, natoms, log_path, proc.returncode, indep


# ---- log parsing -------------------------------------------------
RMSD_RE  = re.compile(r"Overall\s+rmsd\s*[:=]?\s*([0-9.]+)", re.I)
CC_RE    = re.compile(r"Overall\s+CC\s*[:=]?\s*([0-9.]+)", re.I)
APPLY_RE = re.compile(r"^applying\s+(\S+)", re.M)


def parse_log(log_path: Path):
    """Return (cc, rmsd, found_op).  cc set in correlate mode, rmsd otherwise."""
    if not log_path.is_file():
        return None, None, None
    txt = log_path.read_text(errors="ignore")
    cc = None
    for m in CC_RE.finditer(txt):
        cc = float(m.group(1))
    rmsd = None
    for m in RMSD_RE.finditer(txt):
        rmsd = float(m.group(1))
    found = None
    for m in APPLY_RE.finditer(txt):
        found = m.group(1)
    return cc, rmsd, found


# ---- main --------------------------------------------------------
def main():
    if not ORIGINS.is_file():
        sys.exit(f"origins.com not found at {ORIGINS}")
    if ROOT.exists():
        shutil.rmtree(ROOT)
    ROOT.mkdir(parents=True)

    print(f"Launching {len(TESTS)} altindex tests "
          f"(max {MAX_PARALLEL} workers in parallel)...")
    results = []
    with ProcessPoolExecutor(max_workers=MAX_PARALLEL) as ex:
        futures = {ex.submit(run_one, t): t for t in TESTS}
        for fut in as_completed(futures):
            res = fut.result()
            sg, label = res[0], res[1]
            print(f"  done: {sg} / {label}  (rc={res[5]})")
            results.append(res)

    # restore original test order
    ordered = []
    for sg, label, op in TESTS:
        for r in results:
            if r[0] == sg and r[1] == label:
                ordered.append(r); break

    # First pass: parse all logs and find identity CC per SG
    parsed = []  # (sg, label, op, natoms, cc, rmsd, indep, found)
    identity_cc = {}     # per-SG identity CC (the reference)
    for sg, label, op, natoms, log, rc, indep in ordered:
        cc, rmsd, found = parse_log(log)
        parsed.append((sg, label, op, natoms, cc, rmsd, indep, found))
        if label == "identity" and cc is not None:
            identity_cc[sg] = cc

    # Acceptance criterion (per SG, vs identity CC):
    #   PASS if  REL_LO * identity_cc  <=  test_cc  <=  REL_HI * identity_cc
    # The lower bound asks "did origins.com find the alt-indexing?".  The
    # upper bound catches the spurious case where origins.com reports a CC
    # *higher* than the trivial identity case — there is no transformation
    # that can produce a better match than the known-correct one, so a higher
    # CC means the search latched onto a false maximum.
    REL_LO, REL_HI = 0.9, 1.05
    npass = nfail = 0
    rows = []
    for sg, label, op, natoms, cc, rmsd, indep, found in parsed:
        ref = identity_cc.get(sg)
        lo = REL_LO * ref if ref is not None else None
        hi = REL_HI * ref if ref is not None else None
        cc_str    = "n/a" if cc    is None else f"{cc:.3f}"
        indep_str = "n/a" if indep is None else f"{indep:.3f}"
        range_str = "n/a" if lo is None else f"{lo:.3f}-{hi:.3f}"
        found_str = "n/a" if found is None else found
        ok = (cc is not None and lo is not None and lo <= cc <= hi)
        verdict = "PASS" if ok else "FAIL"
        if verdict == "PASS": npass += 1
        else:                 nfail += 1
        rows.append((sg, label, op, natoms, cc_str, range_str, indep_str,
                     found_str, verdict))

    out = []
    out.append("Alternate-indexing test bed for origins.com (correlate mode)")
    out.append(f"Pass: {REL_LO:g} * identity_CC(SG) <= CC <= {REL_HI:g} * identity_CC(SG)")
    out.append("(upper bound: CC > identity is a spurious match — no transformation can beat identity)")
    out.append(f"right cell: {A} {B} {C}  90 90 120")
    out.append(f"wrong cell: {WRONG_A} {WRONG_B} {WRONG_C}  90 90 120")
    out.append(f"NRES (ASU): {NRES}    seed: {SEED}")
    out.append("")
    out.append(f"{'SG':<6} {'alt-name':<10} {'alt-op':<12} "
               f"{'natoms':>6} {'CC':>7} {'pass-range':>13} {'indep':>7}   "
               f"{'found-op':<28} VERDICT")
    out.append("-" * 122)
    for sg, label, op, natoms, cc_str, range_str, indep_str, found_str, verdict in rows:
        out.append(f"{sg:<6} {label:<10} {op:<12} {natoms:>6} "
                   f"{cc_str:>7} {range_str:>13} {indep_str:>7}   "
                   f"{found_str:<28} {verdict}")
    out.append("")
    out.append(f"Summary: {npass} passed, {nfail} failed of {len(TESTS)}")
    out.append(f"Per-test logs: {ROOT}/<sg>_<altname>/run.log")

    text = "\n".join(out) + "\n"
    REPORT.write_text(text)
    print()
    print(text)


if __name__ == "__main__":
    main()
