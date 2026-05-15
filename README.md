# origins

Find the alternative crystallographic origin and indexing convention that brings two PDB structures into best agreement.

When two crystal structures of the same (or related) macromolecule are not isomorphous — different cell, different choice of origin within the same space group, or related by a non-trivial alt-indexing operator — naive overlay fails. `origins.com` searches the full space of allowed crystallographic operations (alt-indexing × space-group symop × origin shift) and reports the combination that maximizes agreement with a reference structure, scored by either RMSD or map cross-correlation.

The repository also contains a small set of Python helpers built on `gemmi` for tasks the main script doesn't cover, including a gemmi-only altindex op enumerator and a tool for finding the closest crystallographic operator to a given LSQ rigid-body fit.

## Requirements

- A working CCP4 environment (`$CCP4_SCR`, `$CLIBD`, `$CCP4` set; `pdbset`, `coordconv`, `overlapmap`, `lsqkab` on PATH)
- `gemmi` ≥ 0.6 (the script uses `gemmi sfcalc`)
- For the Python helpers: `ccp4-python` (Python 3.7+) with `gemmi` and `numpy`

## The main tool: `origins.com`

```tcsh
./origins.com [nochains] <reference.pdb> <moving.pdb> [correlate] [otherhand] [altindex] [debug]
```

Arguments:

- `<reference.pdb>` — the structure you want to match against (held fixed)
- `<moving.pdb>` — the structure that will be transformed
- `nochains` — treat each input as a single body rather than splitting into chains
- `correlate` — score by map cross-correlation (gemmi sfcalc → overlapmap) instead of RMSD
- `otherhand` — also try the opposite hand
- `altindex` — search alt-indexing operators (in addition to origin shifts and symops)
- `debug` — preserve all intermediate temp files

Common invocations:

```tcsh
./origins.com nochains 4fg3.pdb 4e7u.pdb                         # RMSD mode
./origins.com nochains 4fg3.pdb 4e7u.pdb correlate               # CC mode
./origins.com nochains 4fg3.pdb 4e7u.pdb correlate otherhand     # CC + hand search
./origins.com nochains 3poq.pdb 3pou.pdb altindex                # alt-indexing search
./origins.com nochains 3poq.pdb 3pou.pdb altindex correlate debug
```

Output is a `neworigin.pdb` containing the moving structure transformed by the best operator combination, plus a `newlabel_neworigin.pdb` annotated with per-atom CC.

## Python helpers

| Script | What it does |
|---|---|
| [`xyz_to_hkl_op.py`](xyz_to_hkl_op.py) | Convert real-space xyz reindexing operators (e.g. `-X,Y,-Z`) to reciprocal-space HKL operators (e.g. `-h,k,-l`) via `M_hkl = (R_xyz^T)^-1`. Use this to get the definitive HKL relabel for an xyz alt-indexing operator. |
| [`stretch_to_cell.py`](stretch_to_cell.py) | Re-orthogonalize a PDB into a target cell with fractional coords preserved. Absorbs cell-distortion non-isomorphism so the residual real-space difference is just rotation+translation. |
| [`lsq_fit.py`](lsq_fit.py) | Kabsch superposition (matched by chain/resSeq/atom-name) with axis/angle decomposition; flags rotations that look like alt-index 2-folds. |
| [`gemmi_altindex.py`](gemmi_altindex.py) | Gemmi-only enumeration of alt-cell symmetry operators for a given cell+SG. Replaces origins.com's `othercell + symop.lib` lookup with a self-contained brute-force search over integer basis transformations. |
| [`nearest_altindex.py`](nearest_altindex.py) | Given two PDBs, do an LSQ rigid-body fit and rank crystallographic `(altindex × symop)` combinations by how well they reproduce the LSQ rotation. Filters out non-metric-preserving combinations and writes the top-ranked transformation as a PDB. |
| [`blur_sweep.py`](blur_sweep.py) | Run `test_altindex.py` at several resolutions to see how map blur affects origins.com's CC scores. |
| [`test_altindex.py`](test_altindex.py), [`test_pointless_hypothesis.py`](test_pointless_hypothesis.py) | Synthetic tests for alt-indexing recovery and pointless's HKL-relabel correctness. |
| [`test_phase_transform.py`](test_phase_transform.py) | Verify the phase-transformation rule `F'(h) = F(Rᵀ·h) · exp(2πi · h · t)` — what's needed to alt-index a refined MTZ alongside the PDB without redoing refinement. P4 and H32 presets. |
| [`test_phase_all_sg.py`](test_phase_all_sg.py) | Sweep the phase-transformation test across all 559 SGs in gemmi's catalog. 329 PASS, 230 SKIP (centric SGs whose holohedry equals their own symmetry), 0 FAIL. |

### Quick examples

```tcsh
# Convert xyz ops to HKL ops
ccp4-python xyz_to_hkl_op.py "y,x,-z" "-X,+Y,-Z"
# -> k,h,-l
# -> -h,k,-l

# Find the closest crystallographic alt-index op for two unrelated PDBs
ccp4-python nearest_altindex.py moving.pdb reference.pdb
# Prints a ranked table; writes <moving>_after_top_altindex.pdb

# Generate altindex op list for any cell+SG (no CCP4 othercell needed)
ccp4-python gemmi_altindex.py "85.71 85.71 332.94 90 90 120" "H 3 2"
```

## Test data

- `3poq.pdb` / `3pou.pdb` — two crystal forms of *E. coli* OmpF porin in H 3 2 (small and large cells respectively). Used as the worked example in this repo's altindex notes.
- `4fg3.pdb` / `4e7u.pdb` — a second alt-indexing test pair.

## Notes on alt-indexing in trigonal/hexagonal space groups

Pointless can report `h,k,l` (i.e., "no relabel needed") for operations that are genuinely distinct under the cell's Laue group when the alt-cell setting is monoclinic. `xyz_to_hkl_op.py` gives the algebraically correct HKL relabel for any xyz operator without consulting pointless. See `test_pointless_hypothesis.py` for a worked verification on H 3 2.

When two structures are related by a rotation that isn't an exact crystallographic symmetry (e.g., a 178° rotation about an axis 28° off the nearest 2-fold), `nearest_altindex.py` will identify the closest crystallographic `(alt × sym)` combination from the full search space — including ones that origins.com's `othercell` lookup misses.

## License

See `LICENSE`.
