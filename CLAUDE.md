# origins.com — Claude Notes

## What this project is

`origins.com` is a tcsh script that translates one PDB file to alternative crystallographic origins and indexing conventions, then scores the result against a reference PDB. Scoring is either RMSD (default) or map cross-correlation (correlate mode).

## Shell constraint — MUST READ

**This is a tcsh script.** All edits must use tcsh idioms only. No bash-isms:
- No `2>&1 |` in pipes (use `|&` for stderr+stdout in tcsh, or redirect stderr with `>& file`)
- No `$(...)` — use backticks `` `...` ``
- No `[[ ]]` — use `if ( )` with tcsh expressions
- No `>&2` for stderr — use `> /dev/stderr` or `|& tee`
- Heredocs use `<< EOF` / `EOF` without quoting

## Dependencies (CCP4 + gemmi)

The script requires a CCP4 environment (`$CCP4_SCR`, `$CLIBD`, `$CCP4`). Key programs:
- `pdbset` — manipulates PDB headers/chains
- `coordconv` — converts fractional→orthogonal coordinates (does NOT write CRYST1)
- `overlapmap` — computes map cross-correlation
- `gemmi sfcalc` — structure factor/map calculation (replaces the old `sfall`)

## gemmi sfcalc requirements

`gemmi sfcalc` requires a CRYST1 record with a valid space group. Watch out for:
- `coordconv` output: never has CRYST1 — must add via `pdbset CELL / SPACE`
- Chain files split by awk: strip CRYST1, so pdbset calls need `SPACE $SG`
- For P1 maps (originmask): use `SPACE P1`

Pattern for stamping a CRYST1 before calling gemmi:
```tcsh
pdbset xyzin ${tempfile}foo.pdb xyzout ${tempfile}foo_sg.pdb << EOF > /dev/null
    CELL $CELL
    SPACE $SG
EOF
gemmi sfcalc --dmin=$reso --to-mtz=${tempfile}foo.mtz ${tempfile}foo_sg.pdb > /dev/null
rm -f ${tempfile}foo_sg.pdb
```

## Test invocations

```
./origins.com nochains 4fg3.pdb 4e7u.pdb                         # RMSD mode
./origins.com nochains 4fg3.pdb 4e7u.pdb correlate               # correlate mode
./origins.com nochains 4fg3.pdb 4e7u.pdb correlate otherhand     # correlate + hand search
./origins.com nochains 4fg3.pdb 4e7u.pdb altindex                # altindex mode
```

## Refactoring history

`sfall` (CCP4) has been replaced throughout with `gemmi sfcalc`. The replacement adds `--blur=$BADD` in correlate mode to produce smooth density maps compatible with `overlapmap`. The `altindex` mode replacement was completed and tested first; `correlate` mode was completed second.

## Known fragile points

- `overlapmap mapout /dev/null` fails on some systems — use a real temp file and `rm` it.
- pdbset at line ~394 (chain file setup) must include `SPACE $SG` or downstream gemmi calls fail.
- originmask pdbset (line ~561) must use `SPACE P1` (it's always computed in P1).
- The H3/R3 detection block near line 99 converts SG names for hexagonal vs. rhombohedral.

## Python helpers (added separately from origins.com)

These standalone scripts complement the tcsh tool. All use the CCP4 Python (`ccp4-python`) with `gemmi` and `numpy`. They are pure Python (Py3) and have a UTF-8 coding header so older `ccp4-python` interpreters don't choke on non-ASCII docstrings.

| Script | Purpose |
|---|---|
| `xyz_to_hkl_op.py` | Convert real-space xyz reindexing operators to reciprocal HKL operators via `M_hkl = (R_xyz^T)^-1`. Definitive HKL relabel for an xyz alt-indexing op, independent of pointless's heuristics. |
| `stretch_to_cell.py` | Re-orthogonalize a PDB into a target unit cell with fractional coords preserved. Use this when comparing two non-isomorphous crystals so that cell distortion is absorbed as an isotropic stretch and the remaining real-space difference is just the alt-indexing rotation/translation. |
| `lsq_fit.py` | Kabsch superposition (rigid-body fit by chain+resSeq+name match) with axis/angle decomposition. Reports whether the rotation looks like a 180° crystallographic alt-indexing 2-fold. |
| `gemmi_altindex.py` | Gemmi-only enumeration of alt-cell symops for a given cell+SG; replaces origins.com's `othercell + symop.lib` lookup. Generates a *superset* of origins.com's altindex op list (origins.com's set is hand-curated by othercell; this enumeration explores more alt-cell settings, including higher-symmetry ones othercell skips). |
| `nearest_altindex.py` | Given two PDBs, LSQ-fit them and rank crystallographic `(altindex × symop)` combinations by rotational deviation from the LSQ rotation. Filters out non-metric-preserving combinations (alt-cell shears that aren't valid rotations in the original cell). Writes the top-ranked transformation as a PDB. |
| `blur_sweep.py` | Sweep `test_altindex.py` across resolutions to study how map blur affects origins.com CC scores. |
| `test_altindex.py`, `test_pointless_hypothesis.py` | Synthetic tests of alt-indexing recovery and pointless's HKL-relabel choices. |

### Usage

```tcsh
# Reciprocal-space op for one or more xyz operators
ccp4-python xyz_to_hkl_op.py "y,x,-z" "+1/3-X,+2/3+Y-X,+2/3-Z"

# Put a PDB into a target cell, fractions preserved
ccp4-python stretch_to_cell.py in.pdb out.pdb 85.71 85.71 332.94 90 90 120

# Best rigid-body fit of moving onto reference (with axis/angle)
ccp4-python lsq_fit.py moving.pdb reference.pdb [out.pdb]

# Generate altindex ops for a cell+SG
ccp4-python gemmi_altindex.py "85.71 85.71 332.94 90 90 120" "H 3 2"

# Find the best (alt × sym) op + write PDB of moving after that op
ccp4-python nearest_altindex.py moving.pdb reference.pdb [altops.txt] [out.pdb]
# If altops.txt is missing, falls back to gemmi_altindex enumeration.
```

## Subtleties / gotchas worth knowing

- **Pointless's HKL relabel is sometimes wrong** for hex/trigonal alt-indexings (it tends to report `h,k,l` for ops that are genuinely distinct under the cell's Laue group). `xyz_to_hkl_op.py` gives the unambiguous answer.
- **`(alt × sym)` rotational deviation is only meaningful when the combined rotation preserves the lattice metric.** For ops that don't (i.e., alt-cell-derived ones whose `R_combᵀ·G·R_comb ≠ G`), `O·R·F` is non-orthogonal in cartesian — applying it shears the molecule. `nearest_altindex.py` filters these out before ranking; otherwise the standard `arccos((tr(R_lsq·R_cartᵀ)-1)/2)` formula returns garbage.
- **origins.com applies alt-cell-derived ops in a "lattCELL"** (the cell where the op *is* a true rotation), not in the original cell. The Python tooling here doesn't replicate that lattCELL re-expression — it only ranks ops that are rotations in the original cell. The gemmi enumeration nevertheless surfaces *other* metric-preserving ops that origins.com's hand-curated list misses.
- **The Δt (translation residual)** reported by `nearest_altindex.py` is the leftover origin shift origins.com would still need to find by grid search after picking the (alt × sym) basis. Small Δt = origins.com is one short search away from the right answer.
