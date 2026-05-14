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
