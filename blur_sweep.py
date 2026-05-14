#!/usr/bin/env ccp4-python
"""
blur_sweep.py — run test_altindex.py at several resolutions and tabulate
how reported CC depends on map blur (BADD = 79*(reso/3)^2).

For each reso, the sub-run produces /tmp/altindex_report.txt.  We snapshot
that file, then assemble a single comparison table at the end:

    SG        alt-name    CC@reso=3  CC@reso=4  CC@reso=6  CC@reso=8  ...

Use this to see whether increased blur narrows the gap between the
identity test and non-identity alt-indexings (suggesting origins.com's CC
is dominated by spurious sharp-peak matches at low blur).
"""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR  = Path(__file__).resolve().parent
TEST_SCRIPT = SCRIPT_DIR / "test_altindex.py"
REPORT_TMP  = Path("/tmp/altindex_report.txt")
SWEEP_DIR   = Path("/tmp/blur_sweep")
RESOS       = [3, 4, 5, 6, 8]    # angstroms; default for origins.com is 3

# Parse a saved report into rows keyed by (sg, alt_name) -> dict of fields
ROW_RE = re.compile(
    r"^(?P<sg>\S+(?:\s\S+)?)\s+(?P<name>\S+)\s+(?P<op>\S+)\s+"
    r"(?P<natoms>\d+)\s+(?P<cc>[\d\.\-na/]+)"
)


def parse_report(text):
    rows = {}
    in_table = False
    for line in text.splitlines():
        if line.startswith("---"):
            in_table = True; continue
        if not in_table:
            continue
        if not line.strip() or line.startswith("Summary") or line.startswith("Per-test"):
            continue
        # The leading SG name has a space ("H 3" or "H 3 2"); split carefully.
        # Use the natoms (first 3+-digit integer) to anchor.
        parts = line.split()
        # find natoms position
        for i, tok in enumerate(parts):
            if tok.isdigit() and int(tok) > 50:
                natoms = int(tok); break
        else:
            continue
        # before natoms: sg... name op
        cc = parts[i + 1]
        op = parts[i - 1]
        name = parts[i - 2]
        sg = " ".join(parts[: i - 2])
        rows[(sg, name)] = {"cc": cc, "op": op, "natoms": natoms}
    return rows


def main():
    if SWEEP_DIR.exists():
        shutil.rmtree(SWEEP_DIR)
    SWEEP_DIR.mkdir()
    saved_reports = {}

    for reso in RESOS:
        print(f"\n=== running test bed at reso={reso} A ===")
        env = os.environ.copy()
        env["RESO"] = str(reso)
        log = SWEEP_DIR / f"driver_reso{reso}.log"
        with open(log, "wb") as f:
            rc = subprocess.run(
                ["ccp4-python", str(TEST_SCRIPT)],
                env=env, stdout=f, stderr=subprocess.STDOUT,
            ).returncode
        print(f"  exit code {rc}")
        if not REPORT_TMP.is_file():
            print("  WARNING: no report produced"); continue
        snap = SWEEP_DIR / f"report_reso{reso}.txt"
        shutil.copy(REPORT_TMP, snap)
        saved_reports[reso] = parse_report(snap.read_text())

    if not saved_reports:
        sys.exit("no reports parsed; nothing to compare")

    # Build comparison table
    keys = sorted({k for r in saved_reports.values() for k in r})
    header = ["SG", "alt-name"] + [f"CC@reso={r}" for r in RESOS]
    out = []
    out.append("Blur sweep: CC vs. resolution (BADD = 79 * (reso/3)^2)")
    out.append(f"reso={RESOS}    BADD={[round(79*(r/3)**2) for r in RESOS]}")
    out.append("")
    fmt = "{:<6} {:<10}" + " {:>10}" * len(RESOS)
    out.append(fmt.format(*header))
    out.append("-" * (18 + 11 * len(RESOS)))
    for sg, name in keys:
        cells = [sg, name]
        for r in RESOS:
            row = saved_reports.get(r, {}).get((sg, name))
            cells.append(row["cc"] if row else "—")
        out.append(fmt.format(*cells))

    text = "\n".join(out) + "\n"
    sweep_summary = SWEEP_DIR / "summary.txt"
    sweep_summary.write_text(text)
    print()
    print(text)
    print(f"Per-reso reports + driver logs in: {SWEEP_DIR}/")


if __name__ == "__main__":
    main()
