"""Build <team_name>_submission.zip in the exact structure the challenge requires.

    python -m src.make_submission --team YOUR_TEAM_NAME

Refuses to build an archive that would be rejected: it re-runs the official
validator on the outputs first, and checks every required file is present. Better
to fail here than to spend a submission finding out.

Structure produced:

    <team_name>_submission.zip
    |- output/matching_results.tsv
    |- output/candidate_pairs.tsv
    |- code/business_entity_resolution/src/...
    |- code/business_entity_resolution/README.md
    |- code/business_entity_resolution/requirements.txt
    |- Documentation_template.md
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import zipfile

from .paths import OUTPUT, ROOT

CODE_DIR = os.path.join(ROOT, "code", "business_entity_resolution")
REQUIRED = [
    (os.path.join(OUTPUT, "matching_results.tsv"), "output/matching_results.tsv"),
    (os.path.join(OUTPUT, "candidate_pairs.tsv"), "output/candidate_pairs.tsv"),
    (os.path.join(CODE_DIR, "README.md"),
     "code/business_entity_resolution/README.md"),
    (os.path.join(CODE_DIR, "requirements.txt"),
     "code/business_entity_resolution/requirements.txt"),
    (os.path.join(ROOT, "Documentation_template.md"), "Documentation_template.md"),
]


def validate() -> bool:
    print("running the official validator ...", flush=True)
    r = subprocess.run(
        ["python3", os.path.join(ROOT, "utils", "validate_submission.py"),
         "--matching", os.path.join(OUTPUT, "matching_results.tsv"),
         "--candidate", os.path.join(OUTPUT, "candidate_pairs.tsv"),
         "--test-dir", os.path.join(ROOT, "dataset", "test")],
        capture_output=True, text=True)
    print((r.stdout or r.stderr).strip().splitlines()[-1])
    return r.returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True, help="team name, used in the filename")
    ap.add_argument("--skip-validate", action="store_true")
    ap.add_argument("--out-dir", default=ROOT)
    args = ap.parse_args()

    missing = [dst for src, dst in REQUIRED if not os.path.isfile(src)]
    if missing:
        print("MISSING required file(s):")
        for m in missing:
            print(f"  {m}")
        return 1

    if not args.skip_validate and not validate():
        print("ABORT: the outputs do not pass the validator; not building the zip.")
        return 1

    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in args.team)
    zip_path = os.path.join(args.out_dir, f"{safe}_submission.zip")

    # Source files only: no __pycache__, no .pyc, nothing generated.
    sources = sorted(
        f for f in os.listdir(os.path.join(CODE_DIR, "src"))
        if f.endswith(".py")
    )
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED,
                         compresslevel=6) as z:
        for src, dst in REQUIRED:
            z.write(src, dst)
            print(f"  + {dst}  ({os.path.getsize(src) / 1e6:.1f} MB)")
        for name in sources:
            src = os.path.join(CODE_DIR, "src", name)
            z.write(src, f"code/business_entity_resolution/src/{name}")
        print(f"  + code/business_entity_resolution/src/  ({len(sources)} modules)")

    size = os.path.getsize(zip_path) / 1e6
    print(f"\nbuilt {zip_path}  ({size:.1f} MB)")
    with zipfile.ZipFile(zip_path) as z:
        bad = z.testzip()
        if bad is not None:
            print(f"ABORT: archive is corrupt at {bad}")
            return 1
        names = set(z.namelist())
    for _, dst in REQUIRED:
        assert dst in names, dst
    print(f"archive verified: {len(names)} entries, all required paths present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
