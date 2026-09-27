"""Unattended v3 pipeline: analyse, train, ablate, predict, validate.

Ordered so the cheap decisive measurements come first and the expensive full-test
run happens once, with the previous validated submission never at risk: output is
written to a staging directory and only promoted over ``output/`` after the
official validator passes on it.

    python -m src.run_v3
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

from .paths import OUTPUT, ROOT, WORK

PY = sys.executable
STAGE = os.path.join(WORK, "stage_output")
LOG = os.path.join(WORK, "v3_run.log")
MIN_FREE_GB = 2.0


def free_gb() -> float:
    st = os.statvfs(ROOT)
    return st.f_bavail * st.f_frsize / 1e9


def say(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


def run(args: list[str], label: str) -> bool:
    say(f"START {label}: {' '.join(args)}")
    t0 = time.time()
    r = subprocess.run([PY, *args])
    ok = r.returncode == 0
    say(f"{'OK' if ok else 'FAIL'} {label} in {time.time() - t0:.0f}s")
    return ok


def validate(matching: str, candidate: str) -> bool:
    r = subprocess.run(
        ["python3", os.path.join(ROOT, "utils", "validate_submission.py"),
         "--matching", matching, "--candidate", candidate,
         "--test-dir", os.path.join(ROOT, "dataset", "test")],
        capture_output=True, text=True)
    say(r.stdout.strip().splitlines()[-1] if r.stdout else "validator: no output")
    return r.returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-reblock", action="store_true",
                    help="train candidates are already current for every country")
    args = ap.parse_args()
    say(f"=== v3 run start, {free_gb():.1f} GB free ===")

    # 1. Protect the validated v2 submission before anything can touch output/.
    os.makedirs(STAGE, exist_ok=True)
    for name in ("matching_results.tsv", "candidate_pairs.tsv"):
        src = os.path.join(OUTPUT, name)
        dst = os.path.join(WORK, f"v2_{name}")
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            say(f"kept v2 fallback: {dst}")

    # 2. Re-block train with the SAME adaptive-k as the test candidates.
    # The test set was retrieved with k=18 for short-address records while the
    # existing train candidates are uniform k=3, and the per-entity and rev_n
    # features depend on how many candidates an entity has. Training on one
    # distribution and predicting on another would miscalibrate exactly those
    # features, so train is regenerated to match rather than left mismatched.
    if args.skip_reblock:
        say("skipping train re-block (candidates already current)")
    elif not run(["-m", "src.blocking_rev", "--split", "train"],
                 "re-block train with adaptive-k"):
        say("ABORT: train re-blocking failed; v2 output left untouched")
        return 1
    run(["-m", "src.block_eval"], "blocking recall with adaptive-k")

    # 3. Loss attribution, after re-blocking: it needs candidates for the current
    # entity sample. Run before it, the old candidate files covered a different
    # bucket range and every validation block came back empty.
    run(["-m", "src.analyze"], "loss attribution")

    # 4. Stage 1 alone, then stage 1 + stage 2, both reported on validation.
    if not run(["-m", "src.train_v3"], "train v3 (stage1 + stage2)"):
        say("ABORT: training failed; v2 output left untouched")
        return 1

    # The gate must NOT trust train_v3's own report(): that number comes from a
    # ~80%-of-validation tuning slice picked while early stopping runs, and a
    # slice-by-row version of that split once understated v3 by 4.8 F0.5 points
    # (fixed, but the failure mode — a partial, in-training slice standing in for
    # the real score — is exactly what caused the previous wrong ABORT). The gate
    # instead calls src.eval_models, a separate module that scores every
    # validation row after training is done, so a future regression in train_v3's
    # internal split can never again silently reject a good model.
    has_stage2 = os.path.exists(f"{WORK}/model_s2.txt")
    if not run(["-m", "src.eval_models"], "honest v3 evaluation (all validation rows)"):
        say("ABORT: eval_models failed; v2 output left untouched")
        return 1
    eval_label = "stage1_stage2" if has_stage2 else "model_s1.txt"
    honest = json.load(open(f"{WORK}/eval_{eval_label}.json"))
    f_v2 = 0.9455
    best = honest["f05"]
    say(f"validation F0.5 (honest, all rows) — v2 {f_v2:.4f} | v3 {best:.4f} "
        f"({honest['label']}, thr={honest['threshold']:.2f} rel={honest['rel_floor']:.2f})")

    if best <= f_v2 + 0.0005:
        say(f"DECISION: v3 ({best:.4f}) does not beat v2 ({f_v2:.4f}); "
            "keeping v2 as the submission and stopping before the full test run")
        return 0
    use_stage2 = has_stage2
    say(f"DECISION: predicting with {'stage1+stage2' if use_stage2 else 'stage1 only'}"
        f" (validation {best:.4f})")

    # Keep v3_results.json in sync with the honest number: predict.py reads its
    # threshold/rel_floor from there, and it must never fall back to the
    # unaudited in-training figure this gate just refused to trust.
    res = json.load(open(f"{WORK}/v3_results.json"))
    res["stage2" if has_stage2 else "stage1"] = {
        "f05": honest["f05"], "threshold": honest["threshold"],
        "rel_floor": honest["rel_floor"]}
    json.dump(res, open(f"{WORK}/v3_results.json", "w"), indent=2)

    # 5. Full test run into staging.
    if free_gb() < MIN_FREE_GB:
        say(f"ABORT: only {free_gb():.1f} GB free, need {MIN_FREE_GB} GB")
        return 1
    # Reuse the candidates the v2 run persisted: retrieval is ~2.5h and the
    # candidate set is unchanged unless blocking parameters change.
    env_args = ["-m", "src.predict", "--out-dir", STAGE, "--from-disk"]
    if not use_stage2:
        env_args.append("--no-stage2")
    if not run(env_args, "predict test -> staging"):
        say("ABORT: prediction failed; v2 output left untouched")
        return 1

    # 6. Promote only after the official validator passes on the staged files.
    m = os.path.join(STAGE, "matching_results.tsv")
    c = os.path.join(STAGE, "candidate_pairs.tsv")
    if not validate(m, c):
        say("ABORT: staged output failed validation; v2 output left untouched")
        return 1
    for name in ("matching_results.tsv", "candidate_pairs.tsv"):
        shutil.move(os.path.join(STAGE, name), os.path.join(OUTPUT, name))
    say(f"PROMOTED v3 to {OUTPUT} (validation F0.5 {best:.4f} vs v2 {f_v2:.4f})")
    if validate(os.path.join(OUTPUT, "matching_results.tsv"),
                os.path.join(OUTPUT, "candidate_pairs.tsv")):
        say("final validator: PASS — ready to upload")
    return 0


if __name__ == "__main__":
    sys.exit(main())
