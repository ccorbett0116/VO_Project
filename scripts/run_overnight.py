"""Overnight chain: Stage 1 (VO reliability) then Stage 2 (learned fusion).

Stage 2 only starts if Stage 1 exits cleanly. All output is teed to
artifacts/overnight.log.

  .venv/bin/python scripts/run_overnight.py            # full run
  .venv/bin/python scripts/run_overnight.py --smoke    # quick end-to-end check
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def run(name, cmd, logfh):
    banner = f"\n{'#' * 70}\n# {name}  ::  {' '.join(cmd)}\n{'#' * 70}"
    print(banner, flush=True)
    logfh.write(banner + "\n")
    logfh.flush()
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        logfh.write(line)
        logfh.flush()
    code = proc.wait()
    msg = f"# {name} finished in {time.time()-t0:.0f}s (exit {code})"
    print(msg, flush=True)
    logfh.write(msg + "\n")
    logfh.flush()
    return code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    os.makedirs("artifacts", exist_ok=True)
    extra = ["--smoke"] if a.smoke else []
    s1 = [PY, f"{HERE}/train_stage1.py"] + extra + (["--no-image"] if a.smoke else [])
    s2 = [PY, f"{HERE}/train_stage2.py"] + extra

    with open("artifacts/overnight.log", "w") as logfh:
        if run("STAGE 1 — VO reliability", s1, logfh) != 0:
            print("Stage 1 failed; not starting Stage 2.", flush=True)
            sys.exit(1)
        if run("STAGE 2 — learned fusion", s2, logfh) != 0:
            print("Stage 2 failed.", flush=True)
            sys.exit(1)
    print("\nOvernight chain complete. See artifacts/stage2/summary.json")


if __name__ == "__main__":
    main()
