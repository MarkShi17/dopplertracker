"""Run the scaling searches strictly one at a time under nice, aborting at 3 GB RSS or 20 min."""
import json
import math
import subprocess
import sys
import time
from pathlib import Path

LIMIT_RSS_KB = 3 * 1024 * 1024
LIMIT_S = 1200
PY = ".venv/bin/python"


def box(side):
    s = side / 2
    la = s / 111.32
    lo = s / (111.32 * math.cos(math.radians(40.85)))
    return [40.85 - la, 40.85 + la, -73.5 - lo, -73.5 + lo]


def run(side, mode, extra=()):
    tag = f"{side}_{mode}" + ("_profile" if extra else "")
    log = Path(f"outputs/scale_{tag}.log")
    cmd = ["nice", "-n", "15", PY, "scripts/experiments/run_search.py", f"outputs/scale_{tag}.json",
           *map(str, box(side)), mode, *extra]
    t0 = time.time()
    peak = 0
    with log.open("w") as fh:
        p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
        status = "ok"
        while p.poll() is None:
            r = subprocess.run(["ps", "-o", "rss=", "-p", str(p.pid)], capture_output=True, text=True).stdout.strip()
            if r:
                peak = max(peak, int(r))
            if peak > LIMIT_RSS_KB:
                status = "aborted: RSS > 3 GB"
            elif time.time() - t0 > LIMIT_S:
                status = "aborted: > 20 min"
            if status != "ok":
                p.kill()
                p.wait()
                break
            time.sleep(1)
    info = dict(tag=tag, status=status, wall=time.time() - t0, peak_rss_mb=peak / 1024)
    print(json.dumps(info), flush=True)
    return info


plan = [(6, "nojet"), (12, "nojet"), (24, "nojet")]  # 6 km all-tracks exceeded 20 min (run earlier, aborted)
for side, mode in plan:
    info = run(side, mode)
    if info["status"] != "ok":
        break
