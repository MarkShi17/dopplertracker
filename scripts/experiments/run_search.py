"""One timed regional search. Usage: run_search.py OUT.json LAT_MIN LAT_MAX LON_MIN LON_MAX {all|nojet} [--profile]

Mirrors the CLI search path (whiten once, ground_grid, search). Truth is used only after the search.
"""
import cProfile
import json
import pstats
import sys
import time
from pathlib import Path

import numpy as np

import dopplertracker.search as S
from dopplertracker.data import load_tracks
from dopplertracker.detection import load_events
from dopplertracker.geo import ground_grid
from dopplertracker.physics import Atmosphere
from dopplertracker.signal import Spectrogram
from dopplertracker.terrain import ConstantElevation

out = sys.argv[1]
bbox = [float(x) for x in sys.argv[2:6]]
mode = sys.argv[6]
profile = "--profile" in sys.argv
D = Path("data/oyster-bay")
tracks, frame = load_tracks(D / "traces.npz")
if mode == "nojet":
    jets = set(json.loads(Path("data/aircraft-types/jets.json").read_text()))
    tracks = [t for t in tracks if (t.metadata or {}).get("type") not in jets]
events = load_events([str(D / "events.json")])
spec = Spectrogram.load(D / "spectrogram.npz").whiten()
points = ground_grid(frame, bbox, 200.0, ConstantElevation(-30.0, 0.0))

counter = {"n": 0}
orig = S.joint_score


def counted(*a, **k):
    counter["n"] += 1
    try:
        return orig(*a, **k)
    except ValueError:  # supersonic relative motion in a bad segment: treat as unscorable
        counter["failed"] = counter.get("failed", 0) + 1
        return (-np.inf, a[4], [], [])


S.joint_score = counted
pr = cProfile.Profile() if profile else None
t0 = time.time()
if pr:
    pr.enable()
import signal


class Stop(Exception):
    pass


def _stop(*_):
    raise Stop


if profile:  # partial profile after PROFILE_SECONDS wall seconds (default 300)
    signal.signal(signal.SIGALRM, _stop)
    signal.alarm(int(__import__("os").environ.get("PROFILE_SECONDS", "300")))
try:
    results = S.search(spec, events, tracks, points, Atmosphere(), "brightness", None, 5.0, 12.0, 0.0, 10**9)
except Stop:
    results = None
if pr:
    pr.disable()
secs = time.time() - t0
if pr:
    pstats.Stats(pr).sort_stats("cumulative").print_stats(14)

if results is None:
    print("PROFILE STOPPED EARLY; scored so far", counter["n"])
    sys.exit(0)
# evaluation (after search)
truth = json.loads(Path(D / "published/provenance.json").read_text())["evaluation_truth"]
tp = frame.to_local([[truth[0], truth[1], -30.0]])[0]
true_ids = {"a1dfeb", "ace391"}


def key(r):
    return frozenset(f.split(":")[0] for f in r["flights"])


def dist(r):
    return float(np.linalg.norm(np.array(r["position"][:2]) - tp[:2]))


pairs = {}
for r in results:  # sorted by score desc
    pairs.setdefault(key(r), r)
ranked = list(pairs)
rank = ranked.index(frozenset(true_ids)) + 1 if frozenset(true_ids) in pairs else None
scores = [pairs[k]["score"] for k in ranked]
res = dict(
    mode=mode, bbox=bbox, seconds=secs, points=len(points), tracks=len(tracks), scored=counter["n"],
    failed_scores=counter.get("failed", 0), positive_results=len(results), pairs=len(pairs), true_rank=rank,
    margin=(scores[0] - scores[1]) if len(scores) > 1 else None, best=scores[0] if scores else None,
    top_flights=results[0]["flights"] if results else None,
    dist_top=dist(results[0]) if results else None,
    dist_true_best=dist(pairs[frozenset(true_ids)]) if rank else None,
    true_score=pairs[frozenset(true_ids)]["score"] if rank else None,
)
print(json.dumps(res))
Path(out).parent.mkdir(parents=True, exist_ok=True)
Path(out).write_text(json.dumps(dict(summary=res, top=results[:50])))
