"""Build data/aircraft-types/jets.json from the OpenSky mirror of ICAO Doc 8643 and count track types."""
import csv
import json
from collections import Counter
from pathlib import Path

from dopplertracker.data import load_tracks

root = Path("data")
with open(root / "aircraft-types/doc8643AircraftTypes.csv", newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
engines = Counter(r["EngineType"] for r in rows)
jets = {r["Designator"] for r in rows if r["EngineType"] == "Jet"}
known = {r["Designator"] for r in rows}
(root / "aircraft-types/jets.json").write_text(json.dumps(sorted(jets)))
print("engine types", dict(engines))
print("jet designators", len(jets))
tracks, _ = load_tracks(root / "oyster-bay/traces.npz")
c = Counter()
for t in tracks:
    ty = (t.metadata or {}).get("type")
    c["jet" if ty in jets else "nonjet" if ty in known else "unknown"] += 1
print(len(tracks), dict(c))
