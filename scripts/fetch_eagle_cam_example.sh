#!/usr/bin/env bash
# Run in a terminal with network access. Downloads public reference inputs only.
# Prepares inputs for the SW Florida eagle cam recording (2024-12-28). It runs no
# localization, refinement, fit or scoring.
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

# Read-only clone of github.com/mjshiggins/dopplertrack at the pinned commit.
ref_repo="${DOPPLERTRACK_REF:-/private/tmp/claude-501/-Users-markshi-berkeley-dopplertracker/21596e3b-0472-406e-8c20-a44c53ac346e/scratchpad/ref-dopplertrack}"
ref_commit=a16b7006dfe38b4880919ca2652f402342686289
video_id=zs2o5Rhlp5w

if [[ ! -x .venv/bin/python ]]; then
  echo 'Create the project environment with uv sync first.' >&2
  exit 2
fi
if ! command -v uvx >/dev/null || ! command -v ffmpeg >/dev/null; then
  echo 'This command requires uv (uvx) and ffmpeg on PATH.' >&2
  exit 2
fi
mkdir -p data/eagle-cam

echo '1/6 Downloading the eagle cam audio (best audio only).'
audio_ok() {
  .venv/bin/python - <<'PY'
import json
from pathlib import Path
p = Path('data/eagle-cam')
try:
    info = json.loads((p / 'recording.info.json').read_text())
    assert info['id'] == 'zs2o5Rhlp5w'
    assert info['ext'] in ('webm', 'm4a', 'opus', 'ogg', 'mp3', 'mp4', 'wav')
    assert (p / ('recording.' + info['ext'])).stat().st_size > 0
except (OSError, ValueError, KeyError, AssertionError):
    raise SystemExit(1)
PY
}
if audio_ok; then
  echo 'Using the already downloaded audio.'
else
  full_ok=0
  for attempt in 1 2 3; do
    if uvx --from yt-dlp yt-dlp --no-playlist --no-overwrites --write-info-json \
        --js-runtimes node --format bestaudio --output 'data/eagle-cam/recording.%(ext)s' \
        "https://www.youtube.com/watch?v=$video_id"; then full_ok=1; break; fi
    echo "Full download attempt $attempt failed; retrying." >&2
  done
  if [[ $full_ok != 1 ]]; then
    echo 'Full download failed; fetching audio section 5:00:00-6:15:00 (offset 18000 s).' >&2
    uvx --from yt-dlp yt-dlp --no-playlist --no-overwrites --write-info-json \
      --js-runtimes node --format bestaudio --download-sections '*5:00:00-6:15:00' \
      --output 'data/eagle-cam/recording.%(ext)s' "https://www.youtube.com/watch?v=$video_id"
    echo '{"section_start_s": 18000, "section_end_s": 22500}' > data/eagle-cam/recording-section.json
  fi
fi

echo '2/6 Downloading the resumable flight archive (about 2 GB).'
if [[ ! -s data/eagle-cam/traces.npz ]]; then
  .venv/bin/python - <<'PY'
import io
import json
import hashlib
from pathlib import Path
from dopplertracker.data import extract_archive, save_tracks
from dopplertracker.download import JoinedFiles, download_file
from dopplertracker.geo import LocalFrame

tag = 'v2024.12.28-planes-readsb-prod-0'
base = f'https://github.com/adsblol/globe_history_2024/releases/download/{tag}'
urls = [f'{base}/{tag}.tar']
sizes = [1951918080]   # from the GitHub release asset listing
directory = Path('data/eagle-cam/archive')
paths = []
for url, size in zip(urls, sizes, strict=True):
    destination = directory / url.rsplit('/', 1)[1]
    print(f'Downloading/resuming {destination.name}', flush=True)
    paths.append(download_file(url, destination, size,
                 progress=lambda done, total: print(f'{done/1e6:.1f}/{total/1e6:.1f} MB saved', flush=True)))
hashes = []
for path in paths:
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(chunk)
    hashes.append(hasher.hexdigest())
frame = LocalFrame(26.7, -81.85)
print('Archive downloaded. Extracting regional flight tracks locally.', flush=True)
with io.BufferedReader(JoinedFiles(paths)) as stream:
    tracks = extract_archive(stream, frame, (26.115, 27.285, -82.505, -81.195),
                             progress=lambda read, kept: print(f'{read} traces read; {kept} retained', flush=True))
temporary = Path('data/eagle-cam/traces.partial.npz')
save_tracks(temporary, tracks, frame)
temporary.replace('data/eagle-cam/traces.npz')
Path('data/eagle-cam/traces-source.json').write_text(json.dumps(
    {'release': tag, 'urls': urls, 'sizes': sizes, 'sha256': hashes}, indent=2))
PY
fi

echo '3/6 Copying published reference files from the original project.'
mkdir -p data/eagle-cam/published
if [[ ! -s data/eagle-cam/published/provenance.json ]]; then
  [[ -d "$ref_repo/docs/results" ]] || { echo "Reference clone not found: $ref_repo" >&2; exit 2; }
  [[ "$(git -C "$ref_repo" rev-parse HEAD)" == "$ref_commit" ]] || { echo "Reference clone is not at $ref_commit" >&2; exit 2; }
  files=()
  for f in "$ref_repo"/docs/results/swfl_*; do
    cp "$f" data/eagle-cam/published/
    files+=("$(basename "$f")")
  done
  .venv/bin/python - "$ref_repo" "$ref_commit" "${files[@]}" <<'PY'
import json
import subprocess
import sys
from pathlib import Path
ref, commit, names = sys.argv[1], sys.argv[2], sys.argv[3:]
blobs = {n: subprocess.run(['git', '-C', ref, 'rev-parse', f'{commit}:docs/results/{n}'],
                           check=True, capture_output=True, text=True).stdout.strip() for n in names}
Path('data/eagle-cam/published/provenance.json').write_text(json.dumps({
    'source_repository': 'https://github.com/mjshiggins/dopplertrack',
    'commit': commit,
    'video_url': 'https://youtu.be/zs2o5Rhlp5w',
    'recording_date': '2024-12-28',
    'files': {n: {'git_blob_sha': b, 'path': f'docs/results/{n}'} for n, b in blobs.items()},
    'evaluation_truth': [26.702359, -81.847766],
    'evaluation_truth_note': 'Nest location published in the original README.',
    'warning': 'These are published reference data from the original project, not outputs of this implementation.',
}, indent=2))
PY
fi

echo '4/6 Fetching weather (Open-Meteo) for later use.'
if [[ ! -s data/eagle-cam/weather-open-meteo.json ]]; then
  levels="1000 975 950 925 900 850 800 700 600 500 400 300"
  vars="temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,surface_pressure"
  for l in $levels; do
    vars+=",temperature_${l}hPa,relative_humidity_${l}hPa,wind_speed_${l}hPa,wind_direction_${l}hPa,geopotential_height_${l}hPa"
  done
  query="latitude=26.7&longitude=-81.85&start_date=2024-12-28&end_date=2024-12-28&wind_speed_unit=ms&timezone=UTC&hourly=$vars"
  for endpoint in https://historical-forecast-api.open-meteo.com/v1/forecast https://archive-api.open-meteo.com/v1/archive; do
    if curl -fsS --retry 3 "$endpoint?$query" -o data/eagle-cam/weather.partial.json \
        && .venv/bin/python -c "
import json
d=json.load(open('data/eagle-cam/weather.partial.json'))
t=d['hourly']['temperature_2m']
assert len(t)==24 and any(v is not None for v in t)
"; then
      mv data/eagle-cam/weather.partial.json data/eagle-cam/weather-open-meteo.json
      echo "$endpoint" > data/eagle-cam/weather-endpoint.txt
      echo "Weather endpoint used: $endpoint"
      break
    fi
    echo "Endpoint $endpoint gave no usable data." >&2
  done
  [[ -s data/eagle-cam/weather-open-meteo.json ]] || { echo 'No weather endpoint worked.' >&2; exit 1; }
fi

echo '5/6 Fetching the NOAA geoid height.'
if [[ ! -s data/eagle-cam/geoid.json ]]; then
  curl -fsS --retry 3 'https://geodesy.noaa.gov/api/geoid/ght?lat=26.7&lon=-81.85' -o data/eagle-cam/geoid.partial.json
  mv data/eagle-cam/geoid.partial.json data/eagle-cam/geoid.json
fi

echo '6/6 Decoding the audio spectrogram and writing the inputs manifest.'
.venv/bin/python - <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from dopplertracker.data import load_tracks
from dopplertracker.signal import Spectrogram

directory = Path('data/eagle-cam')
info = json.loads((directory / 'recording.info.json').read_text())
if info.get('id') != 'zs2o5Rhlp5w':
    raise SystemExit('Recording ID does not match the eagle cam video.')
recording = directory / ('recording.' + info['ext'])
if not recording.is_file():
    raise SystemExit(f'Missing downloaded audio: {recording}')
spec_path = directory / 'spectrogram.npz'
if not spec_path.is_file():
    subprocess.run([sys.executable, '-m', 'dopplertracker', 'spectrogram',
                    str(recording), str(spec_path)], check=True)
spec = Spectrogram.load(spec_path)
tracks, frame = load_tracks(directory / 'traces.npz')

def digest(path):
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()

section = directory / 'recording-section.json'
paths = [recording, spec_path, directory / 'traces.npz', directory / 'traces-source.json',
         directory / 'weather-open-meteo.json', directory / 'geoid.json']
paths += sorted((directory / 'published').iterdir())
manifest = {
    'status': 'inputs prepared; no localization run',
    'recording_url': 'https://youtu.be/zs2o5Rhlp5w',
    'date': '2024-12-28',
    'audio_duration_s': float(spec.times[-1]),
    'audio_section': json.loads(section.read_text()) if section.exists() else 'full recording',
    'track_segments': len(tracks),
    'frame': [frame.lat, frame.lon],
    'bbox': {'lat': [26.115, 27.285], 'lon': [-82.505, -81.195]},
    'release_tag': 'v2024.12.28-planes-readsb-prod-0',
    'files': {str(p): digest(p) for p in paths},
}
(directory / 'inputs.json').write_text(json.dumps(manifest, indent=2))
print(f'Inputs ready: {len(tracks)} flight segments; {spec.times[-1]:.1f} s audio.')
print('No localization has been run.')
PY
