#!/usr/bin/env bash
# Run in a terminal with network access. Downloads public reference inputs only.
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

if [[ ! -x .venv/bin/python ]]; then
  echo 'Create the project environment with uv sync first.' >&2
  exit 2
fi
if ! command -v uvx >/dev/null || ! command -v ffmpeg >/dev/null; then
  echo 'This command requires uv (uvx) and ffmpeg on PATH.' >&2
  exit 2
fi
mkdir -p data/oyster-bay

echo '1/3 Downloading the original one-hour reference audio, not the narrated video.'
if .venv/bin/python - <<'PY'
import json
from pathlib import Path
p = Path('data/oyster-bay')
try:
    info = json.loads((p / 'recording.info.json').read_text())
    assert info['id'] == '4CxVxuMVkMc'
    assert info['ext'] in ('webm', 'm4a', 'opus', 'ogg', 'mp3', 'mp4', 'wav')
    assert (p / ('recording.' + info['ext'])).stat().st_size > 0
except (OSError, ValueError, KeyError, AssertionError):
    raise SystemExit(1)
PY
then
  echo 'Using the already downloaded reference audio.'
else
uvx --from yt-dlp yt-dlp --no-playlist --no-overwrites --write-info-json \
  --js-runtimes node \
  --format bestaudio --output 'data/oyster-bay/recording.%(ext)s' \
  'https://www.youtube.com/watch?v=4CxVxuMVkMc'
fi

echo '2/3 Downloading resumable flight archive parts (about 3.9 GB disk space).'
if [[ ! -s data/oyster-bay/traces.npz ]]; then
  .venv/bin/python - <<'PY'
import io
import json
from pathlib import Path
from dopplertracker.data import extract_archive, save_tracks
from dopplertracker.download import JoinedFiles, download_file
from dopplertracker.geo import LocalFrame

# Pin the original project's production archive; the preferred-release manifest
# currently selects a different staging receiver archive for this date.
tag = 'v2026.09.08-planes-readsb-prod-0'
base = f'https://github.com/adsblol/globe_history_2026/releases/download/{tag}'
urls = [f'{base}/{tag}.tar.{suffix}' for suffix in ('aa', 'ab')]
sizes = [2000000000, 1874165248]
hashes = ['f9804df3f9e9e173928ee239938d955bb661266f76393d1042bf12b7cb85de44',
          '0d41348dbd9f4c64ee00eafd28ad89e8e748d14c3a2c9c0b11b6b6adae7f9579']
directory = Path('data/oyster-bay/archive')
paths = []
for url, size, digest in zip(urls, sizes, hashes, strict=True):
    destination = directory / url.rsplit('/', 1)[1]
    print(f'Downloading/resuming {destination.name}', flush=True)
    paths.append(download_file(url, destination, size, sha256=digest,
                 progress=lambda done, total: print(f'{done/1e6:.1f}/{total/1e6:.1f} MB saved', flush=True)))
frame = LocalFrame(40.85, -73.5)
print('Both parts verified. Extracting regional flight tracks locally.', flush=True)
with io.BufferedReader(JoinedFiles(paths)) as stream:
    tracks = extract_archive(stream, frame, (40.4, 41.3, -74.4, -72.6),
                             progress=lambda read, kept: print(f'{read} traces read; {kept} retained', flush=True))
temporary = Path('data/oyster-bay/traces.partial.npz')
save_tracks(temporary, tracks, frame)
temporary.replace('data/oyster-bay/traces.npz')
Path('data/oyster-bay/traces-source.json').write_text(json.dumps({'release': tag, 'urls': urls, 'sha256': hashes}, indent=2))
PY
fi

echo '3/3 Decoding the real recording and preparing the published event windows.'
.venv/bin/python - <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from dopplertracker.data import load_tracks
from dopplertracker.detection import Event, save_events
from dopplertracker.signal import Spectrogram

directory = Path('data/oyster-bay')
info = json.loads((directory / 'recording.info.json').read_text())
if info.get('id') != '4CxVxuMVkMc':
    raise SystemExit('Recording ID does not match the original reference clip.')
recording = directory / ('recording.' + info['ext'])
if not recording.is_file():
    raise SystemExit(f'Missing downloaded audio: {recording}')
spec_path = directory / 'spectrogram.npz'
subprocess.run([sys.executable, '-m', 'dopplertracker', 'spectrogram',
                str(recording), str(spec_path)], check=True)
spec = Spectrogram.load(spec_path)
if spec.times[-1] < 3130:
    raise SystemExit('Recording is too short to include both reference flyovers.')
events = []
for index in (1, 2):
    source = directory / 'published' / f'oyster_bay_ev{index}.json'
    row = json.loads(source.read_text())
    events.append(Event((row['t_start'], row['t_end']), row['t_cpa'], row['f0'],
                        row['v'], row['d'], harmonics=2))
save_events(events, directory / 'events.json')
tracks, frame = load_tracks(directory / 'traces.npz')

def digest(path):
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()

manifest = {
    'status': 'real inputs prepared; no localization comparison run yet',
    'recording_url': 'https://www.youtube.com/watch?v=4CxVxuMVkMc',
    'date': '2026-09-08', 'track_segments': len(tracks),
    'frame': [frame.lat, frame.lon],
    'files': {str(path): digest(path) for path in
              (recording, spec_path, directory / 'traces.npz', directory / 'events.json')},
    'event_policy': 'Published measured windows and approximate curve parameters, not automatic detection.',
    'flight_archive': (json.loads((directory / 'traces-source.json').read_text())
                       if (directory / 'traces-source.json').exists() else 'existing file; archive provenance unknown'),
}
(directory / 'inputs.json').write_text(json.dumps(manifest, indent=2))
print(f'Real inputs ready: {len(tracks)} flight segments; {spec.times[-1]:.1f} s audio.')
print('No accuracy claim is made until these inputs are used for localization.')
PY
