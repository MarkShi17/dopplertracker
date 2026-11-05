"""Local command-line pipeline. No service credentials or web deployment required."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .advanced import ModelOptions, fit_consensus, fit_projected, select_model
from .atmosphere import WeatherSeries, fetch_open_meteo
from .benchmark import SCENARIOS, run_benchmark, write_demo
from .data import extract_archive, fetch_day, load_tracks, save_tracks
from .detection import detect, fit_event, load_events, save_events
from .evidence import assess_candidate
from .geo import LocalFrame, ground_grid, validate_bbox
from .physics import Atmosphere, closest_approaches
from .refine import refine_candidate, refine_spectral
from .scoring import joint_score
from .search import distinct_candidates, save_results, search
from .signal import Spectrogram, decode
from .stress_benchmark import METHODS, run_stress_benchmark
from .stress_scenes import STRESS_SCENARIOS
from .terrain import ConstantElevation, GridElevation, TerrainTiles


def utc(value):
    try:
        result = float(value)
    except ValueError:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise argparse.ArgumentTypeError("timestamp must include a timezone, e.g. 2026-09-08T20:17:30Z")
        result = parsed.astimezone(timezone.utc).timestamp()
    if not np.isfinite(result):
        raise argparse.ArgumentTypeError("timestamp must be finite")
    return result


def parser():
    root = argparse.ArgumentParser(
        description="Aircraft acoustic geolocation and reproducible method comparisons"
    )
    commands = root.add_subparsers(dest="command", required=True)
    p = commands.add_parser("spectrogram", help="Decode audio/video to a high-resolution low-frequency STFT")
    p.add_argument("recording")
    p.add_argument("output")
    p.add_argument("--window-seconds", type=float, default=4.0)
    p.add_argument("--hop-seconds", type=float, default=0.5)
    p.add_argument("--band", nargs=2, type=float, default=(40, 450))
    for name in ("fit", "detect"):
        p = commands.add_parser(
            name, help="Fit one flyover" if name == "fit" else "Automatically find flyovers"
        )
        p.add_argument("spectrogram")
        p.add_argument("--band", nargs=2, type=float, default=(60, 300))
        p.add_argument("--harmonics", type=int, default=1)
        p.add_argument("--out", required=True)
        p.add_argument("--raw-score", action="store_true", help="Skip robust spectral whitening")
        if name == "fit":
            p.add_argument("--window", nargs=2, type=float, required=True)
        else:
            p.add_argument("--window-seconds", type=float, default=120.0)
            p.add_argument("--stride", type=float, default=60.0)
            p.add_argument("--top", type=int, default=10)
    for name in ("fetch-day", "extract-region"):
        p = commands.add_parser(name, help="Stream ADS-B archives and retain a padded region")
        p.add_argument("source", help="YYYY-MM-DD" if name == "fetch-day" else "Local TAR archive")
        p.add_argument("output")
        p.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("S", "N", "W", "E"))
        p.add_argument("--padding-km", type=float, default=15.0)
        p.add_argument("--jet-types", help="JSON list of known jet ICAO codes; unknown types are retained")
        p.add_argument(
            "--allow-baro", action="store_true", help="Allow approximate corrected barometric altitudes"
        )
        p.add_argument(
            "--baro-offset", type=float, default=0.0, help="Pressure altitude to MSL correction, meters"
        )
        p.add_argument(
            "--geoid-offset", type=float, default=0.0, help="Geoid undulation N: ellipsoid height = MSL + N"
        )
        p.add_argument("--max-gap", type=float, default=30.0)
    for name in ("search", "search-single", "refine"):
        p = commands.add_parser(
            name,
            help="Search actual trajectories"
            if name != "refine"
            else "Continuously refine competing candidates",
        )
        p.add_argument("spectrogram")
        if name == "refine":
            p.add_argument("results")
            p.add_argument("--extra", nargs="+", default=[])
            p.add_argument("--top", type=int, default=5)
            p.add_argument("--radius", type=float, default=1000.0)
            p.add_argument("--fit-drift", action="store_true")
            p.add_argument("--fit-sound-speed", action="store_true")
            p.add_argument(
                "--method",
                choices=["robust-ridge", "brightness", "contrast", "projected", "auto", "consensus"],
                default="robust-ridge",
            )
            p.add_argument("--source-degree", type=int, choices=[0, 1, 2], default=0)
            p.add_argument("--smooth", choices=["none", "spline", "kalman"], default="none")
            p.add_argument("--fit-wind", action="store_true")
            p.add_argument("--fit-altitude", action="store_true")
            p.add_argument("--fit-clock-rate", action="store_true")
            p.add_argument("--loss", choices=["linear", "soft_l1", "huber", "cauchy"], default="soft_l1")
            p.add_argument(
                "--weather",
                help="Weather file from fetch-weather: layered sound speed and wind, interpolated to each "
                "event; replaces --sound-speed/--wind, and --fit-sound-speed/--fit-wind become offsets",
            )
        else:
            p.add_argument("--bbox", nargs=4, type=float, metavar=("S", "N", "W", "E"))
            p.add_argument("--point", nargs=2, type=float, metavar=("LAT", "LON"))
            p.add_argument("--spacing", type=float, default=200.0)
            p.add_argument("--recording-start", type=utc, required=name == "search-single")
            p.add_argument("--clock-tolerance", type=float, default=5.0)
            p.add_argument("--offset-tolerance", type=float, default=12.0)
            p.add_argument(
                "--silence-radius",
                type=float,
                default=0.0,
                help="Optional strict silence filter in meters; default off",
            )
            p.add_argument("--method", choices=["brightness", "contrast"], default="brightness")
            p.add_argument("--max-candidates", type=int, default=100000)
        p.add_argument("--events", nargs="+", required=True)
        p.add_argument("--traces", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--raw-score", action="store_true")
        p.add_argument("--sound-speed", type=float, default=343.0)
        p.add_argument(
            "--wind", nargs=3, type=float, default=(0.0, 0.0, 0.0), metavar=("EAST", "NORTH", "UP")
        )
        p.add_argument("--geoid-offset", type=float, default=0.0)
        terrain = p.add_mutually_exclusive_group(required=True)
        terrain.add_argument(
            "--elevation", type=float, help="Explicit constant ground elevation in MSL meters"
        )
        terrain.add_argument("--dem", help="Local NPZ grid: lat, lon, elevation_m (MSL)")
        terrain.add_argument("--terrain", action="store_true", help="Download/cache public terrain tiles")
    p = commands.add_parser("demo", help="Write reproducible synthetic inputs, no downloads")
    p.add_argument("--out", default="outputs/demo")
    p.add_argument("--scenario", choices=SCENARIOS, default="rpm_drift")
    p.add_argument("--seed", type=int, default=0)
    p = commands.add_parser("benchmark", help="Compare methods on synthetic scenes; writes HTML and JSON")
    p.add_argument("--out", default="outputs/benchmark")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    p = commands.add_parser(
        "stress-benchmark", help="Expanded accuracy/robustness matrix with development or held-out seeds"
    )
    p.add_argument("--out", default="outputs/stress")
    p.add_argument("--seeds", nargs="+", type=int, default=[100, 101, 102, 103, 104])
    p.add_argument("--scenarios", nargs="+", choices=STRESS_SCENARIOS, default=list(STRESS_SCENARIOS))
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    p.add_argument("--split", choices=["development", "held-out"], default="held-out")
    p.add_argument("--workers", type=int, default=1)
    p = commands.add_parser(
        "controls-benchmark",
        help="Separate detection, flight ranking, waveform and negative-control experiments",
    )
    p.add_argument("--out", default="outputs/controls")
    p.add_argument(
        "--suite", choices=["all", "detection", "identification", "audio", "evidence"], default="all"
    )
    p = commands.add_parser(
        "validate", help="Check frozen candidate evidence against nearby frequency-shifted null curves"
    )
    p.add_argument("spectrogram")
    p.add_argument("results")
    p.add_argument("--events", nargs="+", required=True)
    p.add_argument("--traces", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--raw-score", action="store_true")
    p.add_argument(
        "--weather",
        help="Weather file for candidates that do not record their own layered atmosphere",
    )
    p = commands.add_parser(
        "fetch-weather", help="Download hourly Open-Meteo sound-speed and wind profiles for one UTC day"
    )
    p.add_argument("date", help="YYYY-MM-DD (UTC)")
    p.add_argument("output", help="Derived profile JSON; the raw response is saved beside it (.raw.json)")
    p.add_argument("--lat", type=float, required=True, help="Query point, e.g. the search region centre")
    p.add_argument("--lon", type=float, required=True)
    p.add_argument(
        "--geoid-offset",
        type=float,
        default=0.0,
        help="Geoid undulation N at the query point: ellipsoid height = MSL + N",
    )
    return root


def note(message):
    print(message, file=sys.stderr, flush=True)


def _terrain(args):
    if args.geoid_offset == 0:
        note("Geoid offset is 0 m. Supply local undulation for consistent terrain/GNSS heights.")
    if args.elevation is not None:
        return ConstantElevation(args.elevation, args.geoid_offset)
    if args.dem:
        return GridElevation(args.dem, args.geoid_offset)
    return TerrainTiles(geoid_offset=args.geoid_offset)


def _extra_events(spec, events, chosen, candidate, extra, all_tracks, atmosphere):
    combined = list(zip(events, chosen, candidate["f0"], strict=True))
    for event in extra:
        options = []
        for track in all_tracks:
            hits = closest_approaches(track, np.array(candidate["position"]), atmosphere)
            if not any(
                abs(heard - candidate["clock"] - event.cpa) < 15
                and event.distance_range[0] <= distance <= event.distance_range[1]
                and event.speed_range[0] <= speed <= event.speed_range[1]
                for heard, distance, speed in hits
            ):
                continue
            score, _, f0, _ = joint_score(
                spec,
                [event],
                [track],
                np.array(candidate["position"]),
                candidate["clock"],
                "contrast",
                atmosphere,
                shift_radius=0,
                shift_steps=1,
            )
            options.append((score, track, f0[0]))
        if not options:
            raise ValueError(f"no flight matches extra event near {event.cpa:.1f}s")
        score, track, f0 = max(options, key=lambda item: item[0])
        if score <= 0:
            raise ValueError("extra event has no spectral evidence")
        combined.append((event, track, f0))
    combined.sort(key=lambda item: item[0].cpa)
    new_events, new_tracks, f0 = map(list, zip(*combined, strict=True))
    return new_events, new_tracks, {**candidate, "f0": f0, "flights": [t.id for t in new_tracks]}


def execute(args):
    if args.command == "refine":
        if args.top < 1:
            raise ValueError("--top must be positive")
        advanced = (
            args.source_degree != 0
            or args.smooth != "none"
            or args.fit_wind
            or args.fit_altitude
            or args.fit_clock_rate
            or args.loss != "soft_l1"
        )
        if advanced and args.method not in ("projected", "auto", "consensus"):
            raise ValueError(
                "advanced source, smoothing, loss and nuisance options require --method projected, auto or consensus"
            )
        if args.method in ("brightness", "contrast") and (args.fit_drift or args.fit_sound_speed):
            raise ValueError(
                "--fit-drift and --fit-sound-speed require a ridge or projected refinement method"
            )
        if args.weather and (args.sound_speed != 343.0 or any(args.wind)):
            raise ValueError("--weather replaces --sound-speed and --wind; do not combine them")
        if args.weather and args.fit_sound_speed and args.method == "robust-ridge":
            raise ValueError(
                "--fit-sound-speed with --weather requires --method projected, auto or consensus"
            )
    output = getattr(args, "out", getattr(args, "output", None))
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
    if args.command == "fetch-weather":
        if args.geoid_offset == 0:
            note("Geoid offset is 0 m: profile heights stay MSL, while tracks use ellipsoid heights.")
        url, raw = fetch_open_meteo(args.date, args.lat, args.lon)
        destination = Path(args.output)
        raw_path = destination.with_name(destination.stem + ".raw.json")
        raw_path.write_text(json.dumps(raw, indent=2))
        series = WeatherSeries.from_open_meteo(raw, args.geoid_offset)
        series.metadata.update(
            url=url,
            raw_response=raw_path.name,
            fetched_utc=datetime.now(timezone.utc).isoformat(),
        )
        series.save(destination)
        levels = series.metadata["pressure_levels_hpa"]
        print(f"{len(series.times)} hourly profiles, {len(levels)} pressure levels -> {destination}")
        return 0
    if args.command == "demo":
        print(json.dumps(write_demo(args.out, args.scenario, args.seed), indent=2))
        return 0
    if args.command == "benchmark":

        def progress(scenario, seed, rows):
            note(f"{scenario}, seed {seed}: " + ", ".join(f"{r['method']} {r['error_m']:.1f}m" for r in rows))

        report = run_benchmark(args.out, args.seeds, args.scenarios, progress=progress)
        print(json.dumps(report["summary"], indent=2))
        return 0
    if args.command == "stress-benchmark":
        report = run_stress_benchmark(
            args.out,
            args.seeds,
            args.scenarios,
            args.methods,
            args.split,
            args.workers,
            progress=lambda done, total, scenario, seed: note(
                f"{done}/{total} fits: {scenario}, seed {seed}"
            ),
        )
        print(json.dumps(report["summary"], indent=2, allow_nan=False))
        return 0
    if args.command == "controls-benchmark":
        from .audio_benchmark import run_audio_benchmark
        from .detection_benchmark import run_detection_benchmark
        from .evidence_controls import run_evidence_controls
        from .identification import run_identification

        directory = Path(args.out)
        directory.mkdir(parents=True, exist_ok=True)
        suites = {
            "detection": run_detection_benchmark,
            "identification": run_identification,
            "audio": lambda: run_audio_benchmark(output_dir=directory / "audio"),
            "evidence": lambda: run_evidence_controls(range(200, 220)),
        }
        for name, run in suites.items():
            if args.suite not in ("all", name):
                continue
            note(f"Running {name} controls")
            report = run()
            destination = directory / f"{name}.json"
            destination.write_text(json.dumps(report, indent=2, allow_nan=False))
            print(str(destination), flush=True)
        return 0
    if args.command == "spectrogram":
        spec = decode(
            args.recording, window_seconds=args.window_seconds, hop_seconds=args.hop_seconds, band=args.band
        )
        spec.save(args.output)
        print(f"{len(spec.times)} frames, {len(spec.frequencies)} frequency bins -> {args.output}")
        return 0
    if args.command in ("fetch-day", "extract-region"):
        validate_bbox(args.bbox)
        frame = LocalFrame((args.bbox[0] + args.bbox[1]) / 2, (args.bbox[2] + args.bbox[3]) / 2)
        jets = set(json.loads(Path(args.jet_types).read_text())) if args.jet_types else set()
        if not jets:
            note("No jet classification file supplied: retaining all aircraft types.")
        options = dict(
            jet_types=jets,
            allow_baro=args.allow_baro,
            baro_offset=args.baro_offset,
            geoid_offset=args.geoid_offset,
            max_gap=args.max_gap,
            padding_km=args.padding_km,
            progress=lambda read, kept: note(f"{read} traces read; {kept} segments retained"),
        )
        if args.command == "fetch-day":
            note("Streaming a daily archive can transfer several GB; only regional segments are saved.")
            tracks = fetch_day(args.source, frame, args.bbox, **options)
        else:
            with open(args.source, "rb") as stream:
                tracks = extract_archive(stream, frame, args.bbox, **options)
        save_tracks(args.output, tracks, frame)
        print(f"{len(tracks)} flight segments -> {args.output}")
        return 0
    spec = Spectrogram.load(args.spectrogram)
    if not args.raw_score:
        spec = spec.whiten()
    if args.command == "validate":
        tracks, frame = load_tracks(args.traces)
        events = load_events(args.events)
        lookup = {track.id: track for track in tracks}
        payload = json.loads(Path(args.results).read_text())
        if payload.get("frame") is not None and not np.allclose(
            payload["frame"], [frame.lat, frame.lon], rtol=0, atol=1e-9
        ):
            raise ValueError("results and traces have different coordinate frames")
        weather = WeatherSeries.load(args.weather) if args.weather else None
        for candidate in payload["candidates"]:
            chosen = [lookup[id] for id in candidate["flights"]]
            medium = weather.for_events(events, chosen, candidate["clock"]) if weather else Atmosphere()
            candidate["evidence"] = assess_candidate(spec, events, chosen, candidate, medium)
        Path(args.out).write_text(json.dumps(payload, indent=2, allow_nan=False))
        print(
            f"{sum(r['evidence']['accepted'] for r in payload['candidates'])}/{len(payload['candidates'])} candidates pass heuristic evidence checks"
        )
        return 0
    if args.command in ("fit", "detect"):
        if args.command == "fit":
            events = [fit_event(spec, args.window, args.band, args.harmonics)]
        else:
            events = detect(
                spec,
                args.band,
                args.window_seconds,
                args.stride,
                args.top,
                args.harmonics,
                progress=lambda i, n: note(f"Flyover windows {i}/{n}"),
            )
        save_events(events, args.out)
        for event in events:
            print(
                f"CPA={event.cpa:.1f}s  f0={event.f0:.2f}Hz  speed={event.speed:.1f}m/s  "
                f"slant={event.distance:.0f}m  score={event.score:.3f}"
            )
        return 0
    tracks, frame = load_tracks(args.traces)
    events = load_events(args.events)
    atmosphere = Atmosphere(args.sound_speed, tuple(args.wind))
    elevation = _terrain(args)
    if args.command in ("search", "search-single"):
        if args.command == "search-single" and len(events) != 1:
            raise ValueError("search-single needs exactly one event")
        if args.point:
            lat, lon = args.point
            points = frame.to_local([[lat, lon, float(elevation(lat, lon))]])
        elif args.bbox:
            points = ground_grid(frame, args.bbox, args.spacing, elevation)
        else:
            raise ValueError("supply --bbox or --point")
        note(f"{len(points)} ground points, {len(tracks)} flight segments, {len(events)} events")
        results = search(
            spec,
            events,
            tracks,
            points,
            atmosphere,
            args.method,
            args.recording_start,
            args.clock_tolerance,
            args.offset_tolerance,
            args.silence_radius,
            args.max_candidates,
            progress=lambda phase, i, n: note(f"{phase}: {i}" + (f"/{n}" if n else "")),
        )
    else:
        payload = json.loads(Path(args.results).read_text())
        if payload.get("frame") is not None and not np.allclose(
            payload["frame"], [frame.lat, frame.lon], atol=1e-9, rtol=0
        ):
            raise ValueError("results and flight traces use different coordinate frames")
        lookup = {t.id: t for t in tracks}
        extra = load_events(args.extra) if args.extra else []
        weather = WeatherSeries.load(args.weather) if args.weather else None
        results = []
        for candidate in distinct_candidates(payload["candidates"], args.top):
            try:
                chosen = [lookup[id] for id in candidate["flights"]]
                medium = atmosphere
                if weather is not None:
                    # One profile to match extra events, then one per event at its heard CPA.
                    medium = weather.at(candidate["clock"] + float(np.mean([e.cpa for e in events])))
                active_events, chosen, prepared = _extra_events(
                    spec, events, chosen, candidate, extra, tracks, medium
                )
                if weather is not None:
                    medium = weather.for_events(active_events, chosen, prepared["clock"])
                if args.method in ("projected", "auto", "consensus"):
                    options = ModelOptions(
                        degree=max(args.source_degree, 1 if args.fit_drift else 0),
                        smoothing=args.smooth,
                        fit_sound_speed=args.fit_sound_speed,
                        fit_wind=args.fit_wind,
                        fit_altitude=args.fit_altitude,
                        fit_clock_rate=args.fit_clock_rate,
                        loss=args.loss,
                        radius=args.radius,
                    )
                    if args.method == "auto":
                        from dataclasses import replace

                        result = select_model(
                            spec,
                            active_events,
                            chosen,
                            prepared,
                            [replace(options, degree=d) for d in (0, 1, 2)],
                            medium,
                            frame=frame,
                            elevation=elevation,
                        )
                    elif args.method == "consensus":
                        result = fit_consensus(
                            spec, active_events, chosen, prepared, options, medium, frame, elevation
                        )
                    else:
                        result = fit_projected(
                            spec,
                            active_events,
                            chosen,
                            prepared,
                            options,
                            medium,
                            frame=frame,
                            elevation=elevation,
                        )
                elif args.method == "robust-ridge":
                    result = refine_candidate(
                        spec,
                        active_events,
                        chosen,
                        prepared,
                        medium,
                        args.radius,
                        args.fit_drift,
                        args.fit_sound_speed,
                        frame,
                        elevation,
                    )
                else:
                    result = refine_spectral(
                        spec,
                        active_events,
                        chosen,
                        prepared,
                        args.method,
                        medium,
                        args.radius,
                        frame,
                        elevation,
                    )
                if weather is not None:
                    result.setdefault("atmosphere", medium.to_record())
                result["evidence"] = assess_candidate(spec, active_events, chosen, result, medium)
                results.append(result)
            except (ValueError, KeyError) as exc:
                note(f"Candidate could not be refined: {exc}")
        results.sort(key=lambda r: r["score"], reverse=True)
    save_results(args.out, results, frame)
    print(f"{len(results)} candidates -> {args.out}")
    if results:
        print(json.dumps(results[0], indent=2, allow_nan=False))
        return 0
    note("No candidates survived. Check event windows, date, altitude policy, region and filter ranges.")
    return 2


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        return execute(args)
    except (ValueError, OSError, KeyError) as exc:
        note(f"Error: {exc}")
        return 2
