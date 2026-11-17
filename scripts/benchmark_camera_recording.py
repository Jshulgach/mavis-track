"""Offline camera recording benchmarks using decoded samples from a saved trial.

Never opens cameras or sends Nexus triggers. Outputs live in a new directory.
Example: python scripts/benchmark_camera_recording.py --trial temp_data/TRIAL
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import threading
import time

import cv2
import numpy as np

from mavis_track._camera_recording import CameraRecording


def load_samples(trial, count):
    manifest = json.loads((trial / 'manifest.json').read_text())
    samples = {}
    for camera, filename in sorted(manifest['video_files'].items()):
        path = (trial / filename).resolve()
        if not path.is_relative_to(trial.resolve()):
            raise ValueError('Video path must remain inside the source trial')
        video = cv2.VideoCapture(str(path))
        try:
            total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            frames = []
            for position in np.linspace(0, max(0, total - 1), count, dtype=int):
                video.set(cv2.CAP_PROP_POS_FRAMES, int(position))
                ok, frame = video.read()
                if not ok:
                    raise RuntimeError(f'Cannot decode frame {position}: {path}')
                frames.append(frame)
            samples[int(camera)] = frames
        finally:
            video.release()
    if not samples:
        raise ValueError('Source trial has no camera videos')
    return samples


def parallel(camera_ids, function):
    barrier = threading.Barrier(len(camera_ids))
    def run(camera):
        barrier.wait(timeout=15)
        return function(camera)
    with ThreadPoolExecutor(max_workers=len(camera_ids)) as pool:
        return list(pool.map(run, camera_ids))


def direct_benchmark(folder, samples, frames, stage, sidecar):
    """Unpaced capacity; writer release included, source decode excluded."""
    folder.mkdir()
    lock = threading.Lock()
    stream = (folder / 'frames.jsonl').open('w') if sidecar != 'none' else None
    # Prepare RGBA outside timing to isolate the conversion used by capture.
    rgba = {c: [cv2.cvtColor(f, cv2.COLOR_BGR2RGBA) for f in fs]
            for c, fs in samples.items()} if stage == 'conversion' else {}
    def run(camera):
        source = rgba.get(camera, samples[camera])
        writer = None
        start = time.perf_counter()
        try:
            if stage == 'writer':
                h, w = source[0].shape[:2]
                writer = cv2.VideoWriter(str(folder / f'camera_{camera}.avi'),
                                         cv2.VideoWriter_fourcc(*'MJPG'), 120., (w, h))
                if not writer.isOpened():
                    raise RuntimeError('Cannot open benchmark MJPG writer')
            latencies = []
            for i in range(frames):
                before = time.perf_counter()
                frame = source[i % len(source)]
                if stage == 'conversion':
                    cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                else:
                    writer.write(frame)
                if stream is not None:
                    row = dict(camera_index=camera, video_frame_index=i, sdk_frame_id=i,
                               host_received_monotonic=before, inference_enabled=False,
                               capture_stage_seconds=dict(sdk_get_frame=0., rasterize=0., color_conversion=0.))
                    with lock:
                        stream.write(json.dumps(row) + '\n')
                        if sidecar == 'flush':
                            stream.flush()
                latencies.append(time.perf_counter() - before)
        finally:
            if writer is not None:
                writer.release()
        return dict(camera=camera, frames=frames, seconds=time.perf_counter()-start,
                    mean_frame_ms=statistics.mean(latencies)*1000,
                    p95_frame_ms=float(np.percentile(latencies, 95))*1000)
    try:
        results = parallel(list(samples), run)
    finally:
        if stream is not None:
            stream.close()
    elapsed = max(r['seconds'] for r in results)
    return dict(stage=stage, sidecar=sidecar, cameras=len(samples), seconds=elapsed,
                aggregate_fps=frames*len(samples)/elapsed,
                slowest_camera_fps=min(r['frames']/r['seconds'] for r in results), per_camera=results)


def paced_benchmark(folder, samples, fps, seconds):
    """Exercise the real queues, diagnostics, metadata and finalization at a target rate."""
    recorder = CameraRecording(folder, nominal_fps=fps, camera_ids=list(samples))
    count = max(1, round(fps * seconds))
    start = time.perf_counter() + .1
    def run(camera):
        late = 0
        for i in range(count):
            delay = start + i / fps - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            elif delay < -1/fps:
                late += 1
            recorder.frame(camera, samples[camera][i % len(samples[camera])],
                           dict(sdk_frame_id=i, host_received_monotonic=time.perf_counter(),
                                inference_enabled=False))
        return dict(camera=camera, producer_late_frames=late)
    try:
        producers = parallel(list(samples), run)
    finally:
        before_close = time.perf_counter()
        recorder.close()
    manifest = json.loads((folder / 'manifest.json').read_text())
    return dict(stage='real_recorder', cameras=len(samples), target_fps_per_camera=fps,
                submitted=count*len(samples), written=manifest['frames_written'],
                drops=manifest['queue_items_dropped'], flush_seconds=time.perf_counter()-before_close,
                per_camera=manifest['frames_per_camera'], producers=producers,
                performance=manifest['performance_by_camera'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('.cache/camera-benchmarks'))
    parser.add_argument('--samples', type=int, default=8)
    parser.add_argument('--frames', type=int, default=120, help='Frames per camera per capacity test')
    parser.add_argument('--seconds', type=float, default=3., help='Duration of each paced test')
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--sustained-seconds', type=float, default=30.,
                        help='Final 120 FPS/camera run at default OpenCV threads (0 to skip)')
    args = parser.parse_args()
    if min(args.samples, args.frames, args.seconds, args.repeats) <= 0:
        parser.error('Samples, frames, seconds and repeats must be positive')
    if args.sustained_seconds < 0:
        parser.error('Sustained seconds must be nonnegative')
    samples = load_samples(args.trial, args.samples)
    output = args.output.resolve() / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output.mkdir(parents=True, exist_ok=False)
    default_threads = cv2.getNumThreads()
    report = dict(source=str(args.trial.resolve()), output=str(output),
                  environment=dict(python=platform.python_version(), opencv=cv2.__version__,
                                   cpu_count=os.cpu_count(), default_opencv_threads=default_threads),
                  parameters=vars(args) | {'trial': str(args.trial), 'output': str(output)},
                  limitations=['No SDK acquisition, native rasterization, preview or inference is running.',
                               'Images were decoded from previously saved lossy videos.',
                               'Writes may be cached by the OS; this is not durable disk throughput.',
                               'Wall timings include scheduling; paced producer lateness is reported.'], results=[])
    def record(result, repeat, threads):
        result.update(repeat=repeat, opencv_threads=threads)
        report['results'].append(result)
        (output / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k not in ('per_camera','performance','producers')}), flush=True)
    case = 0
    try:
        for repeat in range(args.repeats):
            # Reverse ordering on alternate passes to reduce warm-cache/order bias.
            settings = list(dict.fromkeys([default_threads, 1]))
            if repeat % 2:
                settings.reverse()
            for threads in settings:
                cv2.setNumThreads(threads)
                for cameras in sorted(set([1, len(samples)])):
                    selected = dict(list(samples.items())[:cameras])
                    for stage, mode in [('conversion','none'), ('writer','none'),
                                        ('writer','flush'), ('writer','buffered')]:
                        if shutil.disk_usage(output).free < 512*1024*1024:
                            raise RuntimeError('Less than 512 MiB free; benchmark stopped')
                        case += 1
                        record(direct_benchmark(output/f'case_{case:03d}', selected, args.frames, stage, mode), repeat, threads)
                for fps in (50, 120):
                    case += 1
                    if shutil.disk_usage(output).free < 512*1024*1024:
                        raise RuntimeError('Less than 512 MiB free; benchmark stopped')
                    record(paced_benchmark(output/f'case_{case:03d}', samples, fps, args.seconds), repeat, threads)
        if args.sustained_seconds:
            cv2.setNumThreads(default_threads)
            if shutil.disk_usage(output).free < 512*1024*1024:
                raise RuntimeError('Less than 512 MiB free; benchmark stopped')
            result = paced_benchmark(output/'sustained', samples, 120, args.sustained_seconds)
            result['sustained_seconds'] = args.sustained_seconds
            record(result, args.repeats, default_threads)
    finally:
        cv2.setNumThreads(default_threads)
    print(f'Report: {output / "report.json"}', flush=True)


if __name__ == '__main__':
    main()
