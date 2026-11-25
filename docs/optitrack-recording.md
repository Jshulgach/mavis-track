# OptiTrack video recording and OptiHub 2 sync

Launch `mavis-track gui --backend optitrack`. The controls are independent:

1. Leave **Enable Inference** unchecked (the default) before **Start Cameras** for a camera-only session.
   Camera startup then does not initialize MediaPipe detectors or load WiLoR
   weights. Inference can be enabled or disabled while capture continues. An
   already executing native inference call may finish, but its result is discarded
   after the toggle. Disabling inference clears overlays and temporal tracking;
   recording continues. Loaded models remain resident for reuse.
2. Enter a **Trial** name matching Nexus and choose the video output folder.
   **Start Recording** creates a unique timestamped session directory. The
   chosen parent folder stays selected for subsequent trials in the window.
   Each camera records independently at its original
   resolution, without tracking overlays or the SDK text overlay. Recording does
   not require calibration or inference.
3. **Stop Recording** flushes the writers in the background. Stopping cameras or
   closing the window finishes the recording before releasing capture. If a
   writer is still blocked after the timeout, the window stays open for retry.

**Show Camera Background** controls the background under inference overlays;
it does not disable inference. Model selection requires stopping cameras.
**Show Camera Preview** (beside **Enable Inference**) hides or shows the camera
tiles while cameras and recording continue. Turning it off skips camera overlay
drawing, display color conversion and Qt image updates. It does not disable
inference or the separately controlled 3D view, and capture still rasterizes
frames for recording. The initial preview setting is saved in session metadata;
changes during recording are logged as `camera_preview_changed` events.
Playback FPS, codec, sync presets and optional Nexus network control are inside
**Recording / sync setup…**. The live window shows a compact recording health
indicator; detailed counters/errors are in its tooltip and saved logs. Review
tools open separately. See [Vicon trial workflow](vicon-sync.md).

### Local camera recording control

In the OptiTrack or [webcam dashboard](webcams.md#record-a-trial), open
**Recording / sync setup…**, check **Enable local UDP control**, enter the
port (default `5015`), and click **Apply UDP settings**. Start cameras, choose
the video output folder, and leave **Send Nexus start/stop over network**
unchecked for camera-only remote recording. The listener binds to
`127.0.0.1` for the current dashboard session. For scripted launches,
`MAVIS_CAPTURE_CONTROL_PORT` can still preconfigure it. A local application can
send a JSON datagram such as:

```json
{"mode":"camera","action":"start","trial_name":"Reach_01","request_id":"abc123"}
```

Use `status` to check `camera_ready` or recover from a missing reply. Replies
echo `request_id` and include `camera_trial` with the last remotely started
trial name, directory, activity flag and recorder error. Use a unique trial
name for each recording. Repeating Start for the most recent trial returns
its state. Stop applies only to the matching remotely started recording.
During finalization, poll Status until `finishing` is false, then inspect the
recording manifest. A Start reply confirms the recorder was created. Frame
counts and timing come from the saved files. Requests with no mode use the
existing Nexus control.

### Headless capture

On the same computer as your trial controller, start MAVIS with one of these
commands. Use the output directory where the controller should create trial folders.

```powershell
mavis-track capture --backend webcam --output-root .\trials --camera-index 0 --camera-index 1
mavis-track capture --backend optitrack --output-root .\trials --sync-preset internal_50hz
```

Webcams use the saved camera profile by default. Use **Choose cameras** in the
dashboard once to save identities, exposure and capture modes, or pass
`--webcam-profile` to use a specific saved profile. OptiTrack requires a
compatible native binding and USB drivers. `--sync-preset` can apply
`internal_50hz` or `vicon_external_50hz` with a compatible binding.
Set `--playback-fps` to match the capture rate and `--codec FFV1` if you
need lossless decoded frames. `--calibration PATH` validates and snapshots
matching rig geometry alongside the videos. The capture command saves raw
camera frames and timestamps. For live landmark overlays and calibrated
reconstruction, use the dashboard.

By default, headless capture and the CLI client use port 5015. Use `--port`
on both commands for a different port. Only clients on the same computer
can connect. Ctrl+C finalizes an active recording and releases the cameras.
The capture process does not send Nexus commands. Use the dashboard for
the existing combined Nexus/MAVIS trigger.

```powershell
mavis-track control status
mavis-track control start --trial-name Reach_01
mavis-track control stop --trial-name Reach_01
mavis-track control status
```

For Python controllers, use the same request and response contract.

```python
from mavis_track.capture import capture_request

status = capture_request("status")
if not status["camera_ready"]:
    raise RuntimeError(status["detail"])
capture_request("start", trial_name="Reach_01")
# End the trial later, then poll status until finishing is false.
capture_request("stop", trial_name="Reach_01")
```

For a shared trial directory, use `output_root` from the status response
and verify `shared_output_supported`. Create `<output_root>/<trial_name>`,
then pass `shared_output_root=output_root` to `capture_request("start", ...)`
or use `--shared-output-root` with the CLI. MAVIS writes video to the
`video` subfolder. After `finishing` becomes false, confirm the manifest
and frame counts before reporting the trial saved. Video and EMG commands
coordinate the trial lifecycle. A shared TTL and visible LED marker are
needed for frame-level alignment.

## Session files

| File | Contents |
| --- | --- |
| `<trial>_<UTC timestamp>_camera_<index>.avi` | Full-resolution video for each camera that supplied frames |
| `frames.jsonl` | One timing/metadata row for each written video frame |
| `sync.jsonl` | Hub configuration/status samples, normally every 200 ms |
| `manifest.json` | Session settings, camera settings/identity when available, counters, completion and errors |
| `diagnostics.json` | SDK gaps/resets, delivery timing and queue-drop counters |
| `events.jsonl` | Recording lifecycle events |
| `control_events.jsonl` | Nexus network requests/results when enabled, independently of frame queue overload |
| `calibration.npz` | Snapshot of the configured calibration if available, identified by SHA-256 |

The manifest's `video_files` mapping is authoritative. Trial names are made safe
for filenames; the original name remains in the manifest. Names contain UTC
recording-start timestamps, not per-exposure times or a cross-computer clock
guarantee. The default MJPG codec is lossy. Optional FFV1 preserves decoded
frames exactly but may require more storage/CPU. Neither format restores sensor
data already compressed by the camera's MJPEG mode. Codec support depends on
OpenCV's installed video backend; unsupported writers fail visibly.

**Playback FPS** in setup sets the AVI's nominal playback speed. It does not change camera
frame rate or resample the captured images. Use `frames.jsonl` for actual timing;
variable frame intervals and missing frames are not represented by AVI playback.
Rows map `camera_index` and zero-based `video_frame_index` to
`host_received_monotonic` in Python `time.perf_counter()` seconds, measured before
rasterization. This clock is shared with `host_sampled_monotonic` in sync samples
and `host_started_monotonic` in the manifest. It measures receipt rather than
exposure, and has no established mapping to another computer's clock.

The extended binding adds `sdk_frame_id`, `sdk_timestamp`,
`hardware_timestamp_valid`, and hardware ticks/frequency when the SDK reports
valid hardware timing. These remain separate clock domains; do not subtract
hardware/SDK timestamps from host timestamps. Not every camera supplies hardware
timing. SDK frame IDs can reveal acquisition gaps. Physical serial numbers are
recorded when supported; camera indices alone do not establish physical identity
or remap calibration.

The original binding can record videos and host timing. Its metadata explicitly
sets `sdk_timing_available` to false, and the GUI reports sync configuration as
unavailable until the extended binding is installed. Build/import verification does not
validate timing against physical exposures.

Capture reads sequential SDK frames while recording instead of draining them
to the newest image. Preview and inference still use replaceable latest-frame
slots. Each camera has an independent encoding/writing thread and a bounded queue of 32 frames. Sync/status events use a separate bounded queue. A slow camera writer does not block the other camera writers; total buffer memory grows with camera count.
When encoding/storage cannot keep up, queue drops are counted in the GUI and
manifest. The count includes dropped video frames or sync samples; it is not a
hardware pulse count or a guarantee of zero SDK/USB losses. Inspect frame IDs and
bench-test the actual camera count, resolution, frame rate and storage device.
Disk space is checked every 100 written frames, stopping below 64 MiB free.

Writer failures are shown in the GUI and leave an incomplete manifest. Sidecars
are flushed incrementally and manifest counters checkpoint every 100 frames.
An interrupted AVI may require repair; these video sessions are separate from
the numeric sessions consumed by `mavis-track replay`/`export`.

## OptiHub 2 (2011) limits

The legacy Camera SDK 1.10 headers expose hub synchronization settings through
`CameraManager::GetSyncSettings`, including input source, trigger selection,
divider and global offset. The GUI reads these settings and records samples
alongside video. Camera startup applies the selected preset (default: internal
50 Hz free run) and enables exposure timing on External Sync Out. Later mode
changes require **Apply Sync Mode** and are disabled during recording.
Startup setup retries when the hub becomes available. GUI recording remains
disabled if setup fails, the hub disappears, or any camera's recent SDK sequence
rate differs from 50 Hz by more than 5%. Allow roughly two seconds after changing
sync for the rate estimate to settle. Recording setup displays observed camera
rates; a configuration readback or playback FPS alone does not establish capture
rate. A sync problem during recording is shown as a warning while Stop remains
available. Rate agreement does not establish electrical phase or input lock.

The SDK's `Frame` documentation restricts synchronization telemetry, including
external lock, to eSync devices. The OptiHub interface has no documented input
level or timestamped edge stream. Therefore the GUI says **Input pulses/level:
unavailable (SDK)** and every manifest sets
`input_edge_recording_supported: false`. An external-source setting does not
prove a signal is connected, active, or locked. Samples describe configuration,
not the electrical signal; no pulse rate or waveform is inferred from camera FPS.

Recording every electrical edge requires an additional digital acquisition
device and a measured clock relationship to exposures (for example, recording
the incoming trigger and the hub exposure output together). Input-divider and
gating settings can prevent some input edges from producing camera exposures.
The hardware's synchronization role is described in the official
[OptiHub guide](https://docs.optitrack.com/v1.1/external-device-setup/custom-synchronization-optihub).

## Vicon Nexus at 50 Hz

Connect the Lock Lab's selected GPO/Sync Out to the OptiHub's **External Sync In**.
In Nexus, confirm the actual system rate is 50 Hz and assign `DV_Normal.gpo` to
the output socket you wired. This program sends one high/low cycle per Vicon
frame after its `MXDVStart` event; confirm pulses are active before expecting
camera frames. See the official [Nexus output settings](https://help.vicon.com/space/Nexus220/897715600/Vicon%2BConnectivity%2Bnodes)
and [GPO definitions](https://vicon-help.atlassian.net/wiki/spaces/Connect/pages/1611203).

Start cameras in MAVIS, open **Recording / sync setup…**, select **Vicon Nexus — external 50 Hz**, then click
**Apply Sync Mode**. The binding applies custom synchronization with external
input, rising-edge trigger, divider 1, and global offset 0 microseconds. It
preserves the current OptiSync/WiredSync topology and output configuration.
The preset expects Vicon to supply 50 Hz; it neither programs Vicon nor measures
the incoming pulse rate. It also sets the recording playback FPS to 50.

To recover without incoming pulses, select **Internal — 50 Hz (free run)** and
Apply. This sets the internal generator to 50 Hz. External mode can start camera
workers without fresh frames, leaving the GUI available while waiting for Vicon
or for switching back to internal mode. Camera failures still report errors.
Changing the combo box alone does nothing, and starting cameras does not apply
a preset automatically. Settings readback verifies SDK configuration, not
electrical lock or exposure alignment. The native setter attempts to restore
the previous settings if application/readback fails.

`expected_frequency_hz` in sync samples is the preset expectation, not a measured
frequency. Recordings still require an independent exposure-alignment check and
the GUI's own Start/Stop Recording actions.

## Install or build the extended native binding

On a configured Windows/Python 3.11 x64 tracking PC, close MAVIS and install the
[prebuilt sync-capable binding](installation.md#install-the-prebuilt-camera-binding):

```powershell
python external/setup_optitrack.py --upgrade-binding
python -m mavis_track doctor --backend optitrack --gui
```

This installs the verified extension without a compiler. The matching legacy
CameraLibrary DLL and USB drivers remain prerequisites.

To build from source instead:

Close MAVIS and use the Python 3.11 x64 build environment described in
[the local SDK guide](https://github.com/Jshulgach/mavis-track/blob/main/external/README.md#illumination-binding-upgrade), with
Visual Studio 2022 C++ tools, CMake and pybind11 installed:

```powershell
python external/build_optitrack.py --restore-sdk --install
```

For Visual Studio 2026, use CMake 4.2 or newer and a separate build directory:

```powershell
python external/build_optitrack.py --restore-sdk --install --generator "Visual Studio 18 2026" --output .cache/optitrack-binding-vs2026
```

Close the GUI before installation, then restart it to load the new extension.
The optimized build exposes `PARALLEL_RASTERIZE_API_VERSION = 1`, recorded in
new trial manifests under `session_metadata.binding_capabilities`.

The build verifies imports and exposes `CAMERA_RECORDING_API_VERSION = 1`,
`get_sync_status()`, `Frame.timing()` and `Camera.capture_info()`. The current
source also adds `OPTIHUB_SYNC_CONTROL_API_VERSION = 1` and
`configure_sync_preset()` for the two explicit presets. It retains the
base binding API version 2 for existing illumination/runtime compatibility.
Earlier prebuilt API v2 archives do not necessarily include these extensions.
SDK initialization, connected-hub operation and electrical timing still require
an on-hardware check after building.
The current source also exposes `configure_exposure_output()` for the setup
dialog's exposure timing output action. This does not generate Nexus remote
start/stop contact closures.

## Recording performance diagnostics

New recordings save `acquisition_diagnostics` and `performance_by_camera` in
both the manifest and diagnostics JSON. Acquisition diagnostics observe every
frame submitted by capture, including frames rejected by a full writer queue.
Their SDK gaps therefore describe missing frame IDs before the writer queue;
`frame_diagnostics` describes gaps among saved frames and includes queue losses.
Neither counter identifies the cause of SDK gaps or measures electrical pulses.

Per-camera stage aggregates contain count, total seconds and maximum seconds
for SDK frame retrieval, rasterization, RGBA-to-BGR conversion, writer queue
wait, and combined video encoding/write. Divide total by count for mean latency.
Capture stage statistics include queue-rejected images. Encoding/write includes
codec and backend I/O time; it does not separate CPU encoding from physical disk
latency. Individual saved frame rows also include these timings. Native binding
rebuilds from current source decode into Python-owned storage and release the
GIL during rasterization. Restart the GUI after installing the rebuilt binding.

### Offline throughput benchmark

Run from the repository using the environment that launches the GUI:

```powershell
.\.venv-wilor\Scripts\python.exe scripts/benchmark_camera_recording.py --trial temp_data/Trial001_20261110T184357_333967Z_c26b35ec
```

This uses decoded samples from each source camera; no cameras or Nexus control
are opened. It compares one/four (or all available) concurrent workers, default
versus single-threaded OpenCV, conversion, MJPG encoding, and shared metadata
with per-frame flushing versus buffering. Paced runs exercise the actual recorder
at 50 and 120 FPS per camera, followed by a 30-second 120 FPS run. Results include
producer lateness, drops, queue timing, finalization time, and environment details.
Use `--sustained-seconds 60` for a longer run or `0` to skip it.

Each invocation creates a new timestamped directory under
`.cache/camera-benchmarks`, retaining `report.json`, test videos, and recorder
sidecars. Source recordings are read only. For meaningful comparisons, close
the live GUI and other heavy workloads and keep the output drive constant.
Short writes may be served by OS caching. The benchmark excludes SDK acquisition,
native rasterization, preview and inference, so a passing result establishes an
offline baseline rather than guaranteeing live capture performance.
