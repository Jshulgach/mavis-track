# Camera workflows

The default `mavis-track demo` processes one webcam or video with no
calibration. The Python frame API can use any caller-owned image source.

## Calibrated webcams

```bash
mavis-track board --backend webcam
mavis-track calibrate --backend webcam
mavis-track gui --backend webcam --advanced-hands
```

Use matching camera resolutions and a accurately printed ChArUco board.
The solver checks held-out reprojection errors, rejects failed candidates,
preserves the prior calibration, and publishes a validated result atomically.
Every fifth capture is reserved from all model fitting for a separate reconstructed
board check. Default gates require at least three reserved views, shape RMS at most
2 mm, scale error at most 2%, and planarity RMS at most 1 mm. These are commissioning
gates, not claims of physical tracking accuracy.
A failed run prints each rejected camera/pair or 3D board metric with its limit,
keeps the published calibration intact, and saves the rejected candidate.
Low per-camera RMS does not guarantee consistent stereo poses or accurate
triangulation. To inspect the last candidate (including files from older versions):

```powershell
./.venv-wilor/Scripts/python.exe -m mavis_track inspect-calibration --backend optitrack --candidate
```

Inspection reports failed status for a rejected candidate; it never publishes it.
The failure reasons and thresholds are saved with new candidates.

Replay the latest complete archive in the configured calibration directory:

```powershell
python -m mavis_track replay-calibration --backend optitrack --latest
```

Close the tracking GUI first, then restart it to load the recomputed calibration.
Replay checks input existence, archive structure and output/source separation
before fitting. It prints the selected paths and skips incomplete/corrupt archives.
With no suitable archive, it reports the searched directory without replacing the
active calibration. A fitted calibration file alone is not a corner-observation
archive. Camera-preview startup already handles missing/rejected calibrations by
keeping 2D video available and disabling calibrated reconstruction.

Stereo fitting preserves each camera's standard or extended distortion model
(rational, thin-prism or tilted), including pairs with different model lengths.
The fitted intrinsics stay fixed during alignment. If a previous calibration used
an extended lens model, replay the saved observations or recalibrate with the fix
to recompute its stereo poses; existing calibration files are not rewritten
automatically.

Calibration count, source indices, image sizes, and matrices are checked when loaded.
Legacy files lacking identity or size metadata produce explicit warnings.

Source indices are not physical camera identities and can change after USB
replugging. Verify physical order and recalibrate after a change. UID-based
remapping is supported when a camera manager supplies reliable UIDs; the current
OptiTrack binding integration does not establish that hardware identity here.

Tracking excludes frames older than 250 ms and cameras whose host receipt times
differ by more than 50 ms. Host receipt timestamps do not establish synchronized
hardware exposures. Fast motion still needs a hardware synchronization assessment.

Advanced multi-camera hand matching currently assumes one hand across views.
Two-hand and multi-person association across cameras remain future work.
Geometry diagnostics are local pixel-noise estimates; they do not include
calibration error or model bias.

## OptiTrack

Launch the tracking window with `mavis-track gui --backend optitrack`. The window
starts within the current screen's available desktop area, including window
borders and the taskbar. Camera and 3D previews resize with the window while
preserving image proportions; the right-hand controls scroll on smaller screens.
Close and relaunch an already running GUI after updating the source.

Camera preview and per-camera 2D hand overlays work without calibration. If the
saved calibration is absent, unreadable, or rejected for the connected cameras,
the GUI shows a calibration notice and keeps the feeds running. It disables the
3D view, reconstruction controls, and reconstructed joint angles; existing
UDP/LSL streams report unavailable hands rather than fabricated 3D observations.

To enable 3D, close the tracking GUI, run
`mavis-track calibrate --backend optitrack`, then reopen the GUI. Each start
validates the saved calibration against the connected camera count, identities
when available, and image sizes before enabling reconstruction.

Within the GUI, **Stop Cameras → select model → Start Cameras** is supported.
Stopping joins the capture workers and releases camera handles. The shared
OptiTrack SDK stays initialized until the application exits, so MediaPipe and
WiLoR can reopen the cameras without attempting to restart a shut-down native
singleton. Start/model selection remains blocked while WiLoR finishes stopping.
Only one capture session can own the cameras at a time.

**Enable Inference** can be switched independently while cameras keep streaming.
**Start Recording** saves per-camera videos, frame timing sidecars and sampled
OptiHub configuration with inference on or off. OptiHub 2 (2011) input levels and
individual electrical edges are unavailable through the legacy SDK; configuration
samples are not pulse recordings. See [recording and sync](optitrack-recording.md)
for session formats, timing limits and the native binding upgrade.
With the sync-control binding extension, **Apply Sync Mode** selects Vicon Nexus
external timing (expected 50 Hz, rising edge, divider 1) or an internal 50 Hz
generator. Applying a mode is explicit and blocked during recording; camera
startup can wait for external pulses without closing the capture session.

Close MAVIS completely before using a separate calibration or vendor camera
application. After updating an older GUI that already reported zero cameras on
restart, fully close and relaunch it once to load the corrected SDK lifetime.

The optional OptiTrack path uses `optitrack_cam` and a separately obtained
Windows Camera SDK. Source installations discover the local
`external/optitrack-sdk-1.10/runtime` directory automatically. Set
`OPTITRACK_CAM_PY_PATH` to use another compatible binding folder. See the
[new-computer installation steps](installation.md#optitrack-on-a-new-windows-computer).
On 2 November 2026, the operator reported that completing the USB driver stage
of the official installer resolved zero-camera discovery on a new PC. The
remaining SDK installation was cancelled after the driver step completed; see
the installation guide for this driver-only workflow. This operator report is
separate from the automated software/import checks.

SDK binaries and headers are excluded from wheel, source distribution, and curated
example downloads. Existing repository history contains earlier development
copies; those are not covered by MAVIS's MIT license.

## Display and calibration controls

The OptiTrack tracking window supports point-sized text, monitor DPI changes, a
resizable sidebar and a text zoom control. Camera tiles use an adaptive grid.
**Fit full image** preserves every pixel; **Fill area** crops the displayed edges
only. Display zoom/cropping never changes detector input or calibration geometry.

OptiTrack auto-capture requires the board to remain still for at least 0.5 seconds
in every camera (corner displacement at most 1 pixel from the settling anchor).
Use a rigid, flat board, hold each pose steady, then change angle/position after
a capture. The capture checks host frame freshness (250 ms), new frames from all
cameras, and receipt-time spread (50 ms). These reject unsuitable frame batches;
host receipt time does not establish hardware exposure synchronization. The
preview explains when it is waiting for a steady board or fresh frames. Each
camera uses its own ChArUco detector during concurrent detection.

The OptiTrack calibration preview uses the same scalable camera tiles, readable
text and a text zoom control. Its compact footer lists these keyboard commands:

| Key | Action |
|---|---|
| Up / Down | Increase / decrease exposure by 1 |
| Shift + Up / Down | Increase / decrease exposure by 10 |
| 0–9 | Select the matching logical camera index |
| A | Target all cameras for exposure and illumination |
| I / K | Increase / decrease IR intensity by 1, within each camera's limits |
| M | Maximum available intensity and supported/available high-power mode |
| H | Toggle high power on capable selected cameras |
| [ / ] | Decrease / increase corner motion tolerance by 0.5 pixels |
| B / N | Decrease / increase maximum corner outlier by 1 pixel |
| J / L | Decrease / increase stable time by 0.1 seconds |
| U / O | Decrease / increase capture countdown by 0.5 seconds |
| Space | Pause / resume automatic board capture |
| C | Capture now, skipping the motion wait and countdown |
| Q | Finish capture and attempt calibration with sufficient observations |
| Esc or close | Cancel without replacing the saved calibration |

Exposure values are SDK settings, not assumed milliseconds. The status line shows
the target cameras and requested values; acquisition pauses briefly after a
change to avoid collecting frames from before that change. A failed setting is
reported in the preview. With PyQt5 unavailable, an OpenCV fallback uses W/S and
Shift+W/S for exposure and displays those keys in its footer.

The Qt preview keeps camera images beside a resizable, scrollable control panel so
small screens and larger text do not collapse the video area. Controls and keyboard
instructions remain accessible by scrolling.

The Qt preview also has **Corner motion**, **Max outlier**, **Stable**, and
**Countdown** controls. Changes restart the wait; settings apply to this run.
Defaults are 5 pixels, 20 pixels, 0.1 seconds, and 0.5 seconds respectively.
Automatic capture can start after about 0.6 seconds of stable detections;
actual speed also depends on detection and storage time. For immediate capture,
click **Capture now (C)** or press **C**. This also works while auto-capture is
paused. It skips only motion checks and the countdown: board detection, finite
corners, fresh/new camera frames and pair receipt-time checks still apply.
Motion is the lower empirical 90th percentile of matched-corner displacement
from the reference pose, allowing isolated detection jitter. The maximum-outlier
limit still rejects large individual jumps. At least six shared corners are
required. The reference stays fixed while steady, so gradual drift cannot evade
the check by moving only a little between consecutive frames.

The preview reports each selected camera's shared-corner count, measured P90
and maximum displacement alongside their limits, stability progress, reset
reason, and capture countdown. If a stationary board repeatedly resets, check
these readings and image contrast before increasing tolerances. OptiTrack
calibration now starts with the tracking exposure default (50 SDK units); use
the existing exposure and IR controls to avoid dark or saturated board images.
Keep the board still during each capture, then vary its position and angle
between captures.

Capture tuning does not change final acceptance limits: maximum held-out view
reprojection error 1.5 pixels, reconstructed board shape RMS 2 mm, absolute scale
error 2%, and planarity RMS 1 mm. These checks do not guarantee accuracy at every
location in the capture volume; validate against known distances in your workspace.

For the estimator accuracy audit and equal-priority hand/whole-body roadmap, see
[the MMPose comparison](mmpose-comparison.md).


## IR illumination

Tracking and OptiTrack calibration include an **IR illumination** panel. Select
**All cameras** or a camera index, enter an SDK intensity and choose **Apply IR**.
**High power** requests the supported camera/hub mode; **Max available** sets
each targeted camera to its reported maximum intensity and enables high power
only where supported and available. Starting preview does not change lighting.

The readback lists each camera's intensity, SDK limits and high-power state.
Unavailable controls are disabled. Mixed hardware is handled per camera:
unsupported targets are skipped with a reason; changes to other cameras remain
visible. For manual all-camera levels, the input range is the intersection of
supported ranges; Max available can use each camera's different maximum.

Flex 3 high-power operation requires a compatible powered OptiHub2. The SDK's
supported/available flags determine which controls can be used. See
[OptiTrack's high-power documentation](https://docs.optitrack.com/v3.2/motive/calibration/.mcal-xml-calibration-files#highpower).

Intensity and high-power output are separate from exposure, gain and the status
indicator LEDs. SDK readback reports settings, not measured optical power.
Continuous IR and camera frame rate are not changed by these controls.
Calibration pauses capture briefly after lighting changes, including partial
failures; image batches acquired before the change are discarded.

These controls require binding API v2. The historical binding restored by
`setup_optitrack.py --from-history` has no illumination methods: preview remains
available, and the panel explains that a binding upgrade is required. `doctor`
reports this separately under `optitrack:illumination`. See the
[prebuilt binding installation](installation.md#install-the-prebuilt-camera-binding).

## Optional WiLoR hand model

The optional experimental WiLoR integration provides a frame API and
single-source and OptiTrack multi-camera mesh/joint preview. Model choice is
separate from camera backend. The OptiTrack grid shares one WiLoR model across
the feeds and retains exposure/IR controls.
See [multi-camera setup and validation status](wilor.md#optitrack-multi-camera-gui).
Calibrated reconstruction remains separate from independent WiLoR camera models.

## Windows USB webcam setup

See [USB webcams](webcams.md) for source selection, native driver controls,
video modes, shared staged calibration and saved settings. The default webcam
Studio previews one source; use the advanced webcam GUI for calibrated multi-camera hands.
