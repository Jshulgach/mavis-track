# Windows USB webcams

MAVIS discovers Windows USB camera names and DirectShow device paths automatically.
Camera models do not need custom setup. Driver support still determines which
controls and video modes are available.

## Set up three cameras

Run these commands from your activated MAVIS environment:

```powershell
mavis-track cameras --backend webcam
```

The setup window previews every opened source. Check the camera names and images,
uncheck any camera you do not want (including the laptop camera). Toggles update
the running camera group automatically. Toggling a camera starts/stops only
that source; the other capture threads keep running. Every known source keeps
a fixed dashboard tile: stopped cameras show a placeholder, and re-enabled
cameras return to the same tile. Unchecked cameras stay in the list so you can
turn them back on. **Stop cameras** releases all streams together and
keeps your selection; **Start selected cameras** opens the checked sources again.
While stopped, toggles change the selection without opening cameras. Turning
every source off stops all streams; select a source again or use Start to resume.

**Retry with MJPEG / up to 15 FPS** is on by default. If a selected camera
fails to start, Setup requests MJPEG and a lower frame rate for the selected
cameras before retrying. Existing previews may pause briefly during the mode
change. Check each camera's actual format and received FPS in Camera Controls.
Uncheck the retry option to keep your preferred modes when a source fails.

You can also enter source indices and use **Find / preview sources** to probe
them. The continue action (**Start calibration** in calibration setup) requires the selected sources to be running, so an
unapplied selection cannot be saved accidentally. The selection order becomes
the calibration order.

Choose a camera or **All cameras** in the settings panel. **Change video mode** enables width, height, requested FPS and format. MJPEG can reduce USB
bandwidth when the driver supports it. These are requests, not a complete list of
advertised modes: verify the actual resolution, format and reported/received FPS
shown below the controls. A driver may report no FPS; received FPS is measured.

Exposure uses driver units, including negative values, rather than a universal
millisecond scale. On DirectShow, MAVIS reads native min/max/step and auto/manual
capabilities. Unsupported controls are disabled. An unknown range is labelled as
driver units; successful writes still require readback where available.
Some OpenCV/driver combinations cannot report auto-exposure state. The panel
distinguishes an acknowledged request from confirmed device readback.

Set exposure to **Manual** and adjust it while checking the image. Reduce motion
blur without making the board too dark. Focus the board where supported.
**Start calibration** switches supported autofocus cameras to manual at their
current focus position, waits for fresh frames and verifies focus readback.
If a driver refuses or cannot confirm the lock, setup stays open with the reason;
set Manual focus in Camera Controls and retry. Optional gain, white balance and zoom controls are shown
according to driver support. **Driver settings** opens the selected
DirectShow camera's native settings panel; close that panel before continuing.

Use **Save and continue** to persist the selected devices and applied
settings. Profiles are local, not part of the repository:
`%LOCALAPPDATA%\mavis-track\calibration\webcam\webcam-settings.json`.
Override this location with `MAVIS_WEBCAM_PROFILE`. Known DirectShow device paths
allow saved settings to follow cameras when numeric USB source indices change.
Where identities cannot be verified, check camera order from the previews.

## Calibrate

```powershell
mavis-track calibrate --backend webcam
```

Camera setup opens first. New calibration does not require an existing
calibration file or camera profile. Select and start the cameras, check that the
printed board is sharp, then click **Start calibration**.
Webcams can use different resolutions. Choose each camera's mode in **Camera
Controls** and verify its actual resolution before starting calibration. For
example, use 320×240 for the LifeCam and 640×480 for the other cameras. MAVIS
fits and saves the calibration at each camera's native resolution. Keep those
modes for tracking with that calibration.
After setup, webcams use the same staged calibration window as OptiTrack:
individual camera intrinsics, then overlapping camera pairs. A connected chain
such as 0–1, 1–2 and 2–3 is sufficient for four cameras.

Use the board dimensions configured for the webcam backend. Generate a matching
board with `mavis-track board --backend webcam` and verify the printed square
size before capture. Vary the board's position, distance and tilt between captures,
and hold it still for each observation. Capture controls, motion diagnostics,
manual `C` capture and observation checkpoints are shared with OptiTrack.
Camera testing and calibration reuse the OptiTrack dashboard palette, grouped
sidebar and action styles. Text size can be increased without shrinking the
camera images. Calibration also exposes Pause/Resume, Next/finish and Cancel
buttons alongside the keyboard shortcuts.

Video mode and lens controls are locked during calibration and tracking. Exposure,
gain and white balance remain available; changes discard stale frames and restart
the capture wait. USB receipt timestamps are not hardware exposure synchronization.

Captured corners must constrain the board pose, as well as meet the minimum
corner count. A narrow partial board with most corners on one row is rejected
with guidance to show more of the board. Intrinsic fitting also checks this
geometry in older saved observations before splitting training/held-out views.
The 1.5 px validation limit applies to the worst held-out view, not the aggregate RMS.

If capture stopped after intrinsics or during alignment, resume on the same rig:

```powershell
mavis-track calibrate --backend webcam --resume-latest
```

Or choose an explicit checkpoint with `--resume "path/to/calibration_observations_....npz"`.
Setup verifies device identities, resolution, format, focus and zoom. Usable
intrinsic captures are reused and validated again; missing views/alignment are
collected live. The source archive stays unchanged, and a new checkpoint records
its origin. Keep the camera mounts and board dimensions unchanged.

Once alignment observations are complete, camera-free recovery is also available:

```powershell
mavis-track replay-calibration --backend webcam --latest
```

Replay alone cannot collect a missing alignment stage.

Quality gates and atomic publication remain unchanged: failed/cancelled calibration
preserves the previous validated file. Archives include actual webcam settings.
Changed resolution, video format, device identity, focus or zoom prevents incompatible calibrated
3D reconstruction. Exposure and color changes do not invalidate camera geometry.

## Track

Both camera dashboards provide **Display → Text size** (80–200%) and
**Fit full image / Fill area**. Text uses point sizes and refreshes after monitor
DPI changes; moving an oversized window to a smaller monitor shrinks it to the
usable desktop. The sidebar resizes beside the video and scrolls on small screens;
smoothing settings use one field per row. Fill area crops only the displayed
image, preserving inference and calibration inputs.

```powershell
mavis-track gui --backend webcam
```

The source numbers belong to the selected Windows capture backend. **Auto**
chooses one backend for the group, preferring DirectShow when it discovers
cameras there. Changing backends remaps cameras by device path where possible.
Check the physical camera names next to the source numbers.

If a selected camera fails during startup, the live dashboard keeps the working
feeds and lists the failed source in **Camera Setup**. The camera setup dialog
also keeps successful previews running. Uncheck failed sources before saving a
different selection. Calibrated 3D still requires a compatible camera set.

The advanced multi-camera dashboard is the default for every camera backend.
The tracking panel keeps a separate calibration notice visible while inference is
off. Before startup it reports whether the configured file exists; starting
cameras validates that file against the selected devices, mode and lens settings.
A loaded calibration remains identified as loaded when a per-camera mode pauses
rig 3D. Hover the notice to see the file path. Mismatch notices report the
expected/current settings so the calibrated configuration can be restored.

The older `--advanced-hands` flag remains accepted. **Start selected cameras** and
**Stop all cameras** start/stop the tracking group. **Choose cameras** opens the
same selection window, first stopping active tracking and then restarting with
your saved sources. **Exposure / camera settings** adjusts live controls with
geometry locked. **Use for 3D** checkboxes only control reconstruction, and are
disabled without usable calibration; they do not start or stop USB streams.
Stop first to change video mode, focus or zoom. Without usable calibration, camera video and per-camera 2D
tracking remain available and calibrated 3D is disabled.

**Model** selects MediaPipe or optional WiLoR. For MediaPipe, select any
combination of **Hands**, **Body**, and **Face**. Start cameras and check
**Enable inference** when ready. Changing the selected regions or disabling
inference keeps the cameras running, as does clearing all three checkboxes.
Raw feeds stay visible during model loading and errors. **Show 3D View**
displays all selected MediaPipe regions together when at least two enabled
cameras with matching calibration provide agreeing views. Keep one subject in
view for Body and Face. Use Hands alone for the existing hand angles and
landmark streams. Webcam WiLoR provides per-camera hand mesh and joint previews.
WiLoR needs its
[separate environment and assets](wilor.md).

`mavis-track gui --studio --backend webcam` explicitly opens the single-camera Studio. Its **USB camera
setup** button stops the current stream before opening setup, then resumes the first
selected source. Studio previews one source; use the advanced command above for
calibrated multi-camera tracking.

Native controls vary between drivers. A supported-looking range is not proof that
a device accepts a requested value: errors and actual readback remain visible.
Three cameras can also exceed a shared USB controller's bandwidth. Test the
selected modes together, and check received FPS before collecting calibration data.
