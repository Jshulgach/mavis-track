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

You can also enter source indices and use **Find / preview sources** to probe
them. **Save and continue** requires the selected sources to be running, so an
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
blur without making the board too dark. Use **Manual focus** and focus the board
where supported; calibration setup requires autofocus to be disabled when manual
focus is available. Optional gain, white balance and zoom controls are shown
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

Camera setup opens first. Every selected camera must have the same **actual**
resolution for the current stereo solver. After setup, webcams use the same
staged calibration window as OptiTrack: individual camera intrinsics, then
overlapping camera pairs. A three-camera chain such as 0–1 and 1–2 is sufficient;
all three cameras do not have to see the board at once.

Use the board dimensions configured for the webcam backend. Generate a matching
board with `mavis-track board --backend webcam` and verify the printed square
size before capture. Vary the board's position, distance and tilt between captures,
and hold it still for each observation. Capture controls, motion diagnostics,
manual `C` capture and observation checkpoints are shared with OptiTrack.

Video mode and lens controls are locked during calibration and tracking. Exposure,
gain and white balance remain available; changes discard stale frames and restart
the capture wait. USB receipt timestamps are not hardware exposure synchronization.

Saved observations support camera-free recovery:

```powershell
mavis-track replay-calibration --backend webcam --latest
```

Quality gates and atomic publication remain unchanged: failed/cancelled calibration
preserves the previous validated file. Archives include actual webcam settings.
Changed resolution, video format, device identity, focus or zoom prevents incompatible calibrated
3D reconstruction. Exposure and color changes do not invalidate camera geometry.

## Track

```powershell
mavis-track gui --backend webcam --advanced-hands
```

This is the multi-camera hand-tracking view. **Start selected cameras** and
**Stop all cameras** start/stop the tracking group. **Choose cameras** opens the
same selection window, first stopping active tracking and then restarting with
your saved sources. **Exposure / camera settings** adjusts live controls with
geometry locked. **Use for 3D** checkboxes only control reconstruction, and are
disabled without usable calibration; they do not start or stop USB streams.
Stop first to change video mode, focus or zoom. Without usable calibration, camera video and per-camera 2D
tracking remain available and calibrated 3D is disabled.

`mavis-track gui --backend webcam` opens the single-camera Studio. Its **USB camera
setup** button stops the current stream before opening setup, then resumes the first
selected source. Studio previews one source; use the advanced command above for
calibrated multi-camera tracking.

Native controls vary between drivers. A supported-looking range is not proof that
a device accepts a requested value: errors and actual readback remain visible.
Three cameras can also exceed a shared USB controller's bandwidth. Test the
selected modes together, and check received FPS before collecting calibration data.
