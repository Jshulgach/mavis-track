# Live landmark analysis

Both advanced camera dashboards include **Cameras** and **Landmark tracker** tabs.
Start the selected cameras, choose an inference model/mode, enable inference, then
open **Landmark tracker**. Switching tabs preserves capture and inference.

Choose a **Signal** (camera/region or calibrated hand), then a **Landmark**.
Hand joints and body landmarks have names; face landmarks use indexed entries
with a few descriptive aliases. Type into the landmark box to search and select
a matching entry. Only the selected landmark's history is retained.

The three plots show x/y/z position, velocity and acceleration. Image signals
show x/y only. Colours are consistent across plots. **Pause plots** freezes the
display while inference continues; **Clear plots** clears this analysis history.
Neither action starts recording or changes camera capture.

## Coordinates and units

| Signal | Position | Velocity / acceleration | Meaning |
| --- | --- | --- | --- |
| Calibrated MediaPipe hand | m | m/s, m/s² | Existing filtered hand estimate in camera-0 rig coordinates; requires usable calibration and accepted reconstruction for the selected point |
| Calibrated WiLoR hand | m | m/s, m/s² | Model mesh joints fitted into the calibrated rig; points without accepted triangulation are excluded |
| Camera hand/body/face | px | px/s, px/s² | Full-source image coordinates: x right, y down; excludes uncalibrated model depth |
| WiLoR Model 3D | m | m/s, m/s² | Model-estimated wrist-relative camera axes; wrist translation is absent, so wrist motion is zero in this space |

WiLoR model coordinates are separate from calibrated rig coordinates. Image
velocity depends on camera geometry, distance and resolution; it is not physical
speed. The units/coordinate frame appear above the plots. Moving cameras still
requires recalibration before using calibrated rig positions.

## Derivatives, timing and gaps

Velocity and acceleration use source-frame **host receipt timestamps**, not GUI
redraw times or model completion times. The inherited camera freshness/skew
checks still apply. These clocks do not establish hardware exposure timing or
synchronization. Repeated source frames never create extra motion observations.

**Fit window** defaults to 0.25 seconds. The fit is causal: it uses only recent
samples, with a linear fit for velocity and a quadratic fit once at least five
samples span 50 ms. Increasing the window reduces noise but smooths fast changes.
Derivatives amplify landmark noise and should be validated for research use.

Missing/non-finite points, rejected reconstruction, body visibility below 0.5,
clock reversals, and gaps longer than 0.5 seconds reset the fit. Reacquisition
starts new derivative estimates; it does not draw a velocity/acceleration spike
across the gap. Low-rate inference can therefore leave derivatives unavailable.
The readout shows unavailable estimates as a dash. Sample age is displayed.

Signal, landmark, model/mode, coordinate-shape and WiLoR track-identity changes
clear incompatible history. Body/face slots and MediaPipe hand labels are not
persistent person identities: use one person in view for interpretation. Multiple
same-labelled hands in a view are treated as ambiguous and excluded.

**History** controls the visible duration (2–60 seconds). Storage is bounded to
4096 selected-point samples and 61 recent fit samples; curves redraw at most
20 times per second and stop redrawing while the tab is hidden. Stopping cameras
retains the plotted history for review; restarting starts a new analysis session.
Existing raw video recording and hand streams are unchanged by the analysis tab.

The GUI extra includes PyQtGraph. If unavailable, the tab explains how to install
`mavis-track[gui]`; camera/inference controls remain available.
