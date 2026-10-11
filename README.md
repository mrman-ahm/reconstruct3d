# Prerecorded video to 3D

This is a Phase 1 research prototype for turning one prerecorded walkthrough video into a sparse reconstruction, dense point cloud, textured mesh, and portable exports. It does not implement live scanning, still-photo modes, a GUI, or a web app.

## Current status

The full CPU pipeline has been run on a 38-second bedroom walkthrough. COLMAP registered 119/125 selected views and OpenMVS produced a dense cloud, mesh, and textured GLB. Visual inspection found substantial missing geometry and severe texture artifacts, so the current result does not yet meet the recognizable-room target. See [the Phase 1 results](docs/phase-1-results.md) for the measured run, hardware, and diagnosis.

## Run

Set up the Python and native tools using [the software setup guide](docs/software-setup.md). Place a walkthrough at `inputs/room.mp4`, then run:

```bash
./reconstruct inputs/room.mp4
```

The default job goes to `outputs/room/`. For a preprocessing-only check:

```bash
./reconstruct inputs/room.mp4 --stop-after keyframes
```

Inspect detected hardware and tool availability with:

```bash
./reconstruct --doctor
```

Available processing presets are `preview`, `balanced` (default), and `high`. Choose an output location with ample free space using `--output /path/to/room-job`. A job contains metadata, selected keyframes, a contact sheet and CSV, sparse model and camera trajectory, dense outputs, exports, logs, and resumable stage status.

Completed stages populate `metadata.json`, `keyframes/`, `sparse/`, `dense/`, `mesh/`, `texture/`, `exports/`, `diagnostics/`, and `logs/` under the job directory.

The supplied walkthrough's inspection artifacts are in [`outputs/whatsapp-video-2026-10-02-keyframe-experiment-02/`](outputs/whatsapp-video-2026-10-02-keyframe-experiment-02/). Its final exports are `exports/scene.glb` (with `scene_dense_mesh_texture_0.png`), `exports/scene.ply`, and `exports/scene_points.ply`; see `diagnostics/final-inspection.png` for the textured and neutral geometry previews.

## Capture guidance

See [the recording guide](docs/recording-guide.md) before filming. A slow, overlapping walkthrough with stable exposure and sharp frames matters more than high frame rate.

## Tests

Run the standard-library test suite with:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Tests cover video metadata parsing, keyframe scoring, command construction, subprocess failures, and resume behavior. Fake executable tests verify the native-stage orchestration and output checks; they do not validate COLMAP or OpenMVS reconstruction quality.
