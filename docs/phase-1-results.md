# Phase 1 — Video to 3D results

**Status: Phase 1 was run end to end on the supplied bedroom walkthrough and the exported GLB was opened and inspected.** Sparse camera recovery was strong; dense geometry is substantial but incomplete, and texturing is poor. This is a useful but not yet satisfactory room reconstruction.

## Machine and software inventory

Detected on 2026-10-02 in the project execution environment. The initial package inventory below predates the local native-tool installation and room run; installed versions and measured results are recorded in the later experiment sections.

| Resource | Detected |
| --- | --- |
| OS / kernel | Kali GNU/Linux Rolling 2026.3, x86_64; Linux 7.0.12+kali-amd64 |
| CPU | 13th Gen Intel Core i7-1355U; 10 cores / 12 threads, AVX2 and AVX-VNNI |
| RAM | 7,462 MiB total; 2,712 MiB available at the final `--doctor` snapshot (available RAM changes over time) |
| GPU | Intel Raptor Lake-P UHD Graphics, integrated/shared memory; no dedicated VRAM reported |
| GPU access | No `/dev/dri` render node in this execution environment; CUDA and `nvcc` absent. OpenCL lists POCL on the CPU only. Vulkan could not start because this environment has no X server connection. No usable GPU compute path was confirmed. |
| Project filesystem | 16.83 GiB free after the user cleared space and the local Python setup completed |
| Python / imaging | Python 3.13.14; NumPy 2.4.6; OpenCV 5.0.0 with FFmpeg decoder support |
| Other software | GCC 15.3.0, Clang 21.1.8, Blender 5.2.0 LTS |
| Not installed at initial inspection | FFmpeg CLI, ffprobe, CMake, Ninja, COLMAP, OpenMVS, OpenMVG, OpenSfM, Meshroom/AliceVision, MVE, pycolmap, Open3D, PyTorch, Trimesh, VGGT, DUSt3R, Nerfstudio |
| Container/install access | Docker CLI exists but daemon access is denied. Initial outbound DNS failures blocked package/source downloads; COLMAP and OpenMVS were later installed under the project-local `native/` directory. Python packages and native tools were kept local; no global/system packages were changed. |

The OpenCV decoder is sufficient for keyframe extraction in this prototype; ffprobe is optional but preferable for richer metadata. Blender 5.2.0 was used headlessly to reopen the GLB, embed its OpenMVS atlas, and render inspection views. The project directory has an empty `.git` placeholder rather than a usable Git repository, so this work could not be committed.

## Research and stack decision

Research snapshot: 2026-10-02. Selection prioritizes CPU execution, camera poses, room-scale dense geometry, textured mesh output, and reasonable reliability on a 7.3 GiB machine.

| Candidate | Practical fit for this machine and phase | Decision |
| --- | --- | --- |
| **COLMAP CPU + OpenMVS CPU** | COLMAP supports CPU SIFT and CPU matching/SfM; exhaustive matching helped on this small revisited image set. OpenMVS supplies CPU PatchMatch, meshing, texturing, and GLB export. OpenMVS 2.4 also has an optional CPU SGM route. The separated tools add build/interop work, and OpenMVS is AGPL-3.0. | **Chosen and run.** Reliable camera recovery and usable dense geometry on this CPU-only machine; this clip exposed significant mesh and texture limitations. [COLMAP CPU options](https://colmap.github.io/faq.html) · [OpenMVS pipeline](https://github.com/cdcseacave/openMVS/blob/develop/docs/wiki/Usage.md) · [OpenMVS license](https://github.com/cdcseacave/openMVS/blob/develop/LICENSE) |
| COLMAP dense MVS | Sparse CPU reconstruction is supported, but COLMAP's PatchMatch dense stage requires CUDA. | Not used as the dense backend here. [COLMAP FAQ](https://colmap.github.io/faq.html) |
| OpenMVS native SfM + MVS | Current v2.4 includes feature extraction, matching, incremental/global SfM options, video keyframe extraction, dense reconstruction, mesh, and texturing. Its documented end-to-end video example uses 360° footage, while the keyframe tool defaults to pinhole imagery and can also process handheld capture. Its newer native SfM has not been benchmarked against COLMAP on our room data. | Candidate for the next controlled backend comparison. [OpenMVS 2.4 usage](https://github.com/cdcseacave/openMVS/blob/develop/docs/wiki/Usage.md) |
| OpenSfM 1.1 | A recent 2026 release has CPU-friendly classical feature choices and OpenCL dense PatchMatch/fusion. On this machine OpenCL only exposes POCL CPU, so dense compute would be software-rendered; its mesh output is documented as experimental and off by default. | Do not make it the default. Revisit if POCL CPU dense performance or a supported accelerator changes. [OpenSfM releases and feature list](https://github.com/OpenSfM/OpenSfM/releases) |
| AliceVision / Meshroom | Its current MVS requirements specify an NVIDIA CUDA-capable GPU and at least 8 GB RAM. This execution environment has integrated Intel graphics, no CUDA access, and 7.3 GiB total RAM. | Poor fit for the detected hardware. [Meshroom requirements](https://github.com/alicevision/meshroom-manual/blob/develop/source/first-steps/install/requirements.rst) |
| MVE | A CPU-oriented photogrammetry pipeline with lower accelerator dependence. The upstream project describes itself as being in maintenance mode, so it is less attractive than the selected split stack for an evolving prototype. | No implementation; useful fallback for a later measured comparison. [MVE project](https://github.com/simonfuhrmann/mve) |
| VGGT / DUSt3R and similar learned geometry | Can infer pose/depth/point maps from image sets, but this machine has no PyTorch installed or accessible GPU. These outputs also need additional surface and texture work for the requested deliverable. | Research only; not a default CPU dependency. [VGGT](https://github.com/facebookresearch/vggt) · [DUSt3R paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_DUSt3R_Geometric_3D_Vision_Made_Easy_CVPR_2024_paper.pdf) |
| Gaussian Splatting | The original method targets real-time radiance-field rendering with a differentiable Gaussian rasterizer. Attractive view synthesis, but a splat is not the same deliverable as a coherent textured mesh, and this common implementation route depends heavily on GPU rasterization. | Not selected for the primary geometry objective. [Original project/paper](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) |

COLMAP is BSD-3-Clause. OpenMVS is AGPL-3.0, which matters before redistributing a combined product. Current upstream COLMAP release reviewed was 4.2.1; current OpenMVS source/release reviewed was 2.4.0. The driver records detected tool paths and startup banners for later jobs.

## Implemented pipeline

```text
video → metadata → sampled candidates → blur/exposure/overlap/change scoring
      → keyframe images + CSV/contact sheet → COLMAP CPU sparse model
      → OpenMVS CPU dense point cloud → mesh → texture → PLY/GLB
```

Keyframe defaults sample up to 4 candidates per second. A candidate is scored at 640 px width using Laplacian blur variance, clipped-pixel fraction, ORB match count, RANSAC inliers, and median inlier motion. Only selected images are written at full resolution. Parameters and candidate reasons are saved. For at most 160 keyframes the command builder chooses exhaustive matching; larger jobs use sequential matching with a 10-frame overlap. A COLMAP vocabulary tree can enable loop detection on larger jobs. This matching threshold is a hardware-aware initial choice, not a benchmark result.

Quality presets cap image sizes at 960/1440/1920 px and keyframes at 60/160/280; the measured balanced run used three CPU threads. OpenMVS resolution level is one in all presets so it does not discard multiple powers of two from room images; high also enables CPU mesh refinement. Preview/high were not benchmarked.

The sparse gate currently requires at least six selected/registered images, 45% registration, and 100 sparse points before dense processing. These are initial stop rules, not validated indoor quality thresholds. Camera intrinsics are shared across frames and COLMAP estimates focal length and radial distortion; video focal length tags are recorded when present but are not passed as a calibrated sensor model.

## Experiments

### Environment smoke — 2026-10-02

- Input video: no real video in project; generated moving-texture AVI only for a software smoke check (3 seconds, 320×240, 12 FPS, 36 frames).
- Smoke counts: 12 candidate frames sampled, 6 selected at 0.0–2.5s, 6 near-identical candidates rejected. Runtime was 1.515 seconds wall time on the current CPU; peak RAM was not measured.
- Inspection: opened the contact sheet and confirmed the saved frames are ordered and the synthetic texture shifts gradually. This is a feature-rich synthetic pattern, not an environment and not a quality benchmark.
- Real video frames / selected frames: not applicable.
- COLMAP/OpenMVS versions: unavailable.
- Sparse registration / point count: not run.
- Dense point count / mesh / texture / GLB: not run.
- Real-room visual quality: **not evaluated**.
- Native pipeline runtime and peak RAM: not measured.
- Default CLI failure path: re-ran the synthetic clip without `--stop-after keyframes`. It wrote metadata and the six keyframes, then exited with status 2 during preflight, recorded the missing COLMAP/OpenMVS programs, and marked the job failed. No sparse/dense command ran.
- Verification: all 32 standard-library tests passed; `compileall`, shell syntax check, `./reconstruct --help`, and an editable package-install smoke check passed. Blender 5.2.0 LTS started headlessly. The fake native executables in unit tests verify command ordering and artifact checks only; their fake geometry is not a reconstruction result.
- Initial finding: the software environment can decode videos through OpenCV's linked FFmpeg and has CPU feature scoring, but the job cannot progress to SfM because the native applications are absent and the sandbox has networking disabled. Blender 5.2.0 also passed a background startup smoke check; it had no model to inspect.

### Local dependency setup — 2026-10-02

- Created/used the project-local `.venv`; copied the already installed NumPy, headless OpenCV, and setuptools package files into it. The venv does not include global Python package directories, and the launcher prefers its interpreter.
- `./reconstruct --help` starts successfully. `./reconstruct --doctor` confirms the local Python imaging stack starts and reports COLMAP/OpenMVS missing.
- Current free storage is 16.83 GiB. No native binaries or build dependencies were downloaded because `pypi.org` and GitHub do not resolve from this execution environment. No system package manager, global Python environment, or system install prefix was modified.

### Native CPU setup — 2026-10-03

- Shell network access was rechecked with `curl -I https://github.com/colmap/colmap/releases/tag/4.2.1` and returned HTTP 200. The project filesystem had about 17 GiB free before native setup.
- Verified the official [COLMAP 4.2.1 release](https://github.com/colmap/colmap/releases/tag/4.2.1) at commit `bd1fcf654d2dd8fefa1466999c190a246f83f4b9` and [OpenMVS v2.4.0 release](https://github.com/cdcseacave/openMVS/releases/tag/v2.4.0). COLMAP publishes no Linux release archive. OpenMVS publishes an Ubuntu x64 archive; its six needed CLI tools were extracted to `native/install/bin` without Viewer, Tests, or unrelated importers. The binaries report OpenMVS x64 v2.4.0.
- Installed local build tools CMake 4.4.3 and Ninja 1.13.2 under `native/tools`, micromamba 2.9.0 under `native/micromamba`, and build libraries under `native/mamba`. Key conda-forge build packages resolved to Boost 1.82.0, Ceres Solver 2.2.0, OpenImageIO 2.4.16.0, and SuiteSparse 7.10.1. OpenImageIO pulled FFmpeg libraries as a transitive dependency; its `ffmpeg` and `ffprobe` CLI executables were removed from the local prefix. The launcher and doctor report neither CLI tool.
- Built COLMAP 4.2.1 from the official source tag with CUDA, HIP, GUI, OpenGL, COLMAP dense MVS, ONNX, CGAL, tests, and download support disabled. The only retained source change sets FAISS to one generic CPU variant instead of dynamic SIMD dispatch. `cmake --build native/colmap-local-build --target colmap_main -j2` succeeded, and the finished CLI was copied to `native/install/bin/colmap`; it reports `COLMAP 4.2.1 (Commit bd1fcf6 ... without GPU support)`. Its runtime library path points to `native/mamba/lib`.
- Principal successful commands, all from the project root: `unzip -p native/downloads/OpenMVS_Ubuntu_x64.zip NAME > native/install/bin/NAME` for the six CLI tools; `PIP_CACHE_DIR=$PWD/native/cache/pip python3 -m pip install --target $PWD/native/tools cmake ninja`; `MAMBA_ROOT_PREFIX=$PWD/native/mamba XDG_CACHE_HOME=$PWD/native/cache/xdg CONDA_PKGS_DIRS=$PWD/native/cache/conda-pkgs native/micromamba/bin/micromamba install -y --no-rc -n base -r $PWD/native/mamba -c conda-forge --override-channels boost-cpp ceres-solver eigen gflags glog openimageio metis sqlite suitesparse pkg-config`; CMake configured with `-DCMAKE_PREFIX_PATH=$PWD/native/mamba`, `-DCMAKE_INSTALL_PREFIX=$PWD/native/install`, and the CPU-only flags above. Full reproduction commands are in [the setup guide](software-setup.md).
- Failed attempts were retained in local logs: initial vcpkg configuration tried to write its registry cache under read-only `~/.cache` until `XDG_CACHE_HOME` was set; OpenBLAS packaging then lacked `pkg-config`; the next vcpkg plan required a missing Fortran compiler for reference LAPACK. The vcpkg route was abandoned and its build/cache files were removed after the working dependency-prefix build. A micromamba named/prefix environment tried to register under read-only `~/.conda`; installing into the project-local base prefix worked. `cmake --install native/colmap-local-build` expected an unbuilt `libcolmap_exe.a` because only the CLI target was built, so the finished executable was installed directly with `install -m 755`. No system paths or `.venv` were changed.
- Final `./reconstruct --doctor` exited 0. `missing_balanced_pipeline_tools` is empty and `native_startup_error` is null. COLMAP, OpenMVS InterfaceCOLMAP, DensifyPointCloud, ReconstructMesh, TextureMesh, and TransformScene each reported `started: true`. OpenMVS help returns 1 as expected by the doctor. `RefineMesh` was also installed for other presets but is not required by balanced.
- Final disk snapshot: `native/` uses about 1.8 GiB; `native/install` 262 MiB, `native/mamba` 771 MiB, `native/cache` 991 MiB, `native/colmap-local-build` 123 MiB, `native/colmap-src` 31 MiB, and `native/downloads` 165 MiB. These component figures overlap because the package cache and prefix share hard links. The project filesystem has about 15 GiB free (`df -h`); the doctor reported 12.18 GB before the abandoned vcpkg files were cleaned. No video, unit-test suite, or room reconstruction was run in this setup session.

## Room experiment 01 — supplied WhatsApp walkthrough — 2026-10-03

- Input: `WhatsApp Video 2026-10-02 at 8.59.54 PM.mp4`, 6.60 MB, 38.48 seconds, 576×1024, 24.974 FPS, 961 decoded frames. Phone metadata did not provide usable focal-length calibration. The source is compressed and relatively low resolution for photogrammetry.
- Hardware/software: i7-1355U, 10 cores/12 threads, 7.3 GiB RAM, Intel integrated graphics with no CUDA or accessible GPU compute; COLMAP 4.2.1 CPU-only and OpenMVS 2.4.0. Balanced preset, three worker threads, SIMPLE_RADIAL camera.
- Keyframe experiments: the initial default run selected only the first 16.8 seconds because high-motion gaps were not being bridged; sparse mapping then also ingested its contact sheet as an image (24/41 registered). A 2 FPS alternative selected 49 frames but split into disconnected reconstruction components. The revised 4 FPS selector kept 125/161 candidates over the full 38.4 seconds, including sharp bridge frames after motion gaps. These controlled results support 4 FPS for this clip.
- Sparse result: 119/125 registered (95.2%), 15,336 sparse points, no mapping warnings. The camera trajectory runs through the bedroom and revisits the start region, but the reconstructed points include outliers and some scale/shape drift. Exhaustive CPU feature extraction, matching, and mapping took 792.8 seconds (13m13s).
- Dense result: OpenMVS produced 832,744 fused points. The dense pass took 47m23s. Meshing took about 45 seconds and produced 351,668 vertices / 703,014 triangles. Texturing took 4m03s and created one 4096×4096 atlas with 20,431 patches. `scene.glb` is about 18 MB with a 9.3 MB atlas PNG beside it; PLY mesh is 13 MB and colored dense PLY is 55 MB. Blender successfully re-imported the GLB/PNG pair and confirmed the atlas loads.
- Visual inspection: the [keyframe contact sheet](../outputs/whatsapp-video-2026-10-02-keyframe-experiment-02/diagnostics/keyframes-contact-sheet.jpg) confirms broad room coverage. The [neutral geometry view](../outputs/whatsapp-video-2026-10-02-keyframe-experiment-02/diagnostics/front.png) and [textured preview](../outputs/whatsapp-video-2026-10-02-keyframe-experiment-02/diagnostics/final-preview.png) are kept with the run. The camera trajectory is plausible, but it does not prove metric accuracy. The mesh view is fragmented, with large missing surfaces and holes rather than a closed, clean room shell; some local furniture-like surfaces are present but major objects and their spatial relationships are not consistently recognizable. The atlas and textured render are severely patchy/dark. This video therefore demonstrates that the CPU pipeline can recover poses and substantial geometry, but **does not yet meet the “recognizable room” success target**.
- Runtime and resource evidence: measured successful compute stages total roughly 65 minutes (13m13s sparse, 47m23s dense, 45s mesh, 4m03s texture), plus short preprocessing and export. The job record spans 92 minutes because it includes debugging/restarts and idle time; that is not a pure runtime benchmark. OpenMVS logs report process virtual-size peaks of about 2.1 GiB during dense fusion, 1.25 GiB meshing, and 2.36 GiB texturing; these are **VmPeak, not RSS**. Peak resident RAM and GPU usage were not sampled. The reconstruction ran CPU-only. It retained the dense intermediates and fit within the available disk, leaving about 12 GiB free.
- Export check: the textured GLB opens in Blender with its neighboring 4K atlas PNG. Exports also include `scene.ply` and `scene_points.ply`; COLMAP/OpenMVS native intermediates, keyframes, CSV, contact sheet, camera trajectory, and logs remain in the job directory. OpenMVS emits a GLB that references its atlas PNG; the prototype now copies that atlas beside the GLB so the pair remains portable.
- Troubleshooting retained: fixed the COLMAP 4.2 feature-matching option name; isolated only frame images from diagnostics; selected bridge frames after long motion gaps; and adapted meshing/texturing checks to OpenMVS's actual PLY and GLB output behavior. Sparse/dense resume avoids repeating the long dense pass during export fixes.
- Verification: all 34 standard-library tests pass after the fixes. They cover orchestration and resume behavior with fake executables; the real video run and Blender import/render provide the actual quality evidence.

## Phase 1 evaluation

**What works:** video decoding and keyframe diagnostics; CPU COLMAP pose recovery at 95.2% registration; CPU OpenMVS dense cloud, mesh, and texture generation; resumable stages; PLY and GLB exports.

**Quality:** sparse structure is coherent enough to locate the walkthrough, but the textured mesh is incomplete and visibly corrupted. It is not yet a useful recognizable room model.

**Hardware bottleneck:** the i7 CPU can complete the pipeline, but dense MVS dominates at roughly 47 minutes for this short clip. The 7.3 GiB RAM did not stop the run; its true peak use was not measured. The integrated GPU was not available to this process and the chosen pipeline does not depend on CUDA.

**Best current pipeline:** 4 FPS quality/overlap-filtered keyframes → COLMAP CPU exhaustive matching/SfM → OpenMVS CPU PatchMatch/fusion → OpenMVS mesh and texture → Blender-verified GLB. It is the strongest tested local path so far, though the visual result sets a clear quality ceiling for this capture and configuration.

**Next highest-ROI improvement:** run one controlled capture with a native high-resolution phone file (not a messaging-app compressed copy), slow translation, and steadier overlap, then compare keyframe/matching and texture quality at the same settings. The current clip and mesh make it difficult to separate capture limitations from pipeline limitations. Keep subsequent work in Phase 1; do not start still-image or live reconstruction.
