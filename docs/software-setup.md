# Software setup

The default `balanced` pipeline uses COLMAP 4.2.1 for CPU sparse reconstruction and OpenMVS 2.4.0 for CPU dense reconstruction, meshing, and texturing. Both now start from this project's `native/install/bin` through `./reconstruct`; no shell profile or global `PATH` change is needed. This is a startup check, not a reconstruction quality result. See [the Phase 1 log](phase-1-results.md).

## Python environment

The existing `.venv` supplies Python 3.13, NumPy, and headless OpenCV. The launcher uses `.venv/bin/python` when present. OpenCV has FFmpeg decoding built in; `ffmpeg` and `ffprobe` command-line programs are not needed for the default pipeline. Do not rebuild `.venv` as part of the native setup.

## Verified native setup

These steps reproduce the working local CPU setup on Linux x86_64. All downloads, packages, sources, and build files stay under `native/`. They use the official [COLMAP 4.2.1 source tag](https://github.com/colmap/colmap/releases/tag/4.2.1), [OpenMVS 2.4.0 release](https://github.com/cdcseacave/openMVS/releases/tag/v2.4.0), and [micromamba](https://mamba.readthedocs.io/en/stable/user_guide/micromamba.html) for build dependencies. COLMAP has no official Linux release binary; OpenMVS has an Ubuntu x64 release archive that starts on this machine. The [COLMAP installation guide](https://colmap.github.io/install.html) documents building with local dependencies when root access is unavailable.

From the project root:

```bash
mkdir -p native/downloads native/install/bin native/cache native/tools native/micromamba
curl -fL -o native/downloads/OpenMVS_Ubuntu_x64.zip \
  https://github.com/cdcseacave/openMVS/releases/download/v2.4.0/OpenMVS_Ubuntu_x64.zip
for name in InterfaceCOLMAP DensifyPointCloud ReconstructMesh RefineMesh TextureMesh TransformScene; do
  unzip -p native/downloads/OpenMVS_Ubuntu_x64.zip "$name" > "native/install/bin/$name"
  chmod +x "native/install/bin/$name"
done

PIP_CACHE_DIR="$PWD/native/cache/pip" python3 -m pip install --target "$PWD/native/tools" cmake ninja
curl -fL -o native/downloads/micromamba-linux-64.tar.bz2 \
  https://micro.mamba.pm/api/micromamba/linux-64/latest
tar -xjf native/downloads/micromamba-linux-64.tar.bz2 -C native/micromamba bin/micromamba
env XDG_CACHE_HOME="$PWD/native/cache/xdg" \
  MAMBA_ROOT_PREFIX="$PWD/native/mamba" \
  CONDA_PKGS_DIRS="$PWD/native/cache/conda-pkgs" \
  native/micromamba/bin/micromamba install -y --no-rc -n base \
  -r "$PWD/native/mamba" -c conda-forge --override-channels \
  boost-cpp ceres-solver eigen gflags glog openimageio metis sqlite suitesparse pkg-config
```

OpenImageIO's conda package pulls FFmpeg libraries and also ships two command-line binaries. The binaries are not used by this pipeline; remove them from the local dependency prefix:

```bash
rm -f native/mamba/bin/ffmpeg native/mamba/bin/ffprobe
```

Build the official COLMAP source. The one local source adjustment selects one generic CPU FAISS implementation instead of compiling dynamic dispatch variants. CUDA, HIP, GUI, OpenGL, ONNX, COLMAP's CUDA dense MVS, CGAL, tests, and download support are disabled. The build uses two jobs for this machine's 7.3 GiB RAM.

```bash
git clone --depth 1 --branch 4.2.1 https://github.com/colmap/colmap.git native/colmap-src
python3 - <<'PY'
from pathlib import Path
p = Path('native/colmap-src/src/thirdparty/CMakeLists.txt')
s = p.read_text()
assert 'set(FAISS_OPT_LEVEL dd)' in s
p.write_text(s.replace('set(FAISS_OPT_LEVEL dd)', 'set(FAISS_OPT_LEVEL generic)', 1))
PY

env PYTHONPATH="$PWD/native/tools" \
  PATH="$PWD/native/tools/bin:$PWD/native/mamba/bin:$PATH" \
  PKG_CONFIG_PATH="$PWD/native/mamba/lib/pkgconfig" \
  LD_LIBRARY_PATH="$PWD/native/mamba/lib" \
  cmake -S native/colmap-src -B native/colmap-local-build -G Ninja \
  -DCMAKE_MAKE_PROGRAM="$PWD/native/tools/bin/ninja" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$PWD/native/mamba" \
  -DCMAKE_INSTALL_PREFIX="$PWD/native/install" \
  -DCMAKE_INSTALL_RPATH="$PWD/native/mamba/lib" \
  -DCUDA_ENABLED=OFF -DHIP_ENABLED=OFF -DGUI_ENABLED=OFF \
  -DOPENGL_ENABLED=OFF -DMVS_ENABLED=OFF -DTESTS_ENABLED=OFF \
  -DONNX_ENABLED=OFF -DCGAL_ENABLED=OFF -DDOWNLOAD_ENABLED=OFF \
  -DIPO_ENABLED=OFF -DFETCH_ONNX=OFF
env PYTHONPATH="$PWD/native/tools" \
  PATH="$PWD/native/tools/bin:$PWD/native/mamba/bin:$PATH" \
  LD_LIBRARY_PATH="$PWD/native/mamba/lib" \
  cmake --build native/colmap-local-build --target colmap_main -j2
install -m 755 native/colmap-local-build/src/colmap/exe/colmap native/install/bin/colmap
./reconstruct --doctor
```

The COLMAP executable has an embedded runtime search path to `native/mamba/lib`. The launcher supplies `COLMAP_BIN` and `OPENMVS_BIN` only when those variables are unset, so explicit per-command overrides still work. The doctor should report an empty `missing_balanced_pipeline_tools` list, no `native_startup_error`, and `started: true` for COLMAP, InterfaceCOLMAP, DensifyPointCloud, ReconstructMesh, TextureMesh, and TransformScene. OpenMVS help exits with code 1 by design; the doctor accepts its help output.

## Process a room later

Place a real walkthrough in `inputs/`, then run `./reconstruct inputs/room.mp4 --quality balanced`. Use an output filesystem with ample free space for dense depth maps. No room video was processed during native setup, so real reconstruction quality and runtime remain unmeasured.

