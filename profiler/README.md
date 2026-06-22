# pjrt-demo + profiler

End-to-end TPU example that loads a Mosaic MLIR kernel through the PJRT C API
and captures a profile via the `PJRT_Profiler_Extension`. The trace is written
as a serialized `tensorflow.profiler.XSpace` proto, viewable in TensorBoard.

## Requirements

```bash
pip install tensorboard tensorboard-plugin-profile
```

## Build

From the repo root:

```bash
cmake -B build -DXLA_SRC_PATH=/path/to/xla
cmake --build build
```

The output binary is `build/profiler`.

## Run

```bash
./build/profiler \
    profiler/vector_add_new.mlir \
    /path/to/site-packages/libtpu/libtpu.so
```

The binary writes the trace directly into the layout TensorBoard expects,
creating the directory tree automatically:

```
profile_logs/plugins/profile/<YYYY_MM_DD_HH_MM_SS>/trace.xplane.pb
```

Each run gets its own timestamped folder, so TensorBoard lists every
profiling run separately. No manual copying or folder creation is needed.

## Viewing in TensorBoard

Point TensorBoard at the `profile_logs/` directory created by the run:

```bash
tensorboard --logdir profile_logs/ --port 6006
```
