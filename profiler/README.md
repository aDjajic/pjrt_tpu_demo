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

The trace file is written to `trace.xplane.pb` in the working directory.

## Viewing in TensorBoard

TensorBoard's profile plugin expects a specific directory layout and the
`.xplane.pb` extension:

```bash
mkdir -p logs/run1/plugins/profile/session1
cp trace.xplane.pb logs/run1/plugins/profile/session1/trace.xplane.pb

tensorboard --logdir logs/ --port 6006
```

## Known limitation: no device timing

Through `PJRT_Profiler_Extension`'s `PLUGIN_Profiler_Create/Start/Stop/CollectData`
path, libtpu captures host TraceMe events and the HLO graph (so TensorBoard
will show 50/50 host/device op placement) but **does not capture device-side
timing** (MXU / VPU / HBM cycles). All TPU utilization metrics will read 0%.
