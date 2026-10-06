# two_procs_two_chips: two independent processes, two TPU chips

Everything else in this repo is *one process* driving one or more chips
through a single PJRT client. This example is the other model: **two
unrelated OS processes, each owning exactly one chip**, running at the
same time on the same host. There is no communication between them, no
shared PJRT client, no SPMD partitioning. Each process is a plain
single-device PJRT program that happens to be told which chip is its.

```
         host (v6e-4: /dev/vfio/0 .. /dev/vfio/3)
         ┌───────────────────────┐   ┌───────────────────────┐
         │ process A (pid 1234)  │   │ process B (pid 1235)  │
         │ TPU_VISIBLE_CHIPS=0   │   │ TPU_VISIBLE_CHIPS=1   │
         │ libtpu.so  (own copy) │   │ libtpu.so  (own copy) │
         │ PJRT client: 1 device │   │ PJRT client: 1 device │
         └──────────┬────────────┘   └──────────┬────────────┘
                    │ /dev/vfio/0                │ /dev/vfio/1
                 ┌──┴──┐                      ┌──┴──┐
                 │chip0│                      │chip1│
                 └─────┘                      └─────┘
```

## Files

| file | what |
| --- | --- |
| `main.cpp` | the per-process worker (`two-procs-two-chips` target) |
| `matmul_mosaic.mlir` | gridded Pallas matmul kernel, 1024x2048 @ 2048x2048 f32, grid (8, 8, 8) |
| `run_two_procs.sh` | starts two workers at once, lines up their execute windows, prints overlap |

## Build and run

```
cmake -B build -DXLA_SRC_PATH=/path/to/xla
cmake --build build --target two-procs-two-chips

./two_procs_two_chips/run_two_procs.sh /path/to/libtpu.so            # chips 0 and 1
./two_procs_two_chips/run_two_procs.sh /path/to/libtpu.so --chips 2,3
./two_procs_two_chips/run_two_procs.sh /path/to/libtpu.so --compare  # serial vs parallel
./two_procs_two_chips/run_two_procs.sh /path/to/libtpu.so --chips 1,1  # deliberate collision
```

A single worker can also be run by hand:

```
./build/two-procs-two-chips two_procs_two_chips/matmul_mosaic.mlir /path/to/libtpu.so 2 4000
#                                                                                   ^chip ^iters
```

## How one process gets one chip

By default libtpu assumes it owns the whole host: it opens every
`/dev/vfio/N`, builds the full 2x2 topology, and takes an exclusive
`lockf` on `/tmp/libtpu_lockfile` so that a second process fails early
with `libtpu.so already in use by another process`.

The worker overrides that **before `dlopen("libtpu.so")`**, in step `[0]`
of `main.cpp`, with four environment variables:

| variable | value | effect |
| --- | --- | --- |
| `TPU_VISIBLE_CHIPS` | `<chip>` | only this `/dev/vfio/<chip>` is opened; the other chips are left alone for other processes |
| `TPU_PROCESS_BOUNDS` | `1,1,1` | this process is a complete slice on its own (no rendezvous with other processes) |
| `TPU_CHIPS_PER_PROCESS_BOUNDS` | `1,1,1` | the slice is one chip, not the host's 2x2 |
| `ALLOW_MULTIPLE_LIBTPU_LOAD` | `1` | skip the host-wide lockfile check entirely |


## What the launcher does

`run_two_procs.sh` starts two workers in the background, waits until both
have written their `ready.<chip>` barrier file, touches `go`, waits for
both to exit, then prints their logs with a `[chip N]` prefix and a
summary: exit codes, the two execute windows relative to the earlier
start, and the overlap between them. With `--compare` it first runs the
two workers one after the other and reports both wall times.

Every run writes to its own folder, `two_procs_two_chips/logs/<date>-<time>-<pid>/`
(git-ignored): the worker logs, the barrier files and the libtpu driver
logs under `tpu_logs/`.

Sample run on a v6e-4 (`--compare`, output trimmed):

```
################ SERIAL: chip 0, then chip 1
[A: chip 0] [3] Created PJRT client in 1.62252 s
[A: chip 0] [9] Done: 4000 matmuls in 1.34322 s  (0.335805 ms/call, 25.5802 TFLOP/s f32)
[B: chip 1] [3] Created PJRT client in 1.68413 s
[B: chip 1] [9] Done: 4000 matmuls in 1.3429 s  (0.335726 ms/call, 25.5862 TFLOP/s f32)
---- serial summary
exit codes: A (chip 0) -> 0, B (chip 1) -> 0
total wall time (one after the other): 9070 ms

################ PARALLEL: chip 0 and chip 1 at the same time
[A: chip 0] === two-procs-two-chips (pid 455806, chip 0) ===
[A: chip 0] [3] Created PJRT client in 2.13361 s
[A: chip 0] [4] Topology seen by this process: 1 device(s) total, 1 addressable
[A: chip 0]     device[id=0] kind="TPU v6 lite" coords=[0,0,0] core_on_chip=0
[A: chip 0] [8b] Ready; waiting for two_procs_two_chips/logs/20261006-103000-455800/barrier/go ...
[A: chip 0] [9] Done: 4000 matmuls in 1.34286 s  (0.335716 ms/call, 25.5869 TFLOP/s f32)
[A: chip 0] EXEC_WINDOW chip=0 pid=455806 start_ms=1791282392017 end_ms=1791282393359 iters=4000
[A: chip 0] Max abs error over all 1024x2048 elements: 0  (OK)

[B: chip 1] === two-procs-two-chips (pid 455808, chip 1) ===
[B: chip 1] [3] Created PJRT client in 2.32343 s
[B: chip 1] [4] Topology seen by this process: 1 device(s) total, 1 addressable
[B: chip 1]     device[id=0] kind="TPU v6 lite" coords=[0,0,0] core_on_chip=0
[B: chip 1] [8b] Ready; waiting for two_procs_two_chips/logs/20261006-103000-455800/barrier/go ...
[B: chip 1] [9] Done: 4000 matmuls in 1.34252 s  (0.33563 ms/call, 25.5935 TFLOP/s f32)
[B: chip 1] EXEC_WINDOW chip=1 pid=455808 start_ms=1791282392017 end_ms=1791282393359 iters=4000
[B: chip 1] Max abs error over all 1024x2048 elements: 0  (OK)

---- parallel summary
exit codes: A (chip 0) -> 0, B (chip 1) -> 0
total wall time (both processes, incl. libtpu init + compile): 5942 ms

execute windows (ms, relative to the earlier start):
  A: chip 0    [     0 ..   1342]  duration   1342 ms
  B: chip 1    [     0 ..   1342]  duration   1342 ms
  overlap        1342 ms
  -> the two chips were executing at the same time

which physical device did each pid open (from libtpu driver logs):
  [/dev/vfio/0 tpu2:pe2:1] vf_id:0 Driver opened
  [/dev/vfio/1 tpu2:pe2:0] vf_id:0 Driver opened

################ COMPARE
serial   : 9070 ms
parallel : 5942 ms
speedup  : 1.53x
```
