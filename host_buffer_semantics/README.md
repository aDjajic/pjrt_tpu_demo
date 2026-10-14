# host-buffer-semantics

Timing comparison of the two non-zero-copy host buffer semantics of
`PJRT_Client_BufferFromHostBuffer` on TPU:

* `PJRT_HostBufferSemantics_kImmutableOnlyDuringCall`
* `PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes` followed by
  `PJRT_Event_Await(done_with_host_buffer)`

## Build and run

```sh
cmake -B build -DXLA_SRC_PATH=/path/to/xla
cmake --build build --target host-buffer-semantics
./build/host-buffer-semantics /path/to/libtpu.so \
    [--sizes-mb 4,32,128,512,1024] [--iters 5] [--chip 0]
```

## Results (v6e-4, libtpu 0.0.47, PJRT API 0.114, 2026-10-14)

```
size     semantics                             call ms    done ms   ready ms   total ms done@ret  buf@ret  GB/s(tot)
4 MB     memcpy (reference)                      0.153          -          -          -        -        -      27.45
         kImmutableOnlyDuringCall                0.136      0.000      0.193      0.332     5/5      0/5       12.64
         kImmutableUntilTransferCompletes        0.008      0.111      0.195      0.314     0/5      0/5       13.38

32 MB    memcpy (reference)                      1.572          -          -          -        -        -      21.34
         kImmutableOnlyDuringCall                0.835      0.000      1.229      2.058     5/5      0/5       16.30
         kImmutableUntilTransferCompletes        0.010      0.392      1.235      1.638     0/5      0/5       20.48

128 MB   memcpy (reference)                      7.586          -          -          -        -        -      17.69
         kImmutableOnlyDuringCall                2.259      0.000      4.782      7.041     5/5      0/5       19.06
         kImmutableUntilTransferCompletes        0.013      1.649      4.787      6.449     0/5      0/5       20.81

512 MB   memcpy (reference)                     30.075          -          -          -        -        -      17.85
         kImmutableOnlyDuringCall                9.360      0.000     18.980     28.343     5/5      0/5       18.94
         kImmutableUntilTransferCompletes        0.021      9.129     18.999     28.141     0/5      0/5       19.08

1024 MB  memcpy (reference)                     58.838          -          -          -        -        -      18.25
         kImmutableOnlyDuringCall               18.661      0.000     37.910     56.572     5/5      0/5       18.98
         kImmutableUntilTransferCompletes        0.025     18.605     37.917     56.569     0/5      0/5       18.98

```
