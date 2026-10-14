// ===--------------------------------------------------------------------=== //
// host-buffer-semantics: PJRT_Client_BufferFromHostBuffer timing comparison
// ===--------------------------------------------------------------------=== //
//
// Question this program answers:
//
//   With PJRT_HostBufferSemantics_kImmutableOnlyDuringCall, does the runtime
//   (a) make a private host copy of `data` and return immediately, letting
//       the DMA to HBM continue in the background, or
//   (b) block the calling CPU thread until the whole transfer has landed on
//       the device?
//
//   And how does that compare with kImmutableUntilTransferCompletes followed
//   by PJRT_Event_Await(done_with_host_buffer)?
//
// How it measures:
//
//   For every (size, semantics) pair the upload is split into three timed
//   phases on the calling thread:
//
//     t_call   : PJRT_Client_BufferFromHostBuffer()   -> returns
//                This is the only time the CPU is *forced* to spend.
//     t_done   : PJRT_Event_Await(done_with_host_buffer)
//                "it is now safe to reuse `data`". For kImmutableOnlyDuringCall
//                this must already be ready at return (we check IsReady).
//     t_ready  : PJRT_Event_Await(PJRT_Buffer_ReadyEvent(buffer))
//                "the bytes are in HBM". This is the real end of the transfer.
//
//   t_call + t_done + t_ready = wall time until the device buffer is usable.
//
//   PJRT_Event_IsReady is sampled right after the call returns, so the output
//   also shows whether each event was *already* complete at that moment.
//
//   A plain memcpy of the same size is timed as a reference: if t_call for
//   kImmutableOnlyDuringCall is about one memcpy, the runtime is staging a
//   host copy; if it is about t_ready of the other semantics, it is blocking.
//
// Usage:
//   host-buffer-semantics <libtpu_so> [--sizes-mb 4,32,128,512,1024]
//                                     [--iters 5] [--chip 0]
// ===--------------------------------------------------------------------=== //

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "xla/pjrt/c/pjrt_c_api.h"

namespace {

// ===--------------------------------------------------------------------=== //
// Error / event helpers
// ===--------------------------------------------------------------------=== //

bool CheckError(const PJRT_Api* api, PJRT_Error* error, const char* context) {
    if (error == nullptr) return true;
    PJRT_Error_Message_Args msg{};
    msg.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE;
    msg.error = error;
    api->PJRT_Error_Message(&msg);
    std::cerr << context << ": " << std::string(msg.message, msg.message_size)
              << "\n";
    PJRT_Error_Destroy_Args d{};
    d.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
    d.error = error;
    api->PJRT_Error_Destroy(&d);
    return false;
}

bool EventIsReady(const PJRT_Api* api, PJRT_Event* ev) {
    PJRT_Event_IsReady_Args a{};
    a.struct_size = PJRT_Event_IsReady_Args_STRUCT_SIZE;
    a.event = ev;
    if (!CheckError(api, api->PJRT_Event_IsReady(&a), "PJRT_Event_IsReady"))
        return false;
    return a.is_ready;
}

bool EventAwait(const PJRT_Api* api, PJRT_Event* ev, const char* ctx) {
    PJRT_Event_Await_Args a{};
    a.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
    a.event = ev;
    return CheckError(api, api->PJRT_Event_Await(&a), ctx);
}

void EventDestroy(const PJRT_Api* api, PJRT_Event* ev) {
    if (ev == nullptr) return;
    PJRT_Event_Destroy_Args a{};
    a.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
    a.event = ev;
    api->PJRT_Event_Destroy(&a);
}

PJRT_Event* BufferReadyEvent(const PJRT_Api* api, PJRT_Buffer* buf) {
    PJRT_Buffer_ReadyEvent_Args a{};
    a.struct_size = PJRT_Buffer_ReadyEvent_Args_STRUCT_SIZE;
    a.buffer = buf;
    if (!CheckError(api, api->PJRT_Buffer_ReadyEvent(&a),
                    "PJRT_Buffer_ReadyEvent"))
        return nullptr;
    return a.event;
}

void BufferDestroy(const PJRT_Api* api, PJRT_Buffer* buf) {
    if (buf == nullptr) return;
    PJRT_Buffer_Destroy_Args a{};
    a.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
    a.buffer = buf;
    CheckError(api, api->PJRT_Buffer_Destroy(&a), "PJRT_Buffer_Destroy");
}

// ===--------------------------------------------------------------------=== //
// Plugin loading
// ===--------------------------------------------------------------------=== //

using GetPjrtApiFn = const PJRT_Api* (*)();

const PJRT_Api* LoadTpuPjrtApi(const char* plugin_path) {
    void* handle = dlopen(plugin_path, RTLD_LAZY | RTLD_LOCAL);
    if (handle == nullptr) {
        std::cerr << "dlopen(\"" << plugin_path << "\") failed: " << dlerror()
                  << "\n";
        return nullptr;
    }
    dlerror();
    auto get_api = reinterpret_cast<GetPjrtApiFn>(dlsym(handle, "GetPjrtApi"));
    const char* err = dlerror();
    if (err != nullptr || get_api == nullptr) {
        std::cerr << "dlsym(\"GetPjrtApi\") failed: "
                  << (err ? err : "symbol returned null") << "\n";
        return nullptr;
    }
    return get_api();  // handle intentionally leaked
}

// ===--------------------------------------------------------------------=== //
// Timing
// ===--------------------------------------------------------------------=== //

using Clock = std::chrono::steady_clock;

double MsSince(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

const char* SemanticsName(PJRT_HostBufferSemantics s) {
    switch (s) {
        case PJRT_HostBufferSemantics_kImmutableOnlyDuringCall:
            return "kImmutableOnlyDuringCall";
        case PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes:
            return "kImmutableUntilTransferCompletes";
        case PJRT_HostBufferSemantics_kImmutableZeroCopy:
            return "kImmutableZeroCopy";
        case PJRT_HostBufferSemantics_kMutableZeroCopy:
            return "kMutableZeroCopy";
    }
    return "?";
}

struct Sample {
    double call_ms = 0;        // BufferFromHostBuffer() wall time
    double done_await_ms = 0;  // Await(done_with_host_buffer)
    double ready_await_ms = 0; // Await(Buffer_ReadyEvent)
    bool done_ready_at_return = false;
    bool buffer_ready_at_return = false;
    double total_ms() const { return call_ms + done_await_ms + ready_await_ms; }
};

struct Stats {
    double call_med = 0, done_med = 0, ready_med = 0, total_med = 0;
    double call_min = 0, total_min = 0;
    int done_ready_count = 0, buffer_ready_count = 0, n = 0;
};

double Median(std::vector<double> v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    size_t m = v.size() / 2;
    return (v.size() % 2) ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

Stats Summarize(const std::vector<Sample>& s) {
    Stats st;
    std::vector<double> c, d, r, t;
    for (const auto& x : s) {
        c.push_back(x.call_ms);
        d.push_back(x.done_await_ms);
        r.push_back(x.ready_await_ms);
        t.push_back(x.total_ms());
        st.done_ready_count += x.done_ready_at_return;
        st.buffer_ready_count += x.buffer_ready_at_return;
    }
    st.n = static_cast<int>(s.size());
    st.call_med = Median(c);
    st.done_med = Median(d);
    st.ready_med = Median(r);
    st.total_med = Median(t);
    st.call_min = *std::min_element(c.begin(), c.end());
    st.total_min = *std::min_element(t.begin(), t.end());
    return st;
}

// ===--------------------------------------------------------------------=== //
// One timed upload
// ===--------------------------------------------------------------------=== //

// Uploads `data` (rows x cols f32) with the given semantics and returns the
// three timed phases. The device buffer is destroyed before returning.
bool TimedUpload(const PJRT_Api* api, PJRT_Client* client, PJRT_Device* device,
                 const float* data, int64_t rows, int64_t cols,
                 PJRT_HostBufferSemantics semantics, Sample* out) {
    const int64_t dims[2] = {rows, cols};

    PJRT_Client_BufferFromHostBuffer_Args args{};
    args.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
    args.client = client;
    args.data = data;
    args.type = PJRT_Buffer_Type_F32;
    args.dims = dims;
    args.num_dims = 2;
    args.byte_strides = nullptr;
    args.num_byte_strides = 0;
    args.host_buffer_semantics = semantics;
    args.device = device;
    args.memory = nullptr;
    args.device_layout = nullptr;

    // --- phase 1: the call itself ---
    auto t0 = Clock::now();
    if (!CheckError(api, api->PJRT_Client_BufferFromHostBuffer(&args),
                    "PJRT_Client_BufferFromHostBuffer"))
        return false;
    out->call_ms = MsSince(t0);

    PJRT_Event* ready_ev = BufferReadyEvent(api, args.buffer);
    if (ready_ev == nullptr) return false;

    // Snapshot of what was already complete when the call returned.
    out->done_ready_at_return = EventIsReady(api, args.done_with_host_buffer);
    out->buffer_ready_at_return = EventIsReady(api, ready_ev);

    // --- phase 2: safe to reuse host memory ---
    t0 = Clock::now();
    if (!EventAwait(api, args.done_with_host_buffer, "done_with_host_buffer"))
        return false;
    out->done_await_ms = MsSince(t0);
    EventDestroy(api, args.done_with_host_buffer);

    // --- phase 3: bytes are on the device ---
    t0 = Clock::now();
    if (!EventAwait(api, ready_ev, "buffer ready event")) return false;
    out->ready_await_ms = MsSince(t0);
    EventDestroy(api, ready_ev);

    BufferDestroy(api, args.buffer);
    return true;
}

// ===--------------------------------------------------------------------=== //
// Host-side helpers
// ===--------------------------------------------------------------------=== //

void FillPattern(float* p, size_t n, float base) {
    for (size_t i = 0; i < n; ++i) p[i] = base + static_cast<float>(i % 1024);
}

double TimedMemcpyMs(float* dst, const float* src, size_t bytes) {
    auto t0 = Clock::now();
    std::memcpy(dst, src, bytes);
    return MsSince(t0);
}

std::vector<size_t> ParseSizesMb(const std::string& s) {
    std::vector<size_t> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (!tok.empty()) out.push_back(static_cast<size_t>(std::atol(tok.c_str())));
    }
    return out;
}

void PrintUsage(const char* argv0) {
    std::cerr << "Usage: " << argv0
              << " <libtpu_so> [--sizes-mb 4,32,128,512,1024] [--iters 5]"
                 " [--chip 0]\n";
}

}  // namespace

// ===--------------------------------------------------------------------=== //
// main
// ===--------------------------------------------------------------------=== //

int main(int argc, char** argv) {
    if (argc < 2) {
        PrintUsage(argv[0]);
        return 1;
    }
    const char* libtpu_path = argv[1];
    std::vector<size_t> sizes_mb = {4, 32, 128, 512, 1024};
    int iters = 5;
    std::string chip = "0";
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char* flag) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << flag << " needs a value\n";
                std::exit(1);
            }
            return argv[++i];
        };
        if (a == "--sizes-mb") sizes_mb = ParseSizesMb(need("--sizes-mb"));
        else if (a == "--iters") iters = std::atoi(need("--iters"));
        else if (a == "--chip") chip = need("--chip");
        else {
            PrintUsage(argv[0]);
            return 1;
        }
    }
    if (iters < 1) iters = 1;

    // Pin to one chip so the benchmark can share the host with other users.
    setenv("TPU_VISIBLE_CHIPS", chip.c_str(), 1);
    setenv("TPU_PROCESS_BOUNDS", "1,1,1", 0);
    setenv("TPU_CHIPS_PER_PROCESS_BOUNDS", "1,1,1", 0);
    setenv("ALLOW_MULTIPLE_LIBTPU_LOAD", "1", 0);

    const PJRT_Api* api = LoadTpuPjrtApi(libtpu_path);
    if (api == nullptr) return 1;
    {
        PJRT_Plugin_Initialize_Args a{};
        a.struct_size = PJRT_Plugin_Initialize_Args_STRUCT_SIZE;
        if (!CheckError(api, api->PJRT_Plugin_Initialize(&a),
                        "PJRT_Plugin_Initialize"))
            return 1;
    }
    PJRT_Client* client = nullptr;
    {
        PJRT_Client_Create_Args a{};
        a.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
        if (!CheckError(api, api->PJRT_Client_Create(&a), "PJRT_Client_Create"))
            return 1;
        client = a.client;
    }
    PJRT_Device* device = nullptr;
    {
        PJRT_Client_AddressableDevices_Args a{};
        a.struct_size = PJRT_Client_AddressableDevices_Args_STRUCT_SIZE;
        a.client = client;
        if (!CheckError(api, api->PJRT_Client_AddressableDevices(&a),
                        "PJRT_Client_AddressableDevices"))
            return 1;
        if (a.num_addressable_devices < 1) {
            std::cerr << "No addressable devices\n";
            return 1;
        }
        device = a.addressable_devices[0];
    }
    std::cout << "PJRT API " << api->pjrt_api_version.major_version << "."
              << api->pjrt_api_version.minor_version << ", chip " << chip
              << ", iters/size " << iters << " (+1 warmup)\n\n";

    const PJRT_HostBufferSemantics kSemantics[2] = {
        PJRT_HostBufferSemantics_kImmutableOnlyDuringCall,
        PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes,
    };

    // ---------------------------------------------------------------------
    // Timing sweep
    // ---------------------------------------------------------------------
    std::printf("%-8s %-34s %10s %10s %10s %10s %8s %8s %10s\n", "size",
                "semantics", "call ms", "done ms", "ready ms", "total ms",
                "done@ret", "buf@ret", "GB/s(tot)");
    std::printf("%-8s %-34s %10s %10s %10s %10s %8s %8s %10s\n", "", "",
                "(median)", "(median)", "(median)", "(median)", "", "", "");

    for (size_t mb : sizes_mb) {
        const size_t bytes = mb * 1024 * 1024;
        const int64_t cols = 1024;
        const int64_t rows = static_cast<int64_t>(bytes / sizeof(float) / cols);
        const size_t n = static_cast<size_t>(rows * cols);

        std::vector<float> host(n), scratch(n);
        FillPattern(host.data(), n, 1.0f);  // also faults the pages in
        FillPattern(scratch.data(), n, 0.0f);

        // Reference: what a plain host memcpy of this size costs.
        std::vector<double> mc;
        for (int i = 0; i < 3; ++i)
            mc.push_back(TimedMemcpyMs(scratch.data(), host.data(), bytes));
        const double memcpy_ms = Median(mc);
        std::printf("%-8s %-34s %10.3f %10s %10s %10s %8s %8s %10.2f\n",
                    (std::to_string(mb) + " MB").c_str(), "memcpy (reference)",
                    memcpy_ms, "-", "-", "-", "-", "-",
                    bytes / memcpy_ms / 1e6);

        for (PJRT_HostBufferSemantics sem : kSemantics) {
            std::vector<Sample> samples;
            for (int it = 0; it < iters + 1; ++it) {
                Sample s;
                if (!TimedUpload(api, client, device, host.data(), rows, cols,
                                 sem, &s))
                    return 1;
                if (it > 0) samples.push_back(s);  // drop warmup
            }
            Stats st = Summarize(samples);
            std::printf("%-8s %-34s %10.3f %10.3f %10.3f %10.3f %5d/%-2d %5d/%-2d %10.2f\n",
                        "", SemanticsName(sem), st.call_med, st.done_med,
                        st.ready_med, st.total_med, st.done_ready_count, st.n,
                        st.buffer_ready_count, st.n,
                        bytes / st.total_med / 1e6);
        }
        std::printf("\n");
    }

    std::cout
        << "Columns: call = time inside PJRT_Client_BufferFromHostBuffer "
           "(CPU blocked);\n"
           "         done = PJRT_Event_Await(done_with_host_buffer) after "
           "the call;\n"
           "         ready = PJRT_Event_Await(PJRT_Buffer_ReadyEvent) after "
           "that;\n"
           "         done@ret / buf@ret = how many iterations had that event "
           "already\n"
           "         complete at the moment the call returned.\n\n";

    PJRT_Client_Destroy_Args d{};
    d.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
    d.client = client;
    CheckError(api, api->PJRT_Client_Destroy(&d), "PJRT_Client_Destroy");
    return 0;
}
