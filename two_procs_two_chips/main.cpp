// ===--------------------------------------------------------------------=== //
// two-procs-two-chips: one OS process == one TPU chip
// ===--------------------------------------------------------------------=== //
//
// This is the per-process *worker*. The launcher (run_two_procs.sh) starts
// two copies of it at the same time, one pinned to chip A and one to chip B.
// Each copy is a completely ordinary single-device PJRT program; the only
// thing that makes it "two chips, two processes" is a handful of environment
// variables that are set BEFORE libtpu.so is dlopen-ed:
//
//   TPU_VISIBLE_CHIPS=<n>              only /dev/vfio/<n> is opened by this
//                                      process. Other chips stay untouched,
//                                      so another process may own them.
//   TPU_PROCESS_BOUNDS=1,1,1           this process is a whole "slice" by
//   TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 itself: a 1x1x1 topology with one
//                                      chip. No rendezvous with other
//                                      processes, no ICI collectives.
//   ALLOW_MULTIPLE_LIBTPU_LOAD=1       skip the host-wide /tmp/libtpu_lockfile
//                                      check that would otherwise make the
//                                      second process bail out with
//                                      "libtpu.so already in use". (libtpu
//                                      already relaxes the lock when
//                                      TPU_CHIPS_PER_PROCESS_BOUNDS is set to
//                                      less than the full host, so this is
//                                      belt-and-braces; it is set explicitly
//                                      because it documents the intent.)
//
// Flow inside one process:
//
//   Mosaic MLIR text file (gridded Pallas matmul, 1024x2048 @ 2048x2048)
//       |
//       v  WrapMosaicInStableHlo (plain single-device wrapper)
//   StableHLO with @tpu_custom_call
//       |
//       v  PJRT_Client_Compile
//   PJRT_LoadedExecutable on the one visible chip
//       |
//       v  PJRT_Client_BufferFromHostBuffer (lhs, rhs)
//       v  PJRT_LoadedExecutable_Execute x <iterations>, enqueued back-to-back
//       v  PJRT_Buffer_ToHostBuffer on the last output (drains the queue)
//   host float[1024x2048], verified against a closed-form expected value
//
// The worker prints absolute wall-clock timestamps for its execute window so
// the launcher can show that the two windows overlap in time.
// ===--------------------------------------------------------------------=== //

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

#include "xla/pjrt/c/pjrt_c_api.h"

namespace {

// Problem size per process. Must agree with matmul_mosaic.mlir:
//   window_bounds   = 128x256 (lhs), 256x256 (rhs), 128x256 (out)
//   iteration_bounds = (M/128, N/256, K/256) = (8, 8, 8)
constexpr size_t M = 1024;
constexpr size_t K = 2048;
constexpr size_t N = 2048;
constexpr double FLOPS_PER_CALL = 2.0 * M * K * N;  // 8.6 GFLOP

// ===--------------------------------------------------------------------=== //
// Error helper
// ===--------------------------------------------------------------------=== //

bool CheckError(const PJRT_Api* api, PJRT_Error* error,
                const char* context) {
    if (error == nullptr) return true;

    PJRT_Error_Message_Args msg_args{};
    msg_args.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE;
    msg_args.error = error;
    api->PJRT_Error_Message(&msg_args);

    std::cerr << context << ": "
              << std::string(msg_args.message, msg_args.message_size)
              << "\n";

    PJRT_Error_Destroy_Args destroy_args{};
    destroy_args.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
    destroy_args.error = error;
    api->PJRT_Error_Destroy(&destroy_args);
    return false;
}

// ===--------------------------------------------------------------------=== //
// File IO
// ===--------------------------------------------------------------------=== //

std::string LoadTextFile(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        std::cerr << "Failed to open " << path << "\n";
        return {};
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

// ===--------------------------------------------------------------------=== //
// StableHLO wrapping (single device, no sharding annotations)
// ===--------------------------------------------------------------------=== //

std::string Base64Encode(const std::string& data) {
    static const char* tbl =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(((data.size() + 2) / 3) * 4);
    size_t i = 0;
    while (i + 3 <= data.size()) {
        uint32_t v = (static_cast<uint8_t>(data[i]) << 16) |
                     (static_cast<uint8_t>(data[i + 1]) << 8) |
                     static_cast<uint8_t>(data[i + 2]);
        out += tbl[(v >> 18) & 63];
        out += tbl[(v >> 12) & 63];
        out += tbl[(v >> 6) & 63];
        out += tbl[v & 63];
        i += 3;
    }
    if (i < data.size()) {
        uint32_t v = static_cast<uint8_t>(data[i]) << 16;
        if (i + 1 < data.size())
            v |= static_cast<uint8_t>(data[i + 1]) << 8;
        out += tbl[(v >> 18) & 63];
        out += tbl[(v >> 12) & 63];
        out += (i + 1 < data.size()) ? tbl[(v >> 6) & 63] : '=';
        out += '=';
    }
    return out;
}

std::string WrapMosaicInStableHlo(const std::string& mosaic_text) {
    const std::string body_b64 = Base64Encode(mosaic_text);
    const std::string json =
        std::string("{\\\"custom_call_config\\\": {\\\"body\\\": \\\"")
        + body_b64 + "\\\","
        + " \\\"needs_layout_passes\\\": true, \\\"shape_invariant_numerics\\\": false}}";

    const std::string lhs_t = "tensor<" + std::to_string(M) + "x" +
                              std::to_string(K) + "xf32>";
    const std::string rhs_t = "tensor<" + std::to_string(K) + "x" +
                              std::to_string(N) + "xf32>";
    const std::string out_t = "tensor<" + std::to_string(M) + "x" +
                              std::to_string(N) + "xf32>";

    return
        "module @wrapper {\n"
        "  func.func public @main(%lhs: " + lhs_t + ", %rhs: " + rhs_t +
        ") -> " + out_t + " {\n"
        "    %out = stablehlo.custom_call @tpu_custom_call(%lhs, %rhs) {\n"
        "      backend_config = \"" + json + "\",\n"
        "      kernel_name = \"matmul_kernel\",\n"
        "      mhlo.frontend_attributes = {kernel_metadata = \"{}\"},\n"
        "      operand_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>],\n"
        "      result_layouts = [dense<[1, 0]> : tensor<2xindex>]\n"
        "    } : (" + lhs_t + ", " + rhs_t + ") -> " + out_t + "\n"
        "    return %out : " + out_t + "\n"
        "  }\n"
        "}\n";
}

// ===--------------------------------------------------------------------=== //
// PJRT plugin loading
// ===--------------------------------------------------------------------=== //

using GetPjrtApiFn = const PJRT_Api* (*)();

const PJRT_Api* LoadTpuPjrtApi(const char* plugin_path) {
    void* handle = dlopen(plugin_path, RTLD_LAZY | RTLD_LOCAL);
    if (handle == nullptr) {
        std::cerr << "dlopen(\"" << plugin_path << "\") failed: "
                  << dlerror() << "\n";
        return nullptr;
    }
    dlerror();
    auto get_api = reinterpret_cast<GetPjrtApiFn>(
        dlsym(handle, "GetPjrtApi"));
    const char* err = dlerror();
    if (err != nullptr || get_api == nullptr) {
        std::cerr << "dlsym(\"GetPjrtApi\") failed: "
                  << (err ? err : "symbol returned null") << "\n";
        dlclose(handle);
        return nullptr;
    }
    // Intentionally leak `handle`: the returned PJRT_Api points into the
    // loaded library, so unloading it would invalidate the function table.
    return get_api();
}

// ===--------------------------------------------------------------------=== //
// Events
// ===--------------------------------------------------------------------=== //

bool AwaitEventAndDestroy(const PJRT_Api* api, PJRT_Event* event,
                          const char* context) {
    PJRT_Event_Await_Args await_args{};
    await_args.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
    await_args.event = event;
    if (!CheckError(api, api->PJRT_Event_Await(&await_args), context)) {
        return false;
    }

    PJRT_Event_Destroy_Args destroy_args{};
    destroy_args.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
    destroy_args.event = event;
    api->PJRT_Event_Destroy(&destroy_args);
    return true;
}

// ===--------------------------------------------------------------------=== //
// Device description (kind + coords), so each process can show what it got
// ===--------------------------------------------------------------------=== //

bool DescribeDevice(const PJRT_Api* api, PJRT_Device* device,
                    const char* prefix) {
    PJRT_Device_GetDescription_Args desc_args{};
    desc_args.struct_size = PJRT_Device_GetDescription_Args_STRUCT_SIZE;
    desc_args.device = device;
    if (!CheckError(api, api->PJRT_Device_GetDescription(&desc_args),
                    "PJRT_Device_GetDescription")) {
        return false;
    }
    PJRT_DeviceDescription* description = desc_args.device_description;

    PJRT_DeviceDescription_Kind_Args kind_args{};
    kind_args.struct_size = PJRT_DeviceDescription_Kind_Args_STRUCT_SIZE;
    kind_args.device_description = description;
    if (!CheckError(api, api->PJRT_DeviceDescription_Kind(&kind_args),
                    "PJRT_DeviceDescription_Kind")) {
        return false;
    }

    PJRT_DeviceDescription_Id_Args id_args{};
    id_args.struct_size = PJRT_DeviceDescription_Id_Args_STRUCT_SIZE;
    id_args.device_description = description;
    if (!CheckError(api, api->PJRT_DeviceDescription_Id(&id_args),
                    "PJRT_DeviceDescription_Id")) {
        return false;
    }

    std::cout << prefix << "device[id=" << id_args.id << "] kind=\""
              << std::string(kind_args.device_kind, kind_args.device_kind_size)
              << "\"";

    PJRT_DeviceDescription_Attributes_Args attr_args{};
    attr_args.struct_size = PJRT_DeviceDescription_Attributes_Args_STRUCT_SIZE;
    attr_args.device_description = description;
    if (!CheckError(api, api->PJRT_DeviceDescription_Attributes(&attr_args),
                    "PJRT_DeviceDescription_Attributes")) {
        return false;
    }
    for (size_t i = 0; i < attr_args.num_attributes; ++i) {
        const PJRT_NamedValue& nv = attr_args.attributes[i];
        std::string name(nv.name, nv.name_size);
        if (name == "coords" && nv.type == PJRT_NamedValue_kInt64List) {
            std::cout << " coords=[";
            for (size_t j = 0; j < nv.value_size; ++j) {
                if (j > 0) std::cout << ",";
                std::cout << nv.int64_array_value[j];
            }
            std::cout << "]";
        } else if (name == "core_on_chip" &&
                   nv.type == PJRT_NamedValue_kInt64) {
            std::cout << " core_on_chip=" << nv.int64_value;
        }
    }
    std::cout << "\n";
    return true;
}

// ===--------------------------------------------------------------------=== //
// Time helpers
// ===--------------------------------------------------------------------=== //

int64_t NowEpochMs() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(
               system_clock::now().time_since_epoch()).count();
}

double SecondsSince(std::chrono::steady_clock::time_point t0) {
    using namespace std::chrono;
    return duration<double>(steady_clock::now() - t0).count();
}

// Set an env var only if the user has not already set it in the shell, so
// the launcher can override any of these.
void SetEnvDefault(const char* name, const char* value) {
    if (std::getenv(name) == nullptr) setenv(name, value, /*overwrite=*/0);
}

void PrintEnv(const char* name) {
    const char* v = std::getenv(name);
    std::cout << "    " << name << "=" << (v ? v : "<unset>") << "\n";
}

// File-based start barrier (see step 8b in main). No-op unless
// TWO_PROCS_BARRIER_DIR is set. Returns false on timeout.
bool WaitAtBarrier(const char* chip_str) {
    const char* dir = std::getenv("TWO_PROCS_BARRIER_DIR");
    if (dir == nullptr || *dir == '\0') return true;

    // The launcher gives each worker a role tag (A/B) so the ready files
    // stay distinct even when both workers target the same chip.
    const char* tag = std::getenv("TWO_PROCS_BARRIER_TAG");
    const std::string ready_path =
        std::string(dir) + "/ready." + ((tag && *tag) ? tag : chip_str);
    const std::string go_path = std::string(dir) + "/go";
    {
        std::ofstream ready(ready_path);
        if (!ready) {
            std::cerr << "Cannot create barrier file " << ready_path << "\n";
            return false;
        }
    }
    std::cout << "[8b] Ready; waiting for " << go_path << " ...\n";
    auto t0 = std::chrono::steady_clock::now();
    while (access(go_path.c_str(), F_OK) != 0) {
        if (SecondsSince(t0) > 120.0) {
            std::cerr << "Barrier timeout: " << go_path
                      << " never appeared\n";
            return false;
        }
        usleep(2000);
    }
    return true;
}

}  // namespace

// ===--------------------------------------------------------------------=== //
// main
// ===--------------------------------------------------------------------=== //

void PrintUsage(const char* argv0) {
    std::cerr
        << "Usage:\n"
        << "  " << argv0
        << " <mosaic_mlir_file> <libtpu_so> <chip_index> [<iterations>]\n\n"
        << "Arguments:\n"
        << "  <mosaic_mlir_file>   Mosaic MLIR source (matmul_mosaic.mlir)\n"
        << "  <libtpu_so>          TPU PJRT plugin exporting GetPjrtApi\n"
        << "  <chip_index>         Which TPU chip this process owns\n"
        << "                       (0..3 on a v6e-4; becomes TPU_VISIBLE_CHIPS)\n"
        << "  <iterations>         How many matmuls to enqueue (default 4000)\n\n"
        << "Normally started by run_two_procs.sh, twice, with two different\n"
        << "chip indices.\n";
}

int main(int argc, char** argv) {
    if (argc != 4 && argc != 5) {
        PrintUsage(argc > 0 ? argv[0] : "two-procs-two-chips");
        return 1;
    }
    const char* mosaic_path = argv[1];
    const char* libtpu_path = argv[2];
    const char* chip_str = argv[3];
    // 4000 x ~0.35 ms on a v6e chip is about 1.5 s of device time per
    // process: long enough that the two execute windows visibly overlap.
    const int iterations = (argc == 5) ? std::atoi(argv[4]) : 4000;
    if (iterations <= 0) {
        std::cerr << "iterations must be > 0\n";
        return 1;
    }

    const pid_t pid = getpid();
    std::cout << "=== two-procs-two-chips (pid " << pid << ", chip "
              << chip_str << ") ===\n";
    std::cout << "Mosaic MLIR: " << mosaic_path << "\n";
    std::cout << "PJRT plugin: " << libtpu_path << "\n";
    std::cout << "Iterations:  " << iterations << "\n\n";

    // ---- 0. Pin this process to one chip BEFORE libtpu is loaded --------
    //
    // libtpu reads these when the client is created, but it is cleaner to
    // have them in place before dlopen so nothing in the library's static
    // initialisers can observe a different value.
    setenv("TPU_VISIBLE_CHIPS", chip_str, /*overwrite=*/1);
    SetEnvDefault("TPU_PROCESS_BOUNDS", "1,1,1");
    SetEnvDefault("TPU_CHIPS_PER_PROCESS_BOUNDS", "1,1,1");
    SetEnvDefault("ALLOW_MULTIPLE_LIBTPU_LOAD", "1");
    std::cout << "[0] Process-local TPU environment:\n";
    PrintEnv("TPU_VISIBLE_CHIPS");
    PrintEnv("TPU_PROCESS_BOUNDS");
    PrintEnv("TPU_CHIPS_PER_PROCESS_BOUNDS");
    PrintEnv("ALLOW_MULTIPLE_LIBTPU_LOAD");

    // ---- 1. Load PJRT plugin ------------------------------------------
    const PJRT_Api* api = LoadTpuPjrtApi(libtpu_path);
    if (api == nullptr) return 1;
    std::cout << "[1] Loaded PJRT plugin\n";

    // ---- 2. Initialize plugin -----------------------------------------
    {
        PJRT_Plugin_Initialize_Args init_args{};
        init_args.struct_size = PJRT_Plugin_Initialize_Args_STRUCT_SIZE;
        if (!CheckError(api, api->PJRT_Plugin_Initialize(&init_args),
                        "PJRT_Plugin_Initialize")) {
            return 1;
        }
    }
    std::cout << "[2] Initialized plugin\n";

    // ---- 3. Create PJRT client (this is where /dev/vfio/<chip> is opened)
    PJRT_Client* client = nullptr;
    {
        auto t0 = std::chrono::steady_clock::now();
        PJRT_Client_Create_Args create_args{};
        create_args.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
        if (!CheckError(api, api->PJRT_Client_Create(&create_args),
                        "PJRT_Client_Create")) {
            std::cerr << "Hint: \"Device or resource busy\" means another "
                         "process already owns chip " << chip_str << ".\n";
            return 1;
        }
        client = create_args.client;
        std::cout << "[3] Created PJRT client in " << SecondsSince(t0)
                  << " s\n";
    }

    // ---- 4. Exactly one device must be visible ------------------------
    PJRT_Device* device = nullptr;
    {
        PJRT_Client_Devices_Args all_args{};
        all_args.struct_size = PJRT_Client_Devices_Args_STRUCT_SIZE;
        all_args.client = client;
        if (!CheckError(api, api->PJRT_Client_Devices(&all_args),
                        "PJRT_Client_Devices")) {
            return 1;
        }
        PJRT_Client_AddressableDevices_Args dev_args{};
        dev_args.struct_size =
            PJRT_Client_AddressableDevices_Args_STRUCT_SIZE;
        dev_args.client = client;
        if (!CheckError(api, api->PJRT_Client_AddressableDevices(&dev_args),
                        "PJRT_Client_AddressableDevices")) {
            return 1;
        }
        std::cout << "[4] Topology seen by this process: "
                  << all_args.num_devices << " device(s) total, "
                  << dev_args.num_addressable_devices << " addressable\n";
        if (dev_args.num_addressable_devices != 1) {
            std::cerr << "Expected exactly 1 addressable device (chip "
                      << chip_str << "), got "
                      << dev_args.num_addressable_devices
                      << ". Is TPU_VISIBLE_CHIPS being honoured?\n";
            return 1;
        }
        device = dev_args.addressable_devices[0];
        if (!DescribeDevice(api, device, "    ")) return 1;
        // Note: the id is 0 and coords are [0,0,0] in EVERY process. Each
        // process sees a private 1-chip topology; the physical chip index
        // only exists in TPU_VISIBLE_CHIPS (and in the libtpu log line
        // "[/dev/vfio/<n> ...] Driver opened").
    }

    // ---- 5. Load Mosaic MLIR and wrap in StableHLO --------------------
    std::string mosaic_text = LoadTextFile(mosaic_path);
    if (mosaic_text.empty()) return 1;
    std::string stablehlo = WrapMosaicInStableHlo(mosaic_text);
    std::cout << "[5] Loaded Mosaic (" << mosaic_text.size()
              << " bytes) and wrapped into StableHLO ("
              << stablehlo.size() << " bytes)\n";

    // ---- 6. Compile ---------------------------------------------------
    PJRT_LoadedExecutable* executable = nullptr;
    {
        auto t0 = std::chrono::steady_clock::now();
        static constexpr char kFormat[] = "mlir";
        PJRT_Program program{};
        program.struct_size = PJRT_Program_STRUCT_SIZE;
        program.code = const_cast<char*>(stablehlo.data());
        program.code_size = stablehlo.size();
        program.format = kFormat;
        program.format_size = sizeof(kFormat) - 1;

        PJRT_Client_Compile_Args compile_args{};
        compile_args.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE;
        compile_args.client = client;
        compile_args.program = &program;
        // Minimal CompileOptionsProto (xla/pjrt/proto/compile_options.proto):
        //   executable_build_options (field 3) {
        //     num_replicas   (field 4) = 1
        //     num_partitions (field 5) = 1
        //   }
        // Wire format: 0x1A 0x04 | 0x20 0x01 | 0x28 0x01. Without it PJRT
        // rejects the compile with "replica_count=0, computation_count=0".
        static constexpr unsigned char kCompileOptions[] = {
            0x1A, 0x04, 0x20, 0x01, 0x28, 0x01,
        };
        compile_args.compile_options =
            reinterpret_cast<const char*>(kCompileOptions);
        compile_args.compile_options_size = sizeof(kCompileOptions);
        if (!CheckError(api, api->PJRT_Client_Compile(&compile_args),
                        "PJRT_Client_Compile")) {
            return 1;
        }
        executable = compile_args.executable;
        std::cout << "[6] Compiled StableHLO -> PJRT_LoadedExecutable in "
                  << SecondsSince(t0) << " s\n";
    }

    // ---- 7. Host inputs with a closed-form expected result -------------
    //
    //   lhs[i][k] = (7 i + 3 k) mod 16          (small exact integers)
    //   rhs[k][j] = 1 + (j mod 2)               (column j is all 1s or all 2s)
    //   out[i][j] = rowsum(lhs[i]) * (1 + j mod 2)
    //
    // Everything stays an integer below 2^24, so f32 accumulation is exact
    // and the check can demand max_abs_err == 0.
    std::vector<float> host_lhs(M * K), host_rhs(K * N);
    std::vector<double> rowsum(M, 0.0);
    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            float v = static_cast<float>((7 * i + 3 * k) % 16);
            host_lhs[i * K + k] = v;
            rowsum[i] += v;
        }
    }
    for (size_t k = 0; k < K; ++k)
        for (size_t j = 0; j < N; ++j)
            host_rhs[k * N + j] = static_cast<float>(1 + (j % 2));

    // ---- 8. Upload -----------------------------------------------------
    auto upload = [&](const float* data, size_t rows, size_t cols)
        -> PJRT_Buffer* {
        int64_t dims[] = {static_cast<int64_t>(rows),
                          static_cast<int64_t>(cols)};
        PJRT_Client_BufferFromHostBuffer_Args args{};
        args.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
        args.client = client;
        args.data = data;
        args.type = PJRT_Buffer_Type_F32;
        args.dims = dims;
        args.num_dims = 2;
        args.byte_strides = nullptr;
        args.num_byte_strides = 0;
        args.host_buffer_semantics =
            PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes;
        args.device = device;
        args.memory = nullptr;
        args.device_layout = nullptr;
        if (!CheckError(api, api->PJRT_Client_BufferFromHostBuffer(&args),
                        "PJRT_Client_BufferFromHostBuffer")) {
            return nullptr;
        }
        if (!AwaitEventAndDestroy(api, args.done_with_host_buffer,
                                  "done_with_host_buffer await")) {
            return nullptr;
        }
        return args.buffer;
    };
    PJRT_Buffer* buf_lhs = upload(host_lhs.data(), M, K);
    PJRT_Buffer* buf_rhs = upload(host_rhs.data(), K, N);
    if (buf_lhs == nullptr || buf_rhs == nullptr) return 1;
    std::cout << "[7] Uploaded lhs " << M << "x" << K << " and rhs "
              << K << "x" << N << " (f32)\n";

    auto destroy_buffer = [&](PJRT_Buffer* buf) {
        if (buf == nullptr) return;
        PJRT_Buffer_Destroy_Args a{};
        a.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
        a.buffer = buf;
        api->PJRT_Buffer_Destroy(&a);
    };

    // ---- 8b. Optional start barrier -------------------------------------
    //
    // libtpu init and compile take a different amount of time in every
    // process, so without a barrier the two execute windows start a second
    // or so apart and only partially overlap. If the launcher sets
    // TWO_PROCS_BARRIER_DIR, each worker drops a "ready.<chip>" file there
    // once its chip is fully set up and then waits for a "go" file, which
    // the launcher creates as soon as every worker has reported ready.
    // Nothing TPU-specific here; it just lines the windows up.
    if (!WaitAtBarrier(chip_str)) return 1;

    // ---- 9. Execute <iterations> times, back-to-back --------------------
    //
    // Execute is asynchronous: each call only enqueues work on the chip and
    // returns an output buffer whose contents become ready later. We keep
    // the previous output alive until the next launch is enqueued and then
    // destroy it; PJRT defers the actual free until the device is done with
    // it, so HBM usage stays at two outputs instead of <iterations>.
    //
    // The host-side timestamps bracket "first enqueue" .. "last result on
    // host". Downloading the last output waits for its definition event,
    // and launches on one device run in order, so that drains the queue.
    std::cout << "[8] Executing " << iterations << " matmul(s) on chip "
              << chip_str << " ...\n";
    const int64_t exec_start_ms = NowEpochMs();
    auto exec_t0 = std::chrono::steady_clock::now();

    PJRT_Buffer* prev_out = nullptr;
    PJRT_Buffer* last_out = nullptr;
    for (int it = 0; it < iterations; ++it) {
        PJRT_Buffer* inputs[2] = {buf_lhs, buf_rhs};
        PJRT_Buffer** argument_lists[1] = {inputs};
        PJRT_Buffer* out = nullptr;
        PJRT_Buffer** output_lists[1] = {&out};

        PJRT_ExecuteOptions exec_options{};
        exec_options.struct_size = PJRT_ExecuteOptions_STRUCT_SIZE;
        exec_options.launch_id = it;

        PJRT_LoadedExecutable_Execute_Args exec_args{};
        exec_args.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
        exec_args.executable = executable;
        exec_args.options = &exec_options;
        exec_args.argument_lists = argument_lists;
        exec_args.num_devices = 1;
        exec_args.num_args = 2;
        exec_args.output_lists = output_lists;
        exec_args.device_complete_events = nullptr;
        exec_args.execute_device = nullptr;
        if (!CheckError(api, api->PJRT_LoadedExecutable_Execute(&exec_args),
                        "PJRT_LoadedExecutable_Execute")) {
            return 1;
        }
        destroy_buffer(prev_out);
        prev_out = out;
        last_out = out;
    }

    // ---- 10. Download the last output (this is what blocks) ------------
    std::vector<float> host_out(M * N);
    {
        PJRT_Buffer_ToHostBuffer_Args args{};
        args.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
        args.src = last_out;
        args.host_layout = nullptr;
        args.dst = host_out.data();
        args.dst_size = host_out.size() * sizeof(float);
        if (!CheckError(api, api->PJRT_Buffer_ToHostBuffer(&args),
                        "PJRT_Buffer_ToHostBuffer")) {
            return 1;
        }
        if (!AwaitEventAndDestroy(api, args.event,
                                  "PJRT_Buffer_ToHostBuffer await")) {
            return 1;
        }
    }
    const double exec_seconds = SecondsSince(exec_t0);
    const int64_t exec_end_ms = NowEpochMs();
    const double tflops =
        FLOPS_PER_CALL * iterations / exec_seconds / 1e12;
    std::cout << "[9] Done: " << iterations << " matmuls in " << exec_seconds
              << " s  (" << (exec_seconds / iterations * 1e3)
              << " ms/call, " << tflops << " TFLOP/s f32)\n";

    // Machine-readable line for run_two_procs.sh.
    std::cout << "EXEC_WINDOW chip=" << chip_str << " pid=" << pid
              << " start_ms=" << exec_start_ms << " end_ms=" << exec_end_ms
              << " iters=" << iterations << "\n";

    // ---- 11. Verify ------------------------------------------------------
    double max_err = 0.0;
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            double expected = rowsum[i] * (1 + (j % 2));
            double err = host_out[i * N + j] - expected;
            if (err < 0) err = -err;
            if (err > max_err) max_err = err;
        }
    }
    std::cout << "Result (row 0, first 4 | row " << (M - 1) << ", first 4):\n  "
              << host_out[0] << " " << host_out[1] << " " << host_out[2] << " "
              << host_out[3] << " | " << host_out[(M - 1) * N + 0] << " "
              << host_out[(M - 1) * N + 1] << " " << host_out[(M - 1) * N + 2]
              << " " << host_out[(M - 1) * N + 3] << "\n";
    std::cout << "Expected:\n  "
              << rowsum[0] << " " << 2 * rowsum[0] << " " << rowsum[0] << " "
              << 2 * rowsum[0] << " | " << rowsum[M - 1] << " "
              << 2 * rowsum[M - 1] << " " << rowsum[M - 1] << " "
              << 2 * rowsum[M - 1] << "\n";
    std::cout << "Max abs error over all " << M << "x" << N
              << " elements: " << max_err
              << (max_err == 0.0 ? "  (OK)" : "  (MISMATCH)") << "\n";

    // ---- 12. Cleanup -----------------------------------------------------
    destroy_buffer(last_out);
    destroy_buffer(buf_lhs);
    destroy_buffer(buf_rhs);
    {
        PJRT_LoadedExecutable_Destroy_Args a{};
        a.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
        a.executable = executable;
        api->PJRT_LoadedExecutable_Destroy(&a);
    }
    {
        PJRT_Client_Destroy_Args a{};
        a.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
        a.client = client;
        api->PJRT_Client_Destroy(&a);
    }

    std::cout << "\n[done] chip " << chip_str << " pid " << pid << "\n";
    return max_err == 0.0 ? 0 : 2;
}
