// ===--------------------------------------------------------------------=== //
// pjrt-demo: run a Mosaic IR kernel on TPU through PJRT
// ===--------------------------------------------------------------------=== //
//
// End-to-end flow:
//
//   Mosaic MLIR text file
//       |
//       v  WrapMosaicInStableHlo
//   StableHLO with @tpu_custom_call(... backend_config = <mosaic text>)
//       |
//       v  PJRT_Client_Compile
//   PJRT_LoadedExecutable
//       |
//       v  PJRT_Client_BufferFromHostBuffer (x2, for lhs/rhs)
//   PJRT_Buffer* inputs
//       |
//       v  PJRT_LoadedExecutable_Execute
//   PJRT_Buffer* output
//       |
//       v  PJRT_Buffer_ToHostBuffer
//   host float[128]
//
// The only compile-time dependency is xla/pjrt/c/pjrt_c_api.h (plain C).
// The runtime dependency is the PJRT-exporting TPU shared library
// (e.g. libtpu.so) whose path is passed on the command line.
// ===--------------------------------------------------------------------=== //

#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <ctime>
#include <filesystem>

#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_profiler_extension.h"

namespace {

// ===--------------------------------------------------------------------=== //
// Profiler extension helpers
// ===--------------------------------------------------------------------=== //
//
// libtpu exposes the profiler as a PJRT extension. The extension chain starts
// at api->extension_start; each entry has `type` and `next` fields. We walk
// the chain looking for PJRT_Extension_Type_Profiler and cast it to
// PJRT_Profiler_Extension*.

const PJRT_Profiler_Extension* FindProfilerExtension(const PJRT_Api* api) {
    for (const PJRT_Extension_Base* ext = api->extension_start;
         ext != nullptr; ext = ext->next) {
        if (ext->type == PJRT_Extension_Type_Profiler) {
            return reinterpret_cast<const PJRT_Profiler_Extension*>(ext);
        }
    }
    return nullptr;
}

bool CheckProfilerError(const PLUGIN_Profiler_Api* p_api,
                        PLUGIN_Profiler_Error* error, const char* context) {
    if (error == nullptr) return true;

    PLUGIN_Profiler_Error_Message_Args msg_args{};
    msg_args.struct_size = PLUGIN_Profiler_Error_Message_Args_STRUCT_SIZE;
    msg_args.error = error;
    p_api->error_message(&msg_args);
    std::cerr << context << ": "
              << std::string(msg_args.message, msg_args.message_size) << "\n";

    PLUGIN_Profiler_Error_Destroy_Args destroy_args{};
    destroy_args.struct_size = PLUGIN_Profiler_Error_Destroy_Args_STRUCT_SIZE;
    destroy_args.error = error;
    p_api->error_destroy(&destroy_args);
    return false;
}

class TpuProfiler {
public:
    TpuProfiler(const PLUGIN_Profiler_Api* api, const std::string& options_pb)
        : api_(api) {
        PLUGIN_Profiler_Create_Args args{};
        args.struct_size = PLUGIN_Profiler_Create_Args_STRUCT_SIZE;
        args.options = options_pb.data();
        args.options_size = options_pb.size();
        if (!CheckProfilerError(api_, api_->create(&args),
                                "PLUGIN_Profiler_Create")) {
            return;
        }
        profiler_ = args.profiler;
    }

    ~TpuProfiler() {
        if (profiler_ == nullptr) return;
        PLUGIN_Profiler_Destroy_Args args{};
        args.struct_size = PLUGIN_Profiler_Destroy_Args_STRUCT_SIZE;
        args.profiler = profiler_;
        api_->destroy(&args);
    }

    bool Start() {
        PLUGIN_Profiler_Start_Args args{};
        args.struct_size = PLUGIN_Profiler_Start_Args_STRUCT_SIZE;
        args.profiler = profiler_;
        return CheckProfilerError(api_, api_->start(&args),
                                  "PLUGIN_Profiler_Start");
    }

    bool Stop() {
        PLUGIN_Profiler_Stop_Args args{};
        args.struct_size = PLUGIN_Profiler_Stop_Args_STRUCT_SIZE;
        args.profiler = profiler_;
        return CheckProfilerError(api_, api_->stop(&args),
                                  "PLUGIN_Profiler_Stop");
    }

    // Two-call pattern: first call with buffer=nullptr to obtain the size,
    // second call with an allocated buffer to pull the bytes.
    bool CollectToFile(const std::string& path) {
        PLUGIN_Profiler_CollectData_Args args{};
        args.struct_size = PLUGIN_Profiler_CollectData_Args_STRUCT_SIZE;
        args.profiler = profiler_;
        args.buffer = nullptr;
        if (!CheckProfilerError(api_, api_->collect_data(&args),
                                "PLUGIN_Profiler_CollectData (size)")) {
            return false;
        }
        if (args.buffer == nullptr || args.buffer_size_in_bytes == 0) {
            std::cerr << "[profiler] CollectData returned no data\n";
            return false;
        }

        // Off-by-one in plugin_tracer_impl.cc: the plugin allocates a vector
        // of size (profiler_data_size + 1) but only serializes the first
        // profiler_data_size bytes. It then reports buffer_size_in_bytes =
        // vector->size(), so the last byte is a stray zero that proto
        // parsers reject as an invalid field-0 tag. Strip it.
        const size_t written = args.buffer_size_in_bytes > 0
                                   ? args.buffer_size_in_bytes - 1
                                   : 0;

        std::ofstream out(path, std::ios::binary);
        if (!out) {
            std::cerr << "Failed to open " << path << " for writing\n";
            return false;
        }
        out.write(reinterpret_cast<const char*>(args.buffer),
                  static_cast<std::streamsize>(written));
        std::cout << "[profiler] Wrote profiler results ("
                  << written << " bytes) to " << path << "\n";
        return true;
    }

    bool valid() const { return profiler_ != nullptr; }

private:
    const PLUGIN_Profiler_Api* api_;
    PLUGIN_Profiler* profiler_ = nullptr;
};

// ===--------------------------------------------------------------------=== //
// Error helper
// ===--------------------------------------------------------------------=== //
//
// Every PJRT call returns a PJRT_Error* (nullptr on success). On error we
// pull the message out via PJRT_Error_Message and destroy the error via
// PJRT_Error_Destroy.

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

// Builds the TensorBoard/XProf trace path:
//   profile_logs/plugins/profile/<YYYY_MM_DD_HH_MM_SS>/trace.xplane.pb
// and creates the directory tree. The timestamped run folder lets TensorBoard
// list each profiling run separately. Returns "" on failure.
std::string MakeProfileOutputPath() {
    const std::time_t now = std::chrono::system_clock::to_time_t(
        std::chrono::system_clock::now());
    std::tm tm_buf{};
    localtime_r(&now, &tm_buf);
    char stamp[32];
    std::strftime(stamp, sizeof(stamp), "%Y_%m_%d_%H_%M_%S", &tm_buf);

    namespace fs = std::filesystem;
    const fs::path dir =
        fs::path("profile_logs") / "plugins" / "profile" / stamp;

    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec) {
        std::cerr << "Failed to create " << dir << ": " << ec.message() << "\n";
        return {};
    }
    return (dir / "trace.xplane.pb").string();
}

// ===--------------------------------------------------------------------=== //
// StableHLO wrapping
// ===--------------------------------------------------------------------=== //

std::string EscapeMlirString(const std::string& raw) {
    std::string out;
    out.reserve(raw.size() * 2);
    for (char c : raw) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:   out += c; break;
        }
    }
    return out;
}

// Standard base64 encoder. jaxlib's `as_tpu_kernel` uses base64 for the
// JSON backend_config's "body" field. We replicate that format.
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

// Build the StableHLO wrapper with the correct Mosaic backend_config format.
// The JSON shape comes from jaxlib's `CustomCallBackendConfig.to_json()`:
//     {"custom_call_config": {"body": "<base64>"}}
//
// The body is the Mosaic MLIR bytes (text is accepted by ir.Module.parse,
// though jaxlib normally feeds bytecode). We try text first; if libtpu
// rejects it we'd need a bytecode pre-processing step.
std::string WrapMosaicInStableHlo(const std::string& mosaic_text) {
    const std::string body_b64 = Base64Encode(mosaic_text);
    // Inner JSON; escape the embedded quotes for the MLIR string literal.
    const std::string json =
        std::string("{\\\"custom_call_config\\\": {\\\"body\\\": \\\"")
        + body_b64 + "\\\","
        + " \\\"needs_layout_passes\\\": true, \\\"shape_invariant_numerics\\\": false}}";

    return
        "module @wrapper {\n"
        "  func.func @main(\n"
        "      %lhs: tensor<256x256xf32>,\n"
        "      %rhs: tensor<256x256xf32>\n"
        "  ) -> tensor<256x256xf32> {\n"
        "    %out = stablehlo.custom_call @tpu_custom_call(%lhs, %rhs) {\n"
        "      backend_config = \"" + json + "\",\n"
        "      operand_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>],\n"
        "      result_layouts = [dense<[1, 0]> : tensor<2xindex>]"
        "    } : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>\n"
        "    return %out : tensor<256x256xf32>\n"
        "  }\n"
        "}\n";
}

// DIAGNOSTIC: a minimal StableHLO that just returns its first argument.
// Used to distinguish "compile path broken" from "Mosaic custom_call broken".
std::string BuildTrivialPassThroughStableHlo() {
    return
        "module @passthrough {\n"
        "  func.func @main(\n"
        "      %lhs: tensor<128xf32>,\n"
        "      %rhs: tensor<128xf32>\n"
        "  ) -> tensor<128xf32> {\n"
        "    return %lhs : tensor<128xf32>\n"
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
// Wait on a PJRT_Event and destroy it.
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

}  // namespace

// ===--------------------------------------------------------------------=== //
// main
// ===--------------------------------------------------------------------=== //

void PrintUsage(const char* argv0) {
    std::cerr
        << "Usage:\n"
        << "  " << argv0 << " <mosaic_mlir_file> <libtpu_so> [<output_blob_path>]\n"
        << "  " << argv0 << " --load-blob <libtpu_so> <input_blob_path>\n\n"
        << "Arguments:\n"
        << "  <mosaic_mlir_file>   Mosaic MLIR source file (e.g. vector_add.mlir)\n"
        << "  <libtpu_so>          TPU PJRT plugin exporting GetPjrtApi\n"
        << "                       (e.g. .../site-packages/libtpu/libtpu.so)\n"
        << "  <output_blob_path>   Optional. If given, the compiled PJRT\n"
        << "                       executable is serialized to this path\n"
        << "                       after compilation.\n"
        << "  --load-blob          Load a previously serialized PJRT executable\n"
        << "                       instead of compiling from MLIR.\n"
        << "  <input_blob_path>    Path to a serialized PJRT executable.\n";
}

int main(int argc, char** argv) {
    // Scan for the optional --trivial flag (diagnostic only).
    bool trivial = false;
    bool load_blob = false;
    std::vector<char*> positional;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--trivial") == 0) {
            trivial = true;
        } else if (std::strcmp(argv[i], "--load-blob") == 0) {
            load_blob = true;
        } else {
            positional.push_back(argv[i]);
        }
    }

    const char* mosaic_path = nullptr;
    const char* libtpu_path = nullptr;
    const char* blob_out_path = nullptr;
    const char* blob_in_path = nullptr;
    
    if ((positional.size() != 2 && positional.size() != 3 && !load_blob) || 
        (positional.size() != 2 && load_blob)) {
        PrintUsage(argc > 0 ? argv[0] : "pjrt-demo");
        return 1;
    }

    std::cout << "=== pjrt-demo ===\n";
    if (load_blob) {
        blob_in_path = positional[0];
        std::cout << "Serialized blob: " << blob_in_path << "\n";
    } else {
        mosaic_path = positional[0];
        blob_out_path =
           (positional.size() == 3) ? positional[2] : nullptr;
        std::cout << "Mosaic MLIR: " << mosaic_path << "\n";
    }
    libtpu_path = positional[1];
    std::cout << "PJRT plugin: " << libtpu_path << "\n\n";

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

    // ---- 3. Create PJRT client ----------------------------------------
    PJRT_Client* client = nullptr;
    {
        PJRT_Client_Create_Args create_args{};
        create_args.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
        if (!CheckError(api, api->PJRT_Client_Create(&create_args),
                        "PJRT_Client_Create")) {
            return 1;
        }
        client = create_args.client;
    }
    std::cout << "[3] Created PJRT client\n";

    // ---- 4. Get the first addressable device --------------------------
    PJRT_Device* device = nullptr;
    {
        PJRT_Client_AddressableDevices_Args dev_args{};
        dev_args.struct_size =
            PJRT_Client_AddressableDevices_Args_STRUCT_SIZE;
        dev_args.client = client;
        if (!CheckError(api, api->PJRT_Client_AddressableDevices(&dev_args),
                        "PJRT_Client_AddressableDevices")) {
            return 1;
        }
        if (dev_args.num_addressable_devices == 0) {
            std::cerr << "No addressable devices.\n";
            return 1;
        }
        device = dev_args.addressable_devices[0];
        std::cout << "[4] Found " << dev_args.num_addressable_devices
                  << " addressable device(s); using device[0]\n";
    }

    PJRT_LoadedExecutable* executable = nullptr;
    if (!load_blob) {
        // ---- 5. Load Mosaic MLIR and wrap in StableHLO --------------------
        std::string stablehlo;
        if (trivial) {
            stablehlo = BuildTrivialPassThroughStableHlo();
            std::cout << "[5] --trivial: using pass-through StableHLO ("
                    << stablehlo.size() << " bytes), ignoring "
                    << mosaic_path << "\n";
        } else {
            std::string mosaic_text = LoadTextFile(mosaic_path);
            if (mosaic_text.empty()) return 1;
            stablehlo = WrapMosaicInStableHlo(mosaic_text);
            std::cout << "[5] Loaded Mosaic (" << mosaic_text.size()
                    << " bytes) and wrapped into StableHLO ("
                    << stablehlo.size() << " bytes)\n";
        }
    

        // ---- 6. Compile ---------------------------------------------------
        {
            static constexpr char kFormat[] = "mlir";
            PJRT_Program program{};
            program.struct_size = PJRT_Program_STRUCT_SIZE;
            program.code = const_cast<char*>(stablehlo.data());
            program.code_size = stablehlo.size();
            program.format = kFormat;
            program.format_size = sizeof(kFormat) - 1;  // 4, without NUL

            // Minimal CompileOptionsProto (from xla/pjrt/compile_options.proto):
            //   CompileOptionsProto.executable_build_options (field 3, message)
            //     ExecutableBuildOptionsProto.num_replicas   (field 4, int64) = 1
            //     ExecutableBuildOptionsProto.num_partitions (field 5, int64) = 1
            //
            // Wire format:
            //   0x1A 0x04  = tag(field=3, wire=length-delim), length=4
            //     0x20 0x01 = tag(field=4, wire=varint), value=1    (num_replicas)
            //     0x28 0x01 = tag(field=5, wire=varint), value=1    (num_partitions)
            //
            // Without this, PJRT rejects with "Invalid (replica_count,
            // computation_count) pair: (0,0)".
            static constexpr unsigned char kCompileOptions[] = {
                0x1A, 0x04, 0x20, 0x01, 0x28, 0x01,
            };

            PJRT_Client_Compile_Args compile_args{};
            compile_args.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE;
            compile_args.client = client;
            compile_args.program = &program;
            compile_args.compile_options =
                reinterpret_cast<const char*>(kCompileOptions);
            compile_args.compile_options_size = sizeof(kCompileOptions);
            if (!CheckError(api, api->PJRT_Client_Compile(&compile_args),
                            "PJRT_Client_Compile")) {
                return 1;
            }
            executable = compile_args.executable;
        }
        std::cout << "[6] Compiled StableHLO -> PJRT_LoadedExecutable\n";

        // ---- 6b. (optional) Serialize executable to disk ------------------
        if (blob_out_path != nullptr) {
            // Get the un-loaded Executable view of the LoadedExecutable.
            PJRT_LoadedExecutable_GetExecutable_Args get_args{};
            get_args.struct_size =
                PJRT_LoadedExecutable_GetExecutable_Args_STRUCT_SIZE;
            get_args.loaded_executable = executable;
            if (!CheckError(api,
                            api->PJRT_LoadedExecutable_GetExecutable(&get_args),
                            "PJRT_LoadedExecutable_GetExecutable")) {
                return 1;
            }
            PJRT_Executable* plain_exe = get_args.executable;

            // Serialize it.
            PJRT_Executable_Serialize_Args ser_args{};
            ser_args.struct_size = PJRT_Executable_Serialize_Args_STRUCT_SIZE;
            ser_args.executable = plain_exe;
            if (!CheckError(api, api->PJRT_Executable_Serialize(&ser_args),
                            "PJRT_Executable_Serialize")) {
                return 1;
            }

            // Write to disk.
            std::ofstream out(blob_out_path, std::ios::binary);
            if (!out) {
                std::cerr << "Failed to open " << blob_out_path
                        << " for writing\n";
                return 1;
            }
            out.write(ser_args.serialized_bytes,
                    static_cast<std::streamsize>(
                        ser_args.serialized_bytes_size));
            out.close();
            std::cout << "[6b] Serialized executable ("
                    << ser_args.serialized_bytes_size << " bytes) to "
                    << blob_out_path << "\n";

            // Free the backing memory and the plain_exe wrapper.
            ser_args.serialized_executable_deleter(
                ser_args.serialized_executable);

            PJRT_Executable_Destroy_Args exe_destroy{};
            exe_destroy.struct_size = PJRT_Executable_Destroy_Args_STRUCT_SIZE;
            exe_destroy.executable = plain_exe;
            api->PJRT_Executable_Destroy(&exe_destroy);
        }
    } else {
        // ---- 5. Deserialize and load -------------------------------------

        // Read file into memory
        std::ifstream in(blob_in_path, std::ios::binary | std::ios::ate);
        if (!in) {
            std::cerr << "Failed to open " << blob_in_path << " for reading\n";
            return 1;
        }

        std::streamsize size = in.tellg();
        in.seekg(0, std::ios::beg);

        std::vector<char> buffer(size);
        if (!in.read(buffer.data(), size)) {
            std::cerr << "Failed to read file " << blob_in_path << "\n";
            return 1;
        }
        in.close();

        // Deserialize executable
        PJRT_Executable_DeserializeAndLoad_Args deser_args{};
        deser_args.struct_size = PJRT_Executable_DeserializeAndLoad_Args_STRUCT_SIZE;
        deser_args.client = client;  // PJRT_Client*
        deser_args.serialized_executable = buffer.data();
        deser_args.serialized_executable_size = buffer.size();

        if (!CheckError(api, api->PJRT_Executable_DeserializeAndLoad(&deser_args),
                        "PJRT_Executable_DeserializeAndLoad")) {
            return 1;
        }

        executable = deser_args.loaded_executable;

        std::cout << "[5] Deserialized executable from "
                << blob_in_path << "\n";
    }

    // ---- 7. Prepare host-side inputs ----------------------------------
    constexpr size_t ROWS = 256;
    constexpr size_t COLS = 256;
    constexpr size_t N = ROWS * COLS;  // 1024
    std::vector<float> host_lhs(N), host_rhs(N);
    for (size_t i = 0; i < N; ++i) {
        host_lhs[i] = static_cast<float>(i);
        host_rhs[i] = 1.0f;
    }

    // ---- 8. Upload inputs to device -----------------------------------
    auto upload = [&](const std::vector<float>& host_data)
        -> PJRT_Buffer* {
        int64_t dims[] = {static_cast<int64_t>(ROWS),
                          static_cast<int64_t>(COLS)};

        PJRT_Client_BufferFromHostBuffer_Args args{};
        args.struct_size =
            PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
        args.client = client;
        args.data = host_data.data();
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
        // Wait for the H->D transfer to complete before freeing host_data
        // (which happens at end of scope).
        if (!AwaitEventAndDestroy(api, args.done_with_host_buffer,
                                  "done_with_host_buffer await")) {
            return nullptr;
        }
        return args.buffer;
    };

    PJRT_Buffer* buf_lhs = upload(host_lhs);
    PJRT_Buffer* buf_rhs = upload(host_rhs);
    if (buf_lhs == nullptr || buf_rhs == nullptr) return 1;
    std::cout << "[7] Uploaded 2 x " << N << " floats to device\n";

    // ---- 9. Execute ---------------------------------------------------
    PJRT_Buffer* output_buffers[1] = {nullptr};
    PJRT_Buffer** output_list = output_buffers;
    PJRT_Buffer* input_args[2] = {buf_lhs, buf_rhs};
    // Non-const to match PJRT 0.23 (PJRT_Buffer***); implicit const-adding
    // conversion covers the newer type PJRT_Buffer* const* const*.
    PJRT_Buffer** input_list[1] = {input_args};

    // --- Profiler setup --------------------------------------------------
    // 1) Find the extension in the chain.
    // 2) Create the profiler (options = empty proto -> default TPU profiling).
    // 3) Start before Execute, Stop afterwards, CollectToFile.
    const PJRT_Profiler_Extension* prof_ext = FindProfilerExtension(api);
    std::unique_ptr<TpuProfiler> profiler;
    if (prof_ext == nullptr || prof_ext->profiler_api == nullptr) {
        std::cerr << "[profiler] Extension not available in this plugin; "
                     "skipping profiling.\n";
    } else {
        // Hand-rolled tensorflow.ProfileOptions proto (from
        // third_party/tsl/tsl/profiler/protobuf/profiler_options.proto).
        // An empty proto leaves every tracer level at 0 (disabled), so we
        // must set at least device_tracer_level + device_type explicitly.
        //
        // Fields we set (all varint):
        //   2  host_tracer_level   = 2              -> 0x10 0x02
        //   3  device_tracer_level = 1              -> 0x18 0x01
        //   5  version             = 1              -> 0x28 0x01
        //   6  device_type         = TPU (3)        -> 0x30 0x03
        //   7  enable_hlo_proto    = true (1)       -> 0x38 0x01
        static constexpr unsigned char kProfileOptions[] = {
            0x10, 0x02, 0x18, 0x01, 0x28, 0x01, 0x30, 0x03, 0x38, 0x01,
        };
        const std::string options_pb(
            reinterpret_cast<const char*>(kProfileOptions),
            sizeof(kProfileOptions));
        profiler = std::make_unique<TpuProfiler>(prof_ext->profiler_api,
                                                 options_pb);
        if (!profiler->valid()) {
            std::cerr << "[profiler] Create failed; skipping.\n";
            profiler.reset();
        } else if (!profiler->Start()) {
            profiler.reset();
        } else {
            std::cout << "[profiler] Start OK\n";
        }
    }
    constexpr int kIters = 2000;
    for (int iter = 0; iter < kIters; ++iter)
    {
        PJRT_ExecuteOptions exec_options{};
        PJRT_Event *deviceCompleteEvents[1] = {nullptr};
        exec_options.struct_size = PJRT_ExecuteOptions_STRUCT_SIZE;

        PJRT_LoadedExecutable_Execute_Args exec_args{};
        exec_args.struct_size =
            PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
        exec_args.executable = executable;
        exec_args.options = &exec_options;
        exec_args.argument_lists = input_list;
        exec_args.num_devices = 1;
        exec_args.num_args = 2;
        exec_args.output_lists = &output_list;
        exec_args.device_complete_events = deviceCompleteEvents;
        exec_args.execute_device = nullptr;

        if (!CheckError(api,
                        api->PJRT_LoadedExecutable_Execute(&exec_args),
                        "PJRT_LoadedExecutable_Execute")) {
            return 1;
        }

        if (!AwaitEventAndDestroy(api, deviceCompleteEvents[0],
                                  "PJRT_LoadedExecutable_Execute await")) {
            return 1;
        }

        if (iter < kIters - 1) {
            PJRT_Buffer_Destroy_Args d{};
            d.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
            d.buffer = output_buffers[0];
            api->PJRT_Buffer_Destroy(&d);
            output_buffers[0] = nullptr;
        }
    }
    std::cout << "[8] Executed kernel\n";


    // ---- 10. Download output ------------------------------------------
    std::vector<float> host_out(N);
    {
        PJRT_Buffer_ToHostBuffer_Args args{};
        args.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
        args.src = output_buffers[0];
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
    std::cout << "[9] Downloaded output\n\n";

    // ---- 11. Show results ---------------------------------------------
    std::cout << "Result (first 8 elements):\n  ";
    for (int i = 0; i < 8; ++i) {
        std::cout << host_out[i] << " ";
    }
    std::cout << "\n";

    // ---- 12. Cleanup --------------------------------------------------
    auto destroy_buffer = [&](PJRT_Buffer* buf) {
        if (buf == nullptr) return;
        PJRT_Buffer_Destroy_Args a{};
        a.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
        a.buffer = buf;
        api->PJRT_Buffer_Destroy(&a);
    };
    destroy_buffer(buf_lhs);
    destroy_buffer(buf_rhs);
    destroy_buffer(output_buffers[0]);

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
    // --- Profiler teardown ----------------------------------------------
    if (profiler) {
        if (profiler->Stop()) {
            const std::string out_path = MakeProfileOutputPath();
            if (!out_path.empty()) {
                profiler->CollectToFile(out_path);
            }
        }
    }

    std::cout << "\n[done]\n";
    return 0;
}
