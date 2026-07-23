// ===--------------------------------------------------------------------=== //
// pjrt-demo-4tpu: run a gridded Pallas/Mosaic add kernel on 4 TPU devices
// ===--------------------------------------------------------------------=== //
//
// End-to-end flow:
//
//   Mosaic MLIR text file (from matrix_add/add_4tpu_grid.py debug=True dump
//   — the per-device program; grid lives in iteration_bounds/window_params)
//       |
//       v  WrapMosaicInStableHlo
//   StableHLO in the sdy (Shardy) partitioned form JAX emits:
//   full 1024x1024 args + sdy.manual_computation over 256x1024 shards
//       |
//       v  PJRT_Client_Compile (num_partitions = 4, use_spmd_partitioning)
//   PJRT_LoadedExecutable spanning 4 devices
//       |
//       v  PJRT_Client_BufferFromHostBuffer
//   lhs and rhs: each device gets its own 256x1024 row shard
//       |
//       v  PJRT_LoadedExecutable_Execute (num_devices = 4)
//   4 x PJRT_Buffer* output shards (256x1024 each)
//       |
//       v  PJRT_Buffer_ToHostBuffer (x4, stitched by rows)
//   host float[1024x1024]
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

#include "xla/pjrt/c/pjrt_c_api.h"

namespace {

constexpr size_t NUM_DEVICES = 4;
constexpr size_t M = 1024;
constexpr size_t N = 1024;
constexpr size_t M_SHARD = M / NUM_DEVICES;  // 256 rows per device

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
// StableHLO wrapping
// ===--------------------------------------------------------------------=== //
//
// Wrapper u sdy (Shardy) formi, isto ono sto JAX salje kroz PJRT za
// shard_map-ovan pallas_call: argumenti @main su PUNI 1024x1024 tenzori sa
// sdy.sharding anotacijama, a sdy.manual_computation unutra radi nad
// per-device 256x1024 shardovima. Zahteva num_partitions = 4 +
// use_spmd_partitioning u compile options (korak 6).

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

    return
        "module @wrapper attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {\n"
        "  sdy.mesh @mesh = <[\"dev\"=4]> {stablehlo.mesh = {axes = [{name = \"dev\", size = 4 : i64}]}}\n"
        "  func.func public @main(\n"
        "      %lhs: tensor<1024x1024xf32> {sdy.sharding = #sdy.sharding<@mesh, [{\"dev\"}, {}]>},\n"
        "      %rhs: tensor<1024x1024xf32> {sdy.sharding = #sdy.sharding<@mesh, [{\"dev\"}, {}]>}\n"
        "  ) -> tensor<1024x1024xf32> {\n"
        "    %0 = sdy.manual_computation(%lhs, %rhs) in_shardings=[<@mesh, [{\"dev\"}, {}]>, <@mesh, [{\"dev\"}, {}]>] out_shardings=[<@mesh, [{\"dev\"}, {}]>] manual_axes={\"dev\"} (%arg2: tensor<256x1024xf32>, %arg3: tensor<256x1024xf32>) {\n"
        "      %1 = stablehlo.custom_call @tpu_custom_call(%arg2, %arg3) {backend_config = \"" + json + "\",\n"
        "      kernel_name = \"add_kernel\", mhlo.frontend_attributes = {kernel_metadata = \"{}\"}, operand_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>], result_layouts = [dense<[1, 0]> : tensor<2xindex>]} : (tensor<256x1024xf32>, tensor<256x1024xf32>) -> tensor<256x1024xf32>\n"
        "      sdy.return %1 : tensor<256x1024xf32>\n"
        "    } : (tensor<1024x1024xf32>, tensor<1024x1024xf32>) -> tensor<1024x1024xf32>\n"
        "    return %0 : tensor<1024x1024xf32>\n"
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
        << "  <mosaic_mlir_file>   Mosaic MLIR source file (per-device gridded\n"
        << "                       add kernel from add_4tpu_grid.py)\n"
        << "  <libtpu_so>          TPU PJRT plugin exporting GetPjrtApi\n"
        << "  <output_blob_path>   Optional. Serialize the compiled executable.\n"
        << "  --load-blob          Load a previously serialized executable\n"
        << "                       instead of compiling from MLIR.\n";
}

int main(int argc, char** argv) {
    bool load_blob = false;
    std::vector<char*> positional;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--load-blob") == 0) {
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
        PrintUsage(argc > 0 ? argv[0] : "pjrt-demo-4tpu");
        return 1;
    }

    std::cout << "=== pjrt-demo-4tpu ===\n";
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

    // ---- 4. Check we have enough addressable devices ------------------
    {
        PJRT_Client_AddressableDevices_Args dev_args{};
        dev_args.struct_size =
            PJRT_Client_AddressableDevices_Args_STRUCT_SIZE;
        dev_args.client = client;
        if (!CheckError(api, api->PJRT_Client_AddressableDevices(&dev_args),
                        "PJRT_Client_AddressableDevices")) {
            return 1;
        }
        if (dev_args.num_addressable_devices < NUM_DEVICES) {
            std::cerr << "Need " << NUM_DEVICES
                      << " addressable devices, found "
                      << dev_args.num_addressable_devices << ".\n";
            return 1;
        }
        std::cout << "[4] Found " << dev_args.num_addressable_devices
                  << " addressable device(s)\n";
    }

    PJRT_LoadedExecutable* executable = nullptr;
    if (!load_blob) {
        // ---- 5. Load Mosaic MLIR and wrap in StableHLO ----------------
        std::string mosaic_text = LoadTextFile(mosaic_path);
        if (mosaic_text.empty()) return 1;
        std::string stablehlo = WrapMosaicInStableHlo(mosaic_text);
        std::cout << "[5] Loaded Mosaic (" << mosaic_text.size()
                  << " bytes) and wrapped into StableHLO ("
                  << stablehlo.size() << " bytes)\n";

        // ---- 6. Compile for 4 SPMD partitions --------------------------
        {
            static constexpr char kFormat[] = "mlir";
            PJRT_Program program{};
            program.struct_size = PJRT_Program_STRUCT_SIZE;
            program.code = const_cast<char*>(stablehlo.data());
            program.code_size = stablehlo.size();
            program.format = kFormat;
            program.format_size = sizeof(kFormat) - 1;  // 4, without NUL

            // Minimal CompileOptionsProto (from xla/pjrt/proto/compile_options.proto):
            //   CompileOptionsProto.executable_build_options (field 3, message)
            //     ExecutableBuildOptionsProto.num_replicas           (field 4, int64) = 1
            //     ExecutableBuildOptionsProto.num_partitions         (field 5, int64) = 4
            //     ExecutableBuildOptionsProto.use_spmd_partitioning  (field 6, bool)  = true
            //     ExecutableBuildOptionsProto.use_shardy_partitioner (field 19, bool) = true
            //
            // Wire format:
            //   0x1A 0x09       = tag(field=3, wire=length-delim), length=9
            //     0x20 0x01      = tag(field=4, wire=varint), value=1  (num_replicas)
            //     0x28 0x04      = tag(field=5, wire=varint), value=4  (num_partitions)
            //     0x30 0x01      = tag(field=6, wire=varint), value=1  (use_spmd_partitioning)
            //     0x98 0x01 0x01 = tag(field=19, wire=varint; two-byte
            //                      varint tag since 19*8=152>127), value=1
            //                      (use_shardy_partitioner)
            //
            // These must agree with the module's own mhlo.num_partitions = 4 /
            // mhlo.num_replicas = 1 attributes. use_shardy_partitioner routes
            // compilation through the Shardy pipeline that consumes the
            // sdy.mesh / sdy.sharding / sdy.manual_computation annotations;
            // without it the classic SPMD partitioner hits the xla.sdy.*
            // custom calls and fails with a RET_CHECK. This mirrors what JAX
            // sends for the shard_map-ed pallas_call.
            static constexpr unsigned char kCompileOptions[] = {
                0x1A, 0x09, 0x20, 0x01, 0x28, 0x04, 0x30, 0x01,
                0x98, 0x01, 0x01,
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
        std::cout << "[6] Compiled StableHLO -> PJRT_LoadedExecutable"
                  << " (num_partitions = " << NUM_DEVICES << ")\n";

        // ---- 6b. (optional) Serialize executable to disk --------------
        if (blob_out_path != nullptr) {
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

            PJRT_Executable_Serialize_Args ser_args{};
            ser_args.struct_size = PJRT_Executable_Serialize_Args_STRUCT_SIZE;
            ser_args.executable = plain_exe;
            if (!CheckError(api, api->PJRT_Executable_Serialize(&ser_args),
                            "PJRT_Executable_Serialize")) {
                return 1;
            }

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

            ser_args.serialized_executable_deleter(
                ser_args.serialized_executable);

            PJRT_Executable_Destroy_Args exe_destroy{};
            exe_destroy.struct_size = PJRT_Executable_Destroy_Args_STRUCT_SIZE;
            exe_destroy.executable = plain_exe;
            api->PJRT_Executable_Destroy(&exe_destroy);
        }
    } else {
        // ---- 5. Deserialize and load -----------------------------------
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

        PJRT_Executable_DeserializeAndLoad_Args deser_args{};
        deser_args.struct_size = PJRT_Executable_DeserializeAndLoad_Args_STRUCT_SIZE;
        deser_args.client = client;
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

    // ---- 7. Query the executable's device assignment -------------------
    //
    // With num_replicas = 4 the executable spans 4 devices. Buffers must be
    // uploaded to, and argument_lists ordered by, THIS device list — not the
    // client's — otherwise Execute rejects the buffers.
    PJRT_Device* exec_devices[NUM_DEVICES] = {};
    {
        PJRT_LoadedExecutable_AddressableDevices_Args dev_args{};
        dev_args.struct_size =
            PJRT_LoadedExecutable_AddressableDevices_Args_STRUCT_SIZE;
        dev_args.executable = executable;
        if (!CheckError(api,
                        api->PJRT_LoadedExecutable_AddressableDevices(&dev_args),
                        "PJRT_LoadedExecutable_AddressableDevices")) {
            return 1;
        }
        if (dev_args.num_addressable_devices != NUM_DEVICES) {
            std::cerr << "Executable spans "
                      << dev_args.num_addressable_devices
                      << " device(s), expected " << NUM_DEVICES << ".\n";
            return 1;
        }
        for (size_t d = 0; d < NUM_DEVICES; ++d) {
            exec_devices[d] = dev_args.addressable_devices[d];
        }
    }
    std::cout << "[7] Executable spans " << NUM_DEVICES << " devices\n";

    // ---- 8. Prepare host-side inputs -----------------------------------
    //
    // lhs[i] = i, rhs = all ones -> expected out[i] = i + 1. (Float is exact
    // up to 2^24, and M * N = 2^20, so no rounding.)
    std::vector<float> host_lhs(M * N), host_rhs(M * N);
    for (size_t i = 0; i < M * N; ++i) {
        host_lhs[i] = static_cast<float>(i);
        host_rhs[i] = 1.0f;
    }

    // ---- 9. Upload inputs: one lhs + rhs row shard per device ----------
    auto upload = [&](const float* data, size_t rows, size_t cols,
                      PJRT_Device* device) -> PJRT_Buffer* {
        int64_t dims[] = {static_cast<int64_t>(rows),
                          static_cast<int64_t>(cols)};

        PJRT_Client_BufferFromHostBuffer_Args args{};
        args.struct_size =
            PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
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

    PJRT_Buffer* buf_lhs[NUM_DEVICES] = {};
    PJRT_Buffer* buf_rhs[NUM_DEVICES] = {};
    for (size_t d = 0; d < NUM_DEVICES; ++d) {
        // Row shard d covers rows [d * M_SHARD, (d + 1) * M_SHARD); rows are
        // contiguous in row-major order, so the shard is a plain offset.
        // Both inputs are sharded the same way (P("dev", None)) — nothing
        // is replicated.
        const size_t shard_offset = d * M_SHARD * N;
        buf_lhs[d] = upload(host_lhs.data() + shard_offset,
                            M_SHARD, N, exec_devices[d]);
        buf_rhs[d] = upload(host_rhs.data() + shard_offset,
                            M_SHARD, N, exec_devices[d]);
        if (buf_lhs[d] == nullptr || buf_rhs[d] == nullptr) return 1;
    }
    std::cout << "[8] Uploaded " << NUM_DEVICES << " x 2 "
              << M_SHARD << "x" << N << " shards (lhs + rhs)\n";

    // ---- 10. Execute on all 4 devices -----------------------------------
    PJRT_Buffer* input_args[NUM_DEVICES][2];
    PJRT_Buffer** argument_lists[NUM_DEVICES];
    PJRT_Buffer* output_buffers[NUM_DEVICES] = {};
    PJRT_Buffer** output_lists[NUM_DEVICES];
    for (size_t d = 0; d < NUM_DEVICES; ++d) {
        input_args[d][0] = buf_lhs[d];
        input_args[d][1] = buf_rhs[d];
        argument_lists[d] = input_args[d];
        output_lists[d] = &output_buffers[d];
    }
    {
        PJRT_ExecuteOptions exec_options{};
        exec_options.struct_size = PJRT_ExecuteOptions_STRUCT_SIZE;

        PJRT_LoadedExecutable_Execute_Args exec_args{};
        exec_args.struct_size =
            PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
        exec_args.executable = executable;
        exec_args.options = &exec_options;
        exec_args.argument_lists = argument_lists;
        exec_args.num_devices = NUM_DEVICES;
        exec_args.num_args = 2;
        exec_args.output_lists = output_lists;
        exec_args.device_complete_events = nullptr;
        // nullptr = launch on every device in the executable's assignment.
        exec_args.execute_device = nullptr;

        if (!CheckError(api,
                        api->PJRT_LoadedExecutable_Execute(&exec_args),
                        "PJRT_LoadedExecutable_Execute")) {
            return 1;
        }
    }
    std::cout << "[9] Executed kernel on " << NUM_DEVICES << " devices\n";

    // ---- 11. Download output shards and stitch by rows ------------------
    std::vector<float> host_out(M * N);
    for (size_t d = 0; d < NUM_DEVICES; ++d) {
        PJRT_Buffer_ToHostBuffer_Args args{};
        args.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
        args.src = output_buffers[d];
        args.host_layout = nullptr;
        args.dst = host_out.data() + d * M_SHARD * N;
        args.dst_size = M_SHARD * N * sizeof(float);
        if (!CheckError(api, api->PJRT_Buffer_ToHostBuffer(&args),
                        "PJRT_Buffer_ToHostBuffer")) {
            return 1;
        }
        if (!AwaitEventAndDestroy(api, args.event,
                                  "PJRT_Buffer_ToHostBuffer await")) {
            return 1;
        }
    }
    std::cout << "[10] Downloaded " << NUM_DEVICES
              << " output shards (" << M_SHARD << "x" << N << " each)\n\n";

    // ---- 12. Verify ------------------------------------------------------
    //
    // Expected: out[i] == lhs[i] + rhs[i] == i + 1. Sampling rows at the
    // shard boundaries (row 0, first row of shard 1, last row) catches a
    // shard stitched into the wrong place.
    std::cout << "Result (row 0, first 4 | row " << M_SHARD
              << ", first 4 | row " << (M - 1) << ", first 4):\n  ";
    for (int i = 0; i < 4; ++i) std::cout << host_out[i] << " ";
    std::cout << "| ";
    for (int i = 0; i < 4; ++i) std::cout << host_out[M_SHARD * N + i] << " ";
    std::cout << "| ";
    for (int i = 0; i < 4; ++i) std::cout << host_out[(M - 1) * N + i] << " ";
    std::cout << "\nExpected:\n  ";
    for (int i = 0; i < 4; ++i)
        std::cout << (host_lhs[i] + host_rhs[i]) << " ";
    std::cout << "| ";
    for (int i = 0; i < 4; ++i)
        std::cout << (host_lhs[M_SHARD * N + i] + host_rhs[M_SHARD * N + i]) << " ";
    std::cout << "| ";
    for (int i = 0; i < 4; ++i)
        std::cout << (host_lhs[(M - 1) * N + i] + host_rhs[(M - 1) * N + i]) << " ";
    std::cout << "\n";

    float max_err = 0.0f;
    for (size_t i = 0; i < M * N; ++i) {
        float err = host_out[i] - (host_lhs[i] + host_rhs[i]);
        if (err < 0.0f) err = -err;
        if (err > max_err) max_err = err;
    }
    std::cout << "Max abs error over all " << M << "x" << N
              << " elements: " << max_err << "\n";

    // ---- 13. Cleanup -----------------------------------------------------
    auto destroy_buffer = [&](PJRT_Buffer* buf) {
        if (buf == nullptr) return;
        PJRT_Buffer_Destroy_Args a{};
        a.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
        a.buffer = buf;
        api->PJRT_Buffer_Destroy(&a);
    };
    for (size_t d = 0; d < NUM_DEVICES; ++d) {
        destroy_buffer(buf_lhs[d]);
        destroy_buffer(buf_rhs[d]);
        destroy_buffer(output_buffers[d]);
    }

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

    std::cout << "\n[done]\n";
    return 0;
}
