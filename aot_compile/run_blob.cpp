// ===--------------------------------------------------------------------=== //
// pjrt-aot-run: load and execute a blob produced by pjrt-aot
// ===--------------------------------------------------------------------=== //
//
// The other half of the cross-compilation story. pjrt-aot (main.cpp in this
// directory) compiles on a machine with no accelerator and stops at a blob
// on disk. This program picks the blob up on a real TPU host and runs it:
//
//   blob on disk
//       |
//       v  PJRT_Client_Create                 <-- needs real devices
//   PJRT_Client
//       |
//       v  PJRT_Executable_DeserializeAndLoad
//   PJRT_LoadedExecutable
//       |
//       v  PJRT_Client_BufferFromHostBuffer (x2, lhs/rhs, per device)
//   PJRT_Buffer* inputs
//       |
//       v  PJRT_LoadedExecutable_Execute
//   PJRT_Buffer* outputs
//       |
//       v  PJRT_Buffer_ToHostBuffer
//   host float[8][128], checked against lhs + rhs
//
// THIS PROGRAM REQUIRES A TPU. Unlike pjrt-aot it calls PJRT_Client_Create,
// which enumerates and acquires devices; on a TPU-less host it fails (or
// hangs in libtpu's discovery retry loop).
//
// It assumes the blob has the signature that pjrt-aot's wrapper emits:
//
//     (tensor<8x128xf32>, tensor<8x128xf32>) -> tensor<8x128xf32>
//
// and verifies that assumption against the executable's own metadata before
// touching any buffers, so a mismatched blob gives a clear message instead
// of a crash or silent garbage.
//
// Compile-time dependency: xla/pjrt/c/pjrt_c_api.h (plain C).
// Runtime dependency:      libtpu.so, dlopen'd from the path given on argv.
// ===--------------------------------------------------------------------=== //

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "xla/pjrt/c/pjrt_c_api.h"

namespace {

// The shape pjrt-aot's StableHLO wrapper is written for.
constexpr int64_t kRows = 8;
constexpr int64_t kCols = 128;
constexpr size_t kNumElements = static_cast<size_t>(kRows * kCols);  // 1024
constexpr size_t kNumArgs = 2;

// ===--------------------------------------------------------------------=== //
// Error helper
// ===--------------------------------------------------------------------=== //

bool CheckError(const PJRT_Api* api, PJRT_Error* error, const char* context) {
    if (error == nullptr) return true;

    PJRT_Error_Message_Args msg_args{};
    msg_args.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE;
    msg_args.error = error;
    api->PJRT_Error_Message(&msg_args);

    std::cerr << context << ": "
              << std::string(msg_args.message, msg_args.message_size) << "\n";

    PJRT_Error_Destroy_Args destroy_args{};
    destroy_args.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
    destroy_args.error = error;
    api->PJRT_Error_Destroy(&destroy_args);
    return false;
}

bool AwaitEventAndDestroy(const PJRT_Api* api, PJRT_Event* event,
                          const char* context) {
    PJRT_Event_Await_Args await_args{};
    await_args.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
    await_args.event = event;
    const bool ok =
        CheckError(api, api->PJRT_Event_Await(&await_args), context);

    // Destroy the event even if the await failed, or we leak it.
    PJRT_Event_Destroy_Args destroy_args{};
    destroy_args.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
    destroy_args.event = event;
    api->PJRT_Event_Destroy(&destroy_args);
    return ok;
}

// ===--------------------------------------------------------------------=== //
// File IO
// ===--------------------------------------------------------------------=== //

bool ReadBinaryFile(const std::string& path, std::vector<char>* out) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        std::cerr << "Failed to open " << path << " for reading\n";
        return false;
    }
    const std::streamsize size = in.tellg();
    if (size <= 0) {
        std::cerr << path << " is empty\n";
        return false;
    }
    in.seekg(0, std::ios::beg);
    out->resize(static_cast<size_t>(size));
    if (!in.read(out->data(), size)) {
        std::cerr << "Failed to read " << path << "\n";
        return false;
    }
    return true;
}

// ===--------------------------------------------------------------------=== //
// PJRT plugin loading
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
        dlclose(handle);
        return nullptr;
    }
    // Intentionally leaked: the PJRT_Api table points into this library.
    return get_api();
}

}  // namespace

// ===--------------------------------------------------------------------=== //
// main
// ===--------------------------------------------------------------------=== //

void PrintUsage(const char* argv0) {
    std::cerr
        << "Usage:\n"
        << "  " << argv0 << " <blob> <libtpu_so> [flags]\n\n"
        << "Runs a serialized PJRT executable written by pjrt-aot.\n"
        << "REQUIRES A TPU: this creates a PJRT client, unlike pjrt-aot.\n\n"
        << "Arguments:\n"
        << "  <blob>        Serialized executable from pjrt-aot.\n"
        << "  <libtpu_so>   TPU PJRT plugin exporting GetPjrtApi. Must be the\n"
        << "                SAME libtpu build that produced the blob — PJRT\n"
        << "                serialization is not portable across versions.\n\n"
        << "Flags:\n"
        << "  --print=N     Print the first N result elements (default 8).\n"
        << "  --no-verify   Skip the lhs+rhs correctness check.\n";
}

int main(int argc, char** argv) {
    int print_count = 8;
    bool verify = true;
    std::vector<char*> positional;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--no-verify") {
            verify = false;
        } else if (arg.rfind("--print=", 0) == 0) {
            print_count = std::atoi(arg.substr(8).c_str());
        } else if (arg.rfind("--", 0) == 0) {
            std::cerr << "Unknown flag: " << arg << "\n\n";
            PrintUsage(argv[0]);
            return 1;
        } else {
            positional.push_back(argv[i]);
        }
    }
    if (positional.size() != 2) {
        PrintUsage(argc > 0 ? argv[0] : "pjrt-aot-run");
        return 1;
    }
    const char* blob_path = positional[0];
    const char* libtpu_path = positional[1];

    std::cout << std::unitbuf;
    std::cout << "=== pjrt-aot-run (deserialize + execute) ===\n"
              << "Blob:        " << blob_path << "\n"
              << "PJRT plugin: " << libtpu_path << "\n\n";

    // ---- 1. Read the blob ---------------------------------------------
    //
    // Done before loading the plugin: no point spending seconds dlopen'ing
    // a 650 MB library to then discover the path was a typo.
    std::vector<char> blob;
    if (!ReadBinaryFile(blob_path, &blob)) return 1;
    std::cout << "[1] Read " << blob.size() << " bytes from " << blob_path
              << "\n";

    // ---- 2. Load PJRT plugin ------------------------------------------
    const PJRT_Api* api = LoadTpuPjrtApi(libtpu_path);
    if (api == nullptr) return 1;
    std::cout << "[2] Loaded PJRT plugin (C API "
              << api->pjrt_api_version.major_version << "."
              << api->pjrt_api_version.minor_version << ")\n";

    // ---- 3. Initialize plugin -----------------------------------------
    //
    // Note the contrast with pjrt-aot: we do NOT set TPU_SKIP_MDS_QUERY
    // here. On a real TPU host the metadata query is what discovers the
    // local accelerator, so suppressing it would be counterproductive.
    {
        PJRT_Plugin_Initialize_Args init_args{};
        init_args.struct_size = PJRT_Plugin_Initialize_Args_STRUCT_SIZE;
        if (!CheckError(api, api->PJRT_Plugin_Initialize(&init_args),
                        "PJRT_Plugin_Initialize")) {
            return 1;
        }
    }
    std::cout << "[3] Initialized plugin\n";

    // ---- 4. Create client ---------------------------------------------
    PJRT_Client* client = nullptr;
    {
        PJRT_Client_Create_Args create_args{};
        create_args.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
        if (!CheckError(api, api->PJRT_Client_Create(&create_args),
                        "PJRT_Client_Create")) {
            std::cerr << "\nThis program needs real TPU devices. If you are "
                         "on a host without a\nTPU, you can only compile "
                         "(see pjrt-aot), not execute.\n";
            return 1;
        }
        client = create_args.client;
    }
    std::cout << "[4] Created PJRT client\n";

    // ---- 5. Deserialize and load --------------------------------------
    //
    // The counterpart to pjrt-aot's PJRT_Executable_Serialize. This is where
    // a blob built for the wrong TPU generation, or by a different libtpu
    // build, gets rejected.
    //
    // overridden_serialized_compile_options is left null, so the options
    // baked in at AOT time (replica/partition counts, any device assignment)
    // are the ones used.
    PJRT_LoadedExecutable* loaded = nullptr;
    {
        PJRT_Executable_DeserializeAndLoad_Args args{};
        args.struct_size =
            PJRT_Executable_DeserializeAndLoad_Args_STRUCT_SIZE;
        args.client = client;
        args.serialized_executable = blob.data();
        args.serialized_executable_size = blob.size();
        args.overridden_serialized_compile_options = nullptr;
        args.overridden_serialized_compile_options_size = 0;
        if (!CheckError(api, api->PJRT_Executable_DeserializeAndLoad(&args),
                        "PJRT_Executable_DeserializeAndLoad")) {
            std::cerr << "\nA blob must be loaded by the same platform and "
                         "libtpu version that\nproduced it, and the TPU "
                         "generation must match the topology it was\n"
                         "compiled for.\n";
            return 1;
        }
        loaded = args.loaded_executable;
    }
    std::cout << "[5] Deserialized blob -> PJRT_LoadedExecutable\n";

    // ---- 6. Inspect what we loaded, and sanity-check our assumptions --
    //
    // The blob carries its own metadata; read it rather than assuming, so a
    // blob compiled with a different signature or --partitions fails with a
    // message instead of undefined behaviour below.
    size_t num_outputs = 0;
    {
        PJRT_Executable_NumOutputs_Args out_args{};
        out_args.struct_size = PJRT_Executable_NumOutputs_Args_STRUCT_SIZE;

        // The output/replica/partition queries take a plain PJRT_Executable,
        // so get the un-loaded view of the loaded executable first.
        PJRT_LoadedExecutable_GetExecutable_Args get_args{};
        get_args.struct_size =
            PJRT_LoadedExecutable_GetExecutable_Args_STRUCT_SIZE;
        get_args.loaded_executable = loaded;
        if (!CheckError(api,
                        api->PJRT_LoadedExecutable_GetExecutable(&get_args),
                        "PJRT_LoadedExecutable_GetExecutable")) {
            return 1;
        }
        PJRT_Executable* plain = get_args.executable;

        PJRT_Executable_Name_Args name_args{};
        name_args.struct_size = PJRT_Executable_Name_Args_STRUCT_SIZE;
        name_args.executable = plain;
        if (CheckError(api, api->PJRT_Executable_Name(&name_args),
                       "PJRT_Executable_Name")) {
            std::cout << "    name:           "
                      << std::string(name_args.executable_name,
                                     name_args.executable_name_size)
                      << "\n";
        }

        PJRT_Executable_NumReplicas_Args rep_args{};
        rep_args.struct_size = PJRT_Executable_NumReplicas_Args_STRUCT_SIZE;
        rep_args.executable = plain;
        if (CheckError(api, api->PJRT_Executable_NumReplicas(&rep_args),
                       "PJRT_Executable_NumReplicas")) {
            std::cout << "    num_replicas:   " << rep_args.num_replicas
                      << "\n";
        }

        PJRT_Executable_NumPartitions_Args part_args{};
        part_args.struct_size = PJRT_Executable_NumPartitions_Args_STRUCT_SIZE;
        part_args.executable = plain;
        if (CheckError(api, api->PJRT_Executable_NumPartitions(&part_args),
                       "PJRT_Executable_NumPartitions")) {
            std::cout << "    num_partitions: " << part_args.num_partitions
                      << "\n";
        }

        out_args.executable = plain;
        if (!CheckError(api, api->PJRT_Executable_NumOutputs(&out_args),
                        "PJRT_Executable_NumOutputs")) {
            return 1;
        }
        num_outputs = out_args.num_outputs;
        std::cout << "    num_outputs:    " << num_outputs << "\n";

        PJRT_Executable_Destroy_Args destroy{};
        destroy.struct_size = PJRT_Executable_Destroy_Args_STRUCT_SIZE;
        destroy.executable = plain;
        api->PJRT_Executable_Destroy(&destroy);
    }
    if (num_outputs != 1) {
        std::cerr << "This runner handles single-output executables only, but "
                     "the blob has "
                  << num_outputs
                  << ".\nIt expects pjrt-aot's signature:\n"
                     "  (tensor<8x128xf32>, tensor<8x128xf32>) -> "
                     "tensor<8x128xf32>\n";
        return 1;
    }

    // ---- 7. Which devices will this run on? ---------------------------
    //
    // Ask the executable, not the client: for a blob compiled with a static
    // device assignment these are the assigned devices, and the count is the
    // `num_devices` that Execute demands.
    std::vector<PJRT_Device*> devices;
    {
        PJRT_LoadedExecutable_AddressableDevices_Args args{};
        args.struct_size =
            PJRT_LoadedExecutable_AddressableDevices_Args_STRUCT_SIZE;
        args.executable = loaded;
        if (!CheckError(api,
                        api->PJRT_LoadedExecutable_AddressableDevices(&args),
                        "PJRT_LoadedExecutable_AddressableDevices")) {
            return 1;
        }
        if (args.num_addressable_devices == 0) {
            std::cerr << "Executable has no addressable devices.\n";
            return 1;
        }
        devices.assign(args.addressable_devices,
                       args.addressable_devices +
                           args.num_addressable_devices);
    }
    const size_t num_devices = devices.size();
    std::cout << "[6] Executable is addressable on " << num_devices
              << " device(s)\n";

    // ---- 8. Host inputs -----------------------------------------------
    std::vector<float> host_lhs(kNumElements), host_rhs(kNumElements);
    for (size_t i = 0; i < kNumElements; ++i) {
        host_lhs[i] = static_cast<float>(i);
        host_rhs[i] = 1.0f;
    }

    // ---- 9. Upload inputs, once per device ----------------------------
    //
    // Execute wants a [num_devices][num_args] array, and each device needs
    // its own buffers — a PJRT_Buffer belongs to one device. Every device
    // gets the same values here, which keeps the expected result identical
    // across devices and makes the check below meaningful for all of them.
    const int64_t dims[] = {kRows, kCols};

    auto upload = [&](const std::vector<float>& host_data,
                      PJRT_Device* device) -> PJRT_Buffer* {
        PJRT_Client_BufferFromHostBuffer_Args args{};
        args.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
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
        // host_data must stay valid until the H->D copy is done. It outlives
        // this call, but await anyway so the semantics above are honoured.
        if (!AwaitEventAndDestroy(api, args.done_with_host_buffer,
                                  "done_with_host_buffer await")) {
            return nullptr;
        }
        return args.buffer;
    };

    // Flat [device][arg] storage, plus the pointer-to-pointer views Execute
    // expects. Kept alive until after the results are read back.
    std::vector<PJRT_Buffer*> input_storage(num_devices * kNumArgs, nullptr);
    std::vector<PJRT_Buffer*> output_storage(num_devices * num_outputs,
                                             nullptr);
    std::vector<PJRT_Buffer* const*> argument_lists(num_devices);
    std::vector<PJRT_Buffer**> output_lists(num_devices);

    bool upload_ok = true;
    for (size_t d = 0; d < num_devices && upload_ok; ++d) {
        input_storage[d * kNumArgs + 0] = upload(host_lhs, devices[d]);
        input_storage[d * kNumArgs + 1] = upload(host_rhs, devices[d]);
        upload_ok = input_storage[d * kNumArgs + 0] != nullptr &&
                    input_storage[d * kNumArgs + 1] != nullptr;
    }
    for (size_t d = 0; d < num_devices; ++d) {
        argument_lists[d] = &input_storage[d * kNumArgs];
        output_lists[d] = &output_storage[d * num_outputs];
    }

    // A single cleanup path, so an early exit below still releases buffers.
    auto destroy_buffers = [&](std::vector<PJRT_Buffer*>& buffers) {
        for (PJRT_Buffer* buf : buffers) {
            if (buf == nullptr) continue;
            PJRT_Buffer_Destroy_Args a{};
            a.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
            a.buffer = buf;
            api->PJRT_Buffer_Destroy(&a);
        }
        buffers.assign(buffers.size(), nullptr);
    };

    if (!upload_ok) {
        destroy_buffers(input_storage);
        return 1;
    }
    std::cout << "[7] Uploaded " << (num_devices * kNumArgs) << " x "
              << kNumElements << " floats (" << kNumArgs << " args x "
              << num_devices << " device(s))\n";

    // ---- 10. Execute ---------------------------------------------------
    {
        PJRT_ExecuteOptions exec_options{};
        exec_options.struct_size = PJRT_ExecuteOptions_STRUCT_SIZE;

        PJRT_LoadedExecutable_Execute_Args exec_args{};
        exec_args.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
        exec_args.executable = loaded;
        exec_args.options = &exec_options;
        exec_args.argument_lists = argument_lists.data();
        exec_args.num_devices = num_devices;
        exec_args.num_args = kNumArgs;
        exec_args.output_lists = output_lists.data();
        exec_args.device_complete_events = nullptr;
        exec_args.execute_device = nullptr;  // use the compiled-in devices

        if (!CheckError(api, api->PJRT_LoadedExecutable_Execute(&exec_args),
                        "PJRT_LoadedExecutable_Execute")) {
            destroy_buffers(input_storage);
            return 1;
        }
    }
    std::cout << "[8] Executed\n";

    // ---- 11. Download and check ----------------------------------------
    //
    // ToHostBuffer's event also serves as the completion signal for the
    // execution that produced the buffer, which is why passing
    // device_complete_events = nullptr above is safe.
    int exit_code = 0;
    for (size_t d = 0; d < num_devices; ++d) {
        std::vector<float> host_out(kNumElements, 0.0f);
        PJRT_Buffer_ToHostBuffer_Args args{};
        args.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
        args.src = output_storage[d * num_outputs];
        args.host_layout = nullptr;
        args.dst = host_out.data();
        args.dst_size = host_out.size() * sizeof(float);
        if (!CheckError(api, api->PJRT_Buffer_ToHostBuffer(&args),
                        "PJRT_Buffer_ToHostBuffer")) {
            exit_code = 1;
            break;
        }
        if (!AwaitEventAndDestroy(api, args.event,
                                  "PJRT_Buffer_ToHostBuffer await")) {
            exit_code = 1;
            break;
        }

        std::cout << "\ndevice[" << d << "] result (first "
                  << print_count << "):\n  ";
        for (int i = 0; i < print_count && i < static_cast<int>(kNumElements);
             ++i) {
            std::cout << host_out[i] << " ";
        }
        std::cout << "\n";

        if (verify) {
            size_t mismatches = 0;
            size_t first_bad = 0;
            for (size_t i = 0; i < kNumElements; ++i) {
                if (host_out[i] != host_lhs[i] + host_rhs[i]) {
                    if (mismatches == 0) first_bad = i;
                    ++mismatches;
                }
            }
            if (mismatches == 0) {
                std::cout << "  OK: all " << kNumElements
                          << " elements equal lhs + rhs\n";
            } else {
                std::cout << "  MISMATCH: " << mismatches << "/"
                          << kNumElements << " elements differ; first at ["
                          << first_bad << "] got " << host_out[first_bad]
                          << ", expected "
                          << (host_lhs[first_bad] + host_rhs[first_bad])
                          << "\n";
                exit_code = 1;
            }
        }
    }

    // ---- 12. Cleanup ---------------------------------------------------
    destroy_buffers(input_storage);
    destroy_buffers(output_storage);
    {
        PJRT_LoadedExecutable_Destroy_Args a{};
        a.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
        a.executable = loaded;
        api->PJRT_LoadedExecutable_Destroy(&a);
    }
    {
        PJRT_Client_Destroy_Args a{};
        a.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
        a.client = client;
        api->PJRT_Client_Destroy(&a);
    }

    std::cout << (exit_code == 0 ? "\n[done]\n" : "\n[failed]\n");
    return exit_code;
}
