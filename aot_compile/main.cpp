// ===--------------------------------------------------------------------=== //
// pjrt-aot: cross-compile a Mosaic kernel for TPU on a host with NO TPU
// ===--------------------------------------------------------------------=== //
//
// vector_add_newer_version/main.cpp needs a real TPU: PJRT_Client_Create
// enumerates and acquires devices, so it fails on a machine that only has
// libtpu.so and no accelerator. The ahead-of-time (AOT) path avoids the
// client entirely:
//
//   Mosaic MLIR text file
//       |
//       v  WrapMosaicInStableHlo
//   StableHLO with @tpu_custom_call(... backend_config = <mosaic base64>)
//       |
//       |  PJRT_TopologyDescription_Create("v4:2x2x1")   <-- no devices needed
//       v  PJRT_Compile(topology, program, options)      <-- client = nullptr
//   PJRT_Executable                     (un-loaded; cannot be executed here)
//       |
//       v  PJRT_Executable_Serialize
//   blob on disk
//
// The blob is then moved to a real TPU host, where
// PJRT_Executable_DeserializeAndLoad turns it into a PJRT_LoadedExecutable
// (see aot_compile/run_blob.cpp).
//
// The one thing the compiler cannot infer without a device is *which* TPU to
// target, hence the mandatory <topology_name> argument. libtpu accepts names
// of the form "<generation>:<mesh>", e.g.
//
//     v6e:1x1x1    v6e:2x2
//
// One wrinkle makes this work at all: PJRT_Plugin_Initialize is still
// required (it runs libtpu's absl flag parsing — without it
// PJRT_TopologyDescription_Create CHECK-fails in
// InitRequestedTpuPlatformType with "ParseCommandLine is not invoked yet"),
// but by default it queries the GCE metadata server at 169.254.169.254:80
// to discover the local TPU. Off GCE that address is unroutable and libtpu
// retries once a second *forever* — the process parks in nanosleep at 0%
// CPU and never returns. TPU_SKIP_MDS_QUERY=1 skips that query; we set it
// ourselves below so the demo just works. The leftover
// "could not determine TPU accelerator type" warnings on stderr are
// expected and harmless: AOT compilation gets its target from the
// topology name, not from the machine.
//
// Compile-time dependency: xla/pjrt/c/pjrt_c_api.h (plain C).
// Runtime dependency:      libtpu.so, dlopen'd from the path given on argv.
// ===--------------------------------------------------------------------=== //

#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "xla/pjrt/c/pjrt_c_api.h"

namespace {

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

bool WriteBinaryFile(const std::string& path, const char* data, size_t size) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        std::cerr << "Failed to open " << path << " for writing\n";
        return false;
    }
    out.write(data, static_cast<std::streamsize>(size));
    return out.good();
}

// ===--------------------------------------------------------------------=== //
// StableHLO wrapping (identical to vector_add_newer_version)
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
        if (i + 1 < data.size()) v |= static_cast<uint8_t>(data[i + 1]) << 8;
        out += tbl[(v >> 18) & 63];
        out += tbl[(v >> 12) & 63];
        out += (i + 1 < data.size()) ? tbl[(v >> 6) & 63] : '=';
        out += '=';
    }
    return out;
}

// {"custom_call_config": {"body": "<base64 mosaic>"}} — the shape produced by
// jaxlib's CustomCallBackendConfig.to_json().
std::string WrapMosaicInStableHlo(const std::string& mosaic_text) {
    const std::string body_b64 = Base64Encode(mosaic_text);
    const std::string json =
        std::string("{\\\"custom_call_config\\\": {\\\"body\\\": \\\"") +
        body_b64 + "\\\"," +
        " \\\"needs_layout_passes\\\": true,"
        " \\\"shape_invariant_numerics\\\": false}}";

    return
        "module @wrapper {\n"
        "  func.func @main(\n"
        "      %lhs: tensor<8x128xf32>,\n"
        "      %rhs: tensor<8x128xf32>\n"
        "  ) -> tensor<8x128xf32> {\n"
        "    %out = stablehlo.custom_call @tpu_custom_call(%lhs, %rhs) {\n"
        "      backend_config = \"" + json + "\",\n"
        "      operand_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>],\n"
        "      result_layouts = [dense<[1, 0]> : tensor<2xindex>]"
        "    } : (tensor<8x128xf32>, tensor<8x128xf32>) -> tensor<8x128xf32>\n"
        "    return %out : tensor<8x128xf32>\n"
        "  }\n"
        "}\n";
}

// A pure-StableHLO program with no Mosaic custom_call. Useful to separate
// "AOT/topology path broken" from "Mosaic kernel broken".
std::string BuildTrivialStableHlo() {
    return
        "module @trivial {\n"
        "  func.func @main(\n"
        "      %lhs: tensor<8x128xf32>,\n"
        "      %rhs: tensor<8x128xf32>\n"
        "  ) -> tensor<8x128xf32> {\n"
        "    %out = stablehlo.add %lhs, %rhs : tensor<8x128xf32>\n"
        "    return %out : tensor<8x128xf32>\n"
        "  }\n"
        "}\n";
}

// ===--------------------------------------------------------------------=== //
// Minimal protobuf writer
// ===--------------------------------------------------------------------=== //
//
// The other demos in this repo hardcode the CompileOptionsProto bytes. Here
// replica/partition counts and the device assignment depend on the topology,
// so we encode them properly. Only the two wire types we need:
//   wire 0 = varint, wire 2 = length-delimited.

void PbVarint(std::string& out, uint64_t v) {
    while (v >= 0x80) {
        out += static_cast<char>((v & 0x7F) | 0x80);
        v >>= 7;
    }
    out += static_cast<char>(v);
}

void PbTag(std::string& out, uint32_t field, uint32_t wire) {
    PbVarint(out, (static_cast<uint64_t>(field) << 3) | wire);
}

void PbInt64(std::string& out, uint32_t field, int64_t v) {
    // proto3 omits zero-valued scalars; keep that convention so we produce
    // byte-identical output to a real serializer.
    if (v == 0) return;
    PbTag(out, field, 0);
    PbVarint(out, static_cast<uint64_t>(v));
}

void PbMessage(std::string& out, uint32_t field, const std::string& body) {
    if (body.empty()) return;
    PbTag(out, field, 2);
    PbVarint(out, body.size());
    out += body;
}

// xla.DeviceAssignmentProto: replica_count=1, computation_count=2,
// computation_devices=3 (each ComputationDevice.replica_device_ids=1).
//
// Laid out as num_partitions computations, each holding num_replicas device
// ids, filled from 0..(num_replicas*num_partitions - 1).
std::string BuildDeviceAssignment(int64_t num_replicas,
                                  int64_t num_partitions) {
    std::string out;
    PbInt64(out, 1, num_replicas);
    PbInt64(out, 2, num_partitions);
    int64_t next_device = 0;
    for (int64_t c = 0; c < num_partitions; ++c) {
        std::string comp;
        for (int64_t r = 0; r < num_replicas; ++r) {
            // repeated int64 is not packed in this proto's generated code path
            // for older readers, but proto3 packs it; use packed form.
            PbVarint(comp, static_cast<uint64_t>(next_device++));
        }
        std::string comp_msg;
        PbTag(comp_msg, 1, 2);
        PbVarint(comp_msg, comp.size());
        comp_msg += comp;
        PbMessage(out, 3, comp_msg);
    }
    return out;
}

// xla.CompileOptionsProto with just executable_build_options (field 3) set:
//   num_replicas          = 4
//   num_partitions        = 5
//   use_spmd_partitioning = 6
//   device_assignment     = 9
std::string BuildCompileOptions(int64_t num_replicas, int64_t num_partitions,
                                bool use_spmd, bool static_assignment) {
    std::string build_opts;
    PbInt64(build_opts, 4, num_replicas);
    PbInt64(build_opts, 5, num_partitions);
    if (use_spmd) {
        PbTag(build_opts, 6, 0);
        PbVarint(build_opts, 1);
    }
    if (static_assignment) {
        PbMessage(build_opts, 9,
                  BuildDeviceAssignment(num_replicas, num_partitions));
    }

    std::string opts;
    PbMessage(opts, 3, build_opts);
    return opts;
}

// ===--------------------------------------------------------------------=== //
// Bounds parsing ("2x2x1" -> {2, 2, 1})
// ===--------------------------------------------------------------------=== //
//
// libtpu wants exactly three integers: "GetTpuTopologyDescription:
// chips_per_host_bounds must be a list of 3 integers."

bool ParseBounds(const std::string& text, int64_t out[3]) {
    size_t pos = 0;
    for (int i = 0; i < 3; ++i) {
        size_t consumed = 0;
        try {
            out[i] = std::stoll(text.substr(pos), &consumed);
        } catch (const std::exception&) {
            return false;
        }
        if (consumed == 0 || out[i] < 0) return false;
        pos += consumed;
        if (i < 2) {
            if (pos >= text.size() ||
                (text[pos] != 'x' && text[pos] != 'X')) {
                return false;
            }
            ++pos;
        }
    }
    return pos == text.size();
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

// ===--------------------------------------------------------------------=== //
// PJRT_NamedValue pretty-printing (topology / device attributes)
// ===--------------------------------------------------------------------=== //

void PrintNamedValues(const PJRT_NamedValue* values, size_t count,
                      const char* indent) {
    for (size_t i = 0; i < count; ++i) {
        const PJRT_NamedValue& v = values[i];
        std::cout << indent << std::string(v.name, v.name_size) << " = ";
        switch (v.type) {
            case PJRT_NamedValue_kString:
                std::cout << "\"" << std::string(v.string_value, v.value_size)
                          << "\"";
                break;
            case PJRT_NamedValue_kInt64:
                std::cout << v.int64_value;
                break;
            case PJRT_NamedValue_kInt64List: {
                std::cout << "[";
                for (size_t j = 0; j < v.value_size; ++j) {
                    if (j != 0) std::cout << ", ";
                    std::cout << v.int64_array_value[j];
                }
                std::cout << "]";
                break;
            }
            case PJRT_NamedValue_kFloat:
                std::cout << v.float_value;
                break;
            case PJRT_NamedValue_kBool:
                std::cout << (v.bool_value ? "true" : "false");
                break;
            default:
                std::cout << "<unknown type " << v.type << ">";
                break;
        }
        std::cout << "\n";
    }
}

}  // namespace

// ===--------------------------------------------------------------------=== //
// main
// ===--------------------------------------------------------------------=== //

void PrintUsage(const char* argv0) {
    std::cerr
        << "Usage:\n"
        << "  " << argv0
        << " [<mosaic_mlir_file>] <libtpu_so> <topology_name> <out_blob>"
           " [flags]\n\n"
        << "Arguments:\n"
        << "  <mosaic_mlir_file>  Mosaic MLIR source, e.g.\n"
        << "                      vector_add_newer_version/"
           "vector_add_new.mlir.\n"
        << "                      Omit this argument when passing --trivial.\n"
        << "  <libtpu_so>         TPU PJRT plugin exporting GetPjrtApi.\n"
        << "  <topology_name>     Target TPU. Either \"<gen>:<mesh>\":\n"
        << "                        v3:2x2x1   v4:2x2x1  v4:2x4x4\n"
        << "                        v5e:2x2    v5p:2x2x1 v6e:2x2\n"
        << "                      or a slice alias, which libtpu resolves\n"
        << "                      against its own mapping table:\n"
        << "                        v4-8  v4-64  v5e-4\n"
        << "  <out_blob>          Path for the serialized PJRT executable.\n\n"
        << "Flags:\n"
        << "  --trivial              Compile a plain stablehlo.add instead of\n"
        << "                         the Mosaic custom_call. Isolates the AOT\n"
        << "                         path from the kernel.\n"
        << "  --replicas=N           num_replicas   (default 1)\n"
        << "  --partitions=N         num_partitions (default 1)\n"
        << "  --spmd                 Set use_spmd_partitioning.\n"
        << "  --device-assignment    Emit a static device assignment for\n"
        << "                         (replicas x partitions).\n"
        << "  --chips-per-host=XxYxZ Override chips_per_host_bounds. Needed\n"
        << "                         for a mesh smaller than one host: a\n"
        << "                         single chip is v6e:1x1 with\n"
        << "                         --chips-per-host=1x1x1.\n"
        << "  --num-slices=N         Number of MegaScale slices; multiplies\n"
        << "                         the mesh (v6e:2x2 --num-slices=2 is 8\n"
        << "                         devices). Inspection only: compiling a\n"
        << "                         multi-slice blob also needs a MegaScale\n"
        << "                         id via serialized_multi_slice_config,\n"
        << "                         which this tool does not build.\n"
        << "  --wrap=XxYxZ           Make an axis a ring instead of a line:\n"
        << "                         1 joins the last chip on that axis back\n"
        << "                         to the first, 0 leaves it open. Shortens\n"
        << "                         the path collectives take. The default\n"
        << "                         is 0x0x0 for every mesh, full pods\n"
        << "                         included, so pass it if your real target\n"
        << "                         wraps.\n"
        << "  --topology-out=PATH    Also serialize the TopologyDescription.\n"
        << "  --list-devices         Print every device description in the\n"
        << "                         topology.\n"
        << "\nDiagnostic flags — both make the run fail on purpose, and exist\n"
        << "to show why the two workarounds below are needed:\n"
        << "  --skip-plugin-init     Don't call PJRT_Plugin_Initialize.\n"
        << "                         Topology creation then CHECK-fails with\n"
        << "                         \"ParseCommandLine is not invoked yet\".\n"
        << "  --no-skip-mds-query    Don't set TPU_SKIP_MDS_QUERY=1. Plugin\n"
        << "                         init then retries the GCE metadata\n"
        << "                         server forever and never returns.\n";
}

int main(int argc, char** argv) {
    bool trivial = false;
    bool spmd = false;
    bool static_assignment = false;
    bool list_devices = false;
    bool init_plugin = true;
    bool skip_mds_query = true;
    int64_t num_replicas = 1;
    int64_t num_partitions = 1;
    std::string topology_out;
    std::string chips_per_host_bounds_arg;
    std::string wrap_arg;
    int64_t num_slices = 0;  // 0 = leave the option out entirely
    std::vector<char*> positional;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--trivial") {
            trivial = true;
        } else if (arg == "--spmd") {
            spmd = true;
        } else if (arg == "--device-assignment") {
            static_assignment = true;
        } else if (arg == "--list-devices") {
            list_devices = true;
        } else if (arg == "--skip-plugin-init") {
            init_plugin = false;
        } else if (arg == "--no-skip-mds-query") {
            skip_mds_query = false;
        } else if (arg.rfind("--replicas=", 0) == 0) {
            num_replicas = std::stoll(arg.substr(11));
        } else if (arg.rfind("--partitions=", 0) == 0) {
            num_partitions = std::stoll(arg.substr(13));
        } else if (arg.rfind("--topology-out=", 0) == 0) {
            topology_out = arg.substr(15);
        } else if (arg.rfind("--chips-per-host=", 0) == 0) {
            chips_per_host_bounds_arg = arg.substr(17);
        } else if (arg.rfind("--wrap=", 0) == 0) {
            wrap_arg = arg.substr(7);
        } else if (arg.rfind("--num-slices=", 0) == 0) {
            num_slices = std::stoll(arg.substr(13));
        } else if (arg.rfind("--", 0) == 0) {
            std::cerr << "Unknown flag: " << arg << "\n\n";
            PrintUsage(argv[0]);
            return 1;
        } else {
            positional.push_back(argv[i]);
        }
    }

    // --trivial has no kernel to read, so the MLIR path may be omitted.
    const size_t expected_positional = trivial ? 3 : 4;
    if (positional.size() != expected_positional) {
        PrintUsage(argc > 0 ? argv[0] : "pjrt-aot");
        return 1;
    }
    size_t next = 0;
    const char* mosaic_path = trivial ? nullptr : positional[next++];
    const char* libtpu_path = positional[next++];
    const std::string topology_name = positional[next++];
    const char* blob_out_path = positional[next++];

    std::cout << "=== pjrt-aot (cross-compile, no TPU required) ===\n"
              << "PJRT plugin:  " << libtpu_path << "\n"
              << "Topology:     " << topology_name << "\n"
              << "Program:      "
              << (trivial ? "<trivial stablehlo.add>" : mosaic_path) << "\n"
              << "Output blob:  " << blob_out_path << "\n"
              << "Replicas x partitions: " << num_replicas << " x "
              << num_partitions << "\n\n";

    // ---- 0. Defuse libtpu's TPU discovery -----------------------------
    //
    // Must happen before the plugin reads its environment. setenv with
    // overwrite=0 so an explicit setting from the caller's shell wins.
    if (skip_mds_query) {
        setenv("TPU_SKIP_MDS_QUERY", "1", /*overwrite=*/0);
        std::cout << "[0] TPU_SKIP_MDS_QUERY=1 (skip the GCE metadata "
                     "server probe)\n";
    }

    // ---- 1. Load PJRT plugin ------------------------------------------
    const PJRT_Api* api = LoadTpuPjrtApi(libtpu_path);
    if (api == nullptr) return 1;
    std::cout << std::unitbuf;
    std::cout << "[1] Loaded PJRT plugin (C API "
              << api->pjrt_api_version.major_version << "."
              << api->pjrt_api_version.minor_version << ")\n";

    // The AOT entry points are optional in the C API. A plugin that does not
    // implement them cannot cross-compile, and there is no fallback that
    // avoids touching devices — so bail out with a clear message.
    if (api->PJRT_TopologyDescription_Create == nullptr ||
        api->PJRT_Compile == nullptr) {
        std::cerr << "Plugin does not implement the AOT entry points "
                     "(PJRT_TopologyDescription_Create / PJRT_Compile).\n";
        return 1;
    }

    // ---- 2. Initialize plugin -----------------------------------------
    //
    // Required even though we never create a client: initialization is what
    // runs libtpu's absl flag parsing, and PJRT_TopologyDescription_Create
    // reads a flag (the requested TPU platform type).
    //
    // TPU_SKIP_MDS_QUERY is set above; without it this call never returns.
    if (init_plugin) {
        PJRT_Plugin_Initialize_Args init_args{};
        init_args.struct_size = PJRT_Plugin_Initialize_Args_STRUCT_SIZE;
        if (!CheckError(api, api->PJRT_Plugin_Initialize(&init_args),
                        "PJRT_Plugin_Initialize")) {
            return 1;
        }
        std::cout << "[2] Initialized plugin\n";
    } else {
        std::cout << "[2] Skipped PJRT_Plugin_Initialize (--skip-plugin-init);"
                     " expect a CHECK failure in step [3]\n";
    }

    // ---- 3. Create the TopologyDescription ----------------------------
    //
    // This is the substitute for PJRT_Client_Create. It describes the target
    // machine from a name string instead of discovering it, which is exactly
    // what makes compiling on a TPU-less host possible.
    PJRT_TopologyDescription* topology = nullptr;
    {
        // The mesh in the topology name must be divisible by
        // chips_per_host_bounds, which defaults to a full host (2x2x1 on v4,
        // v5e and v6e). So "v6e:1x1" is rejected out of the box:
        //
        //   Topology layout "v6e:1x1" is not divisible by the given (or
        //   default) chips_per_host_bounds "2x2x1"
        //
        // Overriding it here is how you describe a single chip, or any slice
        // smaller than one host.
        //
        // libtpu 0.0.47 accepts exactly three create_options names, verified
        // by probing; everything else fails with "Unexpected arguments: X".
        //
        //   chips_per_host_bounds  Int64List, exactly 3   chips per host
        //   wrap                   Int64List, exactly 3   torus wraparound
        //   num_slices             Int64 SCALAR, not list  number of slices
        //
        // The backing arrays must outlive the Create call, so they live in
        // this scope rather than inside the lambda.
        int64_t chips_per_host_bounds_values[3] = {0, 0, 0};
        int64_t wrap_values[3] = {0, 0, 0};
        std::vector<PJRT_NamedValue> create_options;

        auto add_bounds_option = [&](const char* name,
                                     const std::string& text,
                                     int64_t values[3]) -> bool {
            if (text.empty()) return true;
            if (!ParseBounds(text, values)) {
                std::cerr << "Invalid " << name << " \"" << text
                          << "\": expected three non-negative integers like "
                             "1x1x1.\n";
                return false;
            }
            PJRT_NamedValue nv{};
            nv.struct_size = PJRT_NamedValue_STRUCT_SIZE;
            nv.name = name;
            nv.name_size = std::strlen(name);
            nv.type = PJRT_NamedValue_kInt64List;
            nv.int64_array_value = values;
            nv.value_size = 3;
            create_options.push_back(nv);
            std::cout << "    " << name << " = [" << values[0] << ", "
                      << values[1] << ", " << values[2] << "]\n";
            return true;
        };

        // Each option appears as three things: the literal name libtpu
        // matches on, the raw "1x1x1" text from argv (_arg), and the parsed
        // integers we hand over (_values).
        if (!add_bounds_option("chips_per_host_bounds",
                               chips_per_host_bounds_arg,
                               chips_per_host_bounds_values) ||
            !add_bounds_option("wrap", wrap_arg, wrap_values)) {
            return 1;
        }

        // num_slices is a scalar, and passing it as a 1-element list is
        // rejected ("num_slices must be a ..."). Slices multiply the mesh:
        // v6e:2x2 with num_slices=2 describes 8 devices, not 4.
        if (num_slices > 0) {
            PJRT_NamedValue nv{};
            nv.struct_size = PJRT_NamedValue_STRUCT_SIZE;
            nv.name = "num_slices";
            nv.name_size = std::strlen("num_slices");
            nv.type = PJRT_NamedValue_kInt64;
            nv.int64_value = num_slices;
            nv.value_size = 1;
            create_options.push_back(nv);
            std::cout << "    num_slices = " << num_slices << "\n";
        }

        PJRT_TopologyDescription_Create_Args args{};
        args.struct_size = PJRT_TopologyDescription_Create_Args_STRUCT_SIZE;
        args.topology_name = topology_name.c_str();
        args.topology_name_size = topology_name.size();
        args.create_options =
            create_options.empty() ? nullptr : create_options.data();
        args.num_options = create_options.size();
        if (!CheckError(api, api->PJRT_TopologyDescription_Create(&args),
                        "PJRT_TopologyDescription_Create")) {
            std::cerr << "\nHints:\n"
                         "  - the name must be \"<gen>:<mesh>\", e.g. "
                         "v4:2x2x1, v4:2x4x4, v5e:2x2\n"
                         "  - for a slice smaller than one host (a single "
                         "chip, say) also pass\n"
                         "    --chips-per-host=1x1x1, or the mesh will not "
                         "divide the 2x2x1 default\n";
            return 1;
        }
        topology = args.topology;
    }
    std::cout << "[3] Created TopologyDescription for \"" << topology_name
              << "\"\n";

    // ---- 4. Interrogate the topology ----------------------------------
    {
        PJRT_TopologyDescription_PlatformName_Args name_args{};
        name_args.struct_size =
            PJRT_TopologyDescription_PlatformName_Args_STRUCT_SIZE;
        name_args.topology = topology;
        if (CheckError(api,
                       api->PJRT_TopologyDescription_PlatformName(&name_args),
                       "PJRT_TopologyDescription_PlatformName")) {
            std::cout << "    platform name:    "
                      << std::string(name_args.platform_name,
                                     name_args.platform_name_size)
                      << "\n";
        }

        PJRT_TopologyDescription_PlatformVersion_Args ver_args{};
        ver_args.struct_size =
            PJRT_TopologyDescription_PlatformVersion_Args_STRUCT_SIZE;
        ver_args.topology = topology;
        if (CheckError(
                api, api->PJRT_TopologyDescription_PlatformVersion(&ver_args),
                "PJRT_TopologyDescription_PlatformVersion")) {
            std::cout << "    platform version: "
                      << std::string(ver_args.platform_version,
                                     ver_args.platform_version_size)
                      << "\n";
        }

        PJRT_TopologyDescription_GetDeviceDescriptions_Args dd_args{};
        dd_args.struct_size =
            PJRT_TopologyDescription_GetDeviceDescriptions_Args_STRUCT_SIZE;
        dd_args.topology = topology;
        if (CheckError(
                api,
                api->PJRT_TopologyDescription_GetDeviceDescriptions(&dd_args),
                "PJRT_TopologyDescription_GetDeviceDescriptions")) {
            std::cout << "    devices in mesh:  " << dd_args.num_descriptions
                      << "\n";

            if (dd_args.num_descriptions > 0) {
                PJRT_DeviceDescription_Kind_Args kind_args{};
                kind_args.struct_size =
                    PJRT_DeviceDescription_Kind_Args_STRUCT_SIZE;
                kind_args.device_description = dd_args.descriptions[0];
                if (CheckError(api,
                               api->PJRT_DeviceDescription_Kind(&kind_args),
                               "PJRT_DeviceDescription_Kind")) {
                    std::cout << "    device kind:      "
                              << std::string(kind_args.device_kind,
                                             kind_args.device_kind_size)
                              << "\n";
                }
            }

            if (list_devices) {
                for (size_t i = 0; i < dd_args.num_descriptions; ++i) {
                    PJRT_DeviceDescription* dd = dd_args.descriptions[i];

                    PJRT_DeviceDescription_ToString_Args str_args{};
                    str_args.struct_size =
                        PJRT_DeviceDescription_ToString_Args_STRUCT_SIZE;
                    str_args.device_description = dd;
                    std::string label = "<unavailable>";
                    if (CheckError(api,
                                   api->PJRT_DeviceDescription_ToString(
                                       &str_args),
                                   "PJRT_DeviceDescription_ToString")) {
                        label.assign(str_args.to_string,
                                     str_args.to_string_size);
                    }
                    std::cout << "    device[" << i << "] " << label << "\n";

                    PJRT_DeviceDescription_Attributes_Args attr_args{};
                    attr_args.struct_size =
                        PJRT_DeviceDescription_Attributes_Args_STRUCT_SIZE;
                    attr_args.device_description = dd;
                    if (CheckError(api,
                                   api->PJRT_DeviceDescription_Attributes(
                                       &attr_args),
                                   "PJRT_DeviceDescription_Attributes")) {
                        PrintNamedValues(attr_args.attributes,
                                         attr_args.num_attributes,
                                         "        ");
                    }
                }
            }
        }

        // Topology attributes and fingerprint are only present in newer API
        // versions; guard on the function pointers.
        if (api->PJRT_TopologyDescription_Attributes != nullptr) {
            PJRT_TopologyDescription_Attributes_Args attr_args{};
            attr_args.struct_size =
                PJRT_TopologyDescription_Attributes_Args_STRUCT_SIZE;
            attr_args.topology = topology;
            if (CheckError(api,
                           api->PJRT_TopologyDescription_Attributes(&attr_args),
                           "PJRT_TopologyDescription_Attributes") &&
                attr_args.num_attributes > 0) {
                std::cout << "    topology attributes:\n";
                PrintNamedValues(attr_args.attributes,
                                 attr_args.num_attributes, "      ");
            }
        }
    }

    // ---- 5. Build the program -----------------------------------------
    std::string stablehlo;
    if (trivial) {
        stablehlo = BuildTrivialStableHlo();
        std::cout << "[4] Using trivial StableHLO (" << stablehlo.size()
                  << " bytes)\n";
    } else {
        const std::string mosaic_text = LoadTextFile(mosaic_path);
        if (mosaic_text.empty()) return 1;
        stablehlo = WrapMosaicInStableHlo(mosaic_text);
        std::cout << "[4] Loaded Mosaic (" << mosaic_text.size()
                  << " bytes) and wrapped into StableHLO ("
                  << stablehlo.size() << " bytes)\n";
    }

    // ---- 6. AOT compile against the topology --------------------------
    //
    // PJRT_Compile is the client-free sibling of PJRT_Client_Compile. The
    // `client` field is optional and used only for profile-guided
    // optimizations, so we leave it null — that is the whole point here.
    PJRT_Executable* executable = nullptr;
    {
        static constexpr char kFormat[] = "mlir";
        PJRT_Program program{};
        program.struct_size = PJRT_Program_STRUCT_SIZE;
        program.code = const_cast<char*>(stablehlo.data());
        program.code_size = stablehlo.size();
        program.format = kFormat;
        program.format_size = sizeof(kFormat) - 1;  // 4, without NUL

        const std::string compile_options = BuildCompileOptions(
            num_replicas, num_partitions, spmd, static_assignment);

        PJRT_Compile_Args args{};
        args.struct_size = PJRT_Compile_Args_STRUCT_SIZE;
        args.topology = topology;
        args.program = &program;
        args.compile_options = compile_options.data();
        args.compile_options_size = compile_options.size();
        args.client = nullptr;  // no client -> no device needed
        if (!CheckError(api, api->PJRT_Compile(&args), "PJRT_Compile")) {
            return 1;
        }
        executable = args.executable;
    }
    std::cout << "[5] AOT-compiled StableHLO -> PJRT_Executable\n";

    // ---- 7. Report on the compiled executable -------------------------
    {
        PJRT_Executable_Name_Args name_args{};
        name_args.struct_size = PJRT_Executable_Name_Args_STRUCT_SIZE;
        name_args.executable = executable;
        if (CheckError(api, api->PJRT_Executable_Name(&name_args),
                       "PJRT_Executable_Name")) {
            std::cout << "    name:            "
                      << std::string(name_args.executable_name,
                                     name_args.executable_name_size)
                      << "\n";
        }

        PJRT_Executable_NumReplicas_Args rep_args{};
        rep_args.struct_size = PJRT_Executable_NumReplicas_Args_STRUCT_SIZE;
        rep_args.executable = executable;
        if (CheckError(api, api->PJRT_Executable_NumReplicas(&rep_args),
                       "PJRT_Executable_NumReplicas")) {
            std::cout << "    num_replicas:    " << rep_args.num_replicas
                      << "\n";
        }

        PJRT_Executable_NumPartitions_Args part_args{};
        part_args.struct_size = PJRT_Executable_NumPartitions_Args_STRUCT_SIZE;
        part_args.executable = executable;
        if (CheckError(api, api->PJRT_Executable_NumPartitions(&part_args),
                       "PJRT_Executable_NumPartitions")) {
            std::cout << "    num_partitions:  " << part_args.num_partitions
                      << "\n";
        }

        PJRT_Executable_NumOutputs_Args out_args{};
        out_args.struct_size = PJRT_Executable_NumOutputs_Args_STRUCT_SIZE;
        out_args.executable = executable;
        if (CheckError(api, api->PJRT_Executable_NumOutputs(&out_args),
                       "PJRT_Executable_NumOutputs")) {
            std::cout << "    num_outputs:     " << out_args.num_outputs
                      << "\n";
        }

        PJRT_Executable_SizeOfGeneratedCodeInBytes_Args size_args{};
        size_args.struct_size =
            PJRT_Executable_SizeOfGeneratedCodeInBytes_Args_STRUCT_SIZE;
        size_args.executable = executable;
        if (CheckError(
                api,
                api->PJRT_Executable_SizeOfGeneratedCodeInBytes(&size_args),
                "PJRT_Executable_SizeOfGeneratedCodeInBytes")) {
            std::cout << "    generated code:  " << size_args.size_in_bytes
                      << " bytes\n";
        }
    }

    // ---- 8. Serialize the executable ----------------------------------
    {
        PJRT_Executable_Serialize_Args ser_args{};
        ser_args.struct_size = PJRT_Executable_Serialize_Args_STRUCT_SIZE;
        ser_args.executable = executable;
        if (!CheckError(api, api->PJRT_Executable_Serialize(&ser_args),
                        "PJRT_Executable_Serialize")) {
            return 1;
        }
        const bool ok = WriteBinaryFile(blob_out_path, ser_args.serialized_bytes,
                                        ser_args.serialized_bytes_size);
        std::cout << "[6] Serialized executable ("
                  << ser_args.serialized_bytes_size << " bytes) to "
                  << blob_out_path << "\n";
        ser_args.serialized_executable_deleter(ser_args.serialized_executable);
        if (!ok) return 1;
    }

    // ---- 9. Optionally serialize the topology -------------------------
    //
    // Handy as a cache key / provenance record next to the blob: it pins down
    // exactly which target the executable was built for.
    if (!topology_out.empty()) {
        if (api->PJRT_TopologyDescription_Serialize == nullptr) {
            std::cerr << "Plugin does not implement "
                         "PJRT_TopologyDescription_Serialize\n";
        } else {
            PJRT_TopologyDescription_Serialize_Args ser_args{};
            ser_args.struct_size =
                PJRT_TopologyDescription_Serialize_Args_STRUCT_SIZE;
            ser_args.topology = topology;
            if (CheckError(api,
                           api->PJRT_TopologyDescription_Serialize(&ser_args),
                           "PJRT_TopologyDescription_Serialize")) {
                WriteBinaryFile(topology_out, ser_args.serialized_bytes,
                                ser_args.serialized_bytes_size);
                std::cout << "[7] Serialized topology ("
                          << ser_args.serialized_bytes_size << " bytes) to "
                          << topology_out << "\n";
                ser_args.serialized_topology_deleter(
                    ser_args.serialized_topology);
            }
        }
    }

    // ---- 10. Cleanup ---------------------------------------------------
    {
        PJRT_Executable_Destroy_Args a{};
        a.struct_size = PJRT_Executable_Destroy_Args_STRUCT_SIZE;
        a.executable = executable;
        api->PJRT_Executable_Destroy(&a);
    }
    {
        PJRT_TopologyDescription_Destroy_Args a{};
        a.struct_size = PJRT_TopologyDescription_Destroy_Args_STRUCT_SIZE;
        a.topology = topology;
        api->PJRT_TopologyDescription_Destroy(&a);
    }

    std::cout << "\n[done] Copy " << blob_out_path
              << " to a TPU host and run it there with\n"
                 "       pjrt-aot-run (aot_compile/run_blob.cpp), which "
                 "deserializes and\n"
                 "       executes the blob:\n\n"
                 "         ./build/pjrt-aot-run "
              << blob_out_path << " " << libtpu_path << "\n";
    return 0;
}
