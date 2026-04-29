cmake_minimum_required(VERSION 3.16)
project(pjrt_tpu_api CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ------------------------------------------------------------------------------
# REQUIRED: XLA source tree (PJRT C API header)
# ------------------------------------------------------------------------------
#
# The PJRT C API is defined in xla/xla/pjrt/c/pjrt_c_api.h — a single plain-C
# header with no transitive dependencies.
#
#     cmake -B build -DXLA_SRC_PATH=/path/to/xla

if(NOT DEFINED XLA_SRC_PATH OR XLA_SRC_PATH STREQUAL "")
    message(FATAL_ERROR
        "XLA_SRC_PATH is not set. Invoke cmake with:\n"
        "    cmake -B build -DXLA_SRC_PATH=/path/to/xla")
endif()

if(NOT EXISTS "${XLA_SRC_PATH}/xla/pjrt/c/pjrt_c_api.h")
    message(FATAL_ERROR
        "Cannot find PJRT C API header at\n"
        "    ${XLA_SRC_PATH}/xla/pjrt/c/pjrt_c_api.h\n"
        "Check that XLA_SRC_PATH points to the root of a valid XLA checkout.")
endif()

message(STATUS "XLA source tree: ${XLA_SRC_PATH}")

# ------------------------------------------------------------------------------
# OPTIONAL: jaxlib source tree (Mosaic dialect C API)
# ------------------------------------------------------------------------------
#
# If you provide a jaxlib checkout, main.cpp will additionally register the
# `tpu` dialect in an MLIR context so you can parse/verify Mosaic IR in C++
# (via MLIR's C API). This doesn't give you a Mosaic compiler — that lives
# inside libtpu — but it lets you handle the IR itself.
#
#     cmake -B build \
#       -DXLA_SRC_PATH=/path/to/xla \
#       -DJAXLIB_SRC_PATH=/path/to/jax
#
# Enabling this requires the MLIR C API headers to also be reachable.
# MLIR_SRC_PATH can point at an LLVM checkout (.../llvm-project/mlir);
# otherwise the build assumes the system toolchain provides them.

if(DEFINED JAXLIB_SRC_PATH AND NOT JAXLIB_SRC_PATH STREQUAL "")
    set(MOSAIC_DIALECT_HEADER
        "${JAXLIB_SRC_PATH}/jaxlib/mosaic/dialect/tpu/integrations/c/tpu_dialect.h")
    if(NOT EXISTS "${MOSAIC_DIALECT_HEADER}")
        message(FATAL_ERROR
            "JAXLIB_SRC_PATH is set but cannot find Mosaic C API header at\n"
            "    ${MOSAIC_DIALECT_HEADER}\n"
            "Check that JAXLIB_SRC_PATH points to the root of a valid jax checkout.")
    endif()
    set(HAVE_JAXLIB_MOSAIC ON)
    message(STATUS "jaxlib source tree: ${JAXLIB_SRC_PATH} (Mosaic C API enabled)")
else()
    set(HAVE_JAXLIB_MOSAIC OFF)
    message(STATUS "JAXLIB_SRC_PATH not set (Mosaic C API disabled)")
endif()

# ------------------------------------------------------------------------------
# OPTIONAL: prebuilt shared libraries for linking
# ------------------------------------------------------------------------------
#
# At compile time we only need headers. At link time, if you want a
# statically-resolved binary that talks to real TPUs, you need:
#
#   -DPJRT_TPU_LIB=/path/to/libpjrt_tpu.so
#         (on a TPU VM, typically inside site-packages/libtpu/ or the
#          jaxlib tpu plugin .so)
#
#   -DJAXLIB_MOSAIC_C_LIB=/path/to/libmosaic_c_api.so
#         (only needed if HAVE_JAXLIB_MOSAIC is on and you actually call
#          the mlirGetDialectHandle__tpu__ / mlirTpu* symbols. Otherwise
#          the example compiles and just doesn't link those symbols.)
#
# Either can be omitted; without them the executable compiles but will
# fail at runtime (unresolved symbols / missing plugin).

# ------------------------------------------------------------------------------
# The executable
# ------------------------------------------------------------------------------

function(configure_target target_name)
    target_include_directories(${target_name} PRIVATE "${XLA_SRC_PATH}")

    if(HAVE_JAXLIB_MOSAIC)
        target_include_directories(${target_name} PRIVATE "${JAXLIB_SRC_PATH}")
        target_compile_definitions(${target_name} PRIVATE HAVE_JAXLIB_MOSAIC=1)

        # The Mosaic C API header includes mlir-c/IR.h and mlir-c/Support.h.
        # These come from LLVM/MLIR. If MLIR_SRC_PATH is set, use it; otherwise
        # assume the system toolchain provides them (e.g. via pkg-config).
        if(DEFINED MLIR_SRC_PATH AND NOT MLIR_SRC_PATH STREQUAL "")
            target_include_directories(${target_name} PRIVATE
                "${MLIR_SRC_PATH}/include")
        endif()
    endif()

    # Optional: if the jaxlib Mosaic C API .so is known at configure time,
    # link it statically. Not required — the example dlopens libtpu.so at
    # runtime via a path passed on the command line.
    if(HAVE_JAXLIB_MOSAIC AND DEFINED JAXLIB_MOSAIC_C_LIB
    AND NOT JAXLIB_MOSAIC_C_LIB STREQUAL "")
        target_link_libraries(${target_name} PRIVATE "${JAXLIB_MOSAIC_C_LIB}")
    endif()

    # dlopen / dlsym: the TPU plugin path is supplied at runtime, so we always
    # need libdl linked.
    if(APPLE OR UNIX)
        target_link_libraries(${target_name} PRIVATE ${CMAKE_DL_LIBS})
    endif()
endfunction()

add_executable(pjrt-demo main.cpp)
configure_target(pjrt-demo)

add_executable(pjrt-demo-newer vector_add_newer_version/main.cpp)
configure_target(pjrt-demo-newer)

