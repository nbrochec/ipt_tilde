cmake_minimum_required(VERSION 3.19)
set(CMAKE_CXX_STANDARD 17)
project(ipt_max VERSION 0.0.1)

# Register tests at the top level so `ctest --test-dir build` discovers the
# min-object unit tests defined in subdirectories (e.g. ipt_tilde_test).
enable_testing()

# Universal binaries are not possible with torch; libipt is arm64-only.
set(CMAKE_OSX_ARCHITECTURES arm64)

# --------------------------------------------------------------------------------------
# libipt (git submodule) owns the entire IPT inference core, the r8brain resampler,
# and libtorch. ipt_tilde consumes ONLY its C ABI (libipt/include/ipt.h) via the
# ipt_api target — no torch, model, or buffer internals leak into this project.
# add_subdirectory builds libipt.dylib in-tree and defines ipt_api.
# --------------------------------------------------------------------------------------
add_subdirectory(libipt)

# The libtorch runtime dylibs libipt.dylib depends on. libipt downloads torch
# into its own tree; the Max externals copy these + libipt.dylib into their .mxo
# bundles. (find_package's TORCH_INSTALL_PREFIX lives in libipt's scope, so we
# reference the known layout here for the min targets to use.)
set(IPT_TORCH_LIB_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/libipt/libs/libtorch/lib/python3.12/site-packages/torch/lib")

# Standalone CLI example (ported to the C ABI).
add_subdirectory(app/ipt_example)

# min-api config
include(${CMAKE_SOURCE_DIR}/min-api/script/min-package.cmake)
set(C74_MIN_API_DIR ${CMAKE_SOURCE_DIR}/min-api)
set(C74_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/externals)

# min targets — both ported to the libipt C ABI.
add_subdirectory(ipt_tilde)
add_subdirectory(pipo.ipt)

# Override settings from min-api for C++17 support
set(CMAKE_OSX_DEPLOYMENT_TARGET "10.13" CACHE STRING "Minimum OS X deployment version" FORCE)
message(STATUS "MacOS deployment target: ${CMAKE_OSX_DEPLOYMENT_TARGET}")
