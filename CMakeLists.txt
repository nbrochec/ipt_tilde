cmake_minimum_required(VERSION 3.19)  
set(CMAKE_CXX_STANDARD 17)
project(ipt_max VERSION 0.0.1)

# TODO: Won't be able to compile universal binaries using torch -- find solution
set(CMAKE_OSX_ARCHITECTURES arm64)


add_subdirectory(shared_code)
add_subdirectory(app/torch_tests)

# min-api config
include(${CMAKE_SOURCE_DIR}/min-api/script/min-package.cmake)
set(C74_MIN_API_DIR ${CMAKE_SOURCE_DIR}/min-api)  
set(C74_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/externals)

# min targets
add_subdirectory(ipt_tilde)


# Override settings from min-api for C++17 support  
set(CMAKE_OSX_DEPLOYMENT_TARGET "10.13" CACHE STRING "Minimum OS X deployment version" FORCE)  
message(STATUS "MacOS deployment target: ${CMAKE_OSX_DEPLOYMENT_TARGET}")