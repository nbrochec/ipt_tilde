cmake_minimum_required(VERSION 3.19)  
set(CMAKE_CXX_STANDARD 17)
project(ipt_max)  
  
set(CMAKE_OSX_ARCHITECTURES arm64;x86_64)

# include external libraries:
add_subdirectory(shared_code)

include(${CMAKE_SOURCE_DIR}/min-api/script/min-package.cmake)  
  
set(C74_MIN_API_DIR ${CMAKE_SOURCE_DIR}/min-api)  
set(C74_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/externals)
  
add_subdirectory(ipt_tilde)  
add_subdirectory(asdf_remove)

# Override settings from min-api for C++17 support  
set(CMAKE_OSX_DEPLOYMENT_TARGET "10.13" CACHE STRING "Minimum OS X deployment version" FORCE)  
message(STATUS "MacOS deployment target: ${CMAKE_OSX_DEPLOYMENT_TARGET}")