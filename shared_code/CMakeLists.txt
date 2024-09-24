add_library(shared_code
        INTERFACE
        shared.h)

target_include_directories(shared_code INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})