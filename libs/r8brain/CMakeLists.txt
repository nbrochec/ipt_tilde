

set(R8BRAIN_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/r8brain-free-src")

file(GLOB R8BRAIN_SOURCES
    "${R8BRAIN_SRC_DIR}/*.cpp"
    "${R8BRAIN_SRC_DIR}/*.h"
)

list(REMOVE_ITEM R8BRAIN_SOURCES "${R8BRAIN_SRC_DIR}/example.cpp")

add_library(r8brain STATIC ${R8BRAIN_SOURCES})

# ugly SYSTEM keyword workaround to suppress warnings from r8brain
target_include_directories(r8brain SYSTEM PUBLIC "${R8BRAIN_SRC_DIR}")