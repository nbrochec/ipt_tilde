# Standalone CLI example — consumes libipt's C ABI only (ipt.h + libipt.dylib).

add_executable(ipt_example main.cpp)

target_link_libraries(ipt_example PRIVATE ipt_api)
