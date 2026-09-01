# Standalone CLI example — consumes libipt's C ABI only (ipt.h + libipt).

add_executable(ipt_example main.cpp)

target_link_libraries(ipt_example PRIVATE ipt_api)

# On Windows there is no rpath: ipt.dll (built in libipt's directory) and the
# torch DLL closure must sit next to the executable to be found.
if(WIN32)
    add_custom_command(TARGET ipt_example POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:ipt_api>" ${IPT_TORCH_RUNTIME_DLLS}
                "$<TARGET_FILE_DIR:ipt_example>")
endif()
