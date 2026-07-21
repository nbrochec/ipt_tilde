include(${C74_MIN_API_DIR}/script/min-package.cmake)
include(${C74_MIN_API_DIR}/script/min-pretarget.cmake)

include_directories("${C74_INCLUDES}")

# SOURCE_FILES must be defined for min-object-unittest.cmake. Use the bare
# filename (not an absolute path) so its test-source substitution — which swaps
# ipt_tilde.cpp for ipt_tilde_test.cpp by exact name match — works and the test
# doesn't compile ipt_tilde.cpp twice (the test #includes it).
set(SOURCE_FILES ipt_tilde.cpp)

# ${PROJECT_NAME} will here be name of current directory (thanks to min-pretarget.cmake)
add_library(${PROJECT_NAME} MODULE ${SOURCE_FILES})

# ipt_api brings libipt's public include dir (ipt.h) transitively.
target_link_libraries(${PROJECT_NAME}
        PRIVATE
        ipt_api)

include(${C74_MIN_API_DIR}/script/min-posttarget.cmake)
include(${C74_MIN_API_DIR}/test/min-object-unittest.cmake)

# the unit test #includes ipt_tilde.cpp, so it links libipt for ipt.h + the C ABI
# (replaces a stale reference to a non-existent "shared_code" target).
target_link_libraries(${TEST_NAME} PUBLIC ipt_api)

set_target_properties(${PROJECT_NAME}
        PROPERTIES
        BUILD_WITH_INSTALL_RPATH FALSE
        LINK_FLAGS "-Wl,-rpath,@loader_path/"
)

# Bundle the libipt runtime closure into the .mxo: libipt.dylib plus the three
# torch dylibs it depends on. The external's rpath (@loader_path) finds
# libipt.dylib; libipt.dylib finds the torch dylibs beside it (@loader_path added).
set(_mxo_macos "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${${PROJECT_NAME}_EXTERN_OUTPUT_NAME}.mxo/Contents/MacOS")

add_custom_command(
        TARGET ${PROJECT_NAME}
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:ipt_api>" "${_mxo_macos}/"
        COMMAND ${CMAKE_COMMAND} -E copy
                "${IPT_TORCH_LIB_DIR}/libtorch.dylib"
                "${IPT_TORCH_LIB_DIR}/libtorch_cpu.dylib"
                "${IPT_TORCH_LIB_DIR}/libc10.dylib"
                "${_mxo_macos}/"
        COMMAND install_name_tool -add_rpath "@loader_path" "${_mxo_macos}/libipt.dylib"
)

add_custom_command(
        TARGET ${PROJECT_NAME}
        POST_BUILD
        COMMAND codesign -s - --deep --force "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${${PROJECT_NAME}_EXTERN_OUTPUT_NAME}.mxo"
)