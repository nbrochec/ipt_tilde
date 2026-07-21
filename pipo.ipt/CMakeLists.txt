include(${C74_MIN_API_DIR}/script/min-package.cmake)
include(${C74_MIN_API_DIR}/script/min-pretarget.cmake)

include_directories("${C74_INCLUDES}")

include_directories("pipo-sdk/src")
include_directories("pipo-sdk/bindings/max")
include_directories("../max-sdk-base/max-includes")

# SOURCE_FILES must be defined for min-object-unittest.cmake
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/pipo.ipt.cpp)

# ${PROJECT_NAME} will here be name of current directory (thanks to min-pretarget.cmake)
add_library(${PROJECT_NAME} MODULE ${SOURCE_FILES})

# ipt_api brings libipt's public include dir (ipt.h) transitively.
target_link_libraries(${PROJECT_NAME}
        PRIVATE
        ipt_api)

include(${C74_MIN_API_DIR}/script/min-posttarget.cmake)
include(${C74_MIN_API_DIR}/test/min-object-unittest.cmake)

#target_link_libraries(${TEST_NAME} PUBLIC shared_code)

set_target_properties(${PROJECT_NAME}
        PROPERTIES
        BUILD_WITH_INSTALL_RPATH FALSE
        LINK_FLAGS "-Wl,-rpath,@loader_path/"
)

# Bundle the libipt runtime closure into the .mxo: libipt.dylib plus the three
# torch dylibs it depends on. The external's rpath (@loader_path, set above) finds
# libipt.dylib next to it; libipt.dylib in turn finds the torch dylibs beside it.
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
        # the build-tree libipt.dylib resolves torch via an absolute rpath; add
        # @loader_path so the bundled copy finds the torch dylibs beside it on
        # any machine (makes the .mxo relocatable).
        COMMAND install_name_tool -add_rpath "@loader_path" "${_mxo_macos}/libipt.dylib"
)

add_custom_command(
        TARGET ${PROJECT_NAME}
        POST_BUILD
        COMMAND codesign -s - --deep --force "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${${PROJECT_NAME}_EXTERN_OUTPUT_NAME}.mxo"
)