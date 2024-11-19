include(${C74_MIN_API_DIR}/script/min-package.cmake)
include(${C74_MIN_API_DIR}/script/min-pretarget.cmake)

include_directories("${C74_INCLUDES}")

# SOURCE_FILES must be defined for min-object-unittest.cmake
set(SOURCE_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/ipt_tilde.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/circular_buffer.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ipt_classifier.h
        ${CMAKE_CURRENT_SOURCE_DIR}/leaky_integrator.h
        ${CMAKE_CURRENT_SOURCE_DIR}/energy_threshold.h
        ${CMAKE_CURRENT_SOURCE_DIR}/classifier.h
        ${CMAKE_CURRENT_SOURCE_DIR}/utility.h
)

# ${PROJECT_NAME} will here be name of current directory (thanks to min-pretarget.cmake)
add_library(${PROJECT_NAME} MODULE ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME}
        PUBLIC
        shared_code
        PRIVATE
        r8brain
        "${TORCH_LIBRARIES}")

include(${C74_MIN_API_DIR}/script/min-posttarget.cmake)
include(${C74_MIN_API_DIR}/test/min-object-unittest.cmake)

target_link_libraries(${TEST_NAME} PUBLIC shared_code)

set_target_properties(${PROJECT_NAME}
        PROPERTIES
        BUILD_WITH_INSTALL_RPATH FALSE
        LINK_FLAGS "-Wl,-rpath,@loader_path/"
)

add_custom_command(
        TARGET ${PROJECT_NAME}
        POST_BUILD
        COMMAND cp "${TORCH_INSTALL_PREFIX}/lib/*.dylib" "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${${PROJECT_NAME}_EXTERN_OUTPUT_NAME}.mxo/Contents/MacOS/"
)

add_custom_command(
        TARGET ${PROJECT_NAME}
        POST_BUILD
        COMMAND codesign -s - --deep --force "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${${PROJECT_NAME}_EXTERN_OUTPUT_NAME}.mxo"
)