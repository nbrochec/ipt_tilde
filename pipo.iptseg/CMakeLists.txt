include(${C74_MIN_API_DIR}/script/min-package.cmake)
include(${C74_MIN_API_DIR}/script/min-pretarget.cmake)

include_directories("${C74_INCLUDES}")

# the pipo-sdk submodule lives under pipo.ipt
include_directories("../pipo.ipt/pipo-sdk/src")
include_directories("../pipo.ipt/pipo-sdk/bindings/max")

set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/pipo.iptseg.cpp)

# ${PROJECT_NAME} will here be name of current directory (thanks to min-pretarget.cmake)
add_library(${PROJECT_NAME} MODULE ${SOURCE_FILES})

# No libipt and no torch: the module only reads the distributions produced by
# pipo.ipt, so the external has nothing to bundle.
include(${C74_MIN_API_DIR}/script/min-posttarget.cmake)

if(APPLE)
    add_custom_command(
            TARGET ${PROJECT_NAME}
            POST_BUILD
            COMMAND codesign -s - --deep --force "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${${PROJECT_NAME}_EXTERN_OUTPUT_NAME}.mxo"
    )
endif()
