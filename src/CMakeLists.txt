

add_library(ipt INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/circular_buffer.h
        ${CMAKE_CURRENT_SOURCE_DIR}/classifier.h
        ${CMAKE_CURRENT_SOURCE_DIR}/energy_threshold.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ipt_classifier.h
        ${CMAKE_CURRENT_SOURCE_DIR}/leaky_integrator.h
        ${CMAKE_CURRENT_SOURCE_DIR}/utility.h
)

target_include_directories(ipt INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(ipt
        INTERFACE
        r8brain
        "${TORCH_LIBRARIES}")
