list(APPEND CMAKE_PREFIX_PATH "/Applications/JUCE/install")
find_package(JUCE CONFIG REQUIRED)

juce_add_console_app(torch_tests
        PRODUCT_NAME "Torch Tests"
        VERSION "0.0.1")

target_sources(torch_tests
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)


target_compile_definitions(torch_tests
        PRIVATE
        JUCE_WEB_BROWSER=0
        JUCE_USE_CURL=0)


target_link_libraries(torch_tests
        PRIVATE
        juce::juce_core
        juce::juce_audio_basics
        juce::juce_audio_formats
        "${TORCH_LIBRARIES}"
        PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_warning_flags)