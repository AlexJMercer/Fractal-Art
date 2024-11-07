cmake_minimum_required(VERSION 3.5.0)
project(Fractals VERSION 0.1.0 LANGUAGES C CXX)

add_executable(Fractals main.cpp)

set(SFML_DIR "C:/SFML/SFML-2.6.1/lib/cmake/SFML")
find_package(SFML 2 COMPONENTS system window graphics network audio REQUIRED)

target_include_directories(Fractals PRIVATE "C:/SFML/SFML-2.6.1/include")
target_link_libraries(Fractals PRIVATE sfml-system sfml-window sfml-graphics sfml-network sfml-audio)

# add_custom_command(TARGET Fractals POST_BUILD
#     COMMAND ${CMAKE_COMMAND} -E copy_if_different
#     "C:/SFML/SFML-2.6.1/bin/openal32.dll"
#     "C:/SFML/SFML-2.6.1/bin/sfml-system-2.dll"
#     "C:/SFML/SFML-2.6.1/bin/sfml-system-d-2.dll"
#     "C:/SFML/SFML-2.6.1/bin/sfml-window-2.dll"
#     "C:/SFML/SFML-2.6.1/bin/sfml-window-d-2.dll"
#     "C:/SFML/SFML-2.6.1/bin/sfml-graphics-2.dll"
#     "C:/SFML/SFML-2.6.1/bin/sfml-graphics-d-2.dll"
#     "C:/SFML/SFML-2.6.1/bin/sfml-network-2.dll"
#     "C:/SFML/SFML-2.6.1/bin/sfml-network-d-2.dll"
#     "C:/SFML/SFML-2.6.1/bin/sfml-audio-2.dll"
#     "C:/SFML/SFML-2.6.1/bin/sfml-audio-d-2.dll"
#     $<TARGET_FILE_DIR:Fractals>
# )