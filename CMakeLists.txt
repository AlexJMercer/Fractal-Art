cmake_minimum_required(VERSION 3.20)
project(Fractals LANGUAGES C CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Specify CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 75)

# SFML configuration
set(SFML_DIR "C:/SFML-2.6.2/lib/cmake/SFML") 
find_package(SFML 2.5 COMPONENTS system window graphics audio network REQUIRED)

# Set executables
file(GLOB SOURCE_FILES src/*.cpp src/*.cu)

# Add executable
add_executable(Fractals ${SOURCE_FILES})

# Link SFML and OpenGL libraries
target_link_libraries(Fractals
    sfml-system sfml-window sfml-graphics sfml-audio sfml-network
    cudart
)

# Enable separable compilation for CUDA
set_target_properties(Fractals PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# MSVC-specific options
if(MSVC)
    target_compile_options(Fractals PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler="/EHsc /W4 /O2">
    )
endif()
