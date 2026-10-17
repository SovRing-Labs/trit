cmake_minimum_required(VERSION 3.16)
project(trit_cpp LANGUAGES CXX)

# C++17 required
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Native optimizations (AVX2, FMA, etc.)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mtune=native -O3 -fopenmp")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG")
endif()

# Include current directory for trit.hpp
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Header-only interface library
add_library(trit INTERFACE)
target_include_directories(trit INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(trit INTERFACE -march=native -mtune=native -O3 -fopenmp)

# Test executable
add_executable(trit_test ../tests/trit_test.cpp)
target_link_libraries(trit_test trit)

# Install targets (optional)
install(FILES trit.hpp DESTINATION include/trit)