cmake_minimum_required(VERSION 3.16)
project(fixed_pool CXX)

set(CMAKE_CXX_STANDARD          17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS        OFF)

# ---------------------------------------------------------------------------
# Compiler warnings — treat this project like production code
# ---------------------------------------------------------------------------
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(
        -Wall -Wextra -Wpedantic
        -Wcast-align
        -Wcast-qual
        -Wconversion
        -Wshadow
        -Wno-unused-parameter   # common in header-only templates
    )
elseif (MSVC)
    add_compile_options(/W4 /WX)
endif()

# ---------------------------------------------------------------------------
# Interface library (header-only)
# ---------------------------------------------------------------------------
add_library(fixed_pool INTERFACE)
target_include_directories(fixed_pool INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# ---------------------------------------------------------------------------
# Test targets
# ---------------------------------------------------------------------------

# Debug build: enables POOL_DEBUG, AddressSanitizer, UBSan
add_executable(test_pool_debug
    tests/test_pool.cpp
)
target_link_libraries(test_pool_debug PRIVATE fixed_pool)
target_compile_definitions(test_pool_debug PRIVATE POOL_DEBUG)
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(test_pool_debug PRIVATE
        -g -O0 -fsanitize=address,undefined
    )
    target_link_options(test_pool_debug PRIVATE
        -fsanitize=address,undefined
    )
endif()

# Release build: no debug overhead, optimised
add_executable(test_pool_release
    tests/test_pool.cpp
)
target_link_libraries(test_pool_release PRIVATE fixed_pool)
target_compile_options(test_pool_release PRIVATE -O2 -DNDEBUG)

# ---------------------------------------------------------------------------
# CTest integration
# ---------------------------------------------------------------------------
enable_testing()
add_test(NAME pool_debug   COMMAND test_pool_debug)
add_test(NAME pool_release COMMAND test_pool_release)

# ---------------------------------------------------------------------------
# Stage 5: Benchmark target
#
# IMPORTANT: Never link the benchmark against ASan or POOL_DEBUG.
#   - ASan intercepts malloc/free with its own shadow memory instrumentation.
#     This changes the cost of operator new/delete — the "new/delete" column
#     would measure ASan overhead, not the real allocator.
#   - POOL_DEBUG adds a bitmap check to every alloc/dealloc.
#     This inflates pool timings with debug work that is not in production.
#
# Optimisation level: -O2
#   Both the manual compile instruction in bench_pool.cpp and this CMake
#   target use -O2. This is the same level used for test_pool_release,
#   which means benchmark numbers are directly comparable to the test build.
#   -O3 or -march=native would show higher peak numbers but are
#   non-portable and not representative of a typical deployment build.
#   If you want to measure the -O3 ceiling locally, add it to your own
#   cmake invocation: cmake -DCMAKE_CXX_FLAGS="-O3 -march=native" ...
#
# Build the benchmark:
#   cmake -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build --target bench_pool
#   ./build/bench_pool
#
# Or manually (no CMake), matching these flags exactly:
#   g++ -std=c++17 -O2 -DNDEBUG -Iinclude benchmarks/bench_pool.cpp -o bench_pool
#   ./bench_pool
# ---------------------------------------------------------------------------
add_executable(bench_pool
    benchmarks/bench_pool.cpp
)
target_link_libraries(bench_pool PRIVATE fixed_pool)

# -O2 -DNDEBUG: matches the manual compile instruction in bench_pool.cpp.
# -fno-omit-frame-pointer: preserves call stacks for profiling with perf/VTune.
#   Negligible performance cost (~1%) but essential for readable flame graphs.
# No -march=native: keeps results portable and comparable across machines.
# No sanitizers: they alter allocation behaviour (see note above).
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(bench_pool PRIVATE
        -O2
        -DNDEBUG
        -fno-omit-frame-pointer
    )
elseif (MSVC)
    target_compile_options(bench_pool PRIVATE
        /O2 /DNDEBUG /Zi
    )
endif()

# ---------------------------------------------------------------------------
# Optional: Google Benchmark integration
#
# If you have Google Benchmark installed (e.g. via vcpkg or apt):
#   cmake -B build -DUSE_GBENCH=ON
# This links bench_pool against Google Benchmark for richer reporting.
#
# The bench_pool.cpp file is written to work standalone (no dependency).
# Google Benchmark support would be a separate bench_pool_gbench.cpp.
# ---------------------------------------------------------------------------
option(USE_GBENCH "Link bench_pool against Google Benchmark" OFF)
if (USE_GBENCH)
    find_package(benchmark REQUIRED)
    target_link_libraries(bench_pool PRIVATE benchmark::benchmark)
    target_compile_definitions(bench_pool PRIVATE HAVE_GBENCH)
    message(STATUS "Google Benchmark found — HAVE_GBENCH defined")
endif()

# ---------------------------------------------------------------------------
# Convenience target: run all tests then the benchmark
# ---------------------------------------------------------------------------
add_custom_target(run_all
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    COMMAND $<TARGET_FILE:bench_pool>
    DEPENDS test_pool_debug test_pool_release bench_pool
    COMMENT "Running tests then benchmark"
)
