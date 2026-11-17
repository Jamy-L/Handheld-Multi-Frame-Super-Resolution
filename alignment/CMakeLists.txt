cmake_minimum_required(VERSION 3.28)
project(handheld_alignment LANGUAGES C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
# Locate SDKs installed in the selected Python environment.
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c
            "import pathlib, halide, pybind11; p = pathlib.Path(halide.__file__).parent; print(p); print(p / 'lib64' / 'cmake'); print(p / 'lib' / 'cmake'); print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE python_prefixes
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)
string(REPLACE "\n" ";" python_prefixes "${python_prefixes}")
list(APPEND CMAKE_PREFIX_PATH ${python_prefixes})
find_package(Halide REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

add_halide_generator(downsample_generator SOURCES downsample_generator.cpp)
add_halide_runtime(alignment_runtime TARGETS host)
set(downsample_libraries "")
# Keep these specializations in sync with the switch in kernels.cpp.
foreach(scale IN ITEMS 2 4 8 16)
    set(kernel "alignment_downsample_${scale}")
    add_halide_library(${kernel} FROM downsample_generator
        GENERATOR alignment_downsample USE_RUNTIME alignment_runtime
        TARGETS host PARAMS scale_factor=${scale})
    list(APPEND downsample_libraries ${kernel})
endforeach()

add_halide_generator(gradients_generator SOURCES gradients.cpp)
set(gradient_libraries "")
# Keep these specializations in sync with precompute_reference in kernels.cpp.
foreach(tile IN ITEMS 8 16 32 64)
    set(kernel "alignment_gradients_${tile}")
    add_halide_library(${kernel} FROM gradients_generator
        GENERATOR alignment_gradients USE_RUNTIME alignment_runtime
        TARGETS host PARAMS tile_size=${tile})
    list(APPEND gradient_libraries ${kernel})
endforeach()

add_library(alignment_core STATIC alignment.cpp kernels.cpp)
target_link_libraries(alignment_core PUBLIC Halide::Runtime PRIVATE ${downsample_libraries} ${gradient_libraries})
target_include_directories(alignment_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}"
    PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")

pybind11_add_module(_alignment_native bindings.cpp)
target_link_libraries(_alignment_native PRIVATE alignment_core)
# Explicit opt-in build; keep the existing Python pipeline independent.
set_target_properties(_alignment_native PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../handheld_super_resolution"
)

include(CTest)
if(BUILD_TESTING)
    add_executable(test_downsample test_downsample.cpp)
    target_link_libraries(test_downsample PRIVATE alignment_core)
    add_test(NAME downsample COMMAND test_downsample)
    add_executable(test_gradients test_gradients.cpp)
    target_link_libraries(test_gradients PRIVATE alignment_core)
    add_test(NAME gradients COMMAND test_gradients)
endif()
