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
set(ALIGNMENT_DOWNSAMPLE_SCALES 2 4 8 16)
set(downsample_libraries "")
set(downsample_headers "")
set(downsample_entries "")
if(NOT ALIGNMENT_DOWNSAMPLE_SCALES)
    message(FATAL_ERROR "Compile at least one downsample variant")
endif()
foreach(scale IN LISTS ALIGNMENT_DOWNSAMPLE_SCALES)
    set(kernel "alignment_downsample_${scale}")
    add_halide_library(${kernel} FROM downsample_generator
        GENERATOR alignment_downsample USE_RUNTIME alignment_runtime
        TARGETS host PARAMS scale_factor=${scale})
    list(APPEND downsample_libraries ${kernel})
    string(APPEND downsample_headers "#include \"${kernel}.h\"\n")
    string(APPEND downsample_entries "    {${scale}, ${kernel}},\n")
endforeach()
configure_file(downsample_variants.h.in downsample_variants.h @ONLY)

add_halide_generator(gradients_generator SOURCES gradients.cpp)
set(ALIGNMENT_GRADIENT_TILE_SIZES 8 16 32 64)
set(gradient_libraries "")
set(gradient_headers "")
set(gradient_entries "")
if(NOT ALIGNMENT_GRADIENT_TILE_SIZES)
    message(FATAL_ERROR "Compile at least one gradient variant")
endif()
foreach(tile IN LISTS ALIGNMENT_GRADIENT_TILE_SIZES)
    set(kernel "alignment_gradients_${tile}")
    add_halide_library(${kernel} FROM gradients_generator
        GENERATOR alignment_gradients USE_RUNTIME alignment_runtime
        TARGETS host PARAMS tile_size=${tile})
    list(APPEND gradient_libraries ${kernel})
    string(APPEND gradient_headers "#include \"${kernel}.h\"\n")
    string(APPEND gradient_entries "    {${tile}, ${kernel}},\n")
endforeach()
configure_file(gradient_variants.h.in gradient_variants.h @ONLY)

add_halide_generator(block_matching_generator SOURCES block_matching.cpp)
# Single source of truth; compile the Cartesian product of these choices.
set(ALIGNMENT_BM_TILE_SIZES 8 16 32 64)
set(ALIGNMENT_BM_NORMS L1 L2)
set(ALIGNMENT_BM_SEARCH_RADII 1 4)
set(block_matching_libraries "")
set(bm_headers "")
set(bm_entries "")
if(NOT ALIGNMENT_BM_TILE_SIZES OR NOT ALIGNMENT_BM_NORMS OR NOT ALIGNMENT_BM_SEARCH_RADII)
    message(FATAL_ERROR "Compile at least one block-matching variant")
endif()
foreach(tile IN LISTS ALIGNMENT_BM_TILE_SIZES)
    foreach(radius IN LISTS ALIGNMENT_BM_SEARCH_RADII)
        foreach(norm IN LISTS ALIGNMENT_BM_NORMS)
            set(kernel "alignment_bm_t${tile}_r${radius}_${norm}")
            add_halide_library(${kernel} FROM block_matching_generator
                GENERATOR alignment_block_matching USE_RUNTIME alignment_runtime
                TARGETS host PARAMS tile_size=${tile} search_radius=${radius} norm=${norm})
            list(APPEND block_matching_libraries ${kernel})
            string(APPEND bm_headers "#include \"${kernel}.h\"\n")
            string(APPEND bm_entries "    {${tile}, ${radius}, Norm::${norm}, ${kernel}},\n")
        endforeach()
    endforeach()
endforeach()
configure_file(block_matching_variants.h.in block_matching_variants.h @ONLY)

add_halide_generator(flow_transfer_generator SOURCES flow_transfer.cpp)
set(flow_transfer_libraries "")
set(transfer_headers "")
set(transfer_entries "")
# Include the fixed final 16x16 grid, even if BM uses other tile sizes only.
set(flow_transfer_tile_sizes 16 ${ALIGNMENT_BM_TILE_SIZES})
list(REMOVE_DUPLICATES flow_transfer_tile_sizes)
foreach(tile IN LISTS flow_transfer_tile_sizes)
    set(kernel "alignment_flow_transfer_${tile}")
    add_halide_library(${kernel} FROM flow_transfer_generator
        GENERATOR alignment_flow_transfer USE_RUNTIME alignment_runtime
        TARGETS host PARAMS tile_size=${tile})
    list(APPEND flow_transfer_libraries ${kernel})
    string(APPEND transfer_headers "#include \"${kernel}.h\"\n")
    string(APPEND transfer_entries "    {${tile}, ${kernel}},\n")
endforeach()
configure_file(flow_transfer_variants.h.in flow_transfer_variants.h @ONLY)

add_halide_generator(flow_median_generator SOURCES flow_median.cpp)
add_halide_library(alignment_flow_median FROM flow_median_generator
    GENERATOR alignment_flow_median USE_RUNTIME alignment_runtime TARGETS host)

add_library(alignment_core STATIC alignment.cpp kernels.cpp)
target_link_libraries(alignment_core PUBLIC Halide::Runtime PRIVATE ${downsample_libraries} ${gradient_libraries} ${block_matching_libraries} ${flow_transfer_libraries} alignment_flow_median)
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
    target_include_directories(test_downsample PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
    add_test(NAME downsample COMMAND test_downsample)
    add_executable(test_gradients test_gradients.cpp)
    target_link_libraries(test_gradients PRIVATE alignment_core)
    target_include_directories(test_gradients PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
    add_test(NAME gradients COMMAND test_gradients)
    add_executable(test_block_matching test_block_matching.cpp)
    target_include_directories(test_block_matching PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
    target_link_libraries(test_block_matching PRIVATE alignment_core)
    add_test(NAME block_matching COMMAND test_block_matching)
    add_executable(test_flow_transfer test_flow_transfer.cpp)
    target_include_directories(test_flow_transfer PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
    target_link_libraries(test_flow_transfer PRIVATE alignment_core)
    add_test(NAME flow_transfer COMMAND test_flow_transfer)
    add_executable(test_flow_median test_flow_median.cpp)
    target_link_libraries(test_flow_median PRIVATE alignment_core)
    add_test(NAME flow_median COMMAND test_flow_median)
endif()
