#pragma once
#include "alignment.h"

// Integration boundary for future dedicated AOT Halide kernels.
namespace alignment::kernels {
Image downsample(const Image &image, int scale_factor);
// Returns gx, gy, sum_gxx, sum_gxy, sum_gyy, in that order. Partial tiles are skipped.
std::vector<Image> precompute_reference(const Image &image, int tile_size);
// Dispatches to the tile-size/search-radius/norm variants listed in CMake.
// Uncompiled combinations throw; the AOT kernels take only buffers.
Flow block_matching(const Image &reference, const Image &moving,
                    const Flow &initial_flow, int tile_size, int search_radius,
                    Norm norm);

// Reference includes cached gradients and per-tile gradient products.
Flow ica(const PyramidLevel &reference, const Image &moving,
         const Flow &initial_flow, int iterations);

// Images are the reference and moving images at the target pyramid level.
// Grid transform: target_tile_index = scale_ratio * source_tile_index + offset.
// The offsets are in target-grid index units and refer to tile centers.
// Pixel displacement scale = scale_ratio * target_stride / source.stride.
// Select nearest/horizontal/vertical source flows by target-tile L1 cost.
// Scale before integer rounding. Images and source flow use repeated edges.
Flow transfer_flow(const Flow &source, const Image &reference, const Image &moving,
                   int target_tile_size, int target_stride,
                   float scale_ratio, float offset_x, float offset_y);
}  // namespace alignment::kernels
