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
// grid_scale_ratio >= 1 is source_spacing_px / target_stride, with both grid
// spacings expressed in target-image pixels.
// flow_scale is an independent multiplier applied to the flow values.
// offset_{x,y}_px = first source tile center - first target tile center, in
// target-image pixels. First tile corners align; centers use tile_size/2, so
// offset = source.tile_size * pixel_scale / 2 - target_tile_size / 2.
// Mapping: source_index = (target_index * target_stride - offset_px) /
//                         (grid_scale_ratio * target_stride).
// Select nearest/horizontal/vertical source flows by integer-sampled L1 cost,
// preserving the winner's fractional displacement in target-level pixels.
// Images and source flow use repeated edges.
Flow transfer_flow(const Flow &source, const Image &reference, const Image &moving,
                   int target_tile_size, int target_stride,
                   float grid_scale_ratio, float flow_scale, float offset_x_px, float offset_y_px);
// Independent 3x3 medians of dx/dy on the tile grid, with repeated edges.
Flow median_filter(const Flow &source);
}  // namespace alignment::kernels
