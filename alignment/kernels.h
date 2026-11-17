#pragma once
#include "alignment.h"

// Integration boundary for future dedicated AOT Halide kernels.
namespace alignment::kernels {
Image downsample(const Image &image, int scale_factor);
// Returns gx, gy, sum_gxx, sum_gxy, sum_gyy, in that order. Partial tiles are skipped.
std::vector<Image> precompute_reference(const Image &image, int tile_size);
Flow align_level(const PyramidLevel &reference, const PyramidLevel &moving,
                  const Flow &initial_flow, int tile_size, int search_radius,
                  Norm norm, int ica_iterations);
}  // namespace alignment::kernels
