#pragma once

#include <stdexcept>
#include <vector>
#include "HalideBuffer.h"

namespace alignment {
using Image = Halide::Runtime::Buffer<float>;

// Per-tile displacement in current-level pixels. Components use (tile_x, tile_y).
struct Flow {
    Halide::Runtime::Buffer<float> dx;
    Halide::Runtime::Buffer<float> dy;

    Flow(int tiles_x, int tiles_y) {
        if (tiles_x <= 0 || tiles_y <= 0)
            throw std::invalid_argument("flow must contain at least one tile");
        dx = Halide::Runtime::Buffer<float>(tiles_x, tiles_y);
        dy = Halide::Runtime::Buffer<float>(tiles_x, tiles_y);
        dx.fill(0.0f);
        dy.fill(0.0f);
    }

    int tiles_x() const { return dx.width(); }
    int tiles_y() const { return dx.height(); }
};

enum class Norm { L1, L2 };

struct PyramidLevel {
    Image image;
    // Reference only: gx, gy, sum_gxx, sum_gxy, sum_gyy.
    std::vector<Image> precomputed;
};

struct Pyramid {
    bool is_reference;

    // Downsampling ratios between successive levels, finest to coarsest.
    std::vector<int> scale_factors;
    std::vector<int> tile_sizes;
    std::vector<int> search_radii;
    std::vector<Norm> norm;
    int ica_iterations;
    std::vector<PyramidLevel> levels;
};

struct KernelNotImplemented : std::logic_error {
    using std::logic_error::logic_error;
};

Pyramid build_reference_pyramid(const Image &image, const std::vector<int> &scale_factors,
                                const std::vector<int> &tile_sizes, const std::vector<int> &search_radii,
                                const std::vector<Norm> &norm, int ica_iterations);
Pyramid build_moving_pyramid(const Image &image, const std::vector<int> &scale_factors,
                                const std::vector<int> &tile_sizes, const std::vector<int> &search_radii,
                                const std::vector<Norm> &norm, int ica_iterations);
// Initializes zero flow on the coarsest tile grid.
Flow align(const Pyramid &reference, const Pyramid &moving);
}  // namespace alignment
