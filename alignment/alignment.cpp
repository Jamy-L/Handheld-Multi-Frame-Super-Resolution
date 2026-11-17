#include "alignment.h"
#include "kernels.h"
#include <utility>

namespace alignment {
namespace {
// One implementation, specialized at compile time for reference-only work.
template <bool IsReference>
Pyramid build_pyramid(const Image &image, const std::vector<int> &scale_factors,
                      const std::vector<int> &tile_sizes,
                      const std::vector<int> &search_radii,
                      const std::vector<Norm> &norm, int ica_iterations) {
  const size_t levels = scale_factors.size() + 1;
  if (tile_sizes.size() != levels || search_radii.size() != levels ||
      norm.size() != levels)
    throw std::invalid_argument("tile_sizes and search_radii and norm must "
                                "have len(scale_factors) + 1 entries");
  if (norm.size() != levels)
    throw std::invalid_argument(
        "norm must have len(scale_factors) + 1 entries");
  for (Norm value : norm)
    if (value != Norm::L1 && value != Norm::L2)
      throw std::invalid_argument("norm entries must be L1 or L2");
  if (ica_iterations < 0)
    throw std::invalid_argument("ica_iterations must be nonnegative");
  for (int tile : tile_sizes)
    if (tile <= 0)
      throw std::invalid_argument("tile_sizes must be positive integers");
  for (int radius : search_radii)
    if (radius < 0)
      throw std::invalid_argument("search_radii must be nonnegative integers");
  for (int factor : scale_factors)
    if (factor <= 0 || (factor & (factor - 1)) != 0)
      throw std::invalid_argument("scale_factors must be positive powers of 2");
  Pyramid pyramid{IsReference, scale_factors,  tile_sizes, search_radii,
                  norm,        ica_iterations, {}};
  Image current = image.copy();
  for (size_t i = 0; i < levels; ++i) {
    if (i > 0)
      current = kernels::downsample(current, scale_factors[i - 1]);
    PyramidLevel level{current, {}};
    if constexpr (IsReference)
      level.precomputed = kernels::precompute_reference(current, tile_sizes[i]);
    pyramid.levels.push_back(std::move(level));
  }
  return pyramid;
}
} // namespace
Pyramid build_reference_pyramid(const Image &image,
                                const std::vector<int> &scale_factors,
                                const std::vector<int> &tile_sizes,
                                const std::vector<int> &search_radii,
                                const std::vector<Norm> &norm,
                                int ica_iterations) {
  return build_pyramid<true>(image, scale_factors, tile_sizes, search_radii,
                             norm, ica_iterations);
}
Pyramid build_moving_pyramid(const Image &image,
                             const std::vector<int> &scale_factors,
                             const std::vector<int> &tile_sizes,
                             const std::vector<int> &search_radii,
                             const std::vector<Norm> &norm,
                             int ica_iterations) {
  return build_pyramid<false>(image, scale_factors, tile_sizes, search_radii,
                              norm, ica_iterations);
}
Flow align(const Pyramid &reference, const Pyramid &moving) {
  if (!reference.is_reference || moving.is_reference)
    throw std::invalid_argument(
        "expected reference and moving pyramids in that order");
  if (reference.levels.empty() ||
      reference.levels.size() != moving.levels.size())
    throw std::invalid_argument(
        "pyramids must have the same nonzero number of levels");
  const auto &a = reference;
  const auto &b = moving;
  if (a.scale_factors != b.scale_factors || a.tile_sizes != b.tile_sizes ||
      a.search_radii != b.search_radii || a.norm != b.norm ||
      a.ica_iterations != b.ica_iterations)
    throw std::invalid_argument(
        "reference and moving pyramid options must match");
  for (size_t i = 0; i < reference.levels.size(); ++i) {
    if (reference.levels[i].image.width() != moving.levels[i].image.width() ||
        reference.levels[i].image.height() != moving.levels[i].image.height())
      throw std::invalid_argument("pyramid shapes must match");
  }
  const size_t coarsest = reference.levels.size() - 1;
  const auto &image = reference.levels[coarsest].image;
  const int tile_size = reference.tile_sizes[coarsest];
  const int tiles_x = image.width() / tile_size;
  const int tiles_y = image.height() / tile_size;
  if (tiles_x == 0 || tiles_y == 0)
    throw std::invalid_argument("coarsest image must contain at least one complete tile");

  // One (dx, dy) vector per complete tile, in coarsest-level pixels.
  Flow flow(tiles_x, tiles_y);
  flow = kernels::align_level(reference.levels[coarsest], moving.levels[coarsest],
                             flow, tile_size, reference.search_radii[coarsest],
                             reference.norm[coarsest], reference.ica_iterations);
  // TODO: upscale tile flow and refine at each finer level.
  if (coarsest != 0)
    throw KernelNotImplemented("coarse-to-fine alignment is not implemented");
  return flow;
}
} // namespace alignment
