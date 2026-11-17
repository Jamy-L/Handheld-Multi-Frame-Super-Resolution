#include "kernels.h"
#include "block_matching_variants.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

using namespace alignment;

// Independent exhaustive scalar search; double accumulation avoids reproducing
// the generated reduction. Candidate order also checks deterministic tie handling.
Flow expected_flow(const Image &reference, const Image &moving, const Flow &initial,
                   int tile, int radius, Norm norm) {
  Flow output(initial.tiles_x(), initial.tiles_y(), tile, tile);
  for (int ty = 0; ty < initial.tiles_y(); ++ty)
    for (int tx = 0; tx < initial.tiles_x(); ++tx) {
      const int cx = static_cast<int>(std::nearbyint(initial.dx(tx, ty)));
      const int cy = static_cast<int>(std::nearbyint(initial.dy(tx, ty)));
      double best = std::numeric_limits<double>::infinity();
      for (int sy = -radius; sy <= radius; ++sy)
        for (int sx = -radius; sx <= radius; ++sx) {
          double cost = 0;
          for (int y = ty * tile; y < (ty + 1) * tile; ++y)
            for (int x = tx * tile; x < (tx + 1) * tile; ++x) {
              const int mx = x + cx + sx, my = y + cy + sy;
              const double value = moving(std::clamp(mx, 0, moving.width() - 1),
                                          std::clamp(my, 0, moving.height() - 1));
              const double difference = reference(x, y) - value;
              cost += norm == Norm::L1 ? std::abs(difference) : difference * difference;
            }
          if (cost < best) {
            best = cost;
            output.dx(tx, ty) = static_cast<float>(cx + sx);
            output.dy(tx, ty) = static_cast<float>(cy + sy);
          }
        }
    }
  return output;
}

void check(int tile, int radius, Norm norm) {
  std::mt19937 rng(42);
  for (int pattern = 0; pattern < 3; ++pattern) {
    // Deliberately non-square and not divisible by the tile size.
    Image reference(2 * tile + 3, tile + 5), moving(reference.width(), reference.height());
    for (int y = 0; y < reference.height(); ++y)
      for (int x = 0; x < reference.width(); ++x)
        reference(x, y) = pattern == 0 ? 0.0f : static_cast<float>(rng() % 65536) / 65535.0f;
    moving.fill(0.0f);
    for (int y = 0; y < reference.height(); ++y)
      for (int x = 0; x < reference.width(); ++x) {
        if (pattern != 2) moving(x, y) = reference(x, y);
        else if (x + 1 < reference.width() && y > 0) moving(x + 1, y - 1) = reference(x, y);
      }
    Flow initial(reference.width() / tile, reference.height() / tile, tile, tile);
    if (pattern == 2) {
      initial.dx.fill(0.6f);
      initial.dy.fill(-0.6f);
    }
    Flow expected = expected_flow(reference, moving, initial, tile, radius, norm);
    Flow actual = kernels::block_matching(reference, moving, initial, tile, radius, norm);
    for (int ty = 0; ty < initial.tiles_y(); ++ty)
      for (int tx = 0; tx < initial.tiles_x(); ++tx) {
        if (actual.dx(tx, ty) != expected.dx(tx, ty) || actual.dy(tx, ty) != expected.dy(tx, ty))
          throw std::runtime_error("incorrect block matching: tile=" + std::to_string(tile) +
                                   ", radius=" + std::to_string(radius));
        if (initial.dx(tx, ty) != (pattern == 2 ? 0.6f : 0.0f) ||
            initial.dy(tx, ty) != (pattern == 2 ? -0.6f : 0.0f))
          throw std::runtime_error("initial flow modified");
      }
  }
}

int main() {
  try {
    for (const auto &variant : kernels::detail::block_matching_variants)
      check(variant.tile_size, variant.search_radius, variant.norm);
    // An invalid norm cannot match any compiled variant.
    bool rejected = false;
    try {
      kernels::block_matching(Image(16, 16), Image(16, 16), Flow(1, 1, 16, 16),
                              16, 1, static_cast<Norm>(99));
    } catch (const std::invalid_argument &) { rejected = true; }
    if (!rejected) throw std::runtime_error("unsupported variant accepted");
    std::cout << "Compiled block-matching variants match scalar reference\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
