#include "kernels.h"
#include "flow_transfer_variants.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

using namespace alignment;

float edge(const Image &image, int x, int y) {
  return image(std::clamp(x, 0, image.width() - 1), std::clamp(y, 0, image.height() - 1));
}

Flow scalar_transfer(const Flow &source, const Image &ref, const Image &mov,
                     int tile, int stride, float ratio, float ox, float oy) {
  Flow result((ref.width() - tile) / stride + 1, (ref.height() - tile) / stride + 1, tile, stride);
  const float flow_scale = ratio * stride / source.stride;
  for (int ty = 0; ty < result.tiles_y(); ++ty)
    for (int tx = 0; tx < result.tiles_x(); ++tx) {
      const float u = (tx - ox) / ratio, v = (ty - oy) / ratio;
      const int left = static_cast<int>(std::floor(u)), top = static_cast<int>(std::floor(v));
      const int nx = static_cast<int>(std::floor(u + 0.5f)), ny = static_cast<int>(std::floor(v + 0.5f));
      const int xs[] = {nx, nx == left ? left + 1 : left, nx};
      const int ys[] = {ny, ny, ny == top ? top + 1 : top};
      double best = std::numeric_limits<double>::infinity();
      for (int c = 0; c < 3; ++c) {
        const float candidate_dx = edge(source.dx, xs[c], ys[c]) * flow_scale;
        const float candidate_dy = edge(source.dy, xs[c], ys[c]) * flow_scale;
        const int dx = static_cast<int>(std::nearbyint(candidate_dx));
        const int dy = static_cast<int>(std::nearbyint(candidate_dy));
        double cost = 0;
        for (int y = ty * stride; y < ty * stride + tile; ++y)
          for (int x = tx * stride; x < tx * stride + tile; ++x)
            cost += std::abs(static_cast<double>(edge(ref, x, y)) - edge(mov, x + dx, y + dy));
        if (cost < best) {
          best = cost;
          result.dx(tx, ty) = candidate_dx;
          result.dy(tx, ty) = candidate_dy;
        }
      }
    }
  return result;
}

void compare(int tile, int stride, float ratio, float ox, float oy, bool constant) {
  Image ref(2 * tile + 3, 2 * tile + 5), mov(ref.width(), ref.height());
  std::mt19937 random(123);
  for (int y = 0; y < ref.height(); ++y)
    for (int x = 0; x < ref.width(); ++x) {
      ref(x, y) = constant ? 0.25f : static_cast<float>(random() % 65536) / 65535.0f;
      mov(x, y) = constant ? 0.25f : static_cast<float>(random() % 65536) / 65535.0f;
    }
  Flow source(3, 2, 8, 8);
  for (int y = 0; y < source.tiles_y(); ++y)
    for (int x = 0; x < source.tiles_x(); ++x) {
      source.dx(x, y) = (x - 1) * 0.75f + y * 0.25f;
      source.dy(x, y) = (y - 1) * 0.5f - x * 0.25f;
    }
  auto before_x = source.dx.copy(), before_y = source.dy.copy();
  Flow expected = scalar_transfer(source, ref, mov, tile, stride, ratio, ox, oy);
  Flow actual = kernels::transfer_flow(source, ref, mov, tile, stride, ratio, ox, oy);
  if (actual.tiles_x() != expected.tiles_x() || actual.tiles_y() != expected.tiles_y() ||
      actual.tile_size != tile || actual.stride != stride)
    throw std::runtime_error("incorrect target flow grid");
  for (int y = 0; y < actual.tiles_y(); ++y)
    for (int x = 0; x < actual.tiles_x(); ++x)
      if (actual.dx(x, y) != expected.dx(x, y) || actual.dy(x, y) != expected.dy(x, y))
        throw std::runtime_error("flow transfer differs from scalar search, tile=" + std::to_string(tile));
  for (int y = 0; y < source.tiles_y(); ++y)
    for (int x = 0; x < source.tiles_x(); ++x)
      if (source.dx(x, y) != before_x(x, y) || source.dy(x, y) != before_y(x, y))
        throw std::runtime_error("source flow modified");
}

void candidate_selection(int tile) {
  Image image(tile, tile);
  for (int y = 0; y < tile; ++y)
    for (int x = 0; x < tile; ++x) image(x, y) = float(x) / tile;
  Flow source(2, 2, tile, tile);
  // Target center maps to (0.25, 0.25). The omitted diagonal has perfect flow
  // zero; the actual candidates have shifts 1, 2, 3. Nearest must win with 1.
  source.dx(0, 0) = 1; source.dx(1, 0) = 2;
  source.dx(0, 1) = 3; source.dx(1, 1) = 0;
  auto out = kernels::transfer_flow(source, image, image, tile, tile, 1, -0.25f, -0.25f);
  if (out.dx(0, 0) != 1) throw std::runtime_error("diagonal was included or nearest candidate omitted");
  source.dx(1, 0) = 0;
  out = kernels::transfer_flow(source, image, image, tile, tile, 1, -0.25f, -0.25f);
  if (out.dx(0, 0) != 0) throw std::runtime_error("horizontal candidate cannot win");
  source.dx(1, 0) = 2; source.dx(0, 1) = 0;
  out = kernels::transfer_flow(source, image, image, tile, tile, 1, -0.25f, -0.25f);
  if (out.dx(0, 0) != 0) throw std::runtime_error("vertical candidate cannot win");
  // Scoring rounds after scaling, but the winner retains its fractional value.
  image.fill(0.0f); source.dx.fill(0.4f); source.dy.fill(-0.4f);
  out = kernels::transfer_flow(source, image, image, tile, tile, 1.5f, 0, 0);
  if (std::abs(out.dx(0, 0) - 0.6f) > 1e-6f || std::abs(out.dy(0, 0) + 0.6f) > 1e-6f)
    throw std::runtime_error("fractional winner was rounded or not scaled");
  // Identical rounded samples must retain the nearest candidate's own floats.
  source.dx.fill(0.4f); source.dy.fill(-0.3f);
  source.dx(0, 0) = 0.2f;
  out = kernels::transfer_flow(source, image, image, tile, tile, 1, -0.25f, -0.25f);
  if (out.dx(0, 0) != 0.2f || out.dy(0, 0) != -0.3f)
    throw std::runtime_error("equal quantized costs lost the nearest floating candidate");
  // A non-nearest winner must also return its original float, not the sampled int.
  Image shifted(tile, tile);
  for (int y = 0; y < tile; ++y)
    for (int x = 0; x < tile; ++x) {
      image(x, y) = float(x) / tile;
      shifted(x, y) = float(std::max(0, x - 1)) / tile;
    }
  source.dx.fill(3.3f); source.dy.fill(0);
  source.dx(0, 0) = 2.2f; source.dx(1, 0) = 1.2f;
  out = kernels::transfer_flow(source, image, shifted, tile, tile, 1, -0.25f, -0.25f);
  if (out.dx(0, 0) != 1.2f)
    throw std::runtime_error("non-nearest winner lost its fractional displacement");
  image.fill(0.0f);
  // Exact halfway between grid centers: nearest uses the larger grid index.
  source.dx.fill(0); source.dy.fill(0); source.dx(1, 1) = 2;
  out = kernels::transfer_flow(source, image, image, tile, tile, 1, -0.5f, -0.5f);
  if (out.dx(0, 0) != 2) throw std::runtime_error("nearest-grid tie handling changed");
}

// Verify the caller's grid transform against actual even-index pixel locations,
// rather than repeating the inverse grid transform used by the kernel.
void even_sample_geometry(int target_tile) {
  for (int source_tile : {8, 16, 32})
    for (int factor : {1, 2, 4, 8}) {
      const int source_stride = source_tile, target_stride = target_tile;
      Flow source(3, 2, source_tile, source_stride);
      for (int y = 0; y < source.tiles_y(); ++y)
        for (int x = 0; x < source.tiles_x(); ++x) {
          source.dx(x, y) = static_cast<float>(x + 10 * y);
          source.dy(x, y) = -source.dx(x, y);
        }
      Image image(5 * target_tile + 3, 3 * target_tile + 5);
      image.fill(0.0f); // Equal costs force selection of the geometrically nearest flow.
      const float ratio = float(factor * source_stride) / target_stride;
      const float offset = (factor * (source_tile - 1) * 0.5f - (target_tile - 1) * 0.5f) / target_stride;
      auto actual = kernels::transfer_flow(source, image, image, target_tile,
                                           target_stride, ratio, offset, offset);
      for (int y = 0; y < actual.tiles_y(); ++y)
        for (int x = 0; x < actual.tiles_x(); ++x) {
          // Downsampling discards odd pixels: coarse pixel p lies at fine factor*p.
          const double cx = (x * target_stride + (target_tile - 1) * 0.5) / factor;
          const double cy = (y * target_stride + (target_tile - 1) * 0.5) / factor;
          const int sx = std::clamp(int(std::floor((cx - (source_tile - 1) * 0.5) / source_stride + 0.5)),
                                    0, source.tiles_x() - 1);
          const int sy = std::clamp(int(std::floor((cy - (source_tile - 1) * 0.5) / source_stride + 0.5)),
                                    0, source.tiles_y() - 1);
          if (actual.dx(x, y) != factor * source.dx(sx, sy) ||
              actual.dy(x, y) != factor * source.dy(sx, sy))
            throw std::runtime_error("flow grid disagrees with even-index pixel geometry");
        }
    }
}

int main() {
  try {
    for (const auto &variant : kernels::detail::flow_transfer_variants) {
      const int tile = variant.tile_size;
      for (int stride : {tile, tile / 2})
        for (float ratio : {0.75f, 1.5f, 2.0f})
          for (bool constant : {false, true})
            compare(tile, stride, ratio, 0.25f, -0.35f, constant);
      compare(tile, tile, 1.5f, 20, -20, false); // Repeat source-grid edges.
      candidate_selection(tile);
      even_sample_geometry(tile);
    }
    Image image(16, 16); Flow source(1, 1, 8, 8);
    for (float ratio : {0.0f, -1.0f, std::numeric_limits<float>::infinity()}) {
      bool rejected = false;
      try { kernels::transfer_flow(source, image, image, 8, 8, ratio, 0, 0); }
      catch (const std::invalid_argument &) { rejected = true; }
      if (!rejected) throw std::runtime_error("invalid scale accepted");
    }
    std::cout << "Flow transfer checks passed\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
