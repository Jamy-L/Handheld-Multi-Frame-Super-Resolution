#include "kernels.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

using Image = Halide::Runtime::Buffer<float>;
void check(int tile) {
  for (int extra : {0, 3}) {
    const int width = 2 * tile + extra, height = tile + extra;
    Image image(width, height);
    for (int y = 0; y < height; ++y)
      for (int x = 0; x < width; ++x)
        image(x, y) = float((x * 17 + y * 13) % 113) / 112.0f;
    auto outputs = alignment::kernels::precompute_reference(image, tile);
    if (outputs.size() != 5) throw std::runtime_error("expected five gradient outputs");
    const Image &gx = outputs[0], &gy = outputs[1];
    const Image &sum_gxx = outputs[2], &sum_gxy = outputs[3], &sum_gyy = outputs[4];
    for (size_t i = 0; i < outputs.size(); ++i) {
      const int expected_width = i < 2 ? width : width / tile;
      const int expected_height = i < 2 ? height : height / tile;
      if (outputs[i].width() != expected_width || outputs[i].height() != expected_height)
        throw std::runtime_error("incorrect gradient output shape");
    }
    Image expected_gx(width, height), expected_gy(width, height);
    auto pixel = [&](int x, int y) {
      return image(std::clamp(x, 0, width - 1), std::clamp(y, 0, height - 1));
    };
    for (int y = 0; y < height; ++y)
      for (int x = 0; x < width; ++x) {
        float dx = (pixel(x - 2, y) - 8.0f * pixel(x - 1, y) +
                    8.0f * pixel(x + 1, y) - pixel(x + 2, y)) / 12.0f;
        float dy = (pixel(x, y - 2) - 8.0f * pixel(x, y - 1) +
                    8.0f * pixel(x, y + 1) - pixel(x, y + 2)) / 12.0f;
        expected_gx(x, y) = dx;
        expected_gy(x, y) = dy;
        if (std::abs(gx(x, y) - dx) > 2e-6f || std::abs(gy(x, y) - dy) > 2e-6f)
          throw std::runtime_error("incorrect gradients");
      }
    for (int ty = 0; ty < height / tile; ++ty)
      for (int tx = 0; tx < width / tile; ++tx) {
        double expected_xx = 0, expected_xy = 0, expected_yy = 0;
        for (int y = ty * tile; y < (ty + 1) * tile; ++y)
          for (int x = tx * tile; x < (tx + 1) * tile; ++x) {
            const double dx = expected_gx(x, y);
            const double dy = expected_gy(x, y);
            expected_xx += dx * dx;
            expected_xy += dx * dy;
            expected_yy += dy * dy;
          }
        auto check_sum = [](float actual, double expected) {
          // Up to 4096 float32 additions, compared with a double-precision sum.
          if (std::abs(actual - expected) > 1e-4 * std::max(1.0, std::abs(expected)))
            throw std::runtime_error("incorrect patch accumulation: actual=" + std::to_string(actual) + ", expected=" + std::to_string(expected));
        };
        check_sum(sum_gxx(tx, ty), expected_xx);
        check_sum(sum_gxy(tx, ty), expected_xy);
        check_sum(sum_gyy(tx, ty), expected_yy);
      }
  }
}

int main() {
  try {
    check(8);
    check(16);
    check(32);
    check(64);
    for (int tile : {8, 16, 32, 64}) {
      for (auto shape : {std::pair{tile - 1, tile}, {tile, tile - 1}, {1, 1}}) {
        bool rejected = false;
        try { alignment::kernels::precompute_reference(Image(shape.first, shape.second), tile); }
        catch (const std::invalid_argument &) { rejected = true; }
        if (!rejected) throw std::runtime_error("image without a complete tile accepted");
      }
    }
    for (int tile : {0, 4, 12, 128}) {
      bool rejected = false;
      try { alignment::kernels::precompute_reference(Image(16, 16), tile); }
      catch (const std::invalid_argument &) { rejected = true; }
      if (!rejected) throw std::runtime_error("unsupported tile size accepted");
    }
    std::cout << "Gradient checks passed\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
