#include "routines.h"
#include <algorithm>
#include <array>
#include <iostream>
#include <random>
#include <stdexcept>

using namespace alignment;
float scalar_median(const Image &image, int x, int y) {
  std::array<float, 9> values;
  int index = 0;
  for (int j = -1; j <= 1; ++j)
    for (int i = -1; i <= 1; ++i)
      values[index++] = image(std::clamp(x + i, 0, image.width() - 1),
                              std::clamp(y + j, 0, image.height() - 1));
  std::sort(values.begin(), values.end());
  return values[4];
}
int main() {
  try {
    std::mt19937 random(42);
    for (auto shape : {std::pair{1,1}, {1,7}, {7,1}, {2,2}, {7,5}, {31,31}}) {
      Flow input(shape.first, shape.second, 16, 8);
      for (int y = 0; y < input.tiles_y(); ++y)
        for (int x = 0; x < input.tiles_x(); ++x) {
          input.dx(x, y) = (int(random() % 2001) - 1000) * 0.125f;
          input.dy(x, y) = (int(random() % 2001) - 1000) * 0.0625f;
        }
      auto before_x = input.dx.copy(), before_y = input.dy.copy();
      Flow result = routines::median_filter(input);
      if (result.tiles_x() != shape.first || result.tiles_y() != shape.second ||
          result.tile_size != 16 || result.stride != 8)
        throw std::runtime_error("median changed grid metadata");
      for (int y = 0; y < input.tiles_y(); ++y)
        for (int x = 0; x < input.tiles_x(); ++x) {
          if (result.dx(x, y) != scalar_median(before_x, x, y) ||
              result.dy(x, y) != scalar_median(before_y, x, y))
            throw std::runtime_error("median differs from scalar sort");
          if (input.dx(x, y) != before_x(x, y) || input.dy(x, y) != before_y(x, y))
            throw std::runtime_error("median modified its input");
        }
    }
    Flow input(5, 5, 16, 8);
    input.dx.fill(0.375f); input.dy.fill(-0.625f);
    input.dx(2, 2) = 100; input.dy(2, 2) = -100;
    auto result = routines::median_filter(input);
    for (int y = 0; y < 5; ++y)
      for (int x = 0; x < 5; ++x)
        if (result.dx(x, y) != 0.375f || result.dy(x, y) != -0.625f)
          throw std::runtime_error("median failed to remove outlier or preserve subpixel flow");
    std::cout << "Flow median checks passed\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
