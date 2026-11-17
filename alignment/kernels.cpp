#include "kernels.h"
#include "alignment_downsample_16.h"
#include "alignment_downsample_2.h"
#include "alignment_downsample_4.h"
#include "alignment_downsample_8.h"
#include "alignment_gradients_8.h"
#include "alignment_gradients_16.h"
#include "alignment_gradients_32.h"
#include "alignment_gradients_64.h"
#include <string>

namespace alignment::kernels {
Image downsample(const Image &image, int scale_factor) {
  if (scale_factor <= 0 || (scale_factor & (scale_factor - 1)) != 0)
    throw std::invalid_argument("scale_factor must be a positive power of 2");
  if (image.dimensions() != 2 || image.width() <= 0 || image.height() <= 0 ||
      image.dim(0).min() != 0 || image.dim(1).min() != 0)
    throw std::invalid_argument(
        "downsample expects a nonempty zero-based 2D image");
  if (scale_factor == 1)
    return image.copy();
  const int width =
      image.width() / scale_factor + (image.width() % scale_factor != 0);
  const int height =
      image.height() / scale_factor + (image.height() % scale_factor != 0);
  Image output(width, height);
  // Halide's ABI takes mutable buffer descriptors even for read-only inputs.
  Image input = image;
  int status;
  switch (scale_factor) {
  case 2:
    status = alignment_downsample_2(input.raw_buffer(), output.raw_buffer());
    break;
  case 4:
    status = alignment_downsample_4(input.raw_buffer(), output.raw_buffer());
    break;
  case 8:
    status = alignment_downsample_8(input.raw_buffer(), output.raw_buffer());
    break;
  case 16:
    status = alignment_downsample_16(input.raw_buffer(), output.raw_buffer());
    break;
  default:
    throw std::invalid_argument(
        "unsupported downsample scale: " + std::to_string(scale_factor) +
        "; supported scales are 1, 2, 4, 8, 16");
  }
  if (status != 0)
    throw std::runtime_error("Halide downsample failed with status " +
                             std::to_string(status));
  return output;
}
std::vector<Image> precompute_reference(const Image &image, int tile_size) {
  if (tile_size != 8 && tile_size != 16 && tile_size != 32 && tile_size != 64)
    throw std::invalid_argument("unsupported gradient tile size: " + std::to_string(tile_size) +
                                "; supported tile sizes are 8, 16, 32, 64");
  if (image.dimensions() != 2 || image.width() <= 0 || image.height() <= 0 ||
      image.dim(0).min() != 0 || image.dim(1).min() != 0)
    throw std::invalid_argument("gradients expect a nonempty zero-based 2D image");
  const int width = image.width(), height = image.height();
  if (width < tile_size || height < tile_size)
    throw std::invalid_argument("image must contain at least one complete tile");
  Image gx(width, height), gy(width, height);
  const int tiles_x = width / tile_size, tiles_y = height / tile_size;
  Image sum_gxx(tiles_x, tiles_y), sum_gxy(tiles_x, tiles_y), sum_gyy(tiles_x, tiles_y);
  Image input = image;
  int status;
  switch (tile_size) {
  case 8:
    status = alignment_gradients_8(input.raw_buffer(), gx.raw_buffer(), gy.raw_buffer(),
                                       sum_gxx.raw_buffer(), sum_gxy.raw_buffer(), sum_gyy.raw_buffer());
    break;
  case 16:
    status = alignment_gradients_16(input.raw_buffer(), gx.raw_buffer(), gy.raw_buffer(),
                                       sum_gxx.raw_buffer(), sum_gxy.raw_buffer(), sum_gyy.raw_buffer());
    break;
  case 32:
    status = alignment_gradients_32(input.raw_buffer(), gx.raw_buffer(), gy.raw_buffer(),
                                       sum_gxx.raw_buffer(), sum_gxy.raw_buffer(), sum_gyy.raw_buffer());
    break;
  case 64:
    status = alignment_gradients_64(input.raw_buffer(), gx.raw_buffer(), gy.raw_buffer(),
                                       sum_gxx.raw_buffer(), sum_gxy.raw_buffer(), sum_gyy.raw_buffer());
    break;
  default:
    throw std::invalid_argument("unsupported gradient tile size");
  }
  if (status != 0)
    throw std::runtime_error("Halide gradients failed with status " + std::to_string(status));
  return {gx, gy, sum_gxx, sum_gxy, sum_gyy};
}
Flow align_level(const PyramidLevel &, const PyramidLevel &, const Flow &,
                  int, int, Norm, int) {
  throw KernelNotImplemented("Halide alignment kernel is not implemented");
}
} // namespace alignment::kernels
