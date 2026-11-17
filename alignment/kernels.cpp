#include "kernels.h"
#include "block_matching_variants.h"
#include "downsample_variants.h"
#include "gradient_variants.h"
#include "flow_transfer_variants.h"
#include <cmath>
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
  detail::DownsampleKernel kernel = nullptr;
  for (const auto &variant : detail::downsample_variants)
    if (variant.scale_factor == scale_factor) {
      kernel = variant.kernel;
      break;
    }
  if (!kernel)
    throw std::invalid_argument("unsupported downsample scale (not compiled): " + std::to_string(scale_factor));
  const int width =
      image.width() / scale_factor + (image.width() % scale_factor != 0);
  const int height =
      image.height() / scale_factor + (image.height() % scale_factor != 0);
  Image output(width, height);
  // Halide's ABI takes mutable buffer descriptors even for read-only inputs.
  Image input = image;
  const int status = kernel(input.raw_buffer(), output.raw_buffer());
  if (status != 0)
    throw std::runtime_error("Halide downsample failed with status " +
                             std::to_string(status));
  return output;
}
std::vector<Image> precompute_reference(const Image &image, int tile_size) {
  detail::GradientKernel kernel = nullptr;
  for (const auto &variant : detail::gradient_variants)
    if (variant.tile_size == tile_size) {
      kernel = variant.kernel;
      break;
    }
  if (!kernel)
    throw std::invalid_argument("unsupported gradient tile size (not compiled): " + std::to_string(tile_size));
  if (image.dimensions() != 2 || image.width() <= 0 || image.height() <= 0 ||
      image.dim(0).min() != 0 || image.dim(1).min() != 0)
    throw std::invalid_argument(
        "gradients expect a nonempty zero-based 2D image");
  const int width = image.width(), height = image.height();
  if (width < tile_size || height < tile_size)
    throw std::invalid_argument(
        "image must contain at least one complete tile");
  Image gx(width, height), gy(width, height);
  const int tiles_x = width / tile_size, tiles_y = height / tile_size;
  Image sum_gxx(tiles_x, tiles_y), sum_gxy(tiles_x, tiles_y),
      sum_gyy(tiles_x, tiles_y);
  Image input = image;
  const int status = kernel(input.raw_buffer(), gx.raw_buffer(), gy.raw_buffer(),
                            sum_gxx.raw_buffer(), sum_gxy.raw_buffer(), sum_gyy.raw_buffer());
  if (status != 0)
    throw std::runtime_error("Halide gradients failed with status " +
                             std::to_string(status));
  return {gx, gy, sum_gxx, sum_gxy, sum_gyy};
}
Flow block_matching(const Image &reference, const Image &moving,
                    const Flow &initial_flow, int tile_size, int search_radius,
                    Norm norm) {
  const detail::BlockMatchingVariant *selected = nullptr;
  for (const auto &variant : detail::block_matching_variants)
    if (variant.tile_size == tile_size && variant.search_radius == search_radius && variant.norm == norm) {
      selected = &variant;
      break;
    }
  if (!selected)
    throw std::invalid_argument("block matching variant not compiled: tile=" + std::to_string(tile_size) +
                                ", radius=" + std::to_string(search_radius) +
                                ", norm=" + (norm == Norm::L1 ? "L1" : norm == Norm::L2 ? "L2" : "invalid"));
  if (reference.dimensions() != 2 || moving.dimensions() != 2 ||
      reference.width() < tile_size || reference.height() < tile_size ||
      reference.width() != moving.width() || reference.height() != moving.height() ||
      reference.dim(0).min() != 0 || reference.dim(1).min() != 0 ||
      moving.dim(0).min() != 0 || moving.dim(1).min() != 0)
    throw std::invalid_argument("block matching expects matching zero-based images containing a complete tile");
  const int cols = reference.width() / tile_size, rows = reference.height() / tile_size;
  if (initial_flow.tile_size != tile_size || initial_flow.stride != tile_size ||
      initial_flow.dx.dimensions() != 2 || initial_flow.dy.dimensions() != 2 ||
      initial_flow.dx.width() != cols || initial_flow.dx.height() != rows ||
      initial_flow.dy.width() != cols || initial_flow.dy.height() != rows ||
      initial_flow.dx.dim(0).min() != 0 || initial_flow.dx.dim(1).min() != 0 ||
      initial_flow.dy.dim(0).min() != 0 || initial_flow.dy.dim(1).min() != 0)
    throw std::invalid_argument("initial flow must match the non-overlapping block-matching tile grid");
  Flow output(cols, rows, tile_size, tile_size);
  Image ref = reference, mov = moving;
  auto initial_dx = initial_flow.dx, initial_dy = initial_flow.dy;
  const int status = selected->kernel(ref.raw_buffer(), mov.raw_buffer(),
                                       initial_dx.raw_buffer(), initial_dy.raw_buffer(),
                                       output.dx.raw_buffer(), output.dy.raw_buffer());
  if (status != 0)
    throw std::runtime_error("Halide block matching failed with status " + std::to_string(status));
  return output;
}

Flow ica(const PyramidLevel &, const Image &, const Flow &, int) {
  throw KernelNotImplemented("Halide ICA kernel is not implemented");
}

Flow transfer_flow(const Flow &source, const Image &reference, const Image &moving,
                   int target_tile_size, int target_stride, float scale_ratio,
                   float offset_x, float offset_y) {
  detail::FlowTransferKernel kernel = nullptr;
  for (const auto &variant : detail::flow_transfer_variants)
    if (variant.tile_size == target_tile_size) {
      kernel = variant.kernel;
      break;
    }
  if (!kernel)
    throw std::invalid_argument("flow transfer tile size not compiled: " + std::to_string(target_tile_size));
  if (target_stride <= 0 || source.stride <= 0 || source.tile_size <= 0 ||
      !std::isfinite(scale_ratio) || scale_ratio <= 0 ||
      !std::isfinite(offset_x) || !std::isfinite(offset_y))
    throw std::invalid_argument("flow transfer requires positive strides/scale and finite offsets");
  if (reference.dimensions() != 2 || moving.dimensions() != 2 ||
      reference.width() < target_tile_size || reference.height() < target_tile_size ||
      reference.width() != moving.width() || reference.height() != moving.height() ||
      reference.dim(0).min() != 0 || reference.dim(1).min() != 0 ||
      moving.dim(0).min() != 0 || moving.dim(1).min() != 0)
    throw std::invalid_argument("flow transfer expects matching zero-based images containing a complete tile");
  if (source.dx.dimensions() != 2 || source.dy.dimensions() != 2 ||
      source.dx.width() <= 0 || source.dx.height() <= 0 ||
      source.dx.width() != source.dy.width() || source.dx.height() != source.dy.height() ||
      source.dx.dim(0).min() != 0 || source.dx.dim(1).min() != 0 ||
      source.dy.dim(0).min() != 0 || source.dy.dim(1).min() != 0)
    throw std::invalid_argument("flow transfer expects matching nonempty zero-based flow components");
  const float flow_scale = scale_ratio * target_stride / source.stride;
  if (!std::isfinite(flow_scale) || flow_scale <= 0)
    throw std::invalid_argument("flow displacement scale must be finite and positive");
  const int cols = (reference.width() - target_tile_size) / target_stride + 1;
  const int rows = (reference.height() - target_tile_size) / target_stride + 1;
  Flow output(cols, rows, target_tile_size, target_stride);
  auto source_dx = source.dx, source_dy = source.dy;
  Image ref = reference, mov = moving;
  const int status = kernel(source_dx.raw_buffer(), source_dy.raw_buffer(),
                            ref.raw_buffer(), mov.raw_buffer(), target_stride,
                            scale_ratio, offset_x, offset_y, flow_scale,
                            output.dx.raw_buffer(), output.dy.raw_buffer());
  if (status != 0)
    throw std::runtime_error("Halide flow transfer failed with status " + std::to_string(status));
  return output;
}
} // namespace alignment::kernels
