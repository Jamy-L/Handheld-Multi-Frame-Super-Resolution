#include "routines.h"
#include "ica_variants.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

using namespace alignment;
float edge(const Image &image, int x, int y) {
  return image(std::clamp(x, 0, image.width()-1), std::clamp(y, 0, image.height()-1));
}
double bilinear(const Image &image, double x, double y) {
  const int ix = int(std::floor(x)), iy = int(std::floor(y));
  const double fx = x - ix, fy = y - iy;
  return (1-fy) * ((1-fx)*edge(image,ix,iy) + fx*edge(image,ix+1,iy)) +
         fy * ((1-fx)*edge(image,ix,iy+1) + fx*edge(image,ix+1,iy+1));
}
Flow scalar_ica(const PyramidLevel &ref, const Image &moving, const Flow &initial, int iterations) {
  const int tile = initial.tile_size;
  Flow result(initial.tiles_x(), initial.tiles_y(), tile, tile);
  result.dx = initial.dx.copy(); result.dy = initial.dy.copy();
  for (int iteration = 0; iteration < iterations; ++iteration)
    for (int ty = 0; ty < result.tiles_y(); ++ty)
      for (int tx = 0; tx < result.tiles_x(); ++tx) {
        double bx=0, by=0;
        for (int y = ty*tile; y < (ty+1)*tile; ++y)
          for (int x = tx*tile; x < (tx+1)*tile; ++x) {
            double residual = ref.image(x,y) - bilinear(moving,x+double(result.dx(tx,ty)),y+double(result.dy(tx,ty)));
            bx += ref.precomputed[0](x,y) * residual;
            by += ref.precomputed[1](x,y) * residual;
          }
        double a=ref.precomputed[2](tx,ty), b=ref.precomputed[3](tx,ty), c=ref.precomputed[4](tx,ty);
        double det=a*c-b*b+std::numeric_limits<float>::epsilon();
        result.dx(tx,ty) += float(std::clamp((c*bx-b*by)/det,-1.0,1.0));
        result.dy(tx,ty) += float(std::clamp((a*by-b*bx)/det,-1.0,1.0));
      }
  return result;
}
void check(int tile) {
  Image image(2*tile+3,2*tile+5), moving(image.width(),image.height());
  std::mt19937 rng(42);
  for (int y=0; y<image.height(); ++y)
    for (int x=0; x<image.width(); ++x) {
      image(x,y)=float(rng()%65536)/65535;
      moving(x,y)=float(rng()%65536)/65535;
    }
  PyramidLevel ref{image,routines::precompute_reference(image,tile)};
  Flow initial(2,2,tile,tile); initial.dx.fill(-0.35f); initial.dy.fill(0.6f);
  for (int iterations : {0,1,3}) {
    auto actual=routines::ica(ref,moving,initial,iterations);
    auto expected=scalar_ica(ref,moving,initial,iterations);
    for (int y=0; y<2; ++y)
      for (int x=0; x<2; ++x) {
        if (!std::isfinite(actual.dx(x,y)) || !std::isfinite(actual.dy(x,y)) ||
            std::abs(actual.dx(x,y)-expected.dx(x,y))>3e-5f ||
            std::abs(actual.dy(x,y)-expected.dy(x,y))>3e-5f)
          throw std::runtime_error("ICA differs from scalar reference, tile="+std::to_string(tile));
        if (initial.dx(x,y)!=-0.35f || initial.dy(x,y)!=0.6f)
          throw std::runtime_error("ICA mutated initial flow");
      }
    actual.dx(0,0)=100;
    if (initial.dx(0,0)!=-0.35f) throw std::runtime_error("ICA result aliases input");
  }
  // Flat/singular patches must remain finite and preserve an existing flow.
  image.fill(0.5f); moving.fill(0.25f);
  ref.precomputed=routines::precompute_reference(image,tile);
  auto flat=routines::ica(ref,moving,initial,3);
  if (flat.dx(0,0)!=-0.35f || flat.dy(0,0)!=0.6f)
    throw std::runtime_error("singular patch changed flow");
}
void regularization_and_clipping() {
  constexpr int tile=8;
  Image ref_image(tile,tile), moving(tile,tile);
  ref_image.fill(0.6f); moving.fill(0.5f);
  Image gx(tile,tile),gy(tile,tile),xx(1,1),xy(1,1),yy(1,1);
  gx.fill(0.01f); gy.fill(0.02f); xx.fill(2); xy.fill(1); yy.fill(3);
  PyramidLevel ref{ref_image,{gx,gy,xx,xy,yy}};
  Flow initial(1,1,tile,tile); initial.dx.fill(2.25f); initial.dy.fill(-3.125f);
  auto result=routines::ica(ref,moving,initial,3);
  auto expected=scalar_ica(ref,moving,initial,3);
  if (std::abs(result.dx(0,0)-expected.dx(0,0))>2e-6f ||
      std::abs(result.dy(0,0)-expected.dy(0,0))>2e-6f)
    throw std::runtime_error("determinant must include float epsilon; iterations must feed updated flow");
  // Deliberately strong RHS: clipping applies to each increment, not total flow.
  ref_image.fill(1); moving.fill(0); gx.fill(1); gy.fill(-1);
  xx.fill(1); xy.fill(0); yy.fill(1);
  result=routines::ica(ref,moving,initial,3);
  if (result.dx(0,0)!=5.25f || result.dy(0,0)!=-6.125f)
    throw std::runtime_error("ICA must clip each update to [-1,1]");
  bool rejected=false;
  try { routines::ica(ref,moving,initial,-1); }
  catch (const std::invalid_argument &) { rejected=true; }
  if (!rejected) throw std::runtime_error("negative iteration count accepted");
}
void low_contrast_translation() {
  constexpr int tile = 32, size = 3 * tile;
  auto texture = [](float x, float y) {
    return 0.5f + 0.0375f * std::sin(0.13f*x + 0.07f*y)
                + 0.0375f * std::cos(0.05f*x - 0.17f*y)
                + 0.025f * std::sin(0.19f*x - 0.11f*y);
  };
  Image image(size, size), moving(size, size);
  for (int y = 0; y < size; ++y)
    for (int x = 0; x < size; ++x) {
      image(x, y) = texture(x, y);
      moving(x, y) = texture(x - 0.4f, y + 0.3f);
    }
  PyramidLevel ref{image, routines::precompute_reference(image, tile)};
  Flow initial(3, 3, tile, tile);
  auto result = routines::ica(ref, moving, initial, 50);
  if (std::hypot(result.dx(1, 1) - 0.4f, result.dy(1, 1) + 0.3f) > 0.06f)
    throw std::runtime_error("ICA failed to refine a low-contrast subpixel translation");
}
int main() {
  try {
    for (const auto &variant : routines::detail::ica_variants) check(variant.tile_size);
    regularization_and_clipping();
    low_contrast_translation();
    std::cout << "ICA checks passed\n";
  } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
