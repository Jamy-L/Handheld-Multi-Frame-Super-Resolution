#include "Halide.h"

class Gradients : public Halide::Generator<Gradients> {
public:
  Halide::GeneratorParam<int> tile_size{"tile_size", 16};
  Input<Halide::Buffer<float>> input{"input", 2};
  Output<Halide::Buffer<float>> gx{"gx", 2};
  Output<Halide::Buffer<float>> gy{"gy", 2};
  Output<Halide::Buffer<float>> sum_gxx{"sum_gxx", 2};
  Output<Halide::Buffer<float>> sum_gxy{"sum_gxy", 2};
  Output<Halide::Buffer<float>> sum_gyy{"sum_gyy", 2};

  void generate() {
    const int ts = tile_size;
    if (ts <= 0)
      throw std::invalid_argument("tile_size must be positive");

    Halide::Expr width = input.dim(0).extent();
    Halide::Expr height = input.dim(1).extent();
    Halide::Func bounded = Halide::BoundaryConditions::repeat_edge(input);
    gx(x, y) = (bounded(x - 2, y) - 8.0f * bounded(x - 1, y) +
                8.0f * bounded(x + 1, y) - bounded(x + 2, y)) /
               12.0f;
    gy(x, y) = (bounded(x, y - 2) - 8.0f * bounded(x, y - 1) +
                8.0f * bounded(x, y + 1) - bounded(x, y + 2)) /
               12.0f;

    // tx and ty index non-overlapping tiles; r indexes pixels inside one tile.
    Halide::RDom r(0, ts, 0, ts, "r");
    Halide::Expr gx_ = gx(tx * ts + r.x, ty * ts + r.y);
    Halide::Expr gy_ = gy(tx * ts + r.x, ty * ts + r.y);
    sum_gxx(tx, ty) = 0.0f;
    sum_gxx(tx, ty) += gx_ * gx_;
    sum_gxy(tx, ty) = 0.0f;
    sum_gxy(tx, ty) += gx_ * gy_;
    sum_gyy(tx, ty) = 0.0f;
    sum_gyy(tx, ty) += gy_ * gy_;

    // Full-resolution gradients, and only complete tiles in the accumulation.
    gx.dim(0).set_bounds(0, width);
    gx.dim(1).set_bounds(0, height);
    gy.dim(0).set_bounds(0, width);
    gy.dim(1).set_bounds(0, height);
    sum_gxx.dim(0).set_bounds(0, width / ts);
    sum_gxx.dim(1).set_bounds(0, height / ts);
    sum_gxy.dim(0).set_bounds(0, width / ts);
    sum_gxy.dim(1).set_bounds(0, height / ts);
    sum_gyy.dim(0).set_bounds(0, width / ts);
    sum_gyy.dim(1).set_bounds(0, height / ts);
  }

  void schedule() {
    const int lanes = get_target().natural_vector_size<float>();
    gx.compute_root().parallel(y).vectorize(x, lanes,
                                            Halide::TailStrategy::GuardWithIf);
    gy.compute_root().parallel(y).vectorize(x, lanes,
                                            Halide::TailStrategy::GuardWithIf);
    sum_gxx.compute_root().parallel(ty);
    sum_gxy.compute_root().parallel(ty);
    sum_gyy.compute_root().parallel(ty);
    // Different tile rows are independent; reduction pixels remain serial.
    sum_gxx.update().parallel(ty);
    sum_gxy.update().parallel(ty);
    sum_gyy.update().parallel(ty);
  }

private:
  Halide::Var x{"x"}, y{"y"}, tx{"tx"}, ty{"ty"};
};

HALIDE_REGISTER_GENERATOR(Gradients, alignment_gradients)
