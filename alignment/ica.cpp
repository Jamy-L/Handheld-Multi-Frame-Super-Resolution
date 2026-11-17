#include "Halide.h"

class ICAIteration : public Halide::Generator<ICAIteration> {
public:
  Halide::GeneratorParam<int> tile_size{"tile_size", 16};
  Input<Halide::Buffer<float>> reference{"reference", 2};
  Input<Halide::Buffer<float>> moving{"moving", 2};
  Input<Halide::Buffer<float>> gx{"gx", 2};
  Input<Halide::Buffer<float>> gy{"gy", 2};
  Input<Halide::Buffer<float>> hxx{"hxx", 2};
  Input<Halide::Buffer<float>> hxy{"hxy", 2};
  Input<Halide::Buffer<float>> hyy{"hyy", 2};
  Input<Halide::Buffer<float>> initial_dx{"initial_dx", 2};
  Input<Halide::Buffer<float>> initial_dy{"initial_dy", 2};
  Output<Halide::Buffer<float>> dx{"dx", 2};
  Output<Halide::Buffer<float>> dy{"dy", 2};

  void generate() {
    const int ts = tile_size;
    if (ts != 8 && ts != 16 && ts != 32 && ts != 64)
      throw std::invalid_argument("ICA tile_size must be 8, 16, 32 or 64");
    auto bounded = Halide::BoundaryConditions::repeat_edge(moving);
    Halide::RDom pixel(0, ts, 0, ts, "pixel");
    const Halide::Expr x = tx * ts + pixel.x, y = ty * ts + pixel.y;
    const Halide::Expr wx = x + initial_dx(tx, ty), wy = y + initial_dy(tx, ty);
    const Halide::Expr ix = Halide::cast<int>(Halide::floor(wx));
    const Halide::Expr iy = Halide::cast<int>(Halide::floor(wy));
    const Halide::Expr fx = wx - ix, fy = wy - iy;
    // Extend each of the four samples, including for negative fractional
    // positions.
    const Halide::Expr top =
        bounded(ix, iy) + fx * (bounded(ix + 1, iy) - bounded(ix, iy));
    const Halide::Expr bottom =
        bounded(ix, iy + 1) +
        fx * (bounded(ix + 1, iy + 1) - bounded(ix, iy + 1));
    const Halide::Expr warped = top + fy * (bottom - top);
    const Halide::Expr residual = reference(x, y) - warped;
    rhs(tx, ty) = Halide::Tuple(0.0f, 0.0f);
    rhs(tx, ty) = Halide::Tuple(rhs(tx, ty)[0] + gx(x, y) * residual,
                                rhs(tx, ty)[1] + gy(x, y) * residual);

    const Halide::Expr a = hxx(tx, ty), b = hxy(tx, ty), c = hyy(tx, ty);
    const Halide::Expr determinant =
        a * c - b * b + 1.0f; // + 1 is a regularisation for numeric stability
    const Halide::Expr update_x =
        (c * rhs(tx, ty)[0] - b * rhs(tx, ty)[1]) / determinant;
    const Halide::Expr update_y =
        (a * rhs(tx, ty)[1] - b * rhs(tx, ty)[0]) / determinant;
    // Update flow, clip the update to -1,1 textels at most
    dx(tx, ty) = initial_dx(tx, ty) + Halide::clamp(update_x, -1.0f, 1.0f);
    dy(tx, ty) = initial_dy(tx, ty) + Halide::clamp(update_y, -1.0f, 1.0f);

    const Halide::Expr width = reference.dim(0).extent(),
                       height = reference.dim(1).extent();
    const Halide::Expr cols = width / ts, rows = height / ts;
    reference.dim(0).set_min(0);
    reference.dim(1).set_min(0);
    moving.dim(0).set_bounds(0, width);
    moving.dim(1).set_bounds(0, height);
    gx.dim(0).set_bounds(0, width);
    gx.dim(1).set_bounds(0, height);
    gy.dim(0).set_bounds(0, width);
    gy.dim(1).set_bounds(0, height);
    hxx.dim(0).set_bounds(0, cols);
    hxx.dim(1).set_bounds(0, rows);
    hxy.dim(0).set_bounds(0, cols);
    hxy.dim(1).set_bounds(0, rows);
    hyy.dim(0).set_bounds(0, cols);
    hyy.dim(1).set_bounds(0, rows);
    initial_dx.dim(0).set_bounds(0, cols);
    initial_dx.dim(1).set_bounds(0, rows);
    initial_dy.dim(0).set_bounds(0, cols);
    initial_dy.dim(1).set_bounds(0, rows);
    dx.dim(0).set_bounds(0, cols);
    dx.dim(1).set_bounds(0, rows);
    dy.dim(0).set_bounds(0, cols);
    dy.dim(1).set_bounds(0, rows);
  }

  void schedule() {
    const int lanes = get_target().natural_vector_size<float>();
    rhs.compute_root().parallel(ty);
    rhs.update().parallel(ty);
    dx.compute_root().parallel(ty).vectorize(tx, lanes,
                                             Halide::TailStrategy::GuardWithIf);
    dy.compute_root().parallel(ty).vectorize(tx, lanes,
                                             Halide::TailStrategy::GuardWithIf);
  }

private:
  Halide::Var tx{"tx"}, ty{"ty"};
  Halide::Func rhs{"rhs"};
};

HALIDE_REGISTER_GENERATOR(ICAIteration, alignment_ica_iteration)
