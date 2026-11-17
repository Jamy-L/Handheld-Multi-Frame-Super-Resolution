#include "Halide.h"
#include "alignment.h"
#include <limits>

class BlockMatching : public Halide::Generator<BlockMatching> {
public:
  Halide::GeneratorParam<int> tile_size{"tile_size", 16};
  Halide::GeneratorParam<int> search_radius{"search_radius", 4};
  Halide::GeneratorParam<alignment::Norm> norm{
      "norm",
      alignment::Norm::L1,
      {{"L1", alignment::Norm::L1}, {"L2", alignment::Norm::L2}}};
  Input<Halide::Buffer<float>> reference{"reference", 2};
  Input<Halide::Buffer<float>> moving{"moving", 2};
  Input<Halide::Buffer<float>> initial_dx{"initial_dx", 2};
  Input<Halide::Buffer<float>> initial_dy{"initial_dy", 2};

  Output<Halide::Buffer<float>> dx{"dx", 2};
  Output<Halide::Buffer<float>> dy{"dy", 2};

  void generate() {
    const int ts = tile_size;
    const int radius = search_radius;
    const alignment::Norm metric = norm;
    if (ts != 8 && ts != 16 && ts != 32 && ts != 64)
      throw std::invalid_argument("tile_size must be 8, 16, 32 or 64");
    if (radius < 0 || radius > (std::numeric_limits<int>::max() - 1) / 2)
      throw std::invalid_argument(
          "search_radius must be nonnegative and fit the search extent");
    const Halide::Expr tiles_x = reference.dim(0).extent() / ts;
    const Halide::Expr tiles_y = reference.dim(1).extent() / ts;

    // Stage 1: integer search center from the incoming per-tile flow.
    center(tx, ty) =
        Halide::Tuple(Halide::cast<int>(Halide::round(initial_dx(tx, ty))),
                      Halide::cast<int>(Halide::round(initial_dy(tx, ty))));
    Halide::Func moving_bounded =
        Halide::BoundaryConditions::repeat_edge(moving);

    // Stage 2: sum L1 or squared-L2 distances over one complete tile, per
    // candidate.
    Halide::RDom pixel(0, ts, 0, ts, "pixel");
    const Halide::Expr x = tx * ts + pixel.x;
    const Halide::Expr y = ty * ts + pixel.y;
    const Halide::Expr difference =
        reference(x, y) -
        moving_bounded(x + center(tx, ty)[0] + sx, y + center(tx, ty)[1] + sy);
    // C++ chooses the expression during generation: no runtime norm branch.
    const Halide::Expr distance = metric == alignment::Norm::L1
                                      ? Halide::abs(difference)
                                      : difference * difference;
    cost(tx, ty, sx, sy) = 0.0f;
    cost(tx, ty, sx, sy) += distance;

    // Stage 3: select the lowest-cost candidate. Strict comparison preserves
    // the first minimum: x varies fastest, starting at (-radius, -radius).
    search = Halide::RDom(-radius, 2 * radius + 1, -radius, 2 * radius + 1,
                          "search");
    best(tx, ty) = Halide::Tuple(0, 0, std::numeric_limits<float>::infinity());
    const Halide::Expr candidate = cost(tx, ty, search.x, search.y);
    const Halide::Expr better = candidate < best(tx, ty)[2];
    best(tx, ty) =
        Halide::Tuple(Halide::select(better, search.x, best(tx, ty)[0]),
                      Halide::select(better, search.y, best(tx, ty)[1]),
                      Halide::min(candidate, best(tx, ty)[2]));

    dx(tx, ty) = Halide::cast<float>(center(tx, ty)[0] + best(tx, ty)[0]);
    dy(tx, ty) = Halide::cast<float>(center(tx, ty)[1] + best(tx, ty)[1]);
    reference.dim(0).set_min(0);
    reference.dim(1).set_min(0);
    moving.dim(0).set_bounds(0, reference.dim(0).extent());
    moving.dim(1).set_bounds(0, reference.dim(1).extent());
    initial_dx.dim(0).set_bounds(0, tiles_x);
    initial_dx.dim(1).set_bounds(0, tiles_y);
    initial_dy.dim(0).set_bounds(0, tiles_x);
    initial_dy.dim(1).set_bounds(0, tiles_y);
    dx.dim(0).set_bounds(0, tiles_x);
    dx.dim(1).set_bounds(0, tiles_y);
    dy.dim(0).set_bounds(0, tiles_x);
    dy.dim(1).set_bounds(0, tiles_y);
  }

  void schedule() {
    // Correctness-first CPU schedule: keep candidate costs local to a tile row.
    // Leave both reductions serial; optimize their layout and SIMD later.
    best.compute_root().parallel(ty);
    best.update().reorder(search.x, search.y, tx, ty).parallel(ty);
    cost.compute_at(best, tx);
    dx.compute_root().parallel(ty);
    dy.compute_root().parallel(ty);
  }

private:
  Halide::Var tx{"tx"}, ty{"ty"}, sx{"sx"}, sy{"sy"};
  Halide::RDom search;
  Halide::Func center{"center"}, cost{"cost"}, best{"best"};
};

HALIDE_REGISTER_GENERATOR(BlockMatching, alignment_block_matching)
