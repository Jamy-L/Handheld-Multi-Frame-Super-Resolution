#include "Halide.h"
#include <limits>

class FlowTransfer : public Halide::Generator<FlowTransfer> {
public:
  Halide::GeneratorParam<int> tile_size{"tile_size", 16};
  Input<Halide::Buffer<float>> source_dx{"source_dx", 2};
  Input<Halide::Buffer<float>> source_dy{"source_dy", 2};
  Input<Halide::Buffer<float>> reference{"reference", 2};
  Input<Halide::Buffer<float>> moving{"moving", 2};
  Input<int> target_stride{"target_stride"};
  // Source/target grid spacing ratio, with both measured in target-image pixels.
  Input<float> grid_scale_ratio{"grid_scale_ratio"};
  // Independent multiplier for displacement values, not for grid coordinates.
  Input<float> flow_scale{"flow_scale"};
  // First source tile center minus first target tile center, in target-image pixels.
  Input<float> offset_x_px{"offset_x_px"};
  Input<float> offset_y_px{"offset_y_px"};
  Output<Halide::Buffer<float>> dx{"dx", 2};
  Output<Halide::Buffer<float>> dy{"dy", 2};

  void generate() {
    const int ts = tile_size;
    if (ts <= 0) throw std::invalid_argument("tile_size must be positive");
    auto flow_x = Halide::BoundaryConditions::repeat_edge(source_dx);
    auto flow_y = Halide::BoundaryConditions::repeat_edge(source_dy);
    auto ref = Halide::BoundaryConditions::repeat_edge(reference);
    auto mov = Halide::BoundaryConditions::repeat_edge(moving);

    // The caller supplies the center-to-center offset in target-image pixels.
    // Account for it once; do not add another tile-center correction here.
    const Halide::Expr source_spacing_px = grid_scale_ratio * target_stride;
    const Halide::Expr u = (tx * target_stride - offset_x_px) / source_spacing_px;
    const Halide::Expr v = (ty * target_stride - offset_y_px) / source_spacing_px;
    const Halide::Expr left = Halide::cast<int>(Halide::floor(u));
    const Halide::Expr top = Halide::cast<int>(Halide::floor(v));
    // On an exact half-grid tie, choose the larger index consistently.
    const Halide::Expr near_x = Halide::cast<int>(Halide::floor(u + 0.5f));
    const Halide::Expr near_y = Halide::cast<int>(Halide::floor(v + 0.5f));
    const Halide::Expr other_x = Halide::select(near_x == left, left + 1, left);
    const Halide::Expr other_y = Halide::select(near_y == top, top + 1, top);
    const Halide::Expr cx = Halide::select(c == 1, other_x, near_x);
    const Halide::Expr cy = Halide::select(c == 2, other_y, near_y);
    // Three candidates: nearest, horizontal neighbor, vertical neighbor.
    // Preserve the original subpixel flow, expressed in target-level pixels.
    candidates(tx, ty, c) = Halide::Tuple(flow_x(cx, cy) * flow_scale,
                                         flow_y(cx, cy) * flow_scale);

    Halide::RDom pixel(0, ts, 0, ts, "pixel");
    const Halide::Expr x = tx * target_stride + pixel.x;
    const Halide::Expr y = ty * target_stride + pixel.y;
    // Integer sampling is only a proxy for scoring the floating candidates.
    const Halide::Expr sample_dx = Halide::cast<int>(Halide::round(candidates(tx, ty, c)[0]));
    const Halide::Expr sample_dy = Halide::cast<int>(Halide::round(candidates(tx, ty, c)[1]));
    cost(tx, ty, c) = 0.0f;
    cost(tx, ty, c) += Halide::abs(ref(x, y) - mov(x + sample_dx, y + sample_dy));

    // Prefer nearest, then horizontal, then vertical when costs tie.
    best(tx, ty) = Halide::Tuple(0, std::numeric_limits<float>::infinity());
    choice = Halide::RDom(0, 3, "choice");
    const Halide::Expr candidate_cost = cost(tx, ty, choice.x);
    best(tx, ty) = Halide::Tuple(
        Halide::select(candidate_cost < best(tx, ty)[1], choice.x, best(tx, ty)[0]),
        Halide::min(candidate_cost, best(tx, ty)[1]));
    dx(tx, ty) = candidates(tx, ty, best(tx, ty)[0])[0];
    dy(tx, ty) = candidates(tx, ty, best(tx, ty)[0])[1];

    source_dx.dim(0).set_min(0);
    source_dx.dim(1).set_min(0);
    source_dy.dim(0).set_bounds(0, source_dx.dim(0).extent());
    source_dy.dim(1).set_bounds(0, source_dx.dim(1).extent());
    reference.dim(0).set_min(0);
    reference.dim(1).set_min(0);
    moving.dim(0).set_bounds(0, reference.dim(0).extent());
    moving.dim(1).set_bounds(0, reference.dim(1).extent());
    const Halide::Expr cols = (reference.dim(0).extent() - ts) / target_stride + 1;
    const Halide::Expr rows = (reference.dim(1).extent() - ts) / target_stride + 1;
    dx.dim(0).set_bounds(0, cols);
    dx.dim(1).set_bounds(0, rows);
    dy.dim(0).set_bounds(0, cols);
    dy.dim(1).set_bounds(0, rows);
  }

  void schedule() {
    best.compute_root().parallel(ty);
    best.update().reorder(choice.x, tx, ty).parallel(ty);
    cost.compute_at(best, tx);
    dx.compute_root().parallel(ty);
    dy.compute_root().parallel(ty);
  }

private:
  Halide::Var tx{"tx"}, ty{"ty"}, c{"c"};
  Halide::RDom choice;
  Halide::Func candidates{"candidates"}, cost{"cost"}, best{"best"};
};

HALIDE_REGISTER_GENERATOR(FlowTransfer, alignment_flow_transfer)
