#include "Halide.h"
#include <array>

class FlowMedian : public Halide::Generator<FlowMedian> {
public:
  Input<Halide::Buffer<float>> source_dx{"source_dx", 2};
  Input<Halide::Buffer<float>> source_dy{"source_dy", 2};
  Output<Halide::Buffer<float>> dx{"dx", 2};
  Output<Halide::Buffer<float>> dy{"dy", 2};

  void generate() {
    dx(x, y) = median(Halide::BoundaryConditions::repeat_edge(source_dx));
    dy(x, y) = median(Halide::BoundaryConditions::repeat_edge(source_dy));
    source_dx.dim(0).set_min(0);
    source_dx.dim(1).set_min(0);
    source_dy.dim(0).set_bounds(0, source_dx.dim(0).extent());
    source_dy.dim(1).set_bounds(0, source_dx.dim(1).extent());
    dx.dim(0).set_bounds(0, source_dx.dim(0).extent());
    dx.dim(1).set_bounds(0, source_dx.dim(1).extent());
    dy.dim(0).set_bounds(0, source_dx.dim(0).extent());
    dy.dim(1).set_bounds(0, source_dx.dim(1).extent());
  }

  void schedule() {
    const int lanes = get_target().natural_vector_size<float>();
    dx.compute_root().parallel(y).vectorize(x, lanes, Halide::TailStrategy::GuardWithIf);
    dy.compute_root().parallel(y).vectorize(x, lanes, Halide::TailStrategy::GuardWithIf);
  }

private:
  Halide::Var x{"x"}, y{"y"};

  Halide::Expr median(const Halide::Func &flow) {
    std::array<Halide::Expr, 9> values;
    int index = 0;
    for (int j = -1; j <= 1; ++j)
      for (int i = -1; i <= 1; ++i)
        values[index++] = flow(x + i, y + j);
    // Fixed compare-exchange network, constructed at compile time. No runtime
    // sort, data-dependent branches, integer casts or averaging of components.
    for (int i = 1; i < 9; ++i)
      for (int j = i; j > 0; --j) {
        Halide::Expr lo = Halide::min(values[j - 1], values[j]);
        Halide::Expr hi = Halide::max(values[j - 1], values[j]);
        values[j - 1] = lo;
        values[j] = hi;
      }
    return values[4];
  }
};

HALIDE_REGISTER_GENERATOR(FlowMedian, alignment_flow_median)
