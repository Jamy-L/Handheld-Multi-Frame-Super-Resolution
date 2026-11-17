#include "Halide.h"
#include <vector>

// scale_factor is an AOT specialization, not a runtime scalar input.
class Downsample : public Halide::Generator<Downsample> {
public:
    Halide::GeneratorParam<int> scale_factor{"scale_factor", 2};
    Input<Halide::Buffer<float>> input{"input", 2};
    Output<Halide::Buffer<float>> output{"output", 2};

    void generate() {
        const int factor = scale_factor;
        if (factor < 2 || (factor & (factor - 1)) != 0)
            throw std::invalid_argument("scale_factor must be a power of two >= 2");
        Halide::Expr width = input.dim(0).extent();
        Halide::Expr height = input.dim(1).extent();
        Halide::Func previous{"source"};
        previous(x, y) = input(x, y);
        for (int remaining = factor; remaining > 1; remaining >>= 1) {
            // Clamp separately at every level, including intermediate images.
            Halide::Func bounded{"bounded"}, horizontal{"horizontal"}, stage{"stage"};
            bounded(x, y) = previous(Halide::clamp(x, 0, width - 1),
                                     Halide::clamp(y, 0, height - 1));
            horizontal(x, y) = (bounded(2 * x - 1, y) + 2.0f * bounded(2 * x, y)
                                  + bounded(2 * x + 1, y)) * 0.25f;
            stage(x, y) = (horizontal(x, 2 * y - 1) + 2.0f * horizontal(x, 2 * y)
                             + horizontal(x, 2 * y + 1)) * 0.25f;
            stages.push_back(stage);
            previous = stage;
            width = width / 2 + width % 2;
            height = height / 2 + height % 2;
        }
        output(x, y) = previous(x, y);
    }

    void schedule() {
        const int lanes = get_target().natural_vector_size<float>();
        // Materialize intermediate levels once; fuse the final stage into output.
        for (size_t i = 0; i + 1 < stages.size(); ++i)
            stages[i].compute_root().parallel(y).vectorize(x, lanes, Halide::TailStrategy::GuardWithIf);
        output.compute_root().parallel(y).vectorize(x, lanes, Halide::TailStrategy::GuardWithIf);
    }

private:
    Halide::Var x{"x"}, y{"y"};
    std::vector<Halide::Func> stages;
};

HALIDE_REGISTER_GENERATOR(Downsample, alignment_downsample)
