#include <cmath>
#include <limits>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "alignment.h"

namespace py = pybind11;
using Array = py::array_t<float, py::array::c_style | py::array::forcecast>;

Array checked_array(const py::array &input, int ndim, bool normalized) {
    if (input.dtype().kind() != 'f')
        throw py::type_error("arrays must have a floating-point dtype");
    if (input.ndim() != ndim)
        throw py::value_error(ndim == 2 ? "image must be a 2D grayscale array" :
                                         "flow must have shape (height, width, 2)");
    for (int i = 0; i < ndim; ++i)
        if (input.shape(i) <= 0 || input.shape(i) > std::numeric_limits<int>::max())
            throw py::value_error("dimensions must be nonzero and fit int32");
    // Validate before float32 conversion, so rounding cannot hide invalid values.
    auto values = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(input);
    for (py::ssize_t i = 0; i < values.size(); ++i) {
        const double v = values.data()[i];
        if (!std::isfinite(v) || (normalized && (v < 0 || v > 1)) ||
            std::abs(v) > std::numeric_limits<float>::max())
            throw py::value_error(normalized ? "image values must be finite and in [0, 1]" :
                                              "flow values must be finite and fit float32");
    }
    return Array::ensure(input);
}

alignment::Pyramid build(const py::array &input, const std::vector<int> &scale_factors,
                         const std::vector<int> &tile_sizes, const std::vector<int> &search_radii,
                         const std::vector<alignment::Norm> &norm, int ica_iterations, bool reference) {
    Array image = checked_array(input, 2, true);
    alignment::Image buffer(const_cast<float *>(image.data()),
                            static_cast<int>(image.shape(1)), static_cast<int>(image.shape(0)));
    py::gil_scoped_release release;
    return reference ? alignment::build_reference_pyramid(buffer, scale_factors, tile_sizes, search_radii, norm, ica_iterations) :
                       alignment::build_moving_pyramid(buffer, scale_factors, tile_sizes, search_radii, norm, ica_iterations);
}

py::array_t<float> align(const alignment::Pyramid &reference,
                         const alignment::Pyramid &moving) {
    alignment::Flow result = [&]() {
        py::gil_scoped_release release;
        return alignment::align(reference, moving);
    }();
    const int tile_cols = result.tiles_x();
    const int tile_rows = result.tiles_y();
    py::array_t<float> output({tile_rows, tile_cols, 2});
    for (int y = 0; y < tile_rows; ++y)
        for (int x = 0; x < tile_cols; ++x)
        {
            output.mutable_at(y, x, 0) = result.dx(x, y);
            output.mutable_at(y, x, 1) = result.dy(x, y);
        }
    return output;
}

// Inspection returns independent NumPy copies; native pyramid buffers stay owned by C++.
py::array_t<float> image_array(const alignment::Image &image) {
    py::array_t<float> result({image.height(), image.width()});
    auto out = result.mutable_unchecked<2>();
    for (int y = 0; y < image.height(); ++y)
        for (int x = 0; x < image.width(); ++x)
            out(y, x) = image(x, y);
    return result;
}

PYBIND11_MODULE(_alignment_native, module) {
    module.doc() = "C++ alignment orchestration with dedicated Halide kernel hooks";
    py::register_exception<alignment::KernelNotImplemented>(module, "KernelNotImplemented", PyExc_NotImplementedError);
    py::class_<alignment::PyramidLevel> level(module, "PyramidLevel");
    level.def_property_readonly("image", [](const alignment::PyramidLevel &p) {
        return image_array(p.image);
    });
    const char *names[] = {"gx", "gy", "sum_gxx", "sum_gxy", "sum_gyy"};
    for (size_t i = 0; i < 5; ++i)
        level.def_property_readonly(names[i], [i](const alignment::PyramidLevel &p) -> py::object {
            if (p.precomputed.empty()) return py::none();
            return image_array(p.precomputed.at(i));
        });
    py::class_<alignment::Pyramid>(module, "Pyramid")
        .def_property_readonly("levels", [](const alignment::Pyramid &p) { return p.levels; })
        .def_property_readonly("is_reference", [](const alignment::Pyramid &p) { return p.is_reference; })
        .def_property_readonly("scale_factors", [](const alignment::Pyramid &p) { return p.scale_factors; })
        .def_property_readonly("tile_sizes", [](const alignment::Pyramid &p) { return p.tile_sizes; })
        .def_property_readonly("search_radii", [](const alignment::Pyramid &p) { return p.search_radii; })
        .def_property_readonly("norm", [](const alignment::Pyramid &p) {
            std::vector<std::string> result;
            for (auto value : p.norm)
                result.emplace_back(value == alignment::Norm::L1 ? "L1" : "L2");
            return result;
        })
        .def_property_readonly("ica_iterations", [](const alignment::Pyramid &p) { return p.ica_iterations; });
    auto bind_builder = [&](const char *name, bool reference) {
        module.def(name, [reference](const py::array &image,
                                    const std::vector<int> &scale_factors,
                                    const std::vector<int> &tile_sizes,
                                    const std::vector<int> &search_radii,
                                    const std::vector<std::string> &norm, int ica_iterations) {
            std::vector<alignment::Norm> norms;
            norms.reserve(norm.size());
            for (const auto &value : norm) {
                if (value != "L1" && value != "L2")
                    throw py::value_error("norm entries must be L1 or L2");
                norms.push_back(value == "L1" ? alignment::Norm::L1 : alignment::Norm::L2);
            }
            return build(image, scale_factors, tile_sizes, search_radii,
                         norms, ica_iterations, reference);
        }, py::arg("image"), py::arg("scale_factors"), py::arg("tile_sizes"),
           py::arg("search_radii"), py::arg("norm"), py::arg("ica_iterations"),
           "Build an owned pyramid. N scale factors define N+1 levels, finest to coarsest.");
    };
    bind_builder("build_reference_pyramid", true);
    bind_builder("build_moving_pyramid", false);
    module.def("align", &align, py::arg("reference"), py::arg("moving"),
               "Align coarse to fine, then return flow on full-resolution 16x16 tiles with stride 8. ICA runs for the configured ica_iterations (zero disables it).");
}
