#include "routines.h"
#include "downsample_variants.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

using alignment::Image;

// Independent scalar 3x3 convolution followed by decimation.
Image reference_downsample(Image image, int scale) {
    constexpr int weights[] = {1, 2, 1};
    for (; scale > 1; scale >>= 1) {
        Image next((image.width() + 1) / 2, (image.height() + 1) / 2);
        for (int y = 0; y < next.height(); ++y)
            for (int x = 0; x < next.width(); ++x) {
                float sum = 0;
                for (int j = -1; j <= 1; ++j)
                    for (int i = -1; i <= 1; ++i)
                        sum += weights[i + 1] * weights[j + 1] *
                               image(std::clamp(2 * x + i, 0, image.width() - 1),
                                     std::clamp(2 * y + j, 0, image.height() - 1));
                next(x, y) = sum / 16.0f;
            }
        image = next;
    }
    return image;
}

int main() {
    try {
        std::vector<int> scales{1};
        for (const auto &variant : alignment::routines::detail::downsample_variants)
            scales.push_back(variant.scale_factor);
        // Odd/even dimensions, tiny images, borders, ramps, and impulses.
        for (const auto shape : {std::pair{1, 1}, {1, 9}, {10, 1}, {3, 5}, {32, 24}, {65, 49}}) {
            for (int pattern = 0; pattern < 3; ++pattern) {
                Image image(shape.first, shape.second);
                for (int y = 0; y < image.height(); ++y)
                    for (int x = 0; x < image.width(); ++x)
                        image(x, y) = pattern == 0 ? 0.375f : pattern == 1 ?
                            float((x * 13 + y * 7) % 101) / 100.0f : float(x == 0 && y == 0);
                const Image before = image.copy();
                for (int scale : scales) {
                    Image expected = reference_downsample(image, scale);
                    Image actual = alignment::routines::downsample(image, scale);
                    if (actual.width() != expected.width() || actual.height() != expected.height())
                        throw std::runtime_error("incorrect output dimensions");
                    for (int y = 0; y < actual.height(); ++y)
                        for (int x = 0; x < actual.width(); ++x)
                            if (std::abs(actual(x, y) - expected(x, y)) > 2e-6f)
                                throw std::runtime_error("incorrect downsample values, scale " + std::to_string(scale));
                    actual(0, 0) = -1;  // Output owns independent storage, including scale 1.
                    for (int y = 0; y < image.height(); ++y)
                        for (int x = 0; x < image.width(); ++x)
                            if (image(x, y) != before(x, y))
                                throw std::runtime_error("input was modified");
                }
                for (int invalid : {-2, 0, 3, 6}) {
                    bool rejected = false;
                    try { alignment::routines::downsample(image, invalid); }
                    catch (const std::invalid_argument &) { rejected = true; }
                    if (!rejected) throw std::runtime_error("invalid scale was accepted");
                }
            }
        }
        Image source(35, 27);
        for (int y = 0; y < source.height(); ++y)
            for (int x = 0; x < source.width(); ++x)
                source(x, y) = float((x * 11 + y * 3) % 97) / 96.0f;
        auto pyramid = alignment::build_moving_pyramid(
            source, {2, 4, 1}, {8, 8, 4, 4}, {1, 2, 2, 2},
            {alignment::Norm::L1, alignment::Norm::L2, alignment::Norm::L2, alignment::Norm::L2}, 3);
        if (pyramid.levels.size() != 4)
            throw std::runtime_error("incorrect pyramid size");
        Image expected = source;
        const int factors[] = {1, 2, 4, 1};
        for (size_t level = 0; level < pyramid.levels.size(); ++level) {
            expected = reference_downsample(expected, factors[level]);
            const Image &actual = pyramid.levels[level].image;
            if (actual.width() != expected.width() || actual.height() != expected.height())
                throw std::runtime_error("incorrect pyramid level dimensions");
            for (int y = 0; y < actual.height(); ++y)
                for (int x = 0; x < actual.width(); ++x)
                    if (std::abs(actual(x, y) - expected(x, y)) > 2e-6f)
                        throw std::runtime_error("incorrect pyramid level values");
        }
        std::cout << "Downsample checks passed\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
