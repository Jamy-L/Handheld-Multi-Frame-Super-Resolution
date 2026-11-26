"""Alignment configuration checks without CUDA or alignment kernels."""

import unittest

from handheld_super_resolution.config import Config, GrayscaleMethod


class KernelConfigTests(unittest.TestCase):
    def test_snr_curves(self):
        for snr, expected in (
            (-1, (0.33, 5, 0.01, 0.02)),
            (3, (0.33, 5, 0.01 - 0.008 * 2 / 3, 0.015)),
            (12, (0.29, 4, 0.0012, 0.009)),
            (21, (0.25, 3.5, 0.001, 0.0075)),
            (40, (0.25, 3, 0.001, 0.006)),
        ):
            with self.subTest(snr=snr):
                config = Config()
                config.resolve_snr(snr)
                kernel = config.merging.kernel
                actual = (kernel.sigma_detail, kernel.k_denoise, kernel.D_th, kernel.D_tr)
                for value, reference in zip(actual, expected):
                    self.assertAlmostEqual(value, reference)
                self.assertEqual((kernel.k_stretch, kernel.k_shrink), (4, 2.2))

    def test_explicit_kernel_overrides(self):
        kernel = Config().merging.kernel
        kernel.sigma_detail = 0.42
        kernel.k_stretch = 7
        kernel.resolve_snr(12)
        self.assertEqual((kernel.sigma_detail, kernel.k_stretch), (0.42, 7))
        self.assertEqual(kernel.k_denoise, 4)


class AlignmentConfigTests(unittest.TestCase):
    def test_snr_selected_tiles(self):
        for linear_snr, expected in (
            (0.49, [32, 32, 16, 8, 8]),
            (0.5, [64, 64, 16, 8, 8]),
            (2.69, [64, 64, 16, 8, 8]),
            (2.7, [16, 16, 16, 8, 8]),
            (10, [16, 16, 16, 8, 8]),
        ):
            with self.subTest(linear_snr=linear_snr):
                config = Config()
                config.resolve_snr(linear_snr)
                self.assertEqual(config.alignment.tile_sizes, expected)

    def test_fractional_cap_is_rejected(self):
        config = Config()
        config.alignment.search_radii = [1, 4, 4, 4, 4]
        with self.assertRaisesRegex(ValueError, "level 2.*powers of two"):
            config.resolve_snr(18)

    def test_cumulative_fractional_cap_is_rejected(self):
        config = Config()
        config.alignment.factors = [2] * 3
        config.alignment.search_radii = [1, 0, 2, 0]
        config.alignment.block_matching.metrics = ["L2"] * 4
        with self.assertRaisesRegex(ValueError, "level 3.*powers of two"):
            config.resolve_snr(1)

    def test_clamped_caps_are_allowed(self):
        config = Config()
        config.alignment.search_radii = [1, 15, 4, 4, 4]
        config.resolve_snr(1)
        self.assertEqual(config.alignment.tile_sizes, [64, 64, 8, 8, 8])

    def test_invalid_level_settings(self):
        for name, value in (("factors", [2, 3, 4, 4]),
                            ("search_radii", [1]),
                            ("search_radii", [1, -1, 4, 4, 4]),
                            ("tile_size", 12)):
            with self.subTest(name=name, value=value):
                config = Config()
                setattr(config.alignment, name, value)
                with self.assertRaises(ValueError):
                    config.resolve_snr(18)

    def test_finest_level_uses_actual_grayscale_dimensions(self):
        config = Config()
        config.alignment.factors = []
        config.alignment.search_radii = [1]
        config.alignment.block_matching.metrics = ["L2"]
        config.resolve_snr(1)
        with self.assertRaisesRegex(ValueError, "level 0"):
            config.validate_alignment_shape((64, 64))
        config.validate_alignment_shape((128, 128))
        config.alignment.grey_method = GrayscaleMethod.DEMOSAICING
        config.validate_alignment_shape((64, 64))

    def test_pyramid_uses_ceil_downsampling(self):
        config = Config()
        config.alignment.tile_size = 8
        config.alignment.factors = [2]
        config.alignment.search_radii = [1, 1]
        config.alignment.block_matching.metrics = ["L2", "L2"]
        config.resolve_snr(26)
        config.validate_alignment_shape((30, 30))
        with self.assertRaisesRegex(ValueError, "level 1"):
            config.validate_alignment_shape((28, 28))


if __name__ == "__main__":
    unittest.main()
