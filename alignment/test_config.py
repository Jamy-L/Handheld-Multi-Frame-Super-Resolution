"""Alignment configuration checks without CUDA or native kernels."""

import unittest

from handheld_super_resolution.config import Config, GrayscaleMethod
from handheld_super_resolution.params import sanitize_config, update_snr_config


class AlignmentConfigTests(unittest.TestCase):
    def test_snr_selected_tiles(self):
        for snr, expected in ((10, [64, 64, 16, 8, 8]),
                              (18, [32, 32, 16, 8, 8]),
                              (26, [16, 16, 16, 8, 8])):
            with self.subTest(snr=snr):
                config = Config()
                update_snr_config(config, snr)
                self.assertEqual(config.alignment.tile_sizes, expected)

    def test_fractional_cap_is_rejected(self):
        config = Config()
        config.alignment.search_radii = [1, 4, 4, 4, 4]
        with self.assertRaisesRegex(ValueError, "level 2.*powers of two"):
            update_snr_config(config, 18)

    def test_cumulative_fractional_cap_is_rejected(self):
        config = Config()
        config.alignment.factors = [2] * 3
        config.alignment.search_radii = [1, 0, 2, 0]
        config.alignment.block_matching.metrics = ["L2"] * 4
        with self.assertRaisesRegex(ValueError, "level 3.*powers of two"):
            update_snr_config(config, 10)

    def test_clamped_caps_are_allowed(self):
        config = Config()
        config.alignment.search_radii = [1, 15, 4, 4, 4]
        update_snr_config(config, 10)
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
                    update_snr_config(config, 18)

    def test_finest_level_uses_actual_grayscale_dimensions(self):
        config = Config()
        config.alignment.factors = []
        config.alignment.search_radii = [1]
        config.alignment.block_matching.metrics = ["L2"]
        update_snr_config(config, 10)
        with self.assertRaisesRegex(ValueError, "level 0"):
            sanitize_config(config, (64, 64))
        sanitize_config(config, (128, 128))
        config.alignment.grey_method = GrayscaleMethod.DEMOSAICING
        sanitize_config(config, (64, 64))

    def test_pyramid_uses_ceil_downsampling(self):
        config = Config(mode="grey")
        config.alignment.tile_size = 8
        config.alignment.factors = [2]
        config.alignment.search_radii = [1, 1]
        config.alignment.block_matching.metrics = ["L2", "L2"]
        update_snr_config(config, 26)
        sanitize_config(config, (15, 15))
        with self.assertRaisesRegex(ValueError, "level 1"):
            sanitize_config(config, (14, 14))


if __name__ == "__main__":
    unittest.main()
