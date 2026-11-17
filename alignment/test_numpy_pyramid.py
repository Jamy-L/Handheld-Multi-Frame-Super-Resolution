"""End-to-end NumPy -> C++ -> AOT Halide -> NumPy numerical checks."""
import unittest

import numpy as np

from handheld_super_resolution import alignment


def downsample(image, scale):
    """Independent float64 reference: normalized 3x3 filter, then decimate."""
    image = image.astype(np.float64)
    while scale > 1:
        padded = np.pad(image, 1, mode="edge")
        height, width = image.shape
        filtered = np.zeros_like(image)
        for y, wy in enumerate((1, 2, 1)):
            for x, wx in enumerate((1, 2, 1)):
                filtered += wy * wx * padded[y:y + height, x:x + width] / 16
        image = filtered[::2, ::2]
        scale //= 2
    return image


def gradients(image):
    height, width = image.shape
    p = np.pad(image.astype(np.float64), 2, mode="edge")
    gx = (p[2:height + 2, :width] - 8 * p[2:height + 2, 1:width + 1]
          + 8 * p[2:height + 2, 3:width + 3] - p[2:height + 2, 4:width + 4]) / 12
    gy = (p[:height, 2:width + 2] - 8 * p[1:height + 1, 2:width + 2]
          + 8 * p[3:height + 3, 2:width + 2] - p[4:height + 4, 2:width + 2]) / 12
    return gx, gy


def patch_sums(image, tile):
    height, width = image.shape
    rows, cols = height // tile, width // tile
    return image[:rows * tile, :cols * tile].reshape(rows, tile, cols, tile).sum(axis=(1, 3))


class NumPyPyramidTests(unittest.TestCase):
    def test_reference_and_moving_values(self):
        rng = np.random.default_rng(42)
        base = rng.random((133, 149), dtype=np.float32)
        # Non-square, odd sizes; noncontiguous/read-only inputs and dtype conversion.
        for image in (base, base.T, base[::-1], base.astype(np.float64)):
            with self.subTest(shape=image.shape, strides=image.strides):
                before = image.copy()
                image.setflags(write=False)
                reference = alignment.build_reference_pyramid(
                    image, scale_factors=[2, 4], tile_sizes=[32, 16, 8],
                    search_radii=[1, 4, 4], norm=["L1", "L2", "L2"], ica_iterations=3,
                )
                moving = alignment.build_moving_pyramid(
                    image, scale_factors=[2, 4], tile_sizes=[32, 16, 8],
                    search_radii=[1, 4, 4], norm=["L1", "L2", "L2"], ica_iterations=3,
                )
                expected = image.astype(np.float32).astype(np.float64)
                self.assertEqual(len(reference.levels), 3)
                for ref, mov, factor, tile in zip(reference.levels, moving.levels, (1, 2, 4), (32, 16, 8)):
                    expected = downsample(expected, factor)
                    np.testing.assert_allclose(ref.image, expected, atol=2e-7, rtol=2e-6)
                    np.testing.assert_array_equal(ref.image, mov.image)
                    gx, gy = gradients(expected)
                    np.testing.assert_allclose(ref.gx, gx, atol=3e-7, rtol=2e-5)
                    np.testing.assert_allclose(ref.gy, gy, atol=3e-7, rtol=2e-5)
                    for actual, product in ((ref.sum_gxx, gx * gx), (ref.sum_gxy, gx * gy),
                                            (ref.sum_gyy, gy * gy)):
                        np.testing.assert_allclose(actual, patch_sums(product, tile), atol=1e-5, rtol=1e-4)
                    for array in (ref.image, ref.gx, ref.gy, ref.sum_gxx, ref.sum_gxy, ref.sum_gyy):
                        self.assertEqual(array.dtype, np.float32)
                        self.assertTrue(array.flags.c_contiguous)
                        self.assertFalse(np.shares_memory(array, image))
                    for name in ("gx", "gy", "sum_gxx", "sum_gxy", "sum_gyy"):
                        self.assertIsNone(getattr(mov, name))
                np.testing.assert_array_equal(image, before)

    def test_buffer_ownership(self):
        image = np.linspace(0, 1, 32 * 32, dtype=np.float32).reshape(32, 32)
        before = image.copy()
        pyramid = alignment.build_reference_pyramid(
            image, scale_factors=[], tile_sizes=[8], search_radii=[1],
            norm=["L2"], ica_iterations=3,
        )
        level = pyramid.levels[0]
        image[:] = 0
        np.testing.assert_array_equal(level.image, before)
        snapshot = level.gx
        expected = snapshot.copy()
        snapshot[:] = 0
        np.testing.assert_array_equal(level.gx, expected)
        del pyramid
        np.testing.assert_array_equal(level.image, before)
        del level
        np.testing.assert_array_equal(snapshot, 0)


if __name__ == "__main__":
    unittest.main()
