"""Boundary checks for the alignment scaffold."""
import unittest
from functools import partial
import numpy as np
from handheld_super_resolution import _alignment_native as alignment


OPTIONS = dict(scale_factors=[], tile_sizes=[16], search_radii=[4], norm=["L2"], ica_iterations=3)
reference_builder = partial(alignment.build_reference_pyramid, **OPTIONS)
moving_builder = partial(alignment.build_moving_pyramid, **OPTIONS)


class WrapperTests(unittest.TestCase):
    def test_moving_pyramid_layouts(self):
        base = np.linspace(0, 1, 48, dtype=np.float32).reshape(6, 8)
        for image in (base, base.T, base[::-1, ::2], base.astype(np.float64)):
            with self.subTest(strides=image.strides):
                before = image.copy()
                image.setflags(write=False)
                pyramid = moving_builder(image)
                self.assertEqual(pyramid.scale_factors, [])
                self.assertFalse(pyramid.is_reference)
                np.testing.assert_array_equal(image, before)

    def test_image_validation(self):
        for build in (reference_builder, moving_builder):
            for image in (np.zeros((0, 5)), np.zeros(5), np.zeros((2, 3, 4)),
                          np.full((3, 5), -0.1), np.full((3, 5), 1.1),
                          np.full((3, 5), np.nan), np.full((3, 5), np.inf)):
                with self.subTest(build=build.func.__name__, shape=image.shape), self.assertRaises(ValueError):
                    build(image)
            with self.assertRaises(TypeError):
                build(np.zeros((3, 5), dtype=np.uint16))

    def test_reference_precomputation(self):
        image = np.zeros((32, 32), np.float32)
        reference = reference_builder(image)
        self.assertTrue(reference.is_reference)
        moving = moving_builder(image, scale_factors=[2], tile_sizes=[16, 8],
                                search_radii=[4, 2], norm=["L1", "L2"])
        self.assertEqual(moving.scale_factors, [2])

    def test_options(self):
        image = np.zeros((3, 5), np.float32)
        for norm in ("L1", "L2"):
            pyramid = moving_builder(image, norm=[norm], ica_iterations=0)
            self.assertEqual(pyramid.norm, [norm])
            self.assertEqual(pyramid.ica_iterations, 0)
            self.assertEqual(pyramid.scale_factors, [])
            self.assertEqual(pyramid.tile_sizes, [16])
            self.assertEqual(pyramid.search_radii, [4])
        for options in (dict(scale_factors=[0], tile_sizes=[16, 8], search_radii=[4, 2], norm=["L1", "L2"]),
                        dict(tile_sizes=[]), dict(search_radii=[]), dict(tile_sizes=[0]),
                        dict(search_radii=[-1]), dict(norm=["L3"]), dict(norm=[]), dict(norm=["L1", "L2"]), dict(ica_iterations=-1)):
            for builder in (reference_builder, moving_builder):
                with self.subTest(options=options), self.assertRaises(ValueError):
                    builder(image, **options)
        for options in (dict(tile_sizes=[1.5]), dict(search_radii=[1.5]),
                        dict(scale_factors=[2.5]), dict(ica_iterations=1.5)):
            with self.subTest(options=options), self.assertRaises(TypeError):
                moving_builder(image, **options)

    def test_scale_factors_are_positive_powers_of_two(self):
        image = np.zeros((128, 128), np.float32)
        for builder in (reference_builder, moving_builder):
            for factor in (-2, 0, 3, 6):
                with self.subTest(factor=factor), self.assertRaisesRegex(ValueError, "positive powers of 2"):
                    builder(image, scale_factors=[factor], tile_sizes=[4, 4],
                            search_radii=[1, 1], norm=["L1", "L2"])
            for factor in (1, 2, 4, 8, 16):
                arguments = dict(scale_factors=[factor], tile_sizes=[8, 8],
                                 search_radii=[1, 1], norm=["L1", "L2"])
                with self.subTest(factor=factor):
                    pyramid = builder(image, **arguments)
                    self.assertEqual(pyramid.scale_factors, [factor])

    def test_unsupported_scale(self):
        with self.assertRaisesRegex(ValueError, "unsupported downsample scale"):
            moving_builder(np.zeros((8, 8), np.float32), scale_factors=[32],
                           tile_sizes=[4, 4], search_radii=[1, 1], norm=["L1", "L2"])

    def test_unsupported_reference_tile(self):
        with self.assertRaisesRegex(ValueError, "unsupported gradient tile size"):
            reference_builder(np.zeros((16, 16), np.float32), tile_sizes=[4])

    def test_reference_requires_complete_tile(self):
        for shape in ((15, 16), (16, 15), (1, 1)):
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, "at least one complete tile"):
                reference_builder(np.zeros(shape, np.float32))
        with self.assertRaisesRegex(ValueError, "at least one complete tile"):
            reference_builder(np.zeros((16, 16), np.float32), scale_factors=[2],
                              tile_sizes=[16, 16], search_radii=[1, 1], norm=["L1", "L2"])

    def test_alignment_placeholder(self):
        image = np.zeros((16, 16), np.float32)
        reference = reference_builder(image)
        moving = moving_builder(image)
        with self.assertRaises(NotImplementedError):
            alignment.align(reference, moving)

    def test_alignment_validation(self):
        image = np.zeros((32, 32), np.float32)
        reference = reference_builder(image)
        moving = moving_builder(image)
        with self.assertRaisesRegex(ValueError, 'reference and moving'):
            alignment.align(moving, reference)
        mismatched = moving_builder(np.zeros((32, 40), np.float32))
        with self.assertRaisesRegex(ValueError, 'shapes must match'):
            alignment.align(reference, mismatched)
        mismatched = moving_builder(image, search_radii=[2])
        with self.assertRaisesRegex(ValueError, 'options must match'):
            alignment.align(reference, mismatched)

    def test_multilevel_alignment_reaches_coarsest_kernel(self):
        image = np.zeros((129, 145), np.float32)
        reference = alignment.build_reference_pyramid(
            image, scale_factors=[2, 4], tile_sizes=[32, 16, 8],
            search_radii=[1, 2, 4], norm=["L1", "L2", "L2"], ica_iterations=3,
        )
        moving = alignment.build_moving_pyramid(
            image, scale_factors=[2, 4], tile_sizes=[32, 16, 8],
            search_radii=[1, 2, 4], norm=["L1", "L2", "L2"], ica_iterations=3,
        )
        with self.assertRaisesRegex(NotImplementedError, 'Halide alignment kernel'):
            alignment.align(reference, moving)


if __name__ == '__main__':
    unittest.main()
