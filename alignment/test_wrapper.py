"""Boundary checks for the alignment scaffold."""
import unittest
from functools import partial
import numpy as np
from handheld_super_resolution import alignment


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

    def test_block_matching_dispatch(self):
        image = np.random.default_rng(123).random((64, 64), dtype=np.float32)
        for tile in (8, 16, 32, 64):
            for radius in (1, 4):
                for norm in ("L1", "L2"):
                    reference = alignment.build_reference_pyramid(
                        image, scale_factors=[], tile_sizes=[tile], search_radii=[radius],
                        norm=[norm], ica_iterations=3,
                    )
                    moving = alignment.build_moving_pyramid(
                        image, scale_factors=[], tile_sizes=[tile], search_radii=[radius],
                        norm=[norm], ica_iterations=3,
                    )
                    with self.subTest(tile=tile, radius=radius, norm=norm):
                        flow = alignment.align(reference, moving)
                        self.assertEqual(flow.shape, ((64 - 16) // 8 + 1, (64 - 16) // 8 + 1, 2))
                        self.assertEqual(flow.dtype, np.float32)
                        np.testing.assert_array_equal(flow, 0)

    def test_final_overlapping_grid_from_different_tile_sizes(self):
        image = np.random.default_rng(13).random((65, 73), dtype=np.float32)
        for tile in (8, 16, 32, 64):
            reference = alignment.build_reference_pyramid(
                image, scale_factors=[], tile_sizes=[tile], search_radii=[4],
                norm=["L1"], ica_iterations=0,
            )
            moving = alignment.build_moving_pyramid(
                image, scale_factors=[], tile_sizes=[tile], search_radii=[4],
                norm=["L1"], ica_iterations=0,
            )
            with self.subTest(source_tile=tile):
                flow = alignment.align(reference, moving)
                self.assertEqual(flow.shape, (7, 8, 2))
                np.testing.assert_array_equal(flow, 0)

    def test_uncompiled_block_matching_variant(self):
        image = np.zeros((16, 16), np.float32)
        reference = reference_builder(image, search_radii=[3])
        moving = moving_builder(image, search_radii=[3])
        with self.assertRaisesRegex(ValueError, "variant not compiled.*radius=3"):
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

    def test_multilevel_alignment(self):
        image = np.random.default_rng(321).random((129, 145), dtype=np.float32)
        reference = alignment.build_reference_pyramid(
            image, scale_factors=[2, 4], tile_sizes=[32, 16, 8],
            search_radii=[1, 4, 4], norm=["L1", "L2", "L2"], ica_iterations=3,
        )
        moving = alignment.build_moving_pyramid(
            image, scale_factors=[2, 4], tile_sizes=[32, 16, 8],
            search_radii=[1, 4, 4], norm=["L1", "L2", "L2"], ica_iterations=3,
        )
        flow = alignment.align(reference, moving)
        self.assertEqual(flow.shape, ((129 - 16) // 8 + 1, (145 - 16) // 8 + 1, 2))
        self.assertEqual(flow.dtype, np.float32)
        np.testing.assert_array_equal(flow, 0)

    def test_multilevel_translation(self):
        image = np.random.default_rng(10).random((128, 128), dtype=np.float32)
        # moving(x + 3, y - 2) = reference(x, y), with repeated edge pixels.
        moving_image = image[np.clip(np.arange(128) + 2, 0, 127)[:, None],
                             np.clip(np.arange(128) - 3, 0, 127)[None, :]].copy()
        for norm in ("L1", "L2"):
            reference = alignment.build_reference_pyramid(
                image, scale_factors=[2, 2], tile_sizes=[16, 16, 16],
                search_radii=[4, 4, 4], norm=[norm, norm, norm], ica_iterations=0,
            )
            moving = alignment.build_moving_pyramid(
                moving_image, scale_factors=[2, 2], tile_sizes=[16, 16, 16],
                search_radii=[4, 4, 4], norm=[norm, norm, norm], ica_iterations=0,
            )
            with self.subTest(norm=norm):
                flow = alignment.align(reference, moving)
                self.assertEqual(flow.shape, (15, 15, 2))
                np.testing.assert_array_equal(flow[..., 0], 3)
                np.testing.assert_array_equal(flow[..., 1], -2)

    def test_ica_recovers_subpixel_translation(self):
        y, x = np.mgrid[:128, :128]
        def texture(x, y):
            return (0.5 + 0.15 * np.sin(0.13*x + 0.07*y)
                    + 0.15 * np.cos(0.05*x - 0.17*y)
                    + 0.1 * np.sin(0.19*x - 0.11*y)).astype(np.float32)
        image = texture(x, y)
        moving_image = texture(x - 0.4, y + 0.3)
        errors = []
        for iterations in (0, 20):
            reference = alignment.build_reference_pyramid(
                image, scale_factors=[], tile_sizes=[32], search_radii=[1],
                norm=["L1"], ica_iterations=iterations,
            )
            moving = alignment.build_moving_pyramid(
                moving_image, scale_factors=[], tile_sizes=[32], search_radii=[1],
                norm=["L1"], ica_iterations=iterations,
            )
            flow = alignment.align(reference, moving)
            self.assertTrue(np.isfinite(flow).all())
            interior = flow[2:-2, 2:-2]
            errors.append(float(np.median(np.linalg.norm(interior - [0.4, -0.3], axis=-1))))
            if iterations:
                self.assertTrue(np.any(np.abs(interior - np.rint(interior)) > 1e-4))
                np.testing.assert_allclose(np.median(interior, axis=(0, 1)), [0.4, -0.3], atol=0.06)
        self.assertLess(errors[1], 0.1)
        self.assertLess(errors[1], errors[0] / 3)


if __name__ == '__main__':
    unittest.main()
