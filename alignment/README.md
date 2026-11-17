# Alignment scaffold

Three Python functions delegate to a C++ core, which calls dedicated Halide
kernel hooks. The current Python alignment pipeline remains independent.

```bash
uv sync --extra native
cmake -S alignment -B build/alignment \
  -DPython_EXECUTABLE="$PWD/.venv/bin/python" -DCMAKE_BUILD_TYPE=Release
cmake --build build/alignment -j 2
ctest --test-dir build/alignment --output-on-failure
.venv/bin/python -m unittest discover -s alignment -v
```

Requires CMake 3.28+, C++17, Python development headers, Halide and pybind11.
The extension is built explicitly into `handheld_super_resolution/`; rebuild
after source changes or switching Python versions.

```python
import numpy as np
from handheld_super_resolution import _alignment_native as alignment

image = np.zeros((480, 640), dtype=np.float32)
reference = alignment.build_reference_pyramid(
    image,
    scale_factors=[2, 4, 4],
    tile_sizes=[16, 16, 8, 8],
    search_radii=[1, 4, 4, 4],
    norm=["L1", "L2", "L2", "L2"],
    ica_iterations=3,
)
moving = alignment.build_moving_pyramid(
    image,
    scale_factors=[2, 4, 4],
    tile_sizes=[16, 16, 8, 8],
    search_radii=[1, 4, 4, 4],
    norm=["L1", "L2", "L2", "L2"],
    ica_iterations=3,
)
flow = alignment.align(reference, moving)
```

Both pyramid builders work, including reference gradients and patch sums.
Coarse-to-fine block matching and flow transfer are available; ICA remains disabled.

Images are nonempty 2D grayscale floating arrays with finite values in `[0, 1]`.
Integer inputs are rejected. Floating dtypes and strided layouts are converted
to contiguous float32. Pyramids own their buffers and expose `is_reference` and the supplied options. Levels run from full resolution to coarsest, using the supplied scale factors. Reference and moving pyramids must have matching shapes and levels.

Alignment initializes zero float32 flow internally at the coarsest level, with
one `(dx, dy)` vector per complete tile. C++ uses a dedicated `Flow` struct
with separate float32 `dx` and `dy` buffers indexed by `(tile_x, tile_y)`;
Python flow layout is `(tile_rows, tile_cols, 2)`. Vectors are measured
in the current level's pixels. No initial-flow argument is required. Multi-level `align` performs block
matching at each level and transfers flow to the next finer grid. ICA remains
commented out and unimplemented.
Inputs are not modified. Do not mutate input arrays concurrently during a call;
C++ orchestration releases the GIL.

- `bindings.cpp`: array validation, buffer mapping and the three functions.
- `alignment.h` / `alignment.cpp`: owned pyramids and C++ orchestration. A shared
  template specializes reference-only precomputation at compile time.
- `kernels.h` / `kernels.cpp`: AOT downsample dispatch and remaining kernel stubs.
- `downsample_generator.cpp`: scale-specialized binomial downsampling.
- `CMakeLists.txt`: core and binding targets; add kernel generators here later.

Coarse-to-fine flow resizing/refinement is also deferred until kernels are defined.

Both builders require the source image and all five options. Scale factors are
positive powers of two (1, 2, 4, ...) used as downsampling ratios between successive levels: N factors yield
N+1 levels. Tile sizes (positive integers) and search radii (nonnegative integers)
have one entry per level, ordered finest to coarsest. An empty scale-factor list
builds one level. Norm is a per-level list of `"L1"` or `"L2"` entries, also finest to coarsest; ICA iterations is a nonnegative integer
(zero disables refinement). Options are stored on each pyramid as read-only
properties and must match between reference and moving pyramids. Kernel hooks
receive the relevant arguments.

C++ builders also receive each argument explicitly. Python lists convert to
`std::vector<int>`, which carries its length via `.size()`; no level count is passed.

## Downsampling

Each scale-2 stage applies the separable `[1, 2, 1] / 4` filter in both axes,
then keeps even-coordinate samples, starting at `(0, 0)`. Edges are replicated
at every stage. Output dimensions round up, so odd sizes and 1-pixel axes work.
Scale `2^n` performs exactly `n` such stages; scale 1 makes an independent copy.

Scales 2, 4, 8 and 16 are specialized at build time with a Halide GeneratorParam.
`ALIGNMENT_DOWNSAMPLE_SCALES` in CMake controls compilation and automatically
generates kernel includes and dispatch entries using `downsample_variants.h.in`.
Intermediate levels are materialized once within each specialized pipeline.
Unsupported scales raise `ValueError` (`std::invalid_argument` in C++); there is
no fallback or JIT. To add a scale, edit that list and rebuild; no C++ changes
are needed. Scale 1 remains a plain copy.

The C++ numerical test compares against independent scalar 3x3 convolution for
constant, patterned and impulse images, odd/even and tiny dimensions, all
specializations, and scale-1 copy semantics. Unsupported scales are rejected.

## Gradients and patch accumulation

`gradients.cpp` generates one pipeline with five outputs: full-resolution `gx`
and `gy`, and `sum_gxx`, `sum_gxy`, `sum_gyy` on non-overlapping tiles. `tile_size` is a GeneratorParam;
`ALIGNMENT_GRADIENT_TILE_SIZES` in CMake controls the variants (8, 16, 32, 64
by default) and generates includes and dispatch using `gradient_variants.h.in`.
Edit that list and rebuild; no C++ changes are needed. The generated entry points take the
input buffer followed by the five output buffers; tile size is baked into each
function. Gradients are computed once into their output buffers, then reused by
the patch reduction.

Allocate gradient buffers with shape `(width, height)` in Halide and each patch-sum buffer
with shape `(width / tile_size, height / tile_size)` using integer division.
Incomplete tiles on the right and bottom are skipped; gradients still cover the
whole image. The C++ reference precomputation hook dispatches directly to the compiled
variant for tile sizes 8, 16, 32 and 64. Other tile sizes raise `ValueError`.
Each reference level stores `gx`, `gy`, `sum_gxx`, `sum_gxy`, `sum_gyy`, in that
order. Each reference level must contain at least one complete tile; otherwise
precomputation raises `ValueError`.

## Inspect from Python

`pyramid.levels` is ordered finest to coarsest. Each level exposes `image`, `gx`,
`gy`, `sum_gxx`, `sum_gxy`, and `sum_gyy`. These properties return independent,
C-contiguous float32 NumPy copies with `(height, width)` layout, so editing them
cannot modify the native buffers. Gradient and patch-sum properties are `None`
for moving levels. Reading `levels` itself keeps native buffer ownership alive.

```python
reference = alignment.build_reference_pyramid(
    image,
    scale_factors=[2, 4],
    tile_sizes=[32, 16, 8],
    search_radii=[1, 4, 4],
    norm=["L1", "L2", "L2"],
    ica_iterations=3,
)
for level in reference.levels:
    print(level.image.shape, level.gx.shape, level.sum_gxx.shape)
```

`test_numpy_pyramid.py` exercises the complete NumPy/C++/Halide path against
independent NumPy filtering, gradients and patch reductions, including strided,
read-only and float64 inputs, odd dimensions, and buffer lifetime checks.

## Alignment backbone

The loop runs coarsest to finest: block matching, a commented-out ICA call, then
flow transfer to the next finer grid (except after the finest level). Block
matching dispatches by tile size, search radius and norm, all baked into the
compiled kernels. Flow transfer is implemented; ICA remains an explicit stub.

`Flow` stores its source tile size and stride alongside `dx` and `dy`.
`transfer_flow` takes that source flow, the target reference and moving images,
target tile size, target stride, grid scale ratio, and x/y offsets. Grid coordinates
refer to tile centers and obey `target_index = scale_ratio * source_index + offset`.
Pixel displacements scale by `scale_ratio * target_stride / source.stride`.
The caller derives the transform from the pyramid downsampling factor and both
tile grids. Current grids use non-overlapping square tiles (stride = tile size).

## Compiled block-matching variants

Maintain three lists in `CMakeLists.txt`: `ALIGNMENT_BM_TILE_SIZES`,
`ALIGNMENT_BM_SEARCH_RADII`, and `ALIGNMENT_BM_NORMS`. Nested loops compile their
Cartesian product and generate the dispatch-table entries automatically. Defaults
are tiles 8/16/32/64, radii 1/4, and L1/L2 (16 variants). Edit a list and rebuild;
no C++ switch, declarations, or template specializations need manual edits.

`block_matching.cpp` is the single generator for all combinations. Its three
GeneratorParams are resolved at build time; generated functions take only the
reference/moving images, incoming flow components, and output flow components.
`block_matching_variants.h.in` describes the generated registry, which is filled
from the CMake combinations. Uncompiled combinations raise `ValueError`; there is no
fallback or JIT.

The initial implementation rounds incoming flow, searches integer offsets in
`[-radius, radius]` in both axes, repeats the moving-image edge outside its bounds, and skips
incomplete reference tiles. Equal costs keep the first candidate, with x varying
fastest from `(-radius, -radius)`. Its CPU schedule is a correctness baseline.
`test_block_matching.cpp` checks every entry in the generated registry against
independent scalar search, including boundaries, translation, fractional initial
flow and ties. Further SIMD and cache tuning are deferred.

## Flow transfer / upsampling

`flow_transfer.cpp` maps each target tile center back to the previous grid via
`source_index = (target_index - offset) / scale_ratio`. It gathers the nearest
source flow and its horizontal and vertical neighbors from the bilinear cell,
omitting the opposite diagonal. Source-grid half ties choose the larger index.
Flow-grid reads and both image reads use repeated edges.

Each candidate flow is multiplied by the pixel displacement scale, then rounded
to an integer (nearest-even on half ties). The target tile is matched against the
moving image at that displacement using L1. The candidate with minimum cost wins;
equal costs prefer nearest, then horizontal, then vertical. Returned components
are float32 integer-valued displacements, not interpolated flow averages.

Target tile size is compiled from `ALIGNMENT_BM_TILE_SIZES`, with generated dispatch
from `flow_transfer_variants.h.in`. Target stride and the grid transform remain
runtime arguments. Complete target tiles are enumerated with
`(image_extent - tile_size) / stride + 1`, supporting overlapping strides.

The scalar-reference test covers every compiled tile size, changes of scale and
stride, boundaries, omitted diagonal, each winning candidate, rounding and ties.
Python tests exercise the full coarse-to-fine loop, including recovery of a known
`(dx, dy) = (3, -2)` translation for both L1 and L2 block matching.
