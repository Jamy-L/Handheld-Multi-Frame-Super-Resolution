# Alignment scaffold

Three Python functions delegate to a C++ core, which calls dedicated Halide
kernels. The main pipeline calls this alignment module directly.

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
from handheld_super_resolution import alignment

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
Coarse-to-fine block matching, ICA refinement, and flow transfer are available.

Images are nonempty 2D grayscale floating arrays with finite values in `[0, 1]`.
Integer inputs are rejected. Floating dtypes and strided layouts are converted
to contiguous float32. Pyramids own their buffers and expose `is_reference` and the supplied options. Levels run from full resolution to coarsest, using the supplied scale factors. Reference and moving pyramids must have matching shapes and levels.

Alignment initializes zero float32 flow internally at the coarsest level, with
one `(dx, dy)` vector per complete tile. C++ uses a dedicated `Flow` struct
with separate float32 `dx` and `dy` buffers indexed by `(tile_x, tile_y)`;
Python flow layout is `(tile_rows, tile_cols, 2)`. The final flow always uses
full-resolution 16x16 tiles with stride 8, so its shape is
`((height - 16) // 8 + 1, (width - 16) // 8 + 1, 2)`. Vectors are measured
in the current level's pixels. No initial-flow argument is required. Multi-level `align` performs block
matching and ICA refinement at each level, then transfers flow to the next finer
grid. Set `ica_iterations=0` to disable ICA.
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

The loop runs coarsest to finest: block matching, ICA refinement, then
flow transfer to the next finer grid (except after the finest level). Block
matching dispatches by tile size, search radius and norm, all baked into the
compiled kernels. The C++ ICA wrapper iterates the compiled one-step kernel.

`Flow` stores its source tile size and stride alongside `dx` and `dy`.
`transfer_flow` takes that source flow, the target reference and moving images,
target tile size, target stride, `grid_scale_ratio`, independent `flow_scale`,
and x/y offsets in pixels.
`grid_scale_ratio >= 1` is source grid spacing divided by target grid spacing,
with both spacings expressed in target-image pixels. `flow_scale` is supplied
explicitly and multiplies displacement values only; it is never derived from
grid spacing or stride.
The first tiles share a top-left corner. Their centers are `tile_size/2`, so the
caller passes `source.tile_size * pixel_scale / 2 - target_tile_size / 2` as the
source-minus-target first-center offset, in target-image pixels. Current grids use non-overlapping square tiles (stride = tile size).

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
`source_index = (target_index * target_stride - offset_px) /
(grid_scale_ratio * target_stride)`. The offset is the difference between first
tile centers in target-image pixels, not a difference between tile corners and
not a tile-index offset. The caller accounts for `tile_size/2` centers once;
the kernel applies the supplied offset without another center correction. It gathers the nearest
source flow and its horizontal and vertical neighbors from the bilinear cell,
omitting the opposite diagonal. Source-grid half ties choose the larger index.
Flow-grid reads and both image reads use repeated edges.

Each candidate flow is multiplied by the explicit `flow_scale` argument. For L1
scoring only, its components are rounded to integers (nearest-even on half ties)
to sample the moving image. The minimum-cost candidate wins; equal costs prefer
nearest, then horizontal, then vertical. Output preserves the selected candidate's
full floating-point displacement in target-level pixels, without rounding or
averaging the candidate flows.

Target tile size is compiled from `ALIGNMENT_BM_TILE_SIZES`, with generated dispatch
from `flow_transfer_variants.h.in`. Target stride and the grid transform remain
runtime arguments. Complete target tiles are enumerated with
`(image_extent - tile_size) / stride + 1`, supporting overlapping strides.

The scalar-reference test covers every compiled tile size, changes of scale and
stride, boundaries, omitted diagonal, each winning candidate, rounding and ties.
Python tests exercise the full coarse-to-fine loop, including recovery of a known
`(dx, dy) = (3, -2)` translation for both L1 and L2 block matching.

After block matching at level 0, a final transfer uses the same full-resolution
reference/moving images, tile size 16, stride 8 and pixel scale 1. It selects
among the same three source-flow candidates with L1, without another block-matching or ICA pass. The grid scale ratio is `source.stride / 8`; the independently supplied
`flow_scale=1` preserves pixel flow magnitudes.
The center offset is `source.tile_size/2 - 16/2` in full-resolution pixels, which
also handles a finest-level source tile size other than 16. The image must fit at
least one complete 16x16 output tile.

## Final flow median

After the final 16x16 / stride-8 transfer, `flow_median.cpp` applies independent
3x3 medians to `dx` and `dy` on the flow grid. Boundaries repeat the nearest flow
sample. A fixed min/max network computes each median in Halide; values remain
float32 with no integer rounding. Tile size, stride, and output shape stay the
same. Tests cover isolated outliers, fractional values, corners, odd dimensions,
and grids as small as 1x1 against an independent scalar sort.

For example, transferring 16x16 coarse tiles to 16x16 fine tiles at pixel scale 2
and target stride 16 uses `grid_scale_ratio=2`, `flow_scale=2`, and
`offset_x_px=offset_y_px=8`.
The final same-image transfer from 16x16/stride-16 to 16x16/stride-8 uses
`grid_scale_ratio=2`, `flow_scale=1`, and zero center offset. Non-finite values or grid scale
ratios below 1 are rejected.

Grid geometry and displacement scaling are separate API arguments. Tests vary
`grid_scale_ratio` and `flow_scale` independently, including combinations that
cannot be inferred from source/target strides.

## ICA refinement

`ica.cpp` defines one inverse-compositional translation iteration, compiled for
8/16/32/64 tiles using the BM tile-size list and generated `ica_variants.h` dispatch.
The kernel bilinearly samples the moving image at the incoming floating flow;
each sample uses repeated image edges. It uses cached reference `gx`, `gy` and
per-tile Hessian sums. For residual `reference - warped_moving`, it computes
`bx=sum(gx*residual)`, `by=sum(gy*residual)` and solves:

```
det = Hxx*Hyy - Hxy*Hxy + std::numeric_limits<float>::epsilon()
update_x = (Hyy*bx - Hxy*by) / det
update_y = (Hxx*by - Hxy*bx) / det
```

Each update component is clamped to `[-1, 1]`, then added to the original flow.
The epsilon is the portable float32 machine epsilon (about 1.19e-7).
This regularizes the determinant directly; it does not add to the Hessian diagonal
or clamp the accumulated displacement. C++ repeats the step `ica_iterations`
times, swapping two flow buffers between calls and reusing reference precomputation.
Inputs remain unchanged; zero iterations returns an independent flow copy.

ICA is enabled immediately after BM at each pyramid level when iterations are
positive. It uses the non-overlapping per-level grid; the final overlapping-grid
transfer and median follow refinement. Tests compare iterations to scalar bilinear
warping and the exact regularized solve, exercise clipping and flat patches, and
verify recovery of a fractional translation through the Python binding.
