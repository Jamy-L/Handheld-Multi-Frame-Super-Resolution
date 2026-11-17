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
Motion estimation still raises `NotImplementedError`.

Images are nonempty 2D grayscale floating arrays with finite values in `[0, 1]`.
Integer inputs are rejected. Floating dtypes and strided layouts are converted
to contiguous float32. Pyramids own their buffers and expose `is_reference` and the supplied options. Levels run from full resolution to coarsest, using the supplied scale factors. Reference and moving pyramids must have matching shapes and levels.

Alignment initializes zero float32 flow internally at the coarsest level, with
one `(dx, dy)` vector per complete tile. C++ uses a dedicated `Flow` struct
with separate float32 `dx` and `dy` buffers indexed by `(tile_x, tile_y)`;
Python flow layout is `(tile_rows, tile_cols, 2)`. Vectors are measured
in the current level's pixels. No initial-flow argument is required. Motion
estimation and coarse-to-fine refinement remain unimplemented: `align` reaches
the coarsest alignment kernel stub and raises `NotImplementedError`.
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
A switch in `kernels.cpp` directly calls the corresponding compiled function.
Intermediate levels are materialized once within each specialized pipeline.
Unsupported scales raise `ValueError` (`std::invalid_argument` in C++); there is
no fallback or JIT. To add a scale, extend the CMake specialization list and the
includes and switch in `kernels.cpp`.

The C++ numerical test compares against independent scalar 3x3 convolution for
constant, patterned and impulse images, odd/even and tiny dimensions, all
specializations, and scale-1 copy semantics. Unsupported scales are rejected.

## Gradients and patch accumulation

`gradients.cpp` generates one pipeline with five outputs: full-resolution `gx`
and `gy`, and `sum_gxx`, `sum_gxy`, `sum_gyy` on non-overlapping tiles. `tile_size` is a GeneratorParam;
CMake builds variants for 8, 16, 32 and 64. The generated entry points take the
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
