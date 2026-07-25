# CryoSieve Kernel Conventions

## Scope
- This is authoritative for `src/cryosieve/kernels/`; new code must follow it.
- An index identifies an array element; a coordinate identifies its spatial or
  Fourier-space position. Never reuse one variable for both meanings.

## Axes and Variables
- Spatial arrays use NumPy/CuPy order `(z, y, x)` and x-fastest storage.
- Respect non-default MRC header axis mappings at the I/O boundary.
- Use `k, j, i` for z/y/x indices and `z, y, x` for coordinates.
- Use `kp, jp, ip` and `zp, yp, xp` for transformed indices and coordinates.
- Use `b` for a batch/stack index and `m` for batch size; batch is not z.

## Real and Fourier Space

- The real-space origin is at the array center; integer division is intended:

```cpp
int z = k - nz / 2;
int y = j - ny / 2;
int x = i - nx / 2;
```

- Real arrays normally use `rfftn` over explicitly specified spatial axes.
- For output shape `(nz, ny, nxh)`, where `nxh = nx / 2 + 1`:

```cpp
int z = k < nzh ? k : k - nz;
int y = j < nyh ? j : j - ny;
int x = i;
```

- Keep transformed coordinates, shifted positions, lower indices, and offsets
  separate: `xp`, `xpi`, `ip = floor(xpi)`, `dx = xpi-ip`.

## Names and Domains

- Public functions use `<operation>[_<dimension>][_ft]` in `snake_case`.
- Omit dimension when fixed by the operation: `project`, `backproject`,
  `convolute_ctf`, `convolute_ctf_ft`.
- Include dimension when variants are meaningful: `translate_2d`,
  `translate_2d_ft`, `translate_3d`, `rotate_2d`, `bandpass_2d_ft`.
- A non-`_ft` Fourier operation performs `rfftn`, calls its `_ft` function, and
  performs `irfftn`. `_ft` functions normally modify and return their input.
- `_ft` docstrings state shape, dtype, transformed axes, half-spectrum layout,
  in-place behavior, and return value.
- Use `_operation_kernel` for a private Python handle to CUDA `operation`.
