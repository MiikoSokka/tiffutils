# tiffutils

Microscopy TIFF utilities for loading, saving, reshaping, normalizing, and visualizing TIFF files.

Assumes the tiff file is a stack with the shape of T,Z,C,Y,X (not all dimensions are required).


---

## 🔧 Modules & Functions

### `io/`

- **filefinder.py**
  - `find_matching_filenames(input_path, input_pattern='.*\\.tiff?$', subfolders=False)`
    > Returns a natsorted list of filenames matching a regex pattern in a folder or subfolders.

- **loader.py**
  - `load_and_expand_tiff(path_filename: str, channel_number: int)`
    > Loads TIFF into NumPy. If 3D (ZC,Y,X), reshapes into (Z,C,Y,X). Adds empty channels if below expected `channel_number`.

- **saver.py**
  - `save_tiff(stack, filename, output_folder, dtype=None)`
    > Saves a NumPy array as a TIFF. Can specify output `dtype`. Defaults to input's dtype if not provided.

---

### `processing/`

- **dtype.py**
  - `convert_dtype(array: np.ndarray, dtype: str)`
    > Converts a NumPy array to `uint8`, `uint16`, or `float32`. Float output is normalized to 0–1 range.

- **modify_histogram.py**
  - `normalize_stack_percentile(stack_uint16)`
    > Stretches intensities between 1st and 99th percentiles and normalizes to 0–1 float32. Useful for z-stack contrast adjustment.

---

### `stacker/`

- **projection.py**
  - `mip(array)`
    > Maximum intensity projection along axis 0 (e.g. Z) for a 3D/4D array.

- **stacker.py**
  - *(Functions likely include stacking TIFFs along T or Z axis — detailed docstring can be added later if provided)*

---

## ✅ Requirements

- Python ≥ 3.8
- NumPy
- tifffile
- natsort
- re, os

Install via pip (after structuring as a package):

```bash
pip install .

## Registration transforms (0.2.0)

`register_3d_stack(..., return_transforms=True)` additionally returns a `Registration`: the accepted
stages (rigid shift, then any accepted SimpleITK affine, lens warp and B-spline, in application order)
as plain JSON-able records plus metadata.

```python
registered, shift, err, reg = tiffutils.register_3d_stack(fixed_mask, moving_mask, moving, return_transforms=True)
tiffutils.save_registration(reg, "reg.json")          # atomic write
reg = tiffutils.load_registration("reg.json")
again = tiffutils.apply_registration(moving, reg)     # identical to `registered`
```

`apply_registration` uses the same code path as the fit, so it reproduces the registered stack exactly
(shape must match the fitted stack). Without `return_transforms` the return value is unchanged.
Tests: `python tests/test_registration_transforms.py`.
