# coding: utf-8
# Author: Miiko Sokka

'''
These are functions needed to do quick Cellpose segmentation for multi-round imaging experiment done with Opera Phenix.
They segment the nuclei, stack channels from multiple rounds and crops nuclei into individual small files.
'''

# coding: utf-8
# Author: Miiko Sokka

from pathlib import Path
import csv
import re
from typing import List, Tuple, Any
import numpy as np
from skimage.measure import block_reduce, regionprops
from ..io import load_tiff, save_tiff
from ..processing import histogram_stretch


def segment_nuclei_cpsam_3d(
    vol: np.ndarray,
    downsample_factor: int = 20,
    diameter: float | None = None,
    gpu: bool = True,
    anisotropy: float | None = None,
    min_size: int = 0,
    model: Any | None = None,
    normalize: bool = True,
    verbose: bool = True,
) -> np.ndarray:
    """
    Segment nuclei in a 3D volume using Cellpose-SAM (cpsam) with 3D mode.

    Parameters
    ----------
    vol : np.ndarray
        Input volume with shape:
            - (Z, Y, X)   or
            - (Z, C, Y, X)  (in which case only channel 0 is used)
    downsample_factor : int, default 20
        XY downsampling factor (Z is left unchanged).
    diameter : float or None
        Approximate object diameter in pixels at the *downsampled* scale.
        If None, Cellpose will try to estimate.
    gpu : bool, default True
        Use GPU if available.
    anisotropy : float or None
        Z anisotropy (Z spacing / XY pixel size). If None, Cellpose infers / ignores.
    min_size : int, default 0
        Minimum object size (in voxels) for mask filtering in Cellpose.
    model : cellpose.models.CellposeModel or None
        Pre-created model. If None, a new cpsam model is created.
    normalize : bool, default True
        If True, min-max normalize the input (per volume) to [0, 1] before downsampling.
    verbose : bool, default True
        Print basic progress messages.

    Returns
    -------
    masks_zyx : np.ndarray
        Integer-labeled mask array with shape (Z, Y, X) at the original resolution.
    """

        # --- lazy import so that importing this module doesn't require cellpose ---
    if model is None:
        try:
            from cellpose import models
        except ImportError as e:
            raise ImportError(
                "segment_nuclei_cpsam_3d requires `cellpose` to be installed.\n"
                "Install it in the environment where you run segmentation."
            ) from e
        if verbose:
            print("[segment_nuclei_cpsam_3d] Creating Cellpose-SAM cpsam model...")
        model = models.CellposeModel(gpu=gpu)
        
    if vol.ndim == 3:
        # Z, Y, X
        img_zyx = vol
    elif vol.ndim == 4:
        # Z, C, Y, X -> take first channel
        img_zyx = vol[:, 0, ...]
    else:
        raise ValueError(f"Expected vol with ndim 3 or 4 (ZYX or ZCYX), got shape {vol.shape}")

    # Ensure we have a copy in float32
    img_zyx = np.asarray(img_zyx, dtype=np.float32)

    # Optional normalization to [0, 1]
    if normalize:
        vmin = np.percentile(img_zyx, 1)
        vmax = np.percentile(img_zyx, 99)
        if vmax > vmin:
            img_zyx = (img_zyx - vmin) / (vmax - vmin)
            img_zyx = np.clip(img_zyx, 0.0, 1.0)
        else:
            # fallback to simple scaling if weird distribution
            img_zyx = img_zyx - img_zyx.min()
            if img_zyx.max() > 0:
                img_zyx = img_zyx / img_zyx.max()

    Z, Y, X = img_zyx.shape
    if verbose:
        print(f"[segment_nuclei_cpsam_3d] Input volume: {img_zyx.shape}, dtype={img_zyx.dtype}")

    # Downsample in XY using block_reduce (mean pooling)
    if downsample_factor > 1:
        if verbose:
            print(
                f"[segment_nuclei_cpsam_3d] Downsampling XY by factor {downsample_factor}: "
                f"{Y}x{X} -> {Y // downsample_factor}x{X // downsample_factor}"
            )
        img_ds = block_reduce(
            img_zyx,
            block_size=(1, downsample_factor, downsample_factor),
            func=np.mean,
        )
    else:
        img_ds = img_zyx

    # Create CPSAM model if not provided
    if model is None:
        if verbose:
            print("[segment_nuclei_cpsam_3d] Creating Cellpose-SAM cpsam model...")
        model = models.CellposeModel(gpu=gpu)

    if verbose:
        print("[segment_nuclei_cpsam_3d] Running 3D Cellpose-SAM segmentation...")

    # Cellpose expects 3D array; in our case x has shape (Z, Y, X)
    masks_ds, flows, styles = model.eval(
        img_ds,
        diameter=diameter,
        do_3D=True,
        anisotropy=anisotropy,
        min_size=min_size,
        normalize=False,  # already normalized if requested
        batch_size=1,
        channel_axis=None,  # no separate channel dimension
        z_axis=0,           # Z is axis 0 in img_ds
    )

    # masks_ds is (Z, Y_ds, X_ds)
    masks_ds = np.asarray(masks_ds)

    if downsample_factor > 1:
        # Upsample using nearest-neighbor expansion (repeat)
        Y_ds, X_ds = masks_ds.shape[1:]
        if verbose:
            print(
                f"[segment_nuclei_cpsam_3d] Upsampling masks back to original XY "
                f"via repeat({downsample_factor}): {Y_ds}x{X_ds} -> "
                f"{Y_ds*downsample_factor}x{X_ds*downsample_factor}"
            )

        masks_up = np.repeat(
            np.repeat(masks_ds, downsample_factor, axis=1),
            downsample_factor,
            axis=2,
        )

        # Crop to original size in case Y, X are not exact multiples of the factor
        masks_zyx = masks_up[:, :Y, :X]
    else:
        masks_zyx = masks_ds

    masks_zyx = masks_zyx.astype(np.int32, copy=False)

    if verbose:
        print(f"[segment_nuclei_cpsam_3d] Output masks shape: {masks_zyx.shape}, dtype={masks_zyx.dtype}")

    return masks_zyx


def crop_and_save_nuclei_from_mask(
    mask_zyx: np.ndarray,
    arr_zcyx: np.ndarray,
    output_dir: str | Path,
    basename: str,
    margin_xy: int = 5,
    margin_z: int = 1,
) -> list[Path]:
    """
    As before, but skip saving any nucleus whose bounding box touches
    the edges of the full Z/Y/X volume.
    """

    # Basic checks
    if mask_zyx.ndim != 3:
        raise ValueError(f"mask_zyx must have shape (Z, Y, X), got {mask_zyx.shape}")
    if arr_zcyx.ndim != 4:
        raise ValueError(f"arr_zcyx must have shape (Z, C, Y, X), got {arr_zcyx.shape}")

    Zm, Ym, Xm = mask_zyx.shape
    Zi, Ci, Yi, Xi = arr_zcyx.shape

    if (Zm, Ym, Xm) != (Zi, Yi, Xi):
        raise ValueError(
            f"Mask and image spatial dimensions must match: "
            f"mask={mask_zyx.shape}, img={arr_zcyx.shape}"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    Z, Y, X = Zm, Ym, Xm

    # Measure regions
    props = regionprops(mask_zyx)

    saved_paths: list[Path] = []

    for p in props:
        label_id = p.label

        # bbox is (z_min, y_min, x_min, z_max, y_max, x_max)
        z0_s, y0_s, x0_s, z1_s, y1_s, x1_s = p.bbox

        # ------------------------------------------------------------------
        # Skip nuclei that touch any boundary of the full volume
        # ------------------------------------------------------------------
        touches_edge = (
            z0_s == 0 or y0_s == 0 or x0_s == 0 or
            z1_s == Z or y1_s == Y or x1_s == X
        )

        if touches_edge:
            # print optional message
            # print(f"Skipping nucleus {label_id} (touches boundary)")
            continue
        # ------------------------------------------------------------------

        # Apply padding and clamp
        z0 = max(0, z0_s - margin_z)
        z1 = min(Z, z1_s + margin_z)

        y0 = int(max(0, np.floor(y0_s - margin_xy)))
        y1 = int(min(Y, np.ceil(y1_s + margin_xy)))

        x0 = int(max(0, np.floor(x0_s - margin_xy)))
        x1 = int(min(X, np.ceil(x1_s + margin_xy)))

        # Crop (preserve channels)
        crop = arr_zcyx[z0:z1, :, y0:y1, x0:x1]

        # Filename: basename_nXX.tif
        fname = f"{basename}n{label_id:02d}.tif"
        out_path = output_dir / fname

        save_tiff(histogram_stretch(crop), out_path)
        saved_paths.append(out_path)

    return saved_paths


def _natural_key(s: str):
    """
    Key function for natural sorting (e.g. chr2 < chr10).
    """
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", str(s))
    ]

def _channel_sort_key(label: str):
    """
    Sort so that:
      - 'dapi' comes first
      - then all labels starting with 'chr' (naturally sorted)
      - then everything else (naturally sorted)
    """
    low = label.lower()

    if low == "dapi":
        group = 0
    elif low.startswith("chr"):
        group = 1
    else:
        group = 2

    # Within each group, use natural sort on the label
    return (group, _natural_key(low))

def stack_single_file(
    filename: str | Path,
    data_folder: str | Path,
    channel_coding_txt: str | Path,
) -> tuple[np.ndarray, list[str]]:
    """
    Process a single filename using channel_coding_txt.

    For this one filename:
      * Iterate over each folder specified in `channel_coding_txt`.
      * Load ZCYX TIFF from data_folder/folder/filename.
      * Drop channels labeled 'dapi', 'beads', or 'None'-like, with ONE exception:
          - In the FIRST folder only, keep 'dapi'.
      * Concatenate all kept channels along the C axis.
      * Build channel_list in the same order as channels appear in the combined array.
      * Finally, naturally sort channels (and reorder the array accordingly).

    Parameters
    ----------
    filename : str or Path
        The image filename to process (e.g. 'r01c01f01.tiff').
    data_folder : str or Path
        Root folder containing the subfolders listed in channel_coding_txt.
    channel_coding_txt : str or Path
        Tab-delimited file with header including 'folder' and channel columns like
        'ch0', 'ch1', 'ch2', ... Each row defines the channel labels for one folder.

    Returns
    -------
    (array_zcyx, channel_list)
        array_zcyx : np.ndarray
            Final sorted ZCYX array.
        channel_list : list[str]
            Channel labels, in the same order as the C axis of array_zcyx.
    """
    filename = Path(filename).name  # ensure just the name
    data_folder = Path(data_folder)
    channel_coding_txt = Path(channel_coding_txt)

    # ------------------------------------------------------------------
    # 1) Read channel coding table
    # ------------------------------------------------------------------
    channel_rows: list[dict] = []
    with channel_coding_txt.open("r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None:
            raise RuntimeError(f"No header found in {channel_coding_txt}")
        fieldnames = reader.fieldnames

        if "folder" not in fieldnames:
            raise RuntimeError(
                f"'folder' column not found in {channel_coding_txt} header: {fieldnames}"
            )

        # Channel columns = all columns except 'folder'
        channel_cols = [col for col in fieldnames if col != "folder"]

        for row in reader:
            channel_rows.append(row)

    if not channel_rows:
        raise RuntimeError(f"No rows found in {channel_coding_txt}")

    # ------------------------------------------------------------------
    # 2) Helper to turn a channel column name into its numeric index in the C axis
    #    (e.g. "ch0" -> 0, "ch3" -> 3). If pattern is missing, fallback to order.
    # ------------------------------------------------------------------
    def _channel_index_from_col(col: str, fallback_index: int) -> int:
        m = re.search(r"(\d+)", col)
        if m:
            return int(m.group(1))
        return fallback_index

    # Labels to treat as "None"/to drop
    none_like = {"", "none", "null", "na", "n/a"}

    # ------------------------------------------------------------------
    # 3) Main logic for this single filename
    # ------------------------------------------------------------------
    all_kept_arrays: list[np.ndarray] = []
    all_channels: list[str] = []
    reference_shape: tuple[int, int, int] | None = None  # (Z, Y, X)

    for folder_idx, row in enumerate(channel_rows):
        folder_name = row["folder"]
        tif_path = data_folder / folder_name / filename

        # Load TIFF
        arr = load_tiff(tif_path)

        if arr.ndim != 4:
            raise ValueError(
                f"Expected ZCYX (4D) array from {tif_path}, got shape {arr.shape}"
            )

        Z, C, Y, X = arr.shape

        # Enforce consistent Z/Y/X across folders
        if reference_shape is None:
            reference_shape = (Z, Y, X)
        else:
            if (Z, Y, X) != reference_shape:
                raise ValueError(
                    f"Inconsistent Z/Y/X shape for {tif_path}: {arr.shape}, "
                    f"expected (Z, C, {reference_shape[1]}, {reference_shape[2]})"
                )

        # Decide which channels to keep in this folder
        keep_indices: list[int] = []
        keep_labels: list[str] = []

        for fallback_idx, col in enumerate(channel_cols):
            raw_label = row.get(col, "")
            label = (raw_label or "").strip()
            label_lower = label.lower()

            # Skip None-like channels
            if label_lower in none_like:
                continue

            # Drop 'beads' everywhere
            if label_lower == "beads":
                continue

            # Drop 'dapi' except in the first folder
            if label_lower == "dapi" and folder_idx > 0:
                continue

            # Keep this channel
            ch_idx = _channel_index_from_col(col, fallback_idx)
            if ch_idx < 0 or ch_idx >= C:
                raise IndexError(
                    f"Channel index {ch_idx} from column '{col}' out of range "
                    f"for array with C={C} from {tif_path}"
                )

            keep_indices.append(ch_idx)
            keep_labels.append(label)

        if not keep_indices:
            # Nothing to keep from this folder for this file
            continue

        kept_arr = arr[:, keep_indices, :, :]  # (Z, len(keep_indices), Y, X)
        all_kept_arrays.append(kept_arr)
        all_channels.extend(keep_labels)

    if not all_kept_arrays:
        raise RuntimeError(
            f"No channels were kept for filename '{filename}'. "
            f"Check channel_coding_txt filters or input files."
        )

    # ------------------------------------------------------------------
    # 4) Concatenate kept arrays over the C axis
    # ------------------------------------------------------------------
    if len(all_kept_arrays) == 1:
        combined = all_kept_arrays[0]
    else:
        combined = np.concatenate(all_kept_arrays, axis=1)  # still ZCYX

    num_channels = combined.shape[1]
    if num_channels != len(all_channels):
        raise RuntimeError(
            f"Channel list length ({len(all_channels)}) does not match "
            f"C dimension ({num_channels}) for '{filename}'."
        )

    # ------------------------------------------------------------------
    # 5) Natural sort by channel name and reorder array + list
    # ------------------------------------------------------------------
    sorted_indices = sorted(
        range(num_channels), key=lambda i: _channel_sort_key(all_channels[i])
    )
    combined_sorted = combined[:, sorted_indices, :, :]
    channels_sorted = [all_channels[i] for i in sorted_indices]

    return combined_sorted, channels_sorted