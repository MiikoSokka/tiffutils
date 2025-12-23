# coding: utf-8
# Author: Kristine Yang & Miiko Sokka

import numpy as np
from skimage import io, filters, feature
from scipy import ndimage
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from skimage.registration import phase_cross_correlation
from skimage.transform import AffineTransform, warp
from skimage.filters import threshold_otsu
import tifffile as tiff
import os
import warnings

def preprocess_channel(image, sigma=1.0):
    """
    Preprocess a single channel from a 3D image stack.
    
    Args:
        image (np.ndarray): 3D channel shape (Z, Y, X)
        sigma (float): Gaussian smoothing sigma
    """
    # Simple intensity normalization
    p1, p99 = np.percentile(image, (1, 99))
    img_norm = np.clip((image - p1) / (p99 - p1), 0, 1)
    
    # 3D Gaussian filter
    return ndimage.gaussian_filter(img_norm, sigma)

def detect_features_3d(image, min_sigma=1, max_sigma=3, threshold=0.1):
    """3D feature detection with improved robustness."""
    # Use maximum intensity projection for initial feature detection
    mip = np.max(image, axis=0)

    # Detect blobs on MIP
    blobs_mip = feature.blob_log(
        mip,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=4,
        threshold=threshold
    )
    
    if len(blobs_mip) == 0:
        return np.array([]), mip
    
    features_3d = []
    z_range = np.arange(image.shape[0])
    
    for blob in blobs_mip:
        y, x = int(blob[0]), int(blob[1])
        y_min = max(0, y-1)
        y_max = min(image.shape[1], y+2)
        x_min = max(0, x-1)
        x_max = min(image.shape[2], x+2)
        
        z_profile = image[:, y_min:y_max, x_min:x_max].mean(axis=(1,2))
        z = np.argmax(z_profile)
        features_3d.append([z, y, x])
    
    return np.array(features_3d), mip

def match_features(moving_features, fixed_features, max_distance=500):
    """Enhanced feature matching with separate XY and Z thresholds
    
    Args:
    1) moving_features and
    2) fixed features:
        - numpy.ndarray of shape (N, 3)
        - A list of N feature points from the image.
        - Each row is a (z, y, x) coordinate.
    3) max_distance (float, default=500)
        - The maximum allowed XY distance for a valid match.
        - The maximum allowed Z distance is max_distance * 0.02
    
    Returns:
    1) matched_moving and
    2) matched_fixed
        - numpy.ndarray of shape (K, 3)
        - The subset of moving features that found valid matches.
        - Format: (z, y, x).
    3) confidence (float)
        - A matching quality score between 0.0 and 1.0.
    """

    # If either input feature set is empty, return empty arrays and zero confidence
    if len(moving_features) == 0 or len(fixed_features) == 0:
        return np.array([]), np.array([]), 0.0
    
    # Calculate separate distances for XY and Z
        # - uses Euclidean distance to compute pairwise distances between all moving and fixed features.
        # - xy_distances: Computes distances only in the XY plane ([:, 1:] extracts only Y and X).
        # - z_distances: Computes distances only in Z ([:, 0:1] extracts only the Z coordinate).
    xy_distances = cdist(moving_features[:, 1:], fixed_features[:, 1:])
    z_distances = cdist(moving_features[:, 0:1], fixed_features[:, 0:1])

    # Set maximum distance thresholds
    max_xy_distance = max_distance
    max_z_distance = max_distance * 0.02  # More strict in Z
    
    # Apply distance thresholds to find matches that satisfy both conditions
        # A valid match satisfies both conditions:
        # - At least one fixed feature is within max_xy_distance in XY.
        # - At least one fixed feature is within max_z_distance in Z.
        # Any(axis=1): Ensures each moving feature has at least one corresponding fixed feature within the defined thresholds.
    valid_matches = (xy_distances < max_xy_distance).any(axis=1) & (z_distances < max_z_distance).any(axis=1)

    # Handle cases where no matches are found
    # - If no valid matches exist, return empty arrays and confidence 0.0
    if not np.any(valid_matches):
        return np.array([]), np.array([]), 0.0
    
    # Find best matches considering both XY and Z
        # Computes a weighted combined distance:
        # - Z-distances are doubled (z_distances*2) to give higher importance to Z-alignment.
        # - np.argmin(...) selects the best matching fixed feature for each moving feature.
    combined_distances = np.sqrt(xy_distances**2 + (z_distances*2)**2)  # Weight Z more
    matches = np.argmin(combined_distances, axis=1)

    # Extract matched feature pairs
        # - Extracts the subset of matched moving features.
        # - Extracts the corresponding fixed features that are their best matches
    matched_moving = moving_features[valid_matches]
    matched_fixed = fixed_features[matches[valid_matches]]
    
    # Compute confidence score based on XY and Z match quality separately
        # XY Confidence:
        # - Mean of valid_matches (fraction of moving features that found a match).
        # - Exponentially decreases as average XY distance increases.
        # Z Confidence:
        # - Similar to XY confidence but applied to Z distances.
        # - Uses a stricter threshold, so a high Z confidence indicates good depth alignment.
        # - Final Confidence = Average of XY and Z confidence scores.
    xy_confidence = np.mean(valid_matches) * np.exp(-np.mean(xy_distances[valid_matches, matches[valid_matches]]) / max_xy_distance)
    z_confidence = np.mean(valid_matches) * np.exp(-np.mean(z_distances[valid_matches, matches[valid_matches]]) / max_z_distance)
    confidence = (xy_confidence + z_confidence) / 2
    
    return matched_moving, matched_fixed, confidence

def calculate_transformation(moving_coords, fixed_coords, max_shift=400):
    """
    Calculate transformation with validation and limits.
    
    Args:
        moving_coords: Coordinates from moving image
        fixed_coords: Coordinates from fixed image
        max_shift: Maximum allowed shift in any dimension
    
    Returns:
        translation: Validated translation parameters
        is_valid: Whether the transformation is valid
    """
    if len(moving_coords) == 0 or len(fixed_coords) == 0:
        return np.zeros(3), False
    
    # Calculate centroids
    moving_centroid = np.mean(moving_coords, axis=0)
    fixed_centroid = np.mean(fixed_coords, axis=0)
    
    # Calculate translation
    translation = fixed_centroid - moving_centroid
    
    # Apply maximum shift constraint
    is_valid = np.all(np.abs(translation) < max_shift)
    
    # If shift is too large, apply minimal correction
    if not is_valid:
        translation = np.clip(translation, -max_shift, max_shift)
    
    return translation, is_valid

def apply_transform_3d(image, translation):
    """Apply 3D translation to an image stack."""
    return ndimage.shift(image, translation, order=1, mode='constant', cval=0)

def register_3d_images(fixed_image, moving_image, bead_channel=1, sigma=1.0,
                       max_shift=400, min_confidence=0.5):
    """
    Register 3D multi-channel images with improved validation.
    
    Args:
        fixed_image: Reference image (Z, Y, X, C)
        moving_image: Moving image (Z, Y, X, C)
        bead_channel: Index of the fiducial bead channel (default set to the second channel)
        sigma: Gaussian smoothing sigma
        max_shift: Maximum allowed shift in pixels
        min_confidence: Minimum confidence threshold for accepting registration
    """
    # Extract and preprocess bead channels
    fixed_beads = fixed_image[:, :, :, bead_channel]
    moving_beads = moving_image[:, :, :, bead_channel]
    
    fixed_proc = preprocess_channel(fixed_beads, sigma)
    moving_proc = preprocess_channel(moving_beads, sigma)
    
    # Detect features
    fixed_features, fixed_mip = detect_features_3d(fixed_proc)
    moving_features, moving_mip = detect_features_3d(moving_proc)
    
    # Match features
    matched_moving, matched_fixed, confidence = match_features(
        moving_features, fixed_features, max_distance=max_shift
    )
    
    # Calculate transformation
    translation, is_valid = calculate_transformation(
        matched_moving, matched_fixed, max_shift=max_shift
    )
    
    # Initialize output array
    registered_image = np.zeros_like(moving_image)

    # Threshold the beads to discard any noise
    f_thresh_val = filters.threshold_otsu(fixed_mip)  # Otsu's method for global thresholding
    fixed_mip_thresh = fixed_mip > f_thresh_val

    m_thresh_val = filters.threshold_otsu(moving_mip)  # Otsu's method for global thresholding
    moving_mip_thresh = moving_mip > m_thresh_val

    xy_shift, error, diffphase = phase_cross_correlation(
        fixed_mip_thresh,
        moving_mip_thresh,
    )
    
    print(f'xy shift is: {xy_shift}')

    # Apply xy drift correction
    transform = AffineTransform(translation=[xy_shift[1], xy_shift[0]])
    
    # Only apply transformation if confident and valid
    if confidence >= min_confidence and is_valid:
        print(f"Applying translation: {translation}")
        print(f"Confidence score: {confidence:.3f}")
        
        for c in range(moving_image.shape[-1]):
            registered_image[:, :, :, c] = apply_transform_3d(
                moving_image[:, :, :, c],
                translation
            )
    else:
        print("Registration confidence too low or invalid transformation.")
        print(f"Confidence score: {confidence:.3f}")
        print(f"Calculated translation (not applied): {translation}")
        registered_image = moving_image.copy()
    
    final_image = np.zeros_like(registered_image)
    for z in range(registered_image.shape[0]):
        for c in range(registered_image.shape[-1]):
            warped_slice = warp(
                registered_image[z, :, :, c],
                transform.inverse,
                mode='constant',
                cval=0,
                preserve_range=True
            ).astype(registered_image.dtype)
            final_image[z, :, :, c] = warped_slice
    
    return final_image, translation, confidence


def register_and_save_batch(
    input_paths,
    output_paths,
    bead_channel=1,
    sigma=1.0,
    max_shift=400,
    min_confidence=0.5,
):
    """
    Register a list of 4-channel ZYXC TIFF stacks to the first image
    and save all images (reference + registered) to the corresponding
    output paths as ZCYX TIFFs.

    Parameters
    ----------
    input_paths : list of str or Path
        List of input TIFF file paths. The first image is used as the
        fixed/reference image; all others are registered to it.
    output_paths : list of str or Path
        List of output TIFF file paths (same length/order as input_paths).
    bead_channel : int, optional
        Index of the bead/fiducial channel used for registration.
    sigma : float, optional
        Gaussian smoothing sigma during preprocessing.
    max_shift : float, optional
        Maximum allowed 3D shift (in pixels) for the bead-based registration.
    min_confidence : float, optional
        Minimum confidence threshold to accept the bead-based registration.
        If below this, the moving image is only corrected by XY phase
        correlation, not 3D translation.

    Returns
    -------
    results : list of dict
        For each input (except the reference), a dict with:
        {
            "input_path": ...,
            "output_path": ...,
            "translation": np.ndarray(3,),
            "confidence": float
        }
        For the reference image, translation/confidence are None.
    """
    if len(input_paths) != len(output_paths):
        raise ValueError("input_paths and output_paths must have the same length.")
    if len(input_paths) == 0:
        raise ValueError("input_paths must not be empty.")

    # Ensure everything is str for io/tiff
    input_paths = [str(p) for p in input_paths]
    output_paths = [str(p) for p in output_paths]

    results = []

    # --- Load reference (fixed) image ---
    print(f"Loading fixed/reference image: {input_paths[0]}")
    fixed_image = io.imread(input_paths[0])  # expected shape (Z, Y, X, C)

    # Save the fixed image (just axis reorder ZYXC -> ZCYX)
    fixed_out = np.moveaxis(fixed_image, 3, 1)  # (Z, C, Y, X)
    os.makedirs(os.path.dirname(output_paths[0]), exist_ok=True)
    tiff.imwrite(
        output_paths[0],
        fixed_out,
        imagej=True,
        metadata={"axes": "ZCYX"},
    )
    results.append(
        {
            "input_path": input_paths[0],
            "output_path": output_paths[0],
            "translation": None,
            "confidence": None,
        }
    )
    print(f"Saved fixed/reference image to: {output_paths[0]}")

    # --- Process all moving images ---
    for in_path, out_path in zip(input_paths[1:], output_paths[1:]):
        print(f"\nRegistering moving image:\n  {in_path}\n  -> {out_path}")
        moving_image = io.imread(in_path)  # (Z, Y, X, C)

        registered_image, translation, confidence = register_3d_images(
            fixed_image=fixed_image,
            moving_image=moving_image,
            bead_channel=bead_channel,
            sigma=sigma,
            max_shift=max_shift,
            min_confidence=min_confidence,
        )

        # Reorder axes for ImageJ (ZCYX)
        registered_save = np.moveaxis(registered_image, 3, 1)  # (Z, C, Y, X)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        tiff.imwrite(
            out_path,
            registered_save,
            imagej=True,
            metadata={"axes": "ZCYX"},
        )
        print(f"Saved registered image to: {out_path}")

        results.append(
            {
                "input_path": in_path,
                "output_path": out_path,
                "translation": translation,
                "confidence": float(confidence),
            }
        )

    return results