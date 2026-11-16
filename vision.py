"""Simple reference-image deviation check for controlled inspection photos.

This is image differencing, not a trained defect classifier. It assumes both
photos use the same camera, framing, lighting, scale and part orientation.
"""
from __future__ import annotations
from io import BytesIO
from typing import Dict, Tuple

import numpy as np
from PIL import Image, ImageFilter


def compare_reference_images(reference_bytes: bytes, sample_bytes: bytes, size: int = 512) -> Dict[str, object]:
    ref = Image.open(BytesIO(reference_bytes)).convert("RGB")
    sample = Image.open(BytesIO(sample_bytes)).convert("RGB")
    original_size_match = ref.size == sample.size
    ref = ref.resize((size, size), Image.Resampling.LANCZOS)
    sample = sample.resize((size, size), Image.Resampling.LANCZOS)
    ref_arr = np.asarray(ref, dtype=np.float32)
    sample_arr = np.asarray(sample, dtype=np.float32)

    # Blur suppresses isolated pixel noise. This still assumes controlled,
    # matched camera position, lighting, scale and part orientation.
    ref_blur = np.asarray(ref.filter(ImageFilter.GaussianBlur(radius=1)), dtype=np.float32)
    sample_blur = np.asarray(sample.filter(ImageFilter.GaussianBlur(radius=1)), dtype=np.float32)
    diff = np.abs(ref_blur - sample_blur).mean(axis=2)
    mean_diff = float(diff.mean())
    deviation_threshold = 22.0
    raw_mask = (diff > deviation_threshold).astype(np.uint8) * 255
    mask_image = Image.fromarray(raw_mask, mode="L")
    # Remove isolated speckles and close tiny gaps in the highlighted regions.
    mask_image = mask_image.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(5))
    mask = np.asarray(mask_image) > 0
    deviation_pct = float(mask.mean() * 100.0)
    similarity = float(np.clip(100.0 - (mean_diff / 255.0 * 100.0), 0.0, 100.0))

    heat = np.clip(diff * 4.0, 0, 255).astype(np.uint8)
    heat_rgb = np.zeros((size, size, 3), dtype=np.uint8)
    heat_rgb[:, :, 0] = heat
    heat_image = Image.fromarray(heat_rgb, mode="RGB")

    # Overlay deviation pixels in translucent red on the current sample.
    overlay = np.asarray(sample).copy()
    red = np.zeros_like(overlay)
    red[:, :, 0] = 255
    overlay[mask] = (0.45 * overlay[mask] + 0.55 * red[mask]).astype(np.uint8)
    overlay_image = Image.fromarray(overlay, mode="RGB")
    ys, xs = np.where(mask)
    bbox = None
    if len(xs):
        bbox = {"x_min": int(xs.min()), "y_min": int(ys.min()),
                "x_max": int(xs.max()), "y_max": int(ys.max())}

    return {
        "similarity": similarity,
        "deviation_pct": deviation_pct,
        "mean_pixel_difference": mean_diff,
        "heatmap": heat_image,
        "deviation_overlay": overlay_image,
        "deviation_mask": mask_image,
        "deviation_bbox": bbox,
        "original_size_match": original_size_match,
    }
