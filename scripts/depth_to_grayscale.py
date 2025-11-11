#!/usr/bin/env python3
import argparse
import os
from typing import Tuple

import cv2
import numpy as np


def is_probably_jet(image_bgr: np.ndarray) -> bool:
	"""Heuristic: image is 3-channel, many saturated reds/blues, low greens at extremes."""
	if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
		return False
	h, w, _ = image_bgr.shape
	sample = image_bgr.reshape(-1, 3)

	# Count pixels near blue or red corners
	near_blue = np.sum((sample[:, 0] > 200) & (sample[:, 1] < 80) & (sample[:, 2] < 80))
	near_red = np.sum((sample[:, 2] > 200) & (sample[:, 1] < 80) & (sample[:, 0] < 80))
	ratio = (near_blue + near_red) / float(sample.shape[0] + 1e-6)
	return ratio > 0.01


def invert_jet_to_gray(image_bgr: np.ndarray) -> np.ndarray:
	"""
	Approximate inversion of OpenCV's JET colormap back to 0..255 scalar.
	Method: brute-force nearest color on the 256-color JET LUT.
	"""

	jet_lut = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(-1, 1), cv2.COLORMAP_JET)
	jet_lut = jet_lut.reshape(256, 3).astype(np.int16)

	h, w, _ = image_bgr.shape
	img = image_bgr.astype(np.int16).reshape(-1, 3)

	lut_sq = np.sum(jet_lut ** 2, axis=1).reshape(1, 256) 
	img_sq = np.sum(img ** 2, axis=1).reshape(-1, 1)    
	dot = img @ jet_lut.T                                  
	dist2 = img_sq + lut_sq - 2 * dot
	indices = np.argmin(dist2, axis=1).astype(np.uint8)
	return indices.reshape(h, w)


def normalize_depth_to_gray(depth: np.ndarray, min_val: float = None, max_val: float = None) -> np.ndarray:
	"""Normalize single-channel depth to 0..255 uint8 grayscale."""
	if min_val is None:
		min_val = float(np.nanmin(depth))
	if max_val is None:
		max_val = float(np.nanmax(depth))
	if not np.isfinite(min_val) or not np.isfinite(max_val) or max_val <= min_val:
		vmin, vmax = np.nanpercentile(depth, [2.0, 98.0])
		min_val, max_val = float(vmin), float(vmax if vmax > vmin else vmin + 1.0)
	scaled = np.clip((depth - min_val) / (max_val - min_val), 0.0, 1.0)
	return (scaled * 255.0 + 0.5).astype(np.uint8)


def main():
	parser = argparse.ArgumentParser(
		description="Convert depth image to grayscale. Supports 16-bit depth or JET-colored PNGs."
	)
	parser.add_argument("--input", required=True, help="Path to input depth image (png/jpg).")
	parser.add_argument(
		"--output",
		required=False,
		help="Path to save grayscale PNG. Defaults to <input_dir>/depth.png",
	)
	parser.add_argument(
		"--min",
		type=float,
		default=None,
		help="Optional depth min for normalization (if single-channel depth).",
	)
	parser.add_argument(
		"--max",
		type=float,
		default=None,
		help="Optional depth max for normalization (if single-channel depth).",
	)
	args = parser.parse_args()

	if not os.path.isfile(args.input):
		raise FileNotFoundError(f"Input not found: {args.input}")

	image = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)
	if image is None:
		raise RuntimeError(f"Failed to read: {args.input}")

	if image.ndim == 2:
		# Likely 16-bit or 8-bit single-channel depth
		depth = image.astype(np.float32)
		gray = normalize_depth_to_gray(depth, args.min, args.max)
	else:
		# 3-channel: try to invert from JET
		image_bgr = image
		if not is_probably_jet(image_bgr):
			gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
		else:
			gray = invert_jet_to_gray(image_bgr)

	out_path = args.output
	if not out_path:
		out_dir = os.path.dirname(os.path.abspath(args.input))
		out_path = os.path.join(out_dir, "depth.png")
	os.makedirs(os.path.dirname(out_path), exist_ok=True)

	if not cv2.imwrite(out_path, gray):
		raise RuntimeError(f"Failed to write grayscale depth to: {out_path}")
	print(f"Saved grayscale depth: {out_path}")


if __name__ == "__main__":
	main()


