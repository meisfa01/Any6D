#!/usr/bin/env python3
import argparse
import os
from typing import List, Tuple

import cv2
import numpy as np


GC_BGD = 0
GC_FGD = 1
GC_PR_BGD = 2
GC_PR_FGD = 3


class ClickSegmentationGrabCut:
	def __init__(self, image: np.ndarray, window_name: str = "Create Mask"):
		self.image_bgr = image
		self.window_name = window_name
		self.h, self.w = image.shape[:2]
		self.radius_px = max(3, int(0.01 * max(self.h, self.w)))
		self.fg_points: List[Tuple[int, int]] = []
		self.bg_points: List[Tuple[int, int]] = []
		self.mask_gc = np.full((self.h, self.w), GC_PR_BGD, dtype=np.uint8)
		self.last_vis = self.image_bgr.copy()

	def _stamp_points_on_mask(self, mask: np.ndarray):
		for x, y in self.fg_points:
			cv2.circle(mask, (x, y), self.radius_px, GC_FGD, -1)
		for x, y in self.bg_points:
			cv2.circle(mask, (x, y), self.radius_px, GC_BGD, -1)

	def _run_grabcut(self, iterations: int = 5):
		# If no seeds, skip
		if len(self.fg_points) == 0 and len(self.bg_points) == 0:
			return
		
		mask = np.full((self.h, self.w), GC_PR_BGD, dtype=np.uint8)
		bw = max(5, int(0.02 * min(self.h, self.w)))
		mask[:bw, :] = GC_BGD
		mask[-bw:, :] = GC_BGD
		mask[:, :bw] = GC_BGD
		mask[:, -bw:] = GC_BGD
		
		# User seeds
		self._stamp_points_on_mask(mask)
		bg_model = np.zeros((1, 65), np.float64)
		fg_model = np.zeros((1, 65), np.float64)
		cv2.grabCut(
			self.image_bgr,
			mask,
			None,
			bg_model,
			fg_model,
			iterations,
			cv2.GC_INIT_WITH_MASK,
		)
		self.mask_gc = mask

	def _visualize(self) -> np.ndarray:
		vis = self.image_bgr.copy()
		mask_bin = np.where((self.mask_gc == GC_FGD) | (self.mask_gc == GC_PR_FGD), 255, 0).astype(np.uint8)
		contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)
		# Draw seeds
		for x, y in self.fg_points:
			cv2.circle(vis, (x, y), self.radius_px, (0, 255, 0), -1)
		for x, y in self.bg_points:
			cv2.circle(vis, (x, y), self.radius_px, (0, 0, 255), -1)
		# Instructions
		text = "L: FG click  R: BG click  U: undo  C: clear  Enter/S: save  Q/Esc: quit"
		cv2.putText(vis, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
		cv2.putText(vis, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
		return vis

	def _recompute_and_show(self):
		self._run_grabcut(iterations=5)
		self.last_vis = self._visualize()
		cv2.imshow(self.window_name, self.last_vis)

	def on_mouse(self, event, x, y, flags, param):
		if event == cv2.EVENT_LBUTTONDOWN:
			self.fg_points.append((x, y))
			self._recompute_and_show()
		elif event == cv2.EVENT_RBUTTONDOWN:
			self.bg_points.append((x, y))
			self._recompute_and_show()

	def run(self) -> np.ndarray:
		cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
		cv2.resizeWindow(self.window_name, 1280, 720)
		cv2.setMouseCallback(self.window_name, self.on_mouse)
		
		self.last_vis = self.image_bgr.copy()
		msg = "Click object (L=FG). Add R=BG if needed. S=save, Q=quit."
		cv2.putText(self.last_vis, msg, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
		cv2.putText(self.last_vis, msg, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
		cv2.imshow(self.window_name, self.last_vis)

		while True:
			key = cv2.waitKey(50) & 0xFF
			if key in (13, 10, ord("s"), ord("S")):
				break
			elif key in (ord("u"), ord("U")):
				if self.fg_points:
					self.fg_points.pop()
				elif self.bg_points:
					self.bg_points.pop()
				self._recompute_and_show()
			elif key in (ord("c"), ord("C")):
				self.fg_points.clear()
				self.bg_points.clear()
				self._recompute_and_show()
			elif key in (ord("q"), 27):
				self.fg_points.clear()
				self.bg_points.clear()
				break

		cv2.destroyWindow(self.window_name)
		mask_bin = np.where((self.mask_gc == GC_FGD) | (self.mask_gc == GC_PR_FGD), 255, 0).astype(np.uint8)
		return mask_bin


def main():
	parser = argparse.ArgumentParser(
		description="Click-refine segmentation using GrabCut. Left-click adds FG, right-click adds BG. Saves white-on-black mask."
	)
	parser.add_argument("--image", required=True, help="Path to input color image.")
	parser.add_argument(
		"--output",
		required=False,
		help="Path to save mask PNG. Defaults to <image_dir>/mask.png",
	)
	args = parser.parse_args()

	if not os.path.isfile(args.image):
		raise FileNotFoundError(f"Image not found: {args.image}")

	image_bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
	if image_bgr is None:
		raise RuntimeError(f"Failed to read image: {args.image}")

	tool = ClickSegmentationGrabCut(image_bgr)
	mask = tool.run()

	out_path = args.output
	if not out_path:
		out_dir = os.path.dirname(os.path.abspath(args.image))
		out_path = os.path.join(out_dir, "mask.png")
	os.makedirs(os.path.dirname(out_path), exist_ok=True)

	if not cv2.imwrite(out_path, mask):
		raise RuntimeError(f"Failed to write mask to: {out_path}")
	print(f"Saved mask: {out_path}")


if __name__ == "__main__":
	main()


