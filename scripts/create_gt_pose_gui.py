#!/usr/bin/env python3
import argparse
import os
from typing import Optional, Tuple

import cv2
import numpy as np


def load_K(k_path: str) -> np.ndarray:
	K = np.loadtxt(k_path).reshape(3, 3).astype(np.float64)
	return K


def euler_zyx_to_R(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
	rz = np.deg2rad(rz_deg)
	ry = np.deg2rad(ry_deg)
	rx = np.deg2rad(rx_deg)
	sz, cz = np.sin(rz), np.cos(rz)
	sy, cy = np.sin(ry), np.cos(ry)
	sx, cx = np.sin(rx), np.cos(rx)
	Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
	Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
	Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
	return (Rz @ Ry) @ Rx


def assemble_T(R: np.ndarray, t: Tuple[float, float, float]) -> np.ndarray:
	T = np.eye(4, dtype=np.float64)
	T[:3, :3] = R
	T[:3, 3] = np.array(t, dtype=np.float64)
	return T


def save_pose_txt(T: np.ndarray, out_path: str):
	os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
	with open(out_path, "w") as f:
		for r in range(4):
			f.write(" ".join(f"{v:.18e}" for v in T[r, :]) + "\n")
	print(f"Saved pose: {out_path}")


def project_points(X_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
	x = X_cam[:, 0] / X_cam[:, 2]
	y = X_cam[:, 1] / X_cam[:, 2]
	uv = np.stack([x, y, np.ones_like(x)], axis=1) @ K.T
	return uv[:, :2]


def make_axes_points(scale: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
	orig = np.zeros((1, 3), dtype=np.float64)
	x_end = np.array([[scale, 0.0, 0.0]], dtype=np.float64)
	y_end = np.array([[0.0, scale, 0.0]], dtype=np.float64)
	z_end = np.array([[0.0, 0.0, scale]], dtype=np.float64)
	return orig, np.concatenate([x_end, y_end, z_end], axis=0)


def draw_axes(img: np.ndarray, K: np.ndarray, R: np.ndarray, t: Tuple[float, float, float], scale: float = 0.1):
	orig, ends = make_axes_points(scale)
	Xobj = np.concatenate([orig, ends], axis=0)
	Xcam = (R @ Xobj.T).T + np.array(t, dtype=np.float64)
	depths = Xcam[:, 2]
	if np.any(depths <= 1e-6):
		return img
	uv = project_points(Xcam, K)
	o = tuple(np.round(uv[0]).astype(int))
	xp = tuple(np.round(uv[1]).astype(int))
	yp = tuple(np.round(uv[2]).astype(int))
	zp = tuple(np.round(uv[3]).astype(int))
	draw = img.copy()
	cv2.line(draw, o, xp, (0, 0, 255), 3)   # X red
	cv2.line(draw, o, yp, (0, 255, 0), 3)   # Y green
	cv2.line(draw, o, zp, (255, 0, 0), 3)   # Z blue
	cv2.circle(draw, o, 5, (255, 255, 255), -1)
	return draw


def read_pose_txt(path: str) -> Optional[np.ndarray]:
	if not path or not os.path.isfile(path):
		return None
	try:
		M = np.loadtxt(path).reshape(4, 4).astype(np.float64)
		return M
	except Exception:
		return None


def main():
	parser = argparse.ArgumentParser(description="Interactive GUI to set GT pose with visual 3D axes overlay.")
	parser.add_argument("--image", required=True, help="Path to color image.")
	parser.add_argument("--K", required=True, help="Path to 3x3 intrinsics txt (like K.txt).")
	parser.add_argument("--output", required=True, help="Output pose txt path.")
	parser.add_argument("--init", required=False, help="Optional existing 4x4 pose txt to initialize.")
	parser.add_argument("--scale", type=float, default=0.1, help="Axes length in meters.")
	args = parser.parse_args()

	img = cv2.imread(args.image, cv2.IMREAD_COLOR)
	if img is None:
		raise RuntimeError(f"Failed to read image: {args.image}")
	K = load_K(args.K)

	# Initialize pose
	init_T = read_pose_txt(args.init)
	if init_T is not None:
		R0 = init_T[:3, :3]
		t0 = init_T[:3, 3].tolist()
		# Naive extraction of ZYX Euler from R for initialization (not exact but OK for start)
		ry = -np.arcsin(np.clip(R0[2, 0], -1.0, 1.0))
		rx = np.arctan2(R0[2, 1], R0[2, 2])
		rz = np.arctan2(R0[1, 0], R0[0, 0])
		rx0, ry0, rz0 = np.rad2deg([rx, ry, rz])
	else:
		rx0, ry0, rz0 = 0.0, 0.0, 0.0
		t0 = [0.0, 0.0, 1.0]

	win = "GT Pose (S: save, Q/Esc: quit)"
	cv2.namedWindow(win, cv2.WINDOW_NORMAL)
	cv2.resizeWindow(win, 1280, 800)

	
	def tb_setup(name, val, minv, maxv):
		cv2.createTrackbar(name, win, val, maxv - minv, lambda x: None)
		cv2.setTrackbarMin(name, win, minv)
		cv2.setTrackbarPos(name, win, val)

	ang_min, ang_max = -180, 180
	tb_setup("rx_deg", int(round(rx0)), ang_min, ang_max)
	tb_setup("ry_deg", int(round(ry0)), ang_min, ang_max)
	tb_setup("rz_deg", int(round(rz0)), ang_min, ang_max)
	# Translations in mm
	tx0_mm, ty0_mm, tz0_mm = int(round(t0[0] * 1000)), int(round(t0[1] * 1000)), int(round(t0[2] * 1000))
	tb_setup("tx_mm", tx0_mm, -1000, 1000)
	tb_setup("ty_mm", ty0_mm, -1000, 1000)
	tb_setup("tz_mm", tz0_mm, 200, 3000)  

	while True:
		rx = cv2.getTrackbarPos("rx_deg", win)
		ry = cv2.getTrackbarPos("ry_deg", win)
		rz = cv2.getTrackbarPos("rz_deg", win)
		tx = cv2.getTrackbarPos("tx_mm", win) / 1000.0
		ty = cv2.getTrackbarPos("ty_mm", win) / 1000.0
		tz = cv2.getTrackbarPos("tz_mm", win) / 1000.0

		R = euler_zyx_to_R(rx, ry, rz)
		overlay = draw_axes(img, K, R, (tx, ty, tz), scale=args.scale)
		info = f"rx={rx} ry={ry} rz={rz} | tx={tx:.3f} ty={ty:.3f} tz={tz:.3f} (S: save)"
		cv2.putText(overlay, info, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
		cv2.putText(overlay, info, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
		cv2.imshow(win, overlay)

		key = cv2.waitKey(20) & 0xFF
		if key in (ord("s"), ord("S")):
			T = assemble_T(R, (tx, ty, tz))
			save_pose_txt(T, args.output)
		if key in (ord("q"), 27):
			break

	cv2.destroyWindow(win)


if __name__ == "__main__":
	main()


