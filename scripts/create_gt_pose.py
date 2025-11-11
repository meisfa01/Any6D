#!/usr/bin/env python3
import argparse
import math
import os
from typing import Iterable, Tuple

import numpy as np


def euler_zyx_to_matrix(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
	"""Create rotation matrix from ZYX Euler angles in degrees."""
	rz = math.radians(rz_deg)
	ry = math.radians(ry_deg)
	rx = math.radians(rx_deg)

	sz, cz = math.sin(rz), math.cos(rz)
	sy, cy = math.sin(ry), math.cos(ry)
	sx, cx = math.sin(rx), math.cos(rx)

	# R = Rz * Ry * Rx
	Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
	Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
	Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
	return (Rz @ Ry) @ Rx


def quat_wxyz_to_matrix(w: float, x: float, y: float, z: float) -> np.ndarray:
	"""Create rotation matrix from unit quaternion in (w, x, y, z) order."""
	q = np.array([w, x, y, z], dtype=np.float64)
	n = np.linalg.norm(q)
	if n < 1e-12:
		raise ValueError("Quaternion has near-zero norm.")
	w, x, y, z = (q / n).tolist()
	return np.array(
		[
			[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
			[2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
			[2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
		],
		dtype=np.float64,
	)


def assemble_pose(R: np.ndarray, t: Iterable[float]) -> np.ndarray:
	T = np.eye(4, dtype=np.float64)
	T[:3, :3] = R
	t = np.asarray(list(t), dtype=np.float64).reshape(3)
	T[:3, 3] = t
	return T


def save_pose_txt(T: np.ndarray, out_path: str):
	os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
	with open(out_path, "w") as f:
		for r in range(4):
			f.write(" ".join(f"{v:.18e}" for v in T[r, :]) + "\n")
	print(f"Saved pose: {out_path}")


def main():
	parser = argparse.ArgumentParser(
		description="Create a 4x4 ground-truth pose file compatible with Any6D results format. Either input euler angles or quaternion."
	)
	parser.add_argument(
		"--mode",
		choices=["euler_zyx", "quat_wxyz"],
		default="euler_zyx",
		help="Rotation parameterization.",
	)
	# Euler (degrees)
	parser.add_argument("--rx", type=float, default=0.0, help="Rotation about X in degrees.")
	parser.add_argument("--ry", type=float, default=0.0, help="Rotation about Y in degrees.")
	parser.add_argument("--rz", type=float, default=0.0, help="Rotation about Z in degrees.")
	# Quaternion (w, x, y, z)
	parser.add_argument("--qw", type=float, default=1.0, help="Quaternion w.")
	parser.add_argument("--qx", type=float, default=0.0, help="Quaternion x.")
	parser.add_argument("--qy", type=float, default=0.0, help="Quaternion y.")
	parser.add_argument("--qz", type=float, default=0.0, help="Quaternion z.")
	# Translation (meters)
	parser.add_argument("--tx", type=float, default=0.0, help="Translation X (meters).")
	parser.add_argument("--ty", type=float, default=0.0, help="Translation Y (meters).")
	parser.add_argument("--tz", type=float, default=1.0, help="Translation Z (meters).")
	parser.add_argument(
		"--output",
		required=True,
		help="Output pose txt path, e.g., results/test_ycb/004_sugar_box/004_sugar_box_gt_pose.txt",
	)
	args = parser.parse_args()

	if args.mode == "euler_zyx":
		R = euler_zyx_to_matrix(args.rx, args.ry, args.rz)
	else:
		R = quat_wxyz_to_matrix(args.qw, args.qx, args.qy, args.qz)
	T = assemble_pose(R, (args.tx, args.ty, args.tz))
	save_pose_txt(T, args.output)


if __name__ == "__main__":
	main()


