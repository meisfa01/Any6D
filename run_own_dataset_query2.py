import os

import argparse

import glob

import numpy as np

import cv2

import trimesh

from tqdm import tqdm

import nvdiffrast.torch as dr

from estimater import Any6D

from foundationpose.Utils import visualize_frame_results, align_mesh_to_coordinate


def load_intrinsics(k_path: str) -> np.ndarray:
    K = np.loadtxt(k_path).reshape(3, 3).astype(np.float64)

    return K


def list_images(folder: str):
    exts = ("*.png", "*.jpg", "*.jpeg")

    files = []

    for e in exts:
        files.extend(glob.glob(os.path.join(folder, e)))

    files.sort()

    return files


def read_depth(path: str, scale: float = 1000.0) -> np.ndarray:
    d = cv2.imread(path, cv2.IMREAD_ANYDEPTH)

    if d is None:
        raise RuntimeError(f"Failed to read depth: {path}")

    d = d.astype(np.float32) / float(scale)

    return d


def read_mask(path: str) -> np.ndarray:
    m = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if m is None:
        return None

    if m.ndim == 3:
        m = m[..., 0]

    return (m > 0).astype(bool)


def main():
    parser = argparse.ArgumentParser(description="Estimate poses on a custom sequence without GT mesh/metrics.")

    parser.add_argument("--sequence_root", required=True, help="Path to obj_folder containing rgb/, depth/, meta/K.txt")

    parser.add_argument("--mesh_path", required=True, help="Path to an object mesh .obj to register.")

    parser.add_argument("--mask_dir", required=False, help="Optional dir with binary masks per frame (same ordering).")

    parser.add_argument("--depth_scale", type=float, default=1000.0, help="Scale to convert depth units to meters.")

    parser.add_argument("--stride", type=int, default=1, help="Process every N-th frame.")

    args = parser.parse_args()

    seq_root = args.sequence_root

    rgb_dir = os.path.join(seq_root, "rgb")

    depth_dir = os.path.join(seq_root, "depth")

    meta_dir = os.path.join(seq_root, "meta")

    k_path = os.path.join(meta_dir, "K.txt")

    if not os.path.isfile(k_path):
        raise FileNotFoundError(f"K.txt not found at: {k_path}")

    K = load_intrinsics(k_path)

    rgb_files = list_images(rgb_dir)

    depth_files = list_images(depth_dir)

    if len(rgb_files) == 0 or len(depth_files) == 0:
        raise RuntimeError("No rgb/depth images found. Expected rgb/ and depth/ with frames.")

    if len(rgb_files) != len(depth_files):
        print(
            f"Warning: rgb ({len(rgb_files)}) and depth ({len(depth_files)}) counts differ. Will pair by sorted order.")

    mask_files = None

    if args.mask_dir:

        mask_files = list_images(args.mask_dir)

        if len(mask_files) == 0:
            print(f"Warning: no masks found in {args.mask_dir}; will fall back to depth>0 masks.")

            mask_files = None

    # Load or prepare mesh

    mesh = trimesh.load(args.mesh_path)

    mesh = align_mesh_to_coordinate(mesh)

    save_path = seq_root

    obj_name = os.path.basename(os.path.normpath(seq_root))

    os.makedirs(save_path, exist_ok=True)

    glctx = dr.RasterizeCudaContext()

    est = Any6D(symmetry_tfs=None, mesh=mesh, debug_dir=save_path, debug=1)

    for i in tqdm(range(0, min(len(rgb_files), len(depth_files)), args.stride), desc="Frames"):

        rgb_path = rgb_files[i]

        depth_path = depth_files[i]

        color = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)

        depth = read_depth(depth_path, scale=args.depth_scale)

        if mask_files and i < len(mask_files):

            mask = read_mask(mask_files[i])

        else:

            mask = (depth > 0).astype(bool)

        pred_pose = est.register_any6d(K=K, rgb=color, depth=depth, ob_mask=mask, iteration=5, name=f"{obj_name}")

        np.savetxt(os.path.join(save_path, f"{obj_name}_pose_{i:05d}.txt"), pred_pose)

        try:

            # est_only=True to skip GT panel; output_est=True to suffix _est

            visualize_frame_results(color=color, gt_mesh=None, est=est, K=K,

                                    gt_pose=None, pred_pose=pred_pose, metric=None,

                                    obj_f=obj_name, frame_idx=i, save_path=save_path,

                                    glctx=glctx, name="custom_seq", est_only=True, output_est=True)

        except Exception as e:

            print(f"Visualization failed on frame {i}: {e}")


if __name__ == "__main__":
    main()

