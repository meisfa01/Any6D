import os
import copy
import numpy as np
import trimesh
import cv2
from tqdm import tqdm
import nvdiffrast.torch as dr
from pytorch_lightning import seed_everything
from datetime import datetime

from foundationpose.datareader import Ho3dReader
from estimater import Any6D, ScorePredictor, PoseRefinePredictor
from foundationpose.Utils import visualize_estimation

# no ground truth mesh for these objects available, so skip GT and metrics

if __name__ == '__main__':

    seed_everything(0)
    running_stride = 10

    name = "test_own_dataset_fabian"
    data_root = "/home/stois/repos/Any6D/datasets/own_dataset"
    anchor_path = "/home/stois/repos/Any6D/results/own_anchors"

    date_str = f'{datetime.now():%Y-%m-%d_%H-%M-%S}'
    save_root = f"./results/ho3d_results/{name}/{date_str}"
    save_results_est_path = f'{save_root}'

    os.makedirs(save_results_est_path, exist_ok=True)

    obj_folder =[
        '101_white_cup',
        '102_green_bottle',
        '103_casio_calculator'
        ]

    glctx = dr.RasterizeCudaContext()
    mesh_tmp = copy.deepcopy(trimesh.primitives.Box(extents=np.ones((3)), transform=np.eye(4)))
    mesh = trimesh.Trimesh(vertices=mesh_tmp.vertices.copy(), faces= mesh_tmp.faces.copy())
    est = Any6D(mesh=mesh, scorer=ScorePredictor(), refiner=PoseRefinePredictor(), debug_dir=save_results_est_path, debug=0, glctx=glctx)

    for obj_f in tqdm(obj_folder, desc='Evaluating Object'):

        video_dir = os.path.join(f"{data_root}", obj_f)
        print(video_dir)
        reader = Ho3dReader(video_dir, data_root)
        reader.color_files = reader.color_files[::running_stride]

        obj_save_path = os.path.join(save_results_est_path, obj_f)
        os.makedirs(obj_save_path, exist_ok=True)
        est.debug_dir = obj_save_path

        mesh = trimesh.load(reader.get_reference_view_1_mesh(anchor_path))
        est.reset_object(mesh=mesh, symmetry_tfs=None)

        for i in tqdm(range(0, len(reader.color_files), 1), desc=f"{obj_f} - Frames"):
            color_file = reader.color_files[i]
            color = cv2.cvtColor(cv2.imread(color_file), cv2.COLOR_BGR2RGB)
            depth = reader.get_depth(i)
            mask = reader.get_mask(i).astype(np.bool_)

            pred_pose_q = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=5, name=obj_f)

            try:
                visualize_estimation(color=color, K=reader.K, init_pose=None,
                                     pred_pose=pred_pose_q,
                                     frame_idx=i, save_path=obj_save_path, glctx=glctx,
                                     obj_name=f"{len(reader.color_files)}_{name}",
                                     est_mesh=est.mesh)
            except Exception as e:
                print(f"Error visualizing frame {i} for object {obj_f}: {e}")

    print("Done")




