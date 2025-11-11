import os
import trimesh
import numpy as np
import cv2

import nvdiffrast.torch as dr
import argparse
import pandas as pd

from estimater import Any6D
from foundationpose.Utils import visualize_frame_results, calculate_chamfer_distance_gt_mesh, align_mesh_to_coordinate
from tqdm import tqdm
from sam2_instantmesh import *


# no ground truth mesh for these objects available, so skip these parts

if __name__=='__main__':

    anchor_folder = 'results/own_anchors'
    img_to_3d = True

    results = []

    obj_list = [f for f in os.listdir(anchor_folder) if not f.endswith('.xlsx')]

    print(obj_list)

    glctx = dr.RasterizeCudaContext()

    for obj in tqdm(obj_list, desc='Object'):

        print('This is the current obj:')
        print(obj)

        save_path = f'{anchor_folder}/{obj}'
        mesh_path = os.path.join(f'{anchor_folder}/{obj}/mesh_{obj}.obj')

        color = cv2.cvtColor(cv2.imread(os.path.join(save_path, 'color.png')), cv2.COLOR_BGR2RGB)
        depth = cv2.imread(os.path.join(save_path, 'depth.png'), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 1000.0
        mask = cv2.cvtColor(cv2.imread(os.path.join(save_path, 'mask.png')),cv2.COLOR_BGR2RGB)[...,0].astype(np.bool_)

        if img_to_3d:
            cmin, rmin, cmax, rmax = get_bounding_box(mask).astype(np.int32)
            input_box = np.array([cmin, rmin, cmax, rmax])[None, :]
            mask_refine = running_sam_box(color, input_box)

            input_image = preprocess_image(color, mask_refine, save_path, obj)
            images = diffusion_image_generation(save_path, save_path, obj, input_image=input_image)
            instant_mesh_process(images, save_path, obj)

            mesh = trimesh.load(os.path.join(save_path, f'mesh_{obj}.obj'))
        else:
            mesh = trimesh.load(mesh_path)


        mesh = align_mesh_to_coordinate(mesh)
        mesh.export(os.path.join(save_path, f'center_mesh_{obj}.obj'))

        est = Any6D(symmetry_tfs=None, mesh=mesh, debug_dir=save_path, debug=0)

        intrinsic = np.loadtxt(f'{anchor_folder}/{obj}/K.txt')

        # predicted pose
        pred_pose = est.register_any6d(K=intrinsic, rgb=color, depth=depth, ob_mask=mask, iteration=5, name=f'demo')

        # ground truth pose
        gt_pose = np.loadtxt(os.path.join(save_path, f'{obj}_gt_pose.txt'))

        np.savetxt(os.path.join(save_path, f'{obj}_initial_pose.txt'), pred_pose)
        est.mesh.export(os.path.join(save_path, f'final_mesh_{obj}.obj'))

        glctx = dr.RasterizeCudaContext()

    print("Done")



