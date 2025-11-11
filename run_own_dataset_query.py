import copy

from foundationpose.datareader import Ho3dReader
from estimater import *
from bop_toolkit_lib.pose_error_custom import mssd, mspd, vsd

from metrics import *
import json
from bop_toolkit_lib.renderer_vispy import RendererVispy
from pytorch_lightning import seed_everything
from datetime import datetime

# no ground truth mesh for these objects available, so skip these parts

if __name__ == '__main__':

    seed_everything(0)
    running_stride = 10

    name = "test_own_dataset_fabian"
    data_root = "/home/stois/repos/Any6D/datasets/own_dataset"
    ycbv_modesl_info_path = "/home/stois/repos/Any6D/datasets/own_dataset/models_info.json"
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

    excel_files = []
    glctx = dr.RasterizeCudaContext()
    mesh_tmp = copy.deepcopy(trimesh.primitives.Box(extents=np.ones((3)), transform=np.eye(4)))
    mesh = trimesh.Trimesh(vertices=mesh_tmp.vertices.copy(), faces= mesh_tmp.faces.copy())
    est = Any6D(mesh=mesh, scorer=ScorePredictor(), refiner=PoseRefinePredictor(), debug_dir=save_results_est_path, debug=0, glctx=glctx)

    renderer = RendererVispy(640, 480, mode='depth')
    obj_count = 0

    data = []

    for obj_f in tqdm(obj_folder, desc='Evaluating Object'):

        video_dir = os.path.join(f"{data_root}", obj_f)
        print(video_dir)
        reader = Ho3dReader(video_dir, data_root)
        reader.color_files = reader.color_files[::running_stride]

        ob_id = reader.get_obj_id()

        K_anchor = np.loadtxt(reader.get_reference_K(anchor_path))

        gt_diameter = reader.get_gt_mesh_diamter()
        mesh = trimesh.load(reader.get_reference_view_1_mesh(anchor_path))

        pred_pose_a = np.loadtxt(reader.get_reference_view_1_pose(anchor_path))
        gt_pose_a = np.loadtxt(reader.get_reference_view_1_pose(anchor_path).replace('initial','gt'))

        est.reset_object(mesh=mesh, symmetry_tfs=None)

        for i in tqdm(range(0, len(reader.color_files), 1), desc=f"{obj_f} - Frames"):
            gt_pose_q = reader.get_gt_pose(i)

            if gt_pose_q is None:
                continue

            color_file = reader.color_files[i]
            color = cv2.cvtColor(cv2.imread(color_file), cv2.COLOR_BGR2RGB)
            H, W = color.shape[:2]
            depth = reader.get_depth(i)
            mask = reader.get_mask(i).astype(np.bool_)
            pred_pose_q = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=5, name=obj_f)

            pose_aq = pred_pose_q @ np.linalg.inv(pred_pose_a)  # obtained pose A->Q
            pred_q = pose_aq @ gt_pose_a


            err_R, err_T = compute_RT_distances(pred_q, gt_pose_q)

            pose_recall_th = [(5, 5), (5, 10), (10, 10)]

            for r_th, t_th in pose_recall_th:
                succ_r, succ_t = err_R <= r_th, err_T <= t_th
                succ_pose = np.logical_and(succ_r, succ_t).astype(float)


            pred_q, gt_pose_q = pred_q.astype(np.float16), gt_pose_q.astype(np.float16)

            pred_r, pred_t = pred_q[:3, :3], np.expand_dims(pred_q[:3, 3], axis=1) * 1e3
            gt_r, gt_t = gt_pose_q[:3, :3], np.expand_dims(gt_pose_q[:3, 3], axis=1) * 1e3

            mssd_rec = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
            mspd_rec = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50])

            vsd_delta = 15.0
            vsd_taus = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
            vsd_rec = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])

            vsd_errs = vsd(pred_r, pred_t, gt_r, gt_t, (depth *1e3), reader.K.reshape(3, 3), vsd_delta, vsd_taus, True, (gt_diameter*1e3), renderer, ob_id)
            vsd_errs = np.asarray(vsd_errs)
            all_vsd_recs = np.stack([vsd_errs < rec_i for rec_i in vsd_rec], axis=1)
            mean_vsd = all_vsd_recs.mean()

            mssd_cur_rec = mssd_rec * (gt_diameter * 1e3)


            try:
                visualize_estimation(color=color, K=reader.K, init_pose=gt_pose_q,
                                           pred_pose=pred_pose_q,
                                           frame_idx=i, save_path=save_results_est_path, glctx=glctx,
                                           obj_name=f"{len(reader.color_files)}_{name}",
                                           est_mesh=est.mesh)
            except:
                pass
            obj_count+=1

    print("Done")




