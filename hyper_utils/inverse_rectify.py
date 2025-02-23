import numpy as np
import matplotlib.pyplot as plt
import cv2, os, glob
import scipy.io as io
import depth_utils
from tqdm import tqdm

def InverseRectify(args):
    R = np.load(os.path.join(args.rectified_param_dir, "cam2_R.npy"))
    P = np.load(os.path.join(args.rectified_param_dir, "cam2_P.npy"))
    T = np.array([[0.0], [0.0], [0.0]]) # Rectification does not include translation transformation

    # Q is one of the outputs of cv2.stereoRectify, which is used to converts disparity map into depth
    Q = np.load(os.path.join(args.rectified_param_dir, "Q.npy")).astype(np.float32)

    # inverse matrix. since we want inverse rectification
    R = R.T
    T = -np.matmul(R,T).T # -R.T@T

    # We need intrinsics and distortion coefficients of the original camera to project point clouds to the camera plane
    cameraMatrix_original = io.loadmat(os.path.join(args.stereo_param_dir, "intrinsic_camera1.mat"))['K'].astype(np.float32)
    distCoeffs = io.loadmat(os.path.join(args.stereo_param_dir, "distortion_camera1.mat"))['distortion'].astype(np.float32)

    demo_outputs = sorted(glob.glob(args.demo_output_result_dir%args.date, recursive=True))

    for imfile in tqdm(list(demo_outputs)):

        disparity_cam1 = abs(np.load(imfile)).astype(np.float32)

        depth_map_cam1 = cv2.reprojectImageTo3D(disparity_cam1, Q)
        point_cloud_cam1, point_cloud_cam1_vector = depth_utils.depth_map_to_point_cloud(depth_map_cam1)
        point_cloud_cam2 = depth_utils.transform_point_cloud_to_other_camera(point_cloud_cam1, R, T) # rectification domain에서의 different camera
        point_cloud_cam2_vector = np.asarray(point_cloud_cam2.points)
        point_cloud_cam2_xy, _ = cv2.projectPoints(point_cloud_cam1_vector, 
                                                    R, T, 
                                                    cameraMatrix_original,
                                                    distCoeffs) # shape = (n_points, 1, 2)
        depth_cam2 = depth_utils.depth_from_points(np.stack([point_cloud_cam2_xy[:, 0, 0], 
                                                    point_cloud_cam2_xy[:, 0, 1]], axis=-1),
                                                    point_cloud_cam2_vector,
                                                    disparity_cam1.shape[1], 
                                                    disparity_cam1.shape[0])

        # Remove hole from transformation and reprojection
        depth_cam2_hole_removed = depth_utils.filter_rectification_noise(depth_cam2)
        depth_cam2_hole_removed = depth_cam2_hole_removed[args.y_start:args.y_start + args.crop_cam_H, args.x_start:args.x_start + args.crop_cam_W]
        
        date = imfile.split('/')[-3]
        n_dynamic = imfile.split('/')[-2]

        # Save
        np.save(os.path.join(args.depth_output_dir%(date, n_dynamic), 'depth_%s.npy'%imfile.split('/')[-1][-8:-4]), depth_cam2_hole_removed)
        plt.imsave(os.path.join(args.depth_output_dir%(date, n_dynamic), 'depth_%s.png'%imfile.split('/')[-1][-8:-4]), depth_cam2_hole_removed[:, :, 2], vmin = 400, vmax = 1200, cmap ='nipy_spectral')