import open3d as o3d 
import numpy as np
import matplotlib.pyplot as plt
import cv2, os

def depth_map_to_point_cloud(depth_map, intrinsic=None):
    """
    Convert depth map to point cloud. If the depth map already has 
    unprojected (x,y,z) positions for each pixel, (in this case, keep param intrinsic None
    when call the function) the process becomes much simpler.
    :param depth_map: 
     If the depth map only have z value(depth) without (x, y) position of the unprojected point,
     The depth map must be an undistorted one (Use cv2.undistort or other method).
    :param intrinsic: Intrinsic parameters (focal length and principal point).
    :return: (Open3D point cloud object, point vectors)
    """
    if intrinsic is None: 
        mask1 = (depth_map[:,:,2] >= -np.inf) # Masking can be applied by replacing -np.inf.
        mask2 = (depth_map[:,:,2] <= np.inf)  # Masking can be applied by replacing np.inf.
        mask = mask1*mask2
        points = depth_map[mask==True]
    else:
        fx, fy, cx, cy = intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2]   # Focal lengths and principal point
        rows, cols = depth_map.shape
        xx, yy = np.meshgrid(np.arange(cols), np.arange(rows))
        zz = depth_map
        x = (xx - cx) * zz / fx
        y = (yy - cy) * zz / fy
        points = np.vstack((x.flatten(), y.flatten(), zz.flatten())).T
        
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    return point_cloud, points


def transform_point_cloud_to_other_camera(point_cloud, R, T):
    """
    Transform a coordinate of point cloud to the other camera's coordinate system.
    :param point_cloud: Point cloud from the original camera coordinate.
    :param extrinsics: Extrinsic parameters (rotation and translation) from original camera to other camera.
    :return: Transformed point cloud.
    """
    point_cloud.transform(np.vstack([np.hstack((R, T.reshape(-1, 1))), [0, 0, 0, 1]]))
    return point_cloud


def project_point_cloud_to_image_plane(point_cloud, intrinsic, H=None, W=None, distCoeffs=[0, 0, 0, 0]):
    """
    Refer to https://www.opencv.org.cn/opencvdoc/2.3.2/html/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
    Project the point cloud onto the camera's image plane to get the depth map.
    Currently, this function can only deal with 4-th order radial distortion (ignore tangential and higher order radial distortion)
    Currently, this function has minor difference with the opencv's projection function when considering distortion
    No difference when do not using distortion (i.e. distCoeffs = [0, 0, 0, 0])
    :param point_cloud: point cloud in the camera's coordinate system.
    :param intrinsics_left: Intrinsic parameters of the camera.
    :return: Depth map.
    """
    fx, fy, cx, cy = intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2]
    k1, k2, k3, k4 = distCoeffs
    points = np.asarray(point_cloud.points)
    z = points[:, 2]
    x = points[:, 0] / (z+1e-20)
    y = points[:, 1] / (z+1e-20)
    r2 = x**2 + y**2
    r4 = r2 ** 2
    r6 = r2 ** 3
    xx = x*((1 + k1*r2 + k2*r4 + k3*r6)/(1 + k4*r2))
    yy = y*((1 + k1*r2 + k2*r4 + k3*r6)/(1 + k4*r2))
    u = fx * xx + cx # final project points' x position
    v = fy * yy + cy # final project points' y position

    if W is None: 
        W = int(np.max(u)) + 1 
    if H is None: 
        H = int(np.max(v)) + 1
    depth_map = np.zeros((H, W, 3))
    mask_horizontal = (0 <= u) * (u <= W-1)
    mask_vertical = (0 <= v) * (v <= H-1)
    mask = mask_horizontal * mask_vertical
    u_clipped = u[mask]
    v_clipped = v[mask]
    z_clipped = z[mask]
    depth_map[np.round(v_clipped).astype(int), np.round(u_clipped).astype(int)] = np.stack([x[mask], y[mask], z_clipped], axis=-1)
    # depth_map[v_clipped.astype(int), u_clipped.astype(int)] = z_clipped
    return depth_map


def depth_from_points(points_projected_uv, points_vector_xyz, W, H):
    """ 
    :param points_projected_xyz: shape = (n_points, 3).
    :param points_vector_xyz: point cloud vector before projection
    Construct depth map from 3d points resulting from cv2.projectPoints
    """
    depth_map = np.zeros((H, W, 3))
    u = points_projected_uv[:, 0]
    v = points_projected_uv[:, 1]
    mask_horizontal = (0 <= u) * (u <= W-1)
    mask_vertical = (0 <= v) * (v <= H-1)
    mask = mask_horizontal * mask_vertical
    x_clipped = u[mask]
    y_clipped = v[mask]
    xyz_clipped = points_vector_xyz[mask]
    depth_map[np.round(y_clipped).astype(int), np.round(x_clipped).astype(int)] = xyz_clipped
    return depth_map


def filter_rectification_noise(img):
    """ 
    Fill holes(pixel with zero depth) in the image with mean of adjacent pixel values
    param img: numpy image. (H, W, C)"""
    H, W = img.shape[0], img.shape[1]
    temp_img = np.array(img)
    zero_indices = np.where(img[:, :, 2] == 0)
    for (i, j) in zip(zero_indices[0], zero_indices[1]):
        adj_indicies = [(i, j-1), (i, j+1), (i-1, j), (i+1, j-1)]
        mean = 0
        count = 0
        for adj_i, adj_j in adj_indicies:
            if (adj_i >= H) or (adj_j >= W):
                continue 
            adj_pixel_value = img[adj_i, adj_j]
            adj_depth_value = adj_pixel_value[2]
            if not (adj_depth_value == 0):
                mean += adj_pixel_value 
                count += 1
        if count == 0:
            continue 
        mean /= count 
        temp_img[i, j] = mean
    return temp_img