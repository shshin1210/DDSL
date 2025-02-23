import os
import glob
import numpy as np
import cv2, time
import scipy.io as io
import matplotlib.pyplot as plt
from tqdm import tqdm

def Rectify(args):
    
    # Intrinsic parameters
    cameraMatrix1 = io.loadmat(os.path.join(args.stereo_param_dir, 'intrinsic_camera1.mat'))['K']
    cameraMatrix2 = io.loadmat(os.path.join(args.stereo_param_dir, 'intrinsic_camera2.mat'))['K']
    distCoeffs1 = io.loadmat(os.path.join(args.stereo_param_dir, 'distortion_camera1.mat'))['distortion']
    distCoeffs2 = io.loadmat(os.path.join(args.stereo_param_dir, 'distortion_camera2.mat'))['distortion']

    # We use intrinsics and extrinsic acquired from Matlab's stereo rectification result
    # Extrinsic parameters
    R = io.loadmat(os.path.join(args.stereo_param_dir, 'rotation.mat'))['R']
    T = io.loadmat(os.path.join(args.stereo_param_dir, 'translation.mat'))['T']

    cameraMatrix1 = cameraMatrix1.astype(np.float64)
    cameraMatrix2 = cameraMatrix2.astype(np.float64)
    distCoeffs1 = distCoeffs1.astype(np.float64)
    distCoeffs2 = distCoeffs2.astype(np.float64)
    R = R.astype(np.float64)
    T = T.astype(np.float64)

    # Rectify camera and get new camera parameters
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
    cameraMatrix1, distCoeffs1,
    cameraMatrix2, distCoeffs2,
    (args.cam_H, args.cam_W), R, T.T)

    # Save the camera parameters
    np.save(os.path.join(args.rectified_param_dir, 'cam1_R.npy'), R1)
    np.save(os.path.join(args.rectified_param_dir, 'cam2_R.npy'), R2)
    np.save(os.path.join(args.rectified_param_dir, 'cam1_P.npy'), P1)
    np.save(os.path.join(args.rectified_param_dir, 'cam2_P.npy'), P2)
    np.save(os.path.join(args.rectified_param_dir, 'Q.npy'), Q)
    
    map1x, map1y = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, (args.cam_W, args.cam_H), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, (args.cam_W, args.cam_H), cv2.CV_32FC1)
    
    left_images = sorted(glob.glob(args.stereo_image_camera1_dir%args.date, recursive=True))
    left_images = [image for image in left_images if 'black.png' not in image]
    # left_images = [image for image in left_images]

    right_images = sorted(glob.glob(args.stereo_image_camera2_dir%args.date, recursive=True))
    right_images = [image for image in right_images if 'black.png' not in image]
    # right_images = [image for image in right_images]
    cnt = 0
    print('Date:', args.date)
        
    for (imfile1, imfile2) in tqdm(list(zip(left_images, right_images))):
        if not os.path.exists(os.path.join(args.new_rectdata_dir, '%s'%args.date)):
            os.makedirs(os.path.join(args.new_rectdata_dir, '%s'%args.date))
            print('made', os.path.join(args.new_rectdata_dir, '%s'%args.date))
            
            os.makedirs(os.path.join(args.new_rectdata_dir, '%s'%args.date, 'camera1'))
            os.makedirs(os.path.join(args.new_rectdata_dir, '%s'%args.date, 'camera2'))
            print('made', os.path.join(args.new_rectdata_dir, '%s'%args.date, 'camera1'))

        if not os.path.exists(os.path.join(args.new_rectdata_dir, '%s'%args.date, 'camera1', imfile2.split('/')[-2])):
            os.makedirs(os.path.join(args.new_rectdata_dir, '%s'%args.date, 'camera1', imfile1.split('/')[-2]))
            os.makedirs(os.path.join(args.new_rectdata_dir, '%s'%args.date, 'camera2', imfile2.split('/')[-2])) 
            print('made', os.path.join(args.new_rectdata_dir, '%s'%args.date, 'camera1', imfile2.split('/')[-2]))

        cam1_img = cv2.imread(imfile1) # 왼쪽
        cam2_img = cv2.imread(imfile2) # 오른쪽

        cam1_img_tmp = np.zeros(shape=(args.cam_H, args.cam_W, 3))
        cam2_img_tmp = np.zeros(shape=(args.cam_H, args.cam_W, 3))
        
        cam1_img_tmp[args.y_start:args.y_start + args.crop_cam_H, args.x_start:args.x_start + args.crop_cam_W] = cam1_img
        cam2_img_tmp[args.y_start:args.y_start + args.crop_cam_H, args.x_start:args.x_start + args.crop_cam_W] = cam2_img

        # Remap the images using rectified camera parameters
        rectified_img1 = cv2.remap(cam1_img_tmp, map1x, map1y, cv2.INTER_LINEAR)
        rectified_img2 = cv2.remap(cam2_img_tmp, map2x, map2y, cv2.INTER_LINEAR)

        cv2.imwrite(args.new_rectdata_dir + '/%s/camera1/%s/%s.png'%(args.date,imfile1.split('/')[-2], imfile1.split('/')[-1][:-4]), rectified_img1)
        cv2.imwrite(args.new_rectdata_dir + '/%s/camera2/%s/%s.png'%(args.date,imfile1.split('/')[-2], imfile1.split('/')[-1][:-4]), rectified_img2)

        cnt += 1
        
    print('Waiting for saving images.....')
    time.sleep(5)