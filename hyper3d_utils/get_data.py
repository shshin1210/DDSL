import numpy as np
import matplotlib.pyplot as plt
import cv2, os, sys

import torch
import imageio
import time

class GetData():
    def __init__(self, args):        
        # camera
        self.cam_H, self.cam_W = args.crop_cam_H, args.crop_cam_W

        # directory
        self.real_data_dir = args.real_data_dir
        self.npy_dir = args.npy_dir
        self.illum_directory_dir = args.illum_dir
        self.radiometric_data_dir = args.radiometric_data_dir
        
        # depth
        self.depth_start, self.depth_end = args.depth_start, args.depth_end
        self.depth_arange = np.arange(self.depth_start, self.depth_end +1, 1)

        # datas
        self.args = args
        self.wvls = np.arange(args.min_wvl, args.max_wvl+1, args.wvl_interval)
        self.n_illum = args.n_illum
        self.n_groups = len(os.listdir(args.dynamic_dir%args.date)) # Number of groups in the dynamic scene

        # dsl back ward mapping function
        self.first_proj_col_all = np.load((os.path.join(self.npy_dir%self.args.cal_date, 'dispersive_aware_model.npy'))).reshape(len(self.depth_arange), len(self.wvls), self.cam_H*self.cam_W)
        print('full back ward mapping model loaded...')
        
    # Bring radiometric parameters
    def get_radiometric_data(self):
        """
            We bring camera response function, projector emission function, diffraction grating efficiency here.
            ** Make sure each radiometric paramters are ranging from 440nm to 660nm at 10nm interval
            
            Arguments
            -----------
            None

            Returns
            -----------
            - PEF : Projector radiometric paramters (Projector emission function)
            - CRF : Camera radiometric paramters (camera response function)
            - DG_efficiency_image_first : Diffraction grating efficiency 
        """
        
        PEF = np.load(os.path.join(self.illum_directory_dir, '../DDSL_PEF.npy'))
        CRF = np.load(os.path.join(self.illum_directory_dir, '../DDSL_CRF.npy'))
        DG_efficiency = np.load(os.path.join(self.illum_directory_dir, '../DDSL_DG.npy'))
        
        # DG efficiency for all pixels
        DG_efficiency_image_first = np.zeros(shape=(self.cam_H * self.cam_W, len(self.wvls)))
        DG_efficiency_image_first[:,:] =  DG_efficiency

        return PEF, CRF, DG_efficiency_image_first

    def get_real_captured_images(self, date):
        """
            We get all frame of real captured captured images under M DDSL patterns & black pattern here.

            Arguments
            -----------
            - date : the date of real captured dataset

            Returns
            -----------
            - black_imgs : captured images under black patterns
            - ddsl_imgs : captured images under M DDSL patterns
        """
        
        ddsl_imgs = np.zeros(shape=(self.n_groups, self.n_illum, self.cam_H, self.cam_W, 3))
        black_imgs = np.zeros(shape=(self.n_groups, self.cam_H, self.cam_W, 3))
        
        # black images
        for n_group in range(self.n_groups):
            black_imgs[n_group] = cv2.imread(os.path.join(self.real_data_dir%(date, n_group), 'black.png'), -1)[:,:,::-1]/65535.
            # DDSL images
            for i in range(self.n_illum):
                ddsl_imgs[n_group, i] = cv2.imread(os.path.join(self.real_data_dir%(date, n_group), 'capture_%04d.png'%i), -1)[:,:,::-1]/65535.
            
        print("Total %d frames, ddsl imgs and black imgs created ..."%self.n_groups)
        
        return ddsl_imgs, black_imgs     
    
    def get_depth(self, date, n_group):
        """
            We get raft stereo depth results here.

            Arguments
            -----------
            - date : the date of real captured dataset
            - frame : the frame number of dynamic scene
        """
        print('bringing depth...')
        depths = np.zeros(shape=(self.n_illum, self.cam_H, self.cam_W))
        
        for i in range(self.n_illum):
            depth = np.load(os.path.join(self.real_data_dir%(date, n_group),'depth_%04d.npy'%i))[:,:,2]
            depth = np.round(depth).reshape(self.cam_H, self.cam_W).astype(np.int16)
            depths[i] = depth
        print('depth loaded...')
        
        return depths
    
    def get_dsl_scene_dependent_mapping(self, depths, dynamic):
        """
            We get the hyperspectral illumination for specific scene

            Arguments
            -----------
            - depths : the raft stereo result depth of each illuminated scene
            - dynamic : whether it is dynamic or  not
        """

        if dynamic == False:
            print('Non-dynamic scene DSL dispersive-aware mapping...')
            # Choose valid idx
            first_real_proj_col_all = np.zeros(shape=(len(self.wvls), self.cam_H*self.cam_W))
            depths = depths[self.args.target_idx].reshape(-1)
            
            for i in range(self.cam_H*self.cam_W):
                if (depths[i] < self.depth_start) or (depths[i] > self.depth_end):
                        depths[i] = self.depth_start
                depth_idx = np.where(self.depth_arange == depths[i])[0][0]
                first_real_proj_col_all[:,i]= self.first_proj_col_all[depth_idx,:,i]
        
            first_real_img_illum_idx = first_real_proj_col_all.reshape(len(self.wvls), self.cam_H, self.cam_W)
        
        else:
            print('Dynamic scene DSL dispersive-aware mapping...')
            # Choose valid idx
            first_real_proj_col_all = np.zeros(shape=(self.n_illum, len(self.wvls), self.cam_H*self.cam_W))
            depths = depths.reshape(self.n_illum, -1)
            
            for d in range(len(depths)): # each captured images
                for i in range(self.cam_H*self.cam_W):
                    if (depths[d][i] < self.depth_start) or (depths[d][i] > self.depth_end):
                            depths[d][i] = self.depth_start
                    depth_idx = np.where(self.depth_arange == depths[d][i])[0][0]
                    first_real_proj_col_all[d,:,i]= self.first_proj_col_all[depth_idx,:,i]
            
            first_real_img_illum_idx = first_real_proj_col_all.reshape(self.n_illum, len(self.wvls), self.cam_H, self.cam_W)
            
        return first_real_img_illum_idx
    
    def get_ddsl_data(self, ddsl_imgs, black_imgs, map_xy_list):
        """
            We get scene dependent dispersive-aware mapping

            Arguments
            -----------
            - ddsl_imgs : ddsl captured dynamic images, shape (self.n_illum, self.cam_H, self.cam_W, 3)
            - black_imgs : black captured dynamic images, shape (self.cam_H, self.cam_W, 3)
            - map_xy_list : mapping each ddsl & black pattern, shape (self.n_illum, self.cam_H, self.cam_W, 2)
        """
        
        n_group_ddsl_data = np.zeros(shape=(self.n_illum, self.cam_H, self.cam_W, 3))
        n_group_black = black_imgs[map_xy_list[-1,...,1], map_xy_list[-1,...,0]] # black mapped to frame target
        
        for i in range(self.n_illum):
            if i == self.args.target_idx:
                n_group_ddsl_data[i] = ddsl_imgs[i] - n_group_black
            else:
                if i < self.args.target_idx:
                    changed_idx = i
                elif i > self.args.target_idx:
                    changed_idx = i-1
                n_group_ddsl_data[i] = ddsl_imgs[i, map_xy_list[changed_idx,...,1], map_xy_list[changed_idx,...,0]] - n_group_black
        
        print("per frame ddsl data created ...")
        return n_group_ddsl_data