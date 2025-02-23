import numpy as np
import matplotlib.pyplot as plt
import cv2, os, sys

import torch
import imageio
import time
import torchvision.transforms as tf
from scipy import interpolate

sys.path.append('DDSL')
sys.path.append('../RAFT')
sys.path.append('../RAFT/raft_core')

from hyper3d_utils import get_data
from hyper3d_utils import black_opt_flow

class HypReconDynamic():
    def __init__(self, args):
        
        self.args = args
        
        # device
        self.device = args.cuda_device

        # class
        self.get_data = get_data.GetData(args)
        self.black_opt_flow = black_opt_flow.BlackOpticalFlow(args)
        
        # camera
        self.cam_H, self.cam_W = args.crop_cam_H, args.crop_cam_W

        # depth
        self.depth_start, self.depth_end = args.depth_start, args.depth_end
        self.depth_arange = np.arange(self.depth_start, self.depth_end +1, 1)

        # directory
        self.real_data_dir = args.real_data_dir
        self.npy_dir = args.npy_dir
        self.illum_directory_dir = args.illum_dir
        self.radiometric_data_dir = args.radiometric_data_dir
        
        # datas
        self.x_centers_ddsl = np.load(os.path.join(self.illum_directory_dir, 'x_centers_ddsl_8patt.npy'))
        self.x_centers_dsl = np.load(os.path.join(self.illum_directory_dir, 'x_centers_dsl_%s.npy'%args.cal_date))
        self.wvls = np.arange(args.min_wvl, args.max_wvl+1, args.wvl_interval)
        self.n_illum = args.n_illum
        self.n_groups = len(os.listdir(args.dynamic_dir%args.date)) # Number of groups in the dynamic scene
        self.ddsl_patts = np.array([cv2.imread(os.path.join(self.illum_directory_dir, 'DDSL_pattern', 'illum_%03d.png'%i),-1)/255. for i in range(self.n_illum)])

        # get radiometric data
        self.PEF, self.CRF, self.DG_efficiency_image_first = self.get_data.get_radiometric_data()
        
        # gaussian blur
        self.gaussian_blur = tf.GaussianBlur(kernel_size=(7,7), sigma=(3,3))
    
    def get_ddsl_scene_dependent_mapping(self, dynamic, first_real_img_illum_idx, map_xy_list):
        """
            We get scene dependent dispersive-aware mapping

            Arguments
            -----------
            - dynamic : whether it is dynamic or  not
            - scene dependent dsl dispersed-aware mapping
        """
    
        print('Dynamic scene DDSL dispersive-aware mapping...')
        
        start= time.time()

        # gaussiain blur to ddsl patts
        for i in range(self.n_illum):
            self.ddsl_patts[i] = self.gaussian_blur(torch.tensor(self.ddsl_patts[i]).permute(2,0,1)).permute(1,2,0)

        # create hyperspectral illumination
        hyp_illum = np.zeros(shape=(self.n_illum, len(self.wvls), self.cam_H*self.cam_W, self.n_illum))
        first_real_img_illum_idx_int = np.round(first_real_img_illum_idx).astype(np.int32).reshape(self.n_illum, len(self.wvls), -1)

        indices_expanded = first_real_img_illum_idx_int[:, :, :, np.newaxis]
        k_indices = np.arange(self.n_illum)[np.newaxis, np.newaxis, np.newaxis, :]

        hyp_illum = self.ddsl_patts[...,0][k_indices, 100, indices_expanded]
        
        masking_ftn = hyp_illum.reshape(self.n_illum, len(self.wvls), self.cam_H, self.cam_W, self.n_illum)

        print('weighting time (s):', time.time() - start)
        
        final_masking_ftn = np.zeros(shape=(len(self.wvls), self.cam_H, self.cam_W, self.n_illum))
        for i in range(self.n_illum):
            if i == args.target_idx:
                final_masking_ftn[...,i] = masking_ftn[i,...,i]
            else:
                if i < args.target_idx:
                    changed_idx = i
                elif i > args.target_idx:
                    changed_idx = i -1
                final_masking_ftn[...,i] = masking_ftn[i,:,map_xy_list[changed_idx,:,:,1],map_xy_list[changed_idx,:,:,0],i].transpose(2,0,1)
        
        final_masking_ftn = final_masking_ftn.reshape((len(self.wvls), self.cam_H*self.cam_W, self.n_illum))
        
        return final_masking_ftn
        
    def hyp_optimization(self, ddsl_data, depths, gauss_masking_ftn, n_group):
        """
            We optimize to reconstruct hyperspectral image for specific scene

            Arguments
            -----------
            - ddsl_data : captured scene under ddsl patterns
            - depths : depths of captured scenes
            - gauss_masking_ftn : scene dependent dispersive-aware mapping function
        """
        
        # loss
        losses = []
        
        # learning rate & decay step
        epoch = self.args.epoch 
        lr = self.args.lr 
        decay_step = self.args.decay_step
        gamma = self.args.gamma

        # optimized paramter (CRF & PEF)
        initial_value = torch.ones(size =(self.cam_H*self.cam_W, len(self.wvls)))/2
        initial_value = torch.logit(initial_value)
        _opt_param =  torch.tensor(initial_value, dtype= torch.float, requires_grad=True, device= self.device)

        # optimizer and schedular
        optimizer = torch.optim.Adam([_opt_param], lr = lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=decay_step, gamma = gamma)

        # shape : 3, 47
        PEF_dev = torch.tensor(self.PEF, dtype = torch.float).to(self.device).T
        CRF_dev = torch.tensor(self.CRF, dtype = torch.float).to(self.device)
        DG_efficiency_first_dev = torch.tensor(self.DG_efficiency_image_first.reshape(self.cam_H*self.cam_W,-1), device= self.device) # H x W, wvls
        Masking_ftn = torch.tensor(gauss_masking_ftn.reshape(len(self.wvls), self.cam_H*self.cam_W, -1).transpose(1, 2, 0), device = self.device) # HxW, 13(patt num), wvls

        # depth scalar
        A = self.args.depth_scalar_value *1e+5
        depth_scalar = ((depths.astype(np.int32))**2) / A
        depth_scalar_dev = torch.tensor(depth_scalar, dtype = torch.float).to(self.device).reshape(self.n_illum, -1).T

        # white pattern into multi-spectral channels
        white_patt = torch.ones(size = (self.cam_H , self.cam_W, 3), device=self.device).reshape(-1, 3)
        white_patt_hyp = white_patt @ PEF_dev
        white_patt_hyp = white_patt_hyp.squeeze().unsqueeze(dim = 1).unsqueeze(dim = -1)

        # Real captured RGB image
        GT_I_RGB_FIRST_tensor = torch.tensor(ddsl_data.reshape(self.n_illum, self.cam_H*self.cam_W, 3).transpose(1,0,2), device=self.device) # HxW, 13(patt num), 3

        # weight
        weight_spectral = args.weight_spectral # 0.5
        weight_first = 1

        loss_vis = []
        A_first = CRF_dev.unsqueeze(dim = 0).unsqueeze(dim = 0) * white_patt_hyp * DG_efficiency_first_dev.unsqueeze(dim = -1).unsqueeze(dim = 1) * Masking_ftn.unsqueeze(dim = -1)

        for i in range(epoch):
            # initial loss
            loss = 0

            opt_param = torch.sigmoid(_opt_param)

            Simulated_I_RGB_first = opt_param.unsqueeze(dim = 1).unsqueeze(dim = -1) * A_first / (depth_scalar_dev.unsqueeze(dim = -1).unsqueeze(dim = -1) + 1e-7)
            Simulated_I_RGB_first = Simulated_I_RGB_first.sum(axis = 2)
            
            # first order
            image_loss_first = torch.abs(Simulated_I_RGB_first - GT_I_RGB_FIRST_tensor) / (self.cam_H*self.cam_W)
            loss += weight_first * image_loss_first.sum()

            y_dL2 = (abs(opt_param.reshape(self.cam_H, self.cam_W, -1)[:-1] - opt_param.reshape(self.cam_H, self.cam_W, -1)[1:])).sum()/(self.cam_H*self.cam_W)
            x_dL2 = (abs(opt_param.reshape(self.cam_H, self.cam_W, -1)[:,:-1] - opt_param.reshape(self.cam_H, self.cam_W, -1)[:,1:])).sum()/(self.cam_H*self.cam_W)
            loss += args.weight_spatial*(x_dL2+y_dL2)
            
            hyp_dL2 = ((opt_param[:,:-1] - opt_param[:,1:])**2).sum()/ (self.cam_H*self.cam_W)
            loss += weight_spectral*(hyp_dL2)

            loss = loss.sum()
            loss_vis.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
            scheduler.step()
                
            if i % 500 == 0:
                print(f"Frame {n_group} Epoch : {i}/{epoch}, Loss: {loss.item()}, LR: {optimizer.param_groups[0]['lr']}")

        # reconstructed hyperspectral image
        return opt_param.detach().cpu().numpy().reshape(self.cam_H, self.cam_W, len(self.wvls))
    
    def hyp_recon_dynamic(self):
        """
            We optimize to reconstruct hyperspectral image for specific scene


            Arguments
            -----------
            - ddsl_data : captured scene under ddsl patterns
            - depth : depth of captured scenes
            - gauss_masking_ftn : scene dependent dispersive-aware mapping function
        """
        
        date = self.args.date
        dynamic = False
        
        if self.n_groups > 1:
            dynamic = True

        # bring ddsl, black images for each frames
        ddsl_imgs, black_imgs = self.get_data.get_real_captured_images(date)
        
        # get number of (self.n_group -1) optical flow using black_imgs
        black_flow_list = self.black_opt_flow.black_optical_fow(black_imgs)
        
        # interpolate ddsl frames using cubic interpolation, total number of (self.n_group-1)*self.n_illum
        interpolated_flow_list = self.black_opt_flow.interpolate_optical_flow(black_flow_list)
        
        # Find each frame optical flow to target scene, Find mapping for each optical flow
        optical_flow_list, full_mapping_xy_list = self.black_opt_flow.get_mapping_list(interpolated_flow_list)
        
        hyp_recon_result = np.zeros(shape=(self.n_groups-1, self.cam_H, self.cam_W, len(self.wvls)))
        for n_group in range(1, self.n_groups):
            depths = self.get_data.get_depth(date, n_group)
            first_real_img_illum_idx = self.get_data.get_dsl_scene_dependent_mapping(depths, dynamic)
            
            ddsl_data = self.get_data.get_ddsl_data(ddsl_imgs[n_group], black_imgs[n_group], full_mapping_xy_list[n_group-1])
            gauss_masking_ftn = self.get_ddsl_scene_dependent_mapping(dynamic, first_real_img_illum_idx, full_mapping_xy_list[n_group-1])
            hyp_recon_result[n_group-1] = self.hyp_optimization(ddsl_data, depths, gauss_masking_ftn, n_group)
            
if __name__ == "__main__":
    from hyper3d_utils import Argparser

    argument = Argparser.Argument()
    args = argument.parse()
    
    hyp_recon = HypReconDynamic(args).hyp_recon_dynamic()