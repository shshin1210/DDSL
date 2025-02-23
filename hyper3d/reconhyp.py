import numpy as np
import matplotlib.pyplot as plt
import cv2, os, sys

import torch
import imageio
import time
import torchvision.transforms as tf
from scipy import interpolate

sys.path.append('RAFT')
from core.raft import RAFT
from core.utils import flow_viz
from core.utils.utils import InputPadder

class HypReconDynamic():
    def __init__(self, args):
        
        self.args = args
        
        # device
        self.device = args.device
                
        # camera
        self.cam_H, self.cam_W = args.cam_H, args.cam_W

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
        self.PEF, self.CRF, self.DG_efficiency_image_first = self.get_radiometric_data()
        
        # gaussian blur
        self.gaussian_blur = tf.GaussianBlur(kernel_size=(7,7), sigma=(3,3))

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
        
        # black imagess
        for n_group in range(self.n_groups):
            black_imgs[n_group] = cv2.imread(os.path.join(self.real_data_dir%(date, n_group), 'black.png'), -1)[:,:,::-1]/65535.
            # DDSL images
            for i in range(self.n_illum):
                ddsl_imgs[n_group, i] = cv2.imread(os.path.join(self.real_data_dir%(date, n_group), 'capture_%04d.png'%i), -1)[:,:,::-1]/65535.
            
        print("Total %d frames, ddsl imgs and black imgs created ..."%self.n_groups)
        
        return ddsl_imgs, black_imgs     
    
    def optical_flow_raft(self, image1, image2):
        """
            Use RAFT as optical flow
            
            Arguments
            -----------
            - image1 : image of previous frame
            - image2 : image of next frame
            
            Returns
            -----------
            - flow_up : optical flow between image1 and image2
        """
        
        model = torch.nn.DataParallel(RAFT(args))
        model.load_state_dict(torch.load(args.models_dir))

        model = model.module
        model.to(self.args.device)
        model.eval()

        with torch.no_grad():
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(torch.tensor(image1), torch.tensor(image2))
            
            image1, image2 = image1*args.black_const, image2*args.black_const
            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            
        return flow_up
    
    def image_process(self,  img):
        """
            Process image to make it as a input of RAFT
            
            Arguments
            -----------
            img : numpy array img with shape HxWx3 and range 0 to 1
            
            Return
            -----------
            float tensor on device with shape 3xHxW, range 0 to 255 (uint8)
            
        """
        
        img_uint = (img*255.).astype(np.uint8)
        torch_img = torch.from_numpy(img_uint).permute(2, 0, 1).float()
        
        return torch_img[None].to(self.device)
        
    def black_optical_fow(self, black_imgs):
        """
            Our black optical flow method.
            We get the optical flow between back images
            
            Arguments
            -----------
            - black_imgs : the black images for each dynamic groups
            
            Return
            -----------
            - black flow list : optical flow lists for each dynamic groups
        """
        
        # flows & aligned_images
        black_flow_list = np.zeros(shape=(self.n_groups-1, self.cam_H, self.cam_W, 2))
        
        for i in range(self.n_groups):
            
            if i == self.n_groups -1:
                continue
            else:
                image1 = self.image_process(black_imgs[i])
                image2 = self.image_process(black_imgs[i+1])
                
                flow = self.optical_flow_raft(image1, image2).squeeze().permute(1,2,0)[:-2] # to get rid of padding
                flow = flow.detach().cpu().numpy()

                black_flow_list[i] = flow
        
        return black_flow_list
    
    def process_interp_samples(self, interp_samples):
        """
            Processing Optical flow cubic interpolation data points
            
            Arguments
            -----------
            - interp_samples : The samples for interpolation
            
            Return
            -----------
            - flow list : optical flow lists for each dynamic groups
            
        """
        
        # insert 0 value at 0 index
        interp_samples = np.insert(interp_samples, 0, 0)
        
        # processed_interpolated_samples
        processed_interp_samples = np.zeros(len(interp_samples), dtype=np.float32)

        for i in range(1, len(interp_samples)):
            processed_interp_samples[i] = processed_interp_samples[i-1] + interp_samples[i]
            
        return processed_interp_samples
    
    def process_interp_results(self, interp_result, processed_interp_samples):
        """
            Processing Optical flow cubic interpolated data points
        """
        for i in range(len(processed_interp_samples)):
            start_idx = i * (self.n_illum+1) + 1 
            end_idx = (i + 1) * (self.n_illum+1) + 1
            interp_result[start_idx:end_idx] -= processed_interp_samples[i]
            
        return interp_result
    
    def interpolate_optical_flow(self, black_flow_list):
        """
            Interpolate the optical flows for DDSL patterns

            Argument
            --------
            black_flow_list : the optical flow of black images shape (self.n_groups -1, H, W, 2)
            
            Return
            --------
            interpolated_flow_list : interpolated optical flows
            
        """
        
        interpolated_flow_list = np.zeros(shape=((self.n_groups-1)*(self.n_illum+1), self.cam_H*self.cam_W* 2))
        black_flow_list = black_flow_list.reshape(self.n_groups-1, -1)
        indices_samples = np.array([i*(self.n_illum+1) for i in range(self.n_groups)])
        
        for i in range(self.cam_H*self.cam_W*2):
            interp_samples = black_flow_list[:,i]
            processed_interp_samples = self.process_interp_samples(interp_samples)
            
            interp_func = interpolate.CubicSpline(indices_samples, processed_interp_samples)  # 'linear' interpolation, can be 'cubic' for smoother curve
            y_new = interp_func(np.arange(0, indices_samples[-1]+1,1))
                
            processed_interp_result = self.process_interp_results(y_new, processed_interp_samples)
            interpolated_flow_list[:,i] = processed_interp_result[1:]
            
        return interpolated_flow_list.reshape((self.n_groups-1)* (self.n_illum+1), self.cam_H, self.cam_W, 2)
    
    def get_mapping_list(self, interpolated_flow_list):
        """
            Get the mapping xy list with the given interpolated optical flow

            Argument
            --------
            interpolated_flow_list : the optical flow of each black & DDSL
            (shape : self.n_groups-1, self.n_illum+1, self.cam_H, self.cam_W, 2)
            
            Return
            --------
            interpolated_flow_list : interpolated optical flows
        """
        
        optical_flow_list = np.zeros(shape=((self.n_groups-1)*self.n_illum, self.cam_H, self.cam_W, 2))
        mapping_xy_list = np.zeros(shape=((self.n_groups-1)*self.n_illum, self.cam_H, self.cam_W, 2))
        
        dst_idx = np.array([i for i in range(self.n_illum+1) if i != (args.target_idx)])
        final_dst_idx = []
        for i in range(self.n_groups-1):
            incremented_array = dst_idx + i * (self.n_illum+1)
            final_dst_idx.extend(incremented_array)
        final_dst_idx = np.array(final_dst_idx)
        
        target_indices = []
        for i in range(self.n_groups-1):
            target_indices.extend([args.target_idx + i * (self.n_illum+1)] * self.n_illum)
        target_indices = np.array(target_indices)
        
        for i in range((self.n_groups-1)*self.n_illum):
            optical_flow_list[i] = - interpolated_flow_list[target_indices[i]] + interpolated_flow_list[final_dst_idx[i]] 

            coords_x, coords_y = np.meshgrid(np.arange(self.cam_W), np.arange(self.cam_H))

            flow_x, flow_y = optical_flow_list[i, ..., 0], optical_flow_list[i, ..., 1]
    
            map_x = (coords_x + flow_x).astype(np.float32)
            map_y = (coords_y + flow_y).astype(np.float32)
            map_xy = np.dstack((map_x, map_y))
            
            mapping_xy_list[i] = map_xy
                
        mapping_xy_list = np.round(mapping_xy_list).astype(np.int32)
        mapping_xy_list[...,0][mapping_xy_list[...,0] >= self.cam_W] = self.cam_W-1
        mapping_xy_list[...,1][mapping_xy_list[...,1] >= self.cam_H] = self.cam_H-1
        mapping_xy_list[mapping_xy_list <= 0] = 0
            
        return optical_flow_list, mapping_xy_list.reshape(self.n_groups-1, self.n_illum, self.cam_H, self.cam_W, 2)
    
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
            depths = depths[args.target_idx].reshape(-1)
            
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
        ddsl_imgs, black_imgs = self.get_real_captured_images(date)
        
        # get number of (self.n_group -1) optical flow using black_imgs
        black_flow_list = self.black_optical_fow(black_imgs)
        
        # interpolate ddsl frames using cubic interpolation, total number of (self.n_group-1)*self.n_illum
        interpolated_flow_list = self.interpolate_optical_flow(black_flow_list)
        
        # Find each frame optical flow to target scene, Find mapping for each optical flow
        optical_flow_list, full_mapping_xy_list = self.get_mapping_list(interpolated_flow_list)
        
        hyp_recon_result = np.zeros(shape=(self.n_groups-1, self.cam_H, self.cam_W, len(self.wvls)))
        for n_group in range(1, self.n_groups):
            depths = self.get_depth(date, n_group)
            first_real_img_illum_idx = self.get_dsl_scene_dependent_mapping(depths, dynamic)
            
            ddsl_data = self.get_ddsl_data(ddsl_imgs[n_group], black_imgs[n_group], full_mapping_xy_list[n_group-1])
            gauss_masking_ftn = self.get_ddsl_scene_dependent_mapping(dynamic, first_real_img_illum_idx, full_mapping_xy_list[n_group-1])
            hyp_recon_result[n_group-1] = self.hyp_optimization(ddsl_data, depths, gauss_masking_ftn, n_group)
            
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    
    # device
    parser.add_argument('--device', help="device", type = str, default="cuda:5")
    
    # date
    parser.add_argument('--date', help="captured real image date", type = str, default="1114")
    parser.add_argument('--cal_date', help="calibration date", type = str, default="0919")

    # camera
    parser.add_argument('--cam_H', help="camera height", type = int, default=630)
    parser.add_argument('--cam_W', help="camera width", type = int, default=464)

    # depth
    parser.add_argument('--depth_start', help="minimum of depth range", type = int, default=600)
    parser.add_argument('--depth_end', help="maximum of depth range", type = int, default=900)

    # illumination
    parser.add_argument('--n_illum', help="number of illumination", type = int, default=8)
    parser.add_argument('--min_wvl', help="minimum wavelength reconstructed(nm)", type=int, default=440)
    parser.add_argument('--max_wvl', help="maximum wavelength reconstructed(nm)", type=int, default=660)
    parser.add_argument('--wvl_interval', help="wavelength interval(nm)", type=int, default=10)
    parser.add_argument('--target_idx', help="target illumination captured index for optical flow", type = int, default=4)

    # directory
    parser.add_argument('--real_data_dir', help="real captured data directory", type = str, default="./DSL_v2/dataset/data/realdata/2024%s/camera2/dynamic%02d")
    parser.add_argument('--npy_dir', help="pre-calibrated data directory", type = str, default='./DSL_v2/dataset/image_formation/correspondence_model/2024%s/npy_data')
    parser.add_argument('--illum_dir', help="projected illumination directory", type = str, default='./DSL_v2/dataset/image_formation/illum')
    parser.add_argument('--radiometric_data_dir', help="radiometric dataset directory", type = str, default='./DSL_v2/dataset/image_formation')
    parser.add_argument('--dynamic_dir', help="dynamic scene directory", type = str, default= "./DSL_v2/dataset/data/realdata/2024%s/camera2")
    
    # training
    parser.add_argument('--epoch', help="epoch number for optimization", type = int, default=1000)
    parser.add_argument('--lr', help="learning rate for optimization", type = int, default=0.05)
    parser.add_argument('--decay_step', help="decay step for optimization", type = int, default=400)
    parser.add_argument('--gamma', help="gamma value for optimization", type = int, default=0.5)
    parser.add_argument('--weight_spectral', help="spectral axis weight", type = int, default=7)

    # spatial weight
    parser.add_argument('--weight_spatial', help="spatial axis weight", type = int, default=0.05)

    parser.add_argument('--depth_scalar_value', help="depth scalar value for inverse law", type = int, default= 5)

    # RAFT
    parser.add_argument('--models_dir', help = 'RAFT pre-trained model', default='RAFT/models/raft-things.pth')
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--black_const', help = 'to increase intensity of black image', default=3)
    
    args = parser.parse_args()

    hyp_recon = HypReconDynamic(args).hyp_recon_dynamic()