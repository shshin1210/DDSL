import numpy as np
import sys, os, torch

from scipy import interpolate

sys.path.append('DDSL')
sys.path.append('../RAFT')
sys.path.append('../RAFT/raft_core')

from raft_core.raft import RAFT
from raft_core.utils.utils import InputPadder
    
class BlackOpticalFlow():
    def __init__(self, args):        
        # args
        self.args = args
        
        # device
        self.device = args.cuda_device
        
        # camera
        self.cam_H, self.cam_W = args.crop_cam_H, args.crop_cam_W
        
        # datas
        self.n_groups = len(os.listdir(args.dynamic_dir%args.date)) # Number of groups in the dynamic scene

        # directory
        self.n_illum = args.n_illum
        
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
        
        model = torch.nn.DataParallel(RAFT(self.args))
        model.load_state_dict(torch.load(self.args.models_dir))

        model = model.module
        model.to(self.device)
        model.eval()

        with torch.no_grad():
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(torch.tensor(image1), torch.tensor(image2))
            
            image1, image2 = image1*self.args.black_const, image2*self.args.black_const
            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            
        return flow_up
    
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
        
        dst_idx = np.array([i for i in range(self.n_illum+1) if i != (self.args.target_idx)])
        final_dst_idx = []
        for i in range(self.n_groups-1):
            incremented_array = dst_idx + i * (self.n_illum+1)
            final_dst_idx.extend(incremented_array)
        final_dst_idx = np.array(final_dst_idx)
        
        target_indices = []
        for i in range(self.n_groups-1):
            target_indices.extend([self.args.target_idx + i * (self.n_illum+1)] * self.n_illum)
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