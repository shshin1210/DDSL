import argparse

class Argument:
    def __init__(self):    
        
        # Raft stereo
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--date', help="data date", default='20241114')
        
        self.parser.add_argument('--restore_ckpt', help="restore checkpoint", default='../RAFT-Stereo/models/iraftstereo_rvc.pth')
        self.parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
    
        self.parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
        self.parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

        # Default setting is disparity map based on the left image. 
        self.parser.add_argument('--horizontal_flip', default=True, help="Get disparity map based on the right image")
        # self.parser.add_argument('--horizontal_flip', action='store_true', help="Get disparity map based on the right image")

        # Cuda device
        self.parser.add_argument('--cuda_device', type=str, default='cuda:5', help="cuda device")

        # Architecture choices
        self.parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
        self.parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
        self.parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
        self.parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
        self.parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
        self.parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
        self.parser.add_argument('--context_norm', type=str, default="instance", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
        self.parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
        self.parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")

        # Directory
        self.parser.add_argument('--stereo_param_dir', type = str, help = 'directory for stereo parameters', default='./dataset/params/stereo_parameters')
        self.parser.add_argument('--rectified_param_dir', type = str, help = 'directory for rectified parameters', default='./dataset/params/rectified_parameters')

        self.parser.add_argument('--stereo_image_camera1_dir', type = str, help='directory for stereo images of camera 1 (left camera)', default= '../DDSL/dataset/data/realdata/%s/camera1/*/*.png')
        self.parser.add_argument('--stereo_image_camera2_dir', type = str, help='directory for stereo images of camera 2 (right camera)', default= '../DDSL/dataset/data/realdata/%s/camera2/*/*.png')

        self.parser.add_argument('--new_rectdata_dir', help ='save directory for rectified images', default='../DDSL/dataset/data/rectdata')
        
        self.parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="../DDSL/dataset/data/rectdata/%s/camera1/*/*.png")
        self.parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="../DDSL/dataset/data/rectdata/%s/camera2/*/*.png")
                
        self.parser.add_argument('--disparity_output_dir', help="directory to save output disparity", default='../DDSL/dataset/data/dispdata')
        self.parser.add_argument('--demo_output_result_dir', help="directory that has raft stereo output", default="../DDSL/dataset/data/dispdata/%s/*/*.npy")
        self.parser.add_argument('--depth_output_dir', help="directory to save output depth", default='../DDSL/dataset/data/realdata/%s/camera2/%s')

        # args
        self.parser.add_argument('--cam_H', type = int, default= 768) # 630
        self.parser.add_argument('--cam_W', type = int, default= 1024) # 464
        self.parser.add_argument('--x_start', type = int, default= 100)
        self.parser.add_argument('--y_start', type = int, default= 60)
        self.parser.add_argument('--crop_cam_H', type = int, default= 630)
        self.parser.add_argument('--crop_cam_W', type = int, default= 464)


    def parse(self):
        args = self.parser.parse_args()

        return args
