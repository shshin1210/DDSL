import sys

sys.path.append('DDSL')
sys.path.append('../RAFT-Stereo')
sys.path.append('../RAFT-Stereo/core')

import glob, os
import numpy as np
import torch
import torchvision.transforms.functional as f
from tqdm import tqdm
from pathlib import Path

from raft_stereo import RAFTStereo
from utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt

from hyper3d_utils import Argparser
from hyper3d_utils import rectify
from hyper3d_utils import inverse_rectify

def load_image(imfile, args):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(args.cuda_device)

def recon_depth(args):

    DEVICE = args.cuda_device
    print('device : ', DEVICE)
    
    # Stereo images rectification
    rectify.Rectify(args)

    model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[int(DEVICE[-1])])
    model.load_state_dict(torch.load(args.restore_ckpt))

    model = model.module
    model.to(torch.device(DEVICE))
    model.eval()
    print(int(DEVICE[-1]), DEVICE)

    output_directory = Path(args.disparity_output_dir)
    output_directory.mkdir(exist_ok=True)

    with torch.no_grad():
        left_images = sorted(glob.glob(args.left_imgs%args.date, recursive=True))
        right_images = sorted(glob.glob(args.right_imgs%args.date, recursive=True))
        count = 0
        print(f"Found {len(left_images)} images. Saving files to {output_directory}/")

        for (imfile1, imfile2) in tqdm(list(zip(left_images, right_images))):
            if args.horizontal_flip:
                # apply horizontal flip in addition to exchanging left image and right image
                image1 = f.hflip(load_image(imfile2, args))
                image2 = f.hflip(load_image(imfile1, args))
            else:
                image1 = load_image(imfile1, args)
                image2 = load_image(imfile2, args)

            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)

            _, flow_up = model(image1, image2, iters=args.valid_iters, test_mode=True)
            flow_up = padder.unpad(flow_up).squeeze()

            if args.horizontal_flip:
                flow_up = f.hflip(flow_up)
                
            if not os.path.exists(os.path.join(output_directory, imfile1.split('/')[-4], imfile1.split('/')[-2])):
                os.makedirs(os.path.join(output_directory, imfile1.split('/')[-4], imfile1.split('/')[-2]))   
            
            # save files
            np.save(os.path.join(output_directory, imfile1.split('/')[-4], imfile1.split('/')[-2], "%s.npy"%(imfile1.split('/')[-1][:-4])), flow_up.cpu().numpy().squeeze())
            plt.imshow(-flow_up.cpu().numpy().squeeze(), cmap='jet', vmin=0, vmax=200)
            plt.colorbar(fraction=0.03, pad=0.04)
            plt.title('Disparity map (pixel)', fontdict = {'fontsize' : 18})
            plt.savefig(os.path.join(output_directory, imfile1.split('/')[-4], imfile1.split('/')[-2], "%s.png"%(imfile1.split('/')[-1][:-4])), bbox_inches='tight', dpi=400)

            plt.clf()
            count += 1
    
    # inverse rectification to get depth from camera2
    inverse_rectify.InverseRectify(args)
    
if __name__ == '__main__':
    argument = Argparser.Argument()
    args = argument.parse()

    recon_depth(args)
