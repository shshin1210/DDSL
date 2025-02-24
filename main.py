import numpy as np
import matplotlib.pyplot as plt
import cv2, os, sys

import torch
import imageio
import time
import torchvision.transforms as tf
from scipy import interpolate

from hyper3d_utils import Argparser
from hyper3d_sl import recon3d
from hyper3d_sl import reconhyp

def main(args):
    
    # reconstruct 3d information by RAFT-Stereo
    recon3d.recon_depth(args)
    print('depth reconstructed...')
    
    # reconstruct hyperspectral information
    hyp_recon = reconhyp.HypReconDynamic(args).hyp_recon_dynamic()
    print('hyperspectral reconstructed...')
    
    
if __name__ == "__main__":
    argument = Argparser.Argument()
    args = argument.parse()

    main(args)