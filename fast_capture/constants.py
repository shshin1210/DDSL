import datetime, os
import numpy as npp
from os.path import join

# camera
PATTERN_PATH = './experiments/FastHyp3D_pattern/testing_8patt_high'
BLACK_PATH = './experiments/FastHyp3D_pattern'
SHUTTER_TIME = 9 # 3 # ms / 100ms는 0.1초

WAIT_TIME = 25 # ms 25

GAIN = 0  # this is log scale. /20 for conversion. 7.75db * 20, 0db - previous capture, for material capture
BINNING_RADIUS = 2
PIXEL_FORMAT = "BayerGB16"
ROI = [100, 60, 464, 630]

# Fast capture
NUM_FRAME = 40
NUM_DUMMY = 120
FAST_SAVE_PATH = "captured_dual_cam"

# scene parameters
SCENE_PATH = './experiments'
SCENE_NAME = 'test'
LOG_FN = SCENE_PATH + '/logs/' + datetime.datetime.now().strftime('_%Y_%m_%d_%H_%M') + '.txt'
SCENE_FN = SCENE_PATH + '/test' + datetime.datetime.now().strftime('_%Y_%m_%d_%H_%M')

# screen index
CAMERA_NUM = 0  # camera num
SCREEN_NUM = 1  # for projector