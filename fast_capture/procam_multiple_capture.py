import os
import time
import shutil
import concurrent.futures
import numpy as np
import cv2
import EasyPySpin
import PySpin
from screeninfo import get_monitors
import cam_pyspin
import constants

def ddsl_pattern(num_frame, ddsl_patt_dir, black_patt_dir):
    """ Creates sequential ddsl patterns with 1 single black image (total 14 as pairs)
    
    Parameters
    ----------
    num_frame : integer
        The number of frames.
    ddsl_patt_dir : string
        The directory for ddsl pattern
    black_patt_dir : string
        The directory for black pattern
    """
    
    num_patt_ddsl = len(os.listdir(ddsl_patt_dir))
    ddsl_patt_list = []
    
    for k in range(num_frame):
        ddsl_patt_list += [cv2.cvtColor(cv2.imread(os.path.join(ddsl_patt_dir, 'illum_%03d.png'%i)), cv2.COLOR_BGR2RGB) for i in range(num_patt_ddsl)]
        ddsl_patt_list += [cv2.imread(os.path.join(black_patt_dir, 'black_high.png'))] # white_gray
        # ddsl_patt_list += [cv2.imread(os.path.join(black_patt_dir, 'black.png'))]

    return ddsl_patt_list

def main():
    # Initialize both cameras
    serial_1 = "22312680"
    serial_2 = "22312690"
    cap = EasyPySpin.MultipleVideoCapture(serial_1, serial_2)

    print('Height cam 1: ', cap[0].get(cv2.CAP_PROP_FRAME_HEIGHT), 'Height cam 2: ', cap[1].get(cv2.CAP_PROP_FRAME_HEIGHT))
    print('Width cam 1: ', cap[0].get(cv2.CAP_PROP_FRAME_WIDTH), 'Width cam 2: ', cap[1].get(cv2.CAP_PROP_FRAME_WIDTH))
    
    # Seting properties
    cam_pyspin.configure_cam(cap[0].cam, constants.PIXEL_FORMAT, constants.SHUTTER_TIME*1e3, constants.BINNING_RADIUS, roi = constants.ROI)
    cam_pyspin.configure_cam(cap[1].cam, constants.PIXEL_FORMAT, constants.SHUTTER_TIME*1e3, constants.BINNING_RADIUS, roi = constants.ROI)

    print('Height cam 1: ', cap[0].get(cv2.CAP_PROP_FRAME_HEIGHT), 'Height cam 2: ', cap[1].get(cv2.CAP_PROP_FRAME_HEIGHT))
    print('Width cam 1: ', cap[0].get(cv2.CAP_PROP_FRAME_WIDTH), 'Width cam 2: ', cap[1].get(cv2.CAP_PROP_FRAME_WIDTH))
    
    # Trigger mode ON
    cap.set_pyspin_value("TriggerSelector", "FrameStart")
    cap.set_pyspin_value("TriggerMode", "On")
    cap.set_pyspin_value("TriggerSource", "Software")

    cap[0].auto_software_trigger_execute = True
    cap[1].auto_software_trigger_execute = True
    
    ### Monitor setting
    # Identify all available monitors
    monitors = get_monitors()
    for i, monitor in enumerate(monitors):
        print(f"Monitor {i}: {monitor}")
    projector_monitor = monitors[constants.SCREEN_NUM]

    ## Projected patterns here
    # Create a list of images to display
    shape = (projector_monitor.height, projector_monitor.width, 3)
    image_list_dummy = [np.full(shape, 128, dtype=np.uint8) for _ in range(constants.NUM_DUMMY)]
    image_list_main = ddsl_pattern(constants.NUM_FRAME, constants.PATTERN_PATH, constants.BLACK_PATH)
    image_list = image_list_dummy + image_list_main + image_list_dummy

    ### Display image window
    # Create a named window for displaying images
    window_name = "ProjectionWindow"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.moveWindow(window_name, projector_monitor.x, projector_monitor.y)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Start capturing
    print("Start capturing")
    image_list_captured_1 = []
    image_list_captured_2 = []

    time_start_capture = time.time()

    ### Capture
    for i in range(len(image_list)):
        time_start = time.time()

        # Display the image using OpenCV
        cv2.imshow(window_name, image_list[i]) # screen.imshow
        cv2.waitKey(constants.WAIT_TIME)  # Show the image for a brief moment to trigger the update
        
        # Capture images from both cameras simultaneously
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_cam1 = executor.submit(cap[0].read)
            future_cam2 = executor.submit(cap[1].read)

        ret1, frame1 = future_cam1.result()
        ret2, frame2 = future_cam2.result()

        image_list_captured_1.append(frame1)
        image_list_captured_2.append(frame2)

        time_end = time.time()
        elapsed = time_end - time_start
        fps = 1 / (elapsed + 1e-10)
        print(f"{i} fps: {fps:.2f}")

    ### End capturing
    time_end_capture = time.time()
    print("End capturing")
    print(f"Capture time: {time_end_capture - time_start_capture:.2f} s")

    # Release cameras
    cap.release()

    # Export captured images
    shutil.rmtree(constants.FAST_SAVE_PATH, ignore_errors=True)
    os.makedirs(f"{constants.FAST_SAVE_PATH}", exist_ok=True)
    os.makedirs(f"{constants.FAST_SAVE_PATH}/camera1", exist_ok=True)
    os.makedirs(f"{constants.FAST_SAVE_PATH}/camera2", exist_ok=True)
    
    shutil.rmtree(constants.LOG_FN, ignore_errors=True)
    os.makedirs(f"{constants.SCENE_FN}/camera1", exist_ok=True)
    os.makedirs(f"{constants.SCENE_FN}/camera2", exist_ok=True)

    # make dirs for each dynamic scenes (frames)
    for n_frame in range(constants.NUM_FRAME):
        shutil.rmtree(constants.LOG_FN, ignore_errors=True)
        os.makedirs("%s/camera1/dynamic%02d"%(constants.SCENE_FN, n_frame), exist_ok=True)
        os.makedirs("%s/camera2/dynamic%02d"%(constants.SCENE_FN, n_frame), exist_ok=True)
    
    for i, (img1, img2) in enumerate(zip(image_list_captured_1, image_list_captured_2)):
        img1 = cv2.cvtColor(img1, cv2.COLOR_BayerGB2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BayerGB2RGB)
        
        # 전체적인거 한번에 저장
        cv2.imwrite("%s/camera1/capture_%04d.png"%(constants.FAST_SAVE_PATH,i), img1)
        cv2.imwrite("%s/camera2/capture_%04d.png"%(constants.FAST_SAVE_PATH,i), img2)
        
    # Destroy the OpenCV window
    cv2.destroyWindow(window_name)

if __name__ == "__main__":
    main()
