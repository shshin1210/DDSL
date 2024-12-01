# Dense Dispersed Structured Light for Hyperspectral 3D Imaging of Dynamic Scenes
[DSL](https://shshin1210.github.io/DSL/) (Dense Dispersed Structured Light for Hyperspectral 3D Imaging of Dynamic Scenes) is a method that reconstructs both spectral and geometry information of dynamic scenes.

You should follow the [requirements.txt](https://github.com/shshin1210/DDSL/blob/main/requirements.txt) provided there.

## Installation
```
git clone https://github.com/shshin1210/DDSL.git
cd DDSL
pip install -r requirements.txt
```

## Image system configuration
![imageSystem](https://github.com/user-attachments/assets/9f30ac98-d066-490a-906d-e2cfe842db83)

You should prepare the DDSL imaging system configuration as the figure above.

- You will need conventional RGB projector(Epson CO-FH02), and a conventional RGB stereo cameras(FLIR GS3-U3-32S4C-C) with a diffraction grating film(Edmund 54-509) infront of the projector.

- Calibration between camera-projector, camera-camera camera-diffraction grating must be done in advance.


## Calibration datasets

![Backward mapping model](https://github.com/user-attachments/assets/d3df4cbc-a403-4d23-9c97-0755652b2c1a)

We provide the process of our data-driven backward mapping model in our paper and Supplementary Document.

All calibrated paramters should be prepared:

- Camera-camera & camera-projector intrinsic and extrinsic paramters

- Camera response function & projector emission function & Diffraction grating efficiency

- Dispersive-aware backward model

We provide an expample calibration parameters in our [DDSL Calibration Parameters](https://drive.google.com/drive/folders/17pj5KUlZ_uX8pftq2ic9OumOyM24-VNF?usp=drive_link).


## Dataset Acquisition
We capture dynamic scene under a group of M DDSL patterns and a single black pattern at 6.6 fps.

Here we use software synchronization using programs that displays images in Python via OpenGL. More details for fast capture software synchronization is provided in Supplementary Document.

Please refer to this repository [elerac/pyglimshow](https://github.com/elerac/pyglimshow) which provides code for fast capture.

For fast capture, prepare all imaging system configuration and run the code below by using files provided in directory [fast_capture](https://github.com/shshin1210/DDSL/tree/main/fast_capture)
```
python procam_multiple_capture.py
```

Make sure files inside the cloned directory looks like:
```
|-- cloned files ...
|-- procam_multiple_capture.py
|-- constants.py
|-- cam_pyspin.py
```
You may change some settings for cameras in `cam_pyspin.py`.

We provide example of captured dynamic scene images in [dataset directory](https://github.com/shshin1210/DDSL/tree/main/dataset/data/realdata/20241114) for both stereo cameras.

## Depth Reconstruction

We reconstruct depth by using the RAFT-Stereo. We used the code from [princeton-vl/RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo) and earned accurate depths.

Reconstructed depth results for each M DDSL patterns are provided in each dynamic scenes. [dynamic00](https://github.com/shshin1210/DDSL/tree/main/dataset/data/realdata/20241114/camera2/dynamic00)

You should prepare each M DDSL pattern reconstructed depth results in `npy` file for each dynamic scenes.

We provide a example of dynamic scene dataset in [Example_of_Dynamic_scene_Dataset](https://drive.google.com/drive/folders/17pj5KUlZ_uX8pftq2ic9OumOyM24-VNF?usp=drive_link).

Please make sure each methods RAFT, RAFT-Stereo, DDSL are places as:

```
DDSL
|-- dataset
|-- fast_capture
|-- hyp_recon_dynamic.py
RAFT
|-- ...
RAFT-Stereo
|-- ...
```

## Hyperspectral Reconstruction
For Hyperspectral reconstruction of dynamic scenes under group of M DDSL patterns and a single black pattern, we need optical flow estimation.

### Optical Flow

![Black optical flow](https://github.com/user-attachments/assets/5b9c19fc-b99a-4e84-aa09-6e260ca8f98e)

We estimate optical flow between each black pattern captured images by RAFT. We used the code from [princeton-vl/RAFT](https://github.com/princeton-vl/RAFT) please refer to this repository.

If you have prepared all datasets and imaging system configurataion, start reconstructing hyperspectral reflectance:
```
python hyp_recon_dynamic.py
```

replace any configuration changes in ArgumentParser.
