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

Please refer to this repository [Fast_capture](https://github.com/elerac/pyglimshow) which provides code for fast capture, and input files in directory [fast_capture](https://github.com/shshin1210/DDSL/tree/main/fast_capture) into the cloned directory.

We provide example of captured datasets in [dataset directory](https://github.com/shshin1210/DDSL/tree/main/dataset/data/realdata/20241114) for both stereo cameras.

1. Scene's depth map

   https://github.com/shshin1210/DSL/assets/80568500/52a04828-5dad-4c4d-9d49-382ad86a81db

   - Capture a scene under binary code pattern with a specific exposure time where zoer-order light is valid and first-order dispersion intensity is invalid

   - By utilizing conventional structured light decoding method, you should be able to prepare depth reconstructed result. Save the depth result as npy file.
      
2. Scene under white scan line pattern

   https://github.com/shshin1210/DSL/assets/80568500/c4c52964-c5c3-4915-a6ee-606ef3420bf6
   
   - Capture the scene under white scan line pattern with two different intensity pattern values and exposure time.
   
   - Save it in `path_to_ldr_exp1`, `path_to_ldr_exp2`.

4. Scene under black pattern and white pattern
   
   We need scene captured under black pattern with two different intensity pattern values and exposure time same as step 2.

   - Save it in `path_to_black_exp1`, `path_to_black_exp2`.
   
   Also, capture the scene under white pattern under two different intensity pattern values to calculate the radiance weight (normalization) for two different settings same as step 2.

   - Save it in `path_to_intensity1`, `path_to_intensity2`.

```
dataset
|-- depth.npy
|-- intensity1
|-- intensity2
|-- black_exposure1
|-- black_exposure2
|-- ldr_exposure1
    |-- scene under white scanline pattern 0.png
    |-- scene under white scanline pattern 1.png
    |-- ...
|-- ldr_exposure2
    |-- scene under white scanline pattern 0.png
    |-- scene under white scanline pattern 1.png
    |-- ...
```

## Hyperspectral Reconstruction
If you have prepared all datasets, start reconstructing hyperspectral reflectance:
```
python hyper_sl/hyperspectral_reconstruction.py
```

replace any configuration changes in [ArgParse.py] file (https://github.com/shshin1210/DSL/blob/main/hyper_sl/utils/ArgParser.py).

