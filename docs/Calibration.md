# How to calibration Thermal camera
### Step 1: Create checkerboard
The commonly used printed checkerboards are not visible from thermal cameras. therefore, you need to create a checkerboard in a different way to proceed with calibration. As a method we devised, we made a checkerboard with the temperature difference between acrylic and aluminum plates.
### Step 2: Run capture images
Run calibrate.py and capture images while moving the checkerboard at various angles. the captured images is stored in the streoL and streoR folders in the data folder

### Step 3 : Find checkerboard corners
Run calibrate.py to locate the corner of the checkerboard in the captured image. and calibrate the camera.

### Step 4 : Tune Parameters of Block Matching Algorithm
Run Depth-Perception-Using-Stereo-Camera/python/disparity_params_gui.py 