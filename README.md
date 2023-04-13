# FABOT_V2

## Jetson Nano Setup
```
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install python-pip
sudo -H pip install -U jetson-stats
sudo apt-get install nano
```

##gdm3 remove and lightdm install
```
sudo apt-get install lightdm
sudo apt-get purge gdm3

sudo reboot
```

## Swap Setting
```
sudo apt-get install dphys-swapfile

sudo nano /sbin/dphys-swapfile
sudo nano /etc/dphys-swapfile

#Change the value in the file
#CONF_SWAPSIZE=4096
#CONF_SWAPFACTOR=2
#CONF_MAXSWAP=4096

sudo reboot
```

## OpenCV 4.5.4 with CUDA
```
wget https://github.com/Qengineering/Install-OpenCV-Jetson-Nano/raw/main/OpenCV-4-5-4.sh
sudo chmod 755 ./OpenCV-4-5-4.sh
./OpenCV-4-5-4.sh
```

## PyTorch 1.8 + torchvision v0.9.0
```
wget https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl -O torch-1.8.0-cp36-cp36m-linux_aarch64.whl
sudo apt-get install python3-pip libopenblas-base libopenmpi-dev 
```

## Install Cython, numpy, pytorch 
```
pip3 install Cython
pip3 install numpy torch-1.8.0-cp36-cp36m-linux_aarch64.whl
```

## Install torchvision dependencies 
```
sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev
git clone --branch v0.9.0 https://github.com/pytorch/vision torchvision
cd torchvision
export BUILD_VERSION=0.9.0
python3 setup.py install --user
pip3 install 'pillow<9'
```

## Install yolov5
```
git clone https://github.com/ultralytics/yolov5
cd yolov5

#Download yolov5s.pt weight 
wget https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt
```
Install requirements
```
pip3 install -U PyYAML==5.3.1
pip3 install tqdm
pip3 install cycler==0.10
pip3 install kiwisolver==1.3.1
pip3 install pyparsing==2.4.7
pip3 install python-dateutil==2.8.2
pip3 install --no-deps matplotlib==3.2.2
pip3 install scipy==1.4.1
pip3 install pillow==8.3.2
pip3 install typing-extensions==3.10.0.2
pip3 install psutil
pip3 install seaborn
```
STest yolov5 at WebCam
```
python3 detect.py --source 0
```

## Install realsense
```
git clone https://github.com/jetsonhacks/installRealSenseSDK.git
cd installRealSenseSDK/
sudo nano ~/.bashrc

#Add the text in the file
export PYTHONPATH=$PYTHONPATH:/usr/local/lib/python3.6/pyrealsense2

source ~/.bashrc
```

## Test Realsense
### Download and Youtube link
Google Drive link
```
https://drive.google.com/drive/folders/1HA0tLD2Dx53vsSsnlcgeOu4Hce-Hfys_
```
Youtube link
```
https://www.youtube.com/watch?v=oKaLyow7hWU&ab_channel=SOLIDWORKS
```
Test Realsense
```
cd yolov5_object_mapping
python3 object_mapping.py
```