# fabot_v2

sudo apt-get update
sudo apt-get upgrade
sudo apt-get install python-pip
sudo -H pip install -U jetson-stats

sudo apt-get install lightdm
sudo apt-get purge gdm3

sudo reboot

sudo apt-get install nano

sudo apt-get install dphys-swapfile
## 두 Swap파일의 값이 다음과 같도록 값을 추가하거나, 파일 내 주석을 해제합니다.
# CONF_SWAPSIZE=4096
# CONF_SWAPFACTOR=2
# CONF_MAXSWAP=4096

# /sbin/dphys-swapfile를 엽니다.
sudo nano /sbin/dphys-swapfile
 
# 값을 수정한 후 [Ctrl] + [X], [y], [Enter]를 눌러 저장하고 닫습니다
 
 
# /etc/dphys-swapfile를 편집합니다.
sudo nano /etc/dphys-swapfile
 
# 값을 수정한 후 [Ctrl] + [X], [y], [Enter]를 눌러 저장하고 닫습니다
 
# Jetson Nano 재부팅
sudo reboot

#OpenCV 4.5.4 with CUDA

wget https://github.com/Qengineering/Install-OpenCV-Jetson-Nano/raw/main/OpenCV-4-5-4.sh
sudo chmod 755 ./OpenCV-4-5-4.sh
./OpenCV-4-5-4.sh

#PyTorch 1.8 + torchvision v0.9.0
# PyTorch 1.8.0 다운로드 및 dependencies 설치
wget https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl -O torch-1.8.0-cp36-cp36m-linux_aarch64.whl
sudo apt-get install python3-pip libopenblas-base libopenmpi-dev 
 
# Cython, numpy, pytorch 설치
pip3 install Cython
pip3 install numpy torch-1.8.0-cp36-cp36m-linux_aarch64.whl
 
# torchvision dependencies 설치
sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev
git clone --branch v0.9.0 https://github.com/pytorch/vision torchvision
cd torchvision
export BUILD_VERSION=0.9.0
python3 setup.py install --user
pip3 install 'pillow<9'

#install yolov5
git clone https://github.com/ultralytics/yolov5
cd yolov5

# yolov5s.pt weight 다운로드
wget https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt

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

# 연결된 webcam을 통해 Inference 수행하기
python3 detect.py --source 0

#install realsense
git clone https://github.com/jetsonhacks/installRealSenseSDK.git
cd installRealSenseSDK/
sudo nano ~/.bashrc
export PYTHONPATH=$PYTHONPATH:/usr/local/lib/python3.6/pyrealsense2
source ~/.bashrc
