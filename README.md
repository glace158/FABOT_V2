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

## Install Hector SLAM
### Install YDLidar SDK and driver
YDLidar-SDK Link
```
https://github.com/YDLIDAR/YDLidar-SDK
``` 
YDLidar-SDK Link
```
https://github.com/YDLIDAR/ydlidar_ros_driver
```
### 1. Install Qt4
```
sudo apt-get install qt4-qmake qt4-dev-tools
```
### 2. Git clone Hector-mapping
git clone at ydlidar_ws/src$ 
```
git clone https://github.com/tu-darmstadt-ros-pkg/hector_slam.git

cd ..
catkin_make
```
#### If you issue Opencv path
Modify cv_bridgeConfig.cmake
```
sudo nano /opt/ros/melodic/share/cv_bridge/cmake/cv_bridgeConfig.cmake
```
Change line before
```
set(_include_dirs "include;/usr/include;/usr/include/opencv")
```
Change line after
```
set(_include_dirs "include;/usr/include;/usr/include/opencv4")
```
### 3. Modify launch flie
```
sudo nano ~/ydlidar_ws/src/hector_slam/hector_mapping/launch/mapping_default.launch
```
Change after
```
<?xml version="1.0"?>

<launch>
  <arg name="tf_map_scanmatch_transform_frame_name" default="scanmatcher_frame"/>
  <!--<arg name="base_frame" default="base_footprint"/>-->
  <arg name="base_frame" default="laser_frame"/>
  <!--<arg name="odom_frame" default="nav"/>-->
  <arg name="odom_frame" default="laser_frame"/>
  <arg name="pub_map_odom_transform" default="true"/>
  <arg name="scan_subscriber_queue_size" default="5"/>
  <arg name="scan_topic" default="scan"/>
  <arg name="map_size" default="2048"/>
  
  <node pkg="hector_mapping" type="hector_mapping" name="hector_mapping" output="screen">
    
    <!-- Frame names -->
    <param name="map_frame" value="map" />
    <param name="base_frame" value="$(arg base_frame)" />
    <param name="odom_frame" value="$(arg odom_frame)" />
    
    <!-- Tf use -->
    <param name="use_tf_scan_transformation" value="true"/>
    <param name="use_tf_pose_start_estimate" value="false"/>
    <param name="pub_map_odom_transform" value="$(arg pub_map_odom_transform)"/>
    
    <!-- Map size / start point -->
    <param name="map_resolution" value="0.050"/>
    <param name="map_size" value="$(arg map_size)"/>
    <param name="map_start_x" value="0.5"/>
    <param name="map_start_y" value="0.5" />
    <param name="map_multi_res_levels" value="2" />
    
    <!-- Map update parameters -->
    <param name="update_factor_free" value="0.4"/>
    <param name="update_factor_occupied" value="0.9" />    
    <param name="map_update_distance_thresh" value="0.4"/>
    <param name="map_update_angle_thresh" value="0.06" />
    <param name="laser_z_min_value" value = "-1.0" />
    <param name="laser_z_max_value" value = "1.0" />
    
    <!-- Advertising config --> 
    <param name="advertise_map_service" value="true"/>
    
    <param name="scan_subscriber_queue_size" value="$(arg scan_subscriber_queue_size)"/>
    <param name="scan_topic" value="$(arg scan_topic)"/>
    
    <!-- Debug parameters -->
    <!--
      <param name="output_timing" value="false"/>
      <param name="pub_drawings" value="true"/>
      <param name="pub_debug_output" value="true"/>
    -->
    <param name="tf_map_scanmatch_transform_frame_name" value="$(arg tf_map_scanmatch_transform_frame_name)" />
  </node>
    
  <node pkg="tf" type="static_transform_publisher" name="base_to_broadcaster" args="0 0 0 0 0 0 base_link laser 100"/>
</launch>
```

```
sudo nano ~/ydlidar_ws/src/hector_slam/hector_slam_launch/launch/tutorial.launch
```
Change after
```
<?xml version="1.0"?>

<launch>

  <arg name="geotiff_map_file_path" default="$(find hector_geotiff)/maps"/>

  <!--<param name="/use_sim_time" value="true"/>-->
  <param name="/use_sim_time" value="false"/>
  <node pkg="rviz" type="rviz" name="rviz"
    args="-d $(find hector_slam_launch)/rviz_cfg/mapping_demo.rviz"/>

  <include file="$(find hector_mapping)/launch/mapping_default.launch"/>

  <include file="$(find hector_geotiff_launch)/launch/geotiff_mapper.launch">
    <arg name="trajectory_source_frame_name" value="scanmatcher_frame"/>
    <arg name="map_file_path" value="$(arg geotiff_map_file_path)"/>
  </include>

</launch>
```

### 4. Test Hector SLAM
```
roscore

roslaunch ydlidar_ros_driver TX.launch

roslaunch hector_slam_launch tutorial.launch
```

## RosWeb
### Install rosbridge
```
sudo apt-get install ros-melodic-rosbridge-server
```
### Modify IP
rosWeb/roslisb2.js
```
var ros = new ROSLIB.Ros({
    url : 'ws://**localhost**:9090'
  });
```
opt/ros/melodic/share/rosbridge_server/launch/rosbridge_websocket.launch
```
<arg name="address" default="**localhost**" />
```
### Test RosWeb
Start rosbridge_websocket launch
```
roslaunch rosbridge_server rosbridge_websocket.launch
```
Start main server at ~/rosWeb/
```
python3 main.py
```

