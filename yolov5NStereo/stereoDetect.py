#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import numpy as np
import cv2
import torch
import time
import math
from scipy.stats import t
import pandas as pd
import queue
import warnings
from openpyxl import load_workbook

warnings.filterwarnings(action='ignore')
#import rospy
#from std_msgs.msg import String
FILE = Path(__file__).resolve()

ROOT = FILE.parents[0]  # YOLOv5 root directory
print(ROOT)
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.general import (LOGGER, Profile, check_img_size, check_requirements, cv2,
                           non_max_suppression, scale_boxes)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, smart_inference_mode
from utils.augmentations import letterbox
print(str(ROOT.parents[0]))
# Reading the mapping values for stereo image rectification
cv_file = cv2.FileStorage(str(ROOT)+"/data/stereo_rectify_maps.xml", cv2.FILE_STORAGE_READ)
Left_Stereo_Map_x = cv_file.getNode("Left_Stereo_Map_x").mat()
Left_Stereo_Map_y = cv_file.getNode("Left_Stereo_Map_y").mat()
Right_Stereo_Map_x = cv_file.getNode("Right_Stereo_Map_x").mat()
Right_Stereo_Map_y = cv_file.getNode("Right_Stereo_Map_y").mat()
cv_file.release()

depth_map = None
disparity = None
# These parameters can vary according to the setup
max_depth = 400 # maximum distance the setup can measure (in cm)
min_depth = 0 # minimum distance the setup can measure (in cm)

# Reading the stored the StereoBM parameters
cv_file = cv2.FileStorage(str(ROOT)+"/data/depth_estmation_params_py.xml", cv2.FILE_STORAGE_READ)
numDisparities = int(cv_file.getNode("numDisparities").real())
blockSize = int(cv_file.getNode("blockSize").real())
preFilterType = int(cv_file.getNode("preFilterType").real())
preFilterSize = int(cv_file.getNode("preFilterSize").real())
preFilterCap = int(cv_file.getNode("preFilterCap").real())
textureThreshold = int(cv_file.getNode("textureThreshold").real())
uniquenessRatio = int(cv_file.getNode("uniquenessRatio").real())
speckleRange = int(cv_file.getNode("speckleRange").real())
speckleWindowSize = int(cv_file.getNode("speckleWindowSize").real())
disp12MaxDiff = int(cv_file.getNode("disp12MaxDiff").real())
minDisparity = int(cv_file.getNode("minDisparity").real())
M = cv_file.getNode("M").real()
cv_file.release()

# Creating an object of StereoBM algorithm
stereo = cv2.StereoBM_create()

stereo.setNumDisparities(numDisparities)
stereo.setBlockSize(blockSize)
stereo.setPreFilterType(preFilterType)
stereo.setPreFilterSize(preFilterSize)
stereo.setPreFilterCap(preFilterCap)
stereo.setTextureThreshold(textureThreshold)
stereo.setUniquenessRatio(uniquenessRatio)
stereo.setSpeckleRange(speckleRange)
stereo.setSpeckleWindowSize(speckleWindowSize)
stereo.setDisp12MaxDiff(disp12MaxDiff)
stereo.setMinDisparity(minDisparity)

check_requirements(exclude=('tensorboard', 'thop'))

weights=ROOT / 'best3.pt'  # model path or triton URL
source=ROOT / '0'  # file/dir/URL/glob/screen/0(webcam)
data=ROOT / 'data/coco128.yaml'  # dataset.yaml path
img_size=(640, 480)  # inference size (height, width)
conf_thres=0.25  # confidence threshold
iou_thres=0.45  # NMS IOU threshold
max_det=1000  # maximum detections per image
device=''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
classes=None  # filter by class: --class 0, or --class 0 2 3
agnostic_nms=False  # class-agnostic NMS
augment=False  # augmented inference
line_thickness=3  # bounding box thickness (pixels)
hide_labels=False  # hide labels
hide_conf=False  # hide confidences
half=False
dnn=False

depth_mean_list = []
distance_list = []
sample_size = 5
cur_depth_mean = 0
count = 0

sample_delta = 50
step = 0
mindist = 50
dataframe_list = []
q = queue.Queue(50)
start = False
# Defining callback functions for mouse events
def mouse_click(event,x,y,flags,param):
    global step
    global start
    
    if event == cv2.EVENT_LBUTTONDBLCLK:
            print("click")
            start = True

#source = str(source)
source = 0
# Load model
device = select_device(device)
model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
stride, names, auto = model.stride, model.names, model.pt
img_size = check_img_size(img_size, s=stride)  # check image size

# Dataloader
bs = 0  # batch_size
torch.backends.cudnn.benchmark = True

imgs, fps, frames= [None], 0, 0
transforms = None
cap = cv2.VideoCapture(source)
cap2 = cv2.VideoCapture(source + 1)

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
frames = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')  # infinite stream fallback
fps = max((fps if math.isfinite(fps) else 0) % 100, 0) or 30  # 30 FPS fallback

_, imgs[0] = cap.read()
LOGGER.info(f"Success ({frames} frames {w}x{h} at {fps:.2f} FPS)")

s = np.stack([letterbox(x, img_size, stride=stride, auto=auto)[0].shape for x in imgs])
rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
auto = auto and rect
transforms = transforms  # optional

# Run inference
model.warmup(imgsz=(1 if auto or model.triton else bs, 3, *img_size))  # warmup
dt = (Profile(), Profile(), Profile())


while True:
    #cap2.grab()
    #success, im2 = cap2.retrieve()
    success, im2 = cap2.read()
    #cap.grab()
    #success, im = cap.retrieve()
    success, im = cap.read()

    imgR = cv2.resize(im, img_size)
    imgL = cv2.resize(im2, img_size)
    
    # Applying stereo image rectification on the left image
    Left_nice= cv2.remap(imgL,
                        Left_Stereo_Map_x,
                        Left_Stereo_Map_y,
                        cv2.INTER_LANCZOS4,
                        cv2.BORDER_CONSTANT,
                        0)
    im = Left_nice

    # Applying stereo image rectification on the right image
    Right_nice= cv2.remap(imgR,
                        Right_Stereo_Map_x,
                        Right_Stereo_Map_y,
                        cv2.INTER_LANCZOS4,
                        cv2.BORDER_CONSTANT,
                        0)

    Right_nice = cv2.cvtColor(Right_nice,cv2.COLOR_BGR2GRAY)
    Left_nice = cv2.cvtColor(Left_nice,cv2.COLOR_BGR2GRAY)
    
    # Calculating disparity using the StereoBM algorithm
    disparity = stereo.compute(Left_nice,Right_nice)

    # Converting to float32 
    disparity = disparity.astype(np.float32)

    # Normalizing the disparity map
    disparity = (disparity/16.0 - minDisparity)/numDisparities
    try:
        depth_map = M/(disparity) # for depth in (cm)
    except:
        depth_map = 0
    
    mask_temp = cv2.inRange(depth_map,min_depth,max_depth)
    depth_map = cv2.bitwise_and(depth_map,depth_map,mask=mask_temp)

    if success:
        imgs[0] = im
    else:
        LOGGER.warning('WARNING ⚠️ Video stream unresponsive, please check your IP camera connection.')
        imgs[0] = np.zeros_like(imgs[0])
        cap.open(source)  # re-open stream if signal was lost
    time.sleep(0.0)  # wait time

    im0s = imgs.copy()
    if transforms:
        im = np.stack([transforms(x) for x in im0s])  # transforms
    else:
        im = np.stack([letterbox(x, img_size, stride=stride, auto=auto)[0] for x in im0s])  # resize
        im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        #im = im[..., ::-1].transpose((2,0,1))
        im = np.ascontiguousarray(im)  # contiguous

    with dt[0]:
        
        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

    # Inference
    with dt[1]:
        pred = model(im, augment=augment, visualize=False)

    # NMS
    with dt[2]:
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

    # Process predictions
    for i, det in enumerate(pred):  # per image
        depth_mean = None
        
        im0 = im0s[0].copy()

        im0 = cv2.resize(im0, img_size)
        
        annotator = Annotator(im0, line_width=line_thickness, example=str(names))
        
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
            
            # Write results
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                annotator.box_label(xyxy, label, color=colors(c, True))
                
                # Mask to segment regions with depth less than threshold
                x1,y1,x2,y2 = int(xyxy[0]) ,int(xyxy[1]), int(xyxy[2]) ,int(xyxy[3])
                mask = cv2.inRange(depth_map,10,0)
                temp = depth_map[y1: y2, x1: x2]
                mask[y1: y2, x1: x2] = temp

                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cnts = sorted(contours, key=cv2.contourArea, reverse=True)                    

                mask2 = np.zeros_like(mask)
                cv2.drawContours(mask2, cnts, 0, (255), -1)
                depth_mean, _ = cv2.meanStdDev(depth_map, mask=mask2)
                cur_depth_mean = depth_mean-25
                depth_mean_list.append(depth_mean-25)
                count += 1
                #pub = rospy.Publisher('distance', String, queue_size=10)
                #rospy.init_node('talker', anonymous=True)
                #hello_str = str(count) +"/" + str(int(depth_mean))
                #print(count)
                '''
                if count > sample_size:
                    count = 0
                    #print("update")
                    sample_mean = np.mean(depth_mean_list) 
                    sample_std = np.std(depth_mean_list, ddof=1)
                    sample_std_error = sample_std/sample_size**0.5
                    CI_min, CI_max = t.interval(.95, sample_size - 1, loc=sample_mean, scale=sample_std_error)
                    cur_depth_mean = np.mean([x for x in depth_mean_list if CI_min < x and x < CI_max ])
                    depth_mean_list = []
                '''
                if start:
                    cm = mindist + step * sample_delta
                    q.put(cur_depth_mean)
            
                    if q.full():
                        distance_data = []
                        error_data = []
                        nan_list = []
                        distance_mean = []
                        error_mean = []
                        distance_mean_error = []

                        while not q.empty():
                            val = q.get()
                            distance_data.append(float(val))
                            error_data.append((abs(cm - float(val))/cm)*100)
                            nan_list.append('')
                                
                            distance_mean.append('')
                            error_mean.append('')
                            distance_mean_error.append('')
                    
                        distance_mean[0] = float(np.mean(distance_data))
                        error_mean[0]  = float(np.mean(error_data))
                        distance_mean_error[0] = (abs(cm - float(np.mean(distance_data)) )/cm)*100
                    
                        dataframe = pd.DataFrame(
                            data=np.array((
                            distance_data,
                            error_data,
                            distance_mean,#측정값 평균
                            error_mean,#각각 오차 평균
                            distance_mean_error,#총합 오차
                            nan_list
                            )),
                            index= ["distance", "error", "distance_mean", "error_mean", "distance_mean_error",""]
                            )
                        print(distance_data)
                        #dataframe = pd.DataFrame(distance_data)
                        dataframe_list.append(dataframe)
                        resultframe =  pd.concat(dataframe_list)
                        resultframe.to_excel(excel_writer='sample.xlsx')
                        '''
                        book = load_workbook('sample.xlsx')
                        writer = pd.ExcelWriter('sample.xlsx', engine='openpyxl') 
                        writer.book = book
                        writer.sheets = dict((ws.title, ws) for ws in book.worksheets)

                        newdf = pd.read_excel('sample.xlsx', sheet_name="Sheet1", engine='openpyxl')
                        newdf.to_excel(writer)
                        
                        writer.save()
                        '''
                        start = False
                        step += 1

                #pub.publish(hello_str)
                cv2.putText(im0, "%.2f cm"%(cur_depth_mean), (x1 ,y1 +60),1,3,(255,255,255),5,3)
                
        # Stream results
        im0 = annotator.result()
        
        cv2.imshow(str(source), im0)
        cv2.setMouseCallback(str(source),mouse_click)
        cv2.waitKey(1)  # 1 millisecond
