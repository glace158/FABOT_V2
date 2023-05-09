import argparse
import os
import sys
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
import time
import math

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import LoadStreams
from utils.general import (LOGGER, Profile, check_img_size, check_requirements, cv2,
                           non_max_suppression, scale_boxes)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, smart_inference_mode
from utils.augmentations import letterbox

@smart_inference_mode()
def run(
        weights=ROOT / 'best3.pt',  # model path or triton URL
        source=ROOT / '0',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        img_size=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    #source = str(source)
    source = 0
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, auto = model.stride, model.names, model.pt
    img_size = check_img_size(img_size, s=stride)  # check image size

    # Dataloader
    bs = 0  # batch_size
    #dataset = LoadStreams(source, img_size=img_size, stride=stride, auto=auto, vid_stride=vid_stride)
    torch.backends.cudnn.benchmark = True
    n = 1
    imgs, fps, frames, threads = [None], 0, 0, None
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
        cap2.grab()
        success, im2 = cap2.retrieve()
        
        cap.grab()
        success, im = cap.retrieve()
        #im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
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
            im = np.ascontiguousarray(im)  # contiguous

        #for path, im, im0s, vid_cap, s in dataset:
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
            im0 = im0s[0].copy()

            im0 = cv2.resize(im0, (640,640))
            
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))
                
            # Stream results
            im0 = annotator.result()
            
            im2 = cv2.resize(im2, (640,640))
            cv2.imshow(str(source+1), im2)
            cv2.imshow(str(source), im0)
            cv2.waitKey(1)  # 1 millisecond

if __name__ == "__main__":
    check_requirements(exclude=('tensorboard', 'thop'))
    run()