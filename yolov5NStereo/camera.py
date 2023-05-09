import cv2
from queue import Queue
import time
from threading import Thread

class Camera:
    def __init__(self, cam_num, fps, state=True):
        self.cam = cv2.VideoCapture(cam_num, cv2.CAP_V4L)
        
        frame = cv2.imread("./static/cameraload.png")
        buffer = cv2.imencode('.png', frame)
        self.loadimg = buffer[1].tobytes()
        
        self.q = Queue()
        self.fps = fps
        self.prev_time = 0
        self.state = state
        self.cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cam.set(3, 80)
        self.cam.set(4, 40)
        print("----------Camera----------")
        print("Camera", cam_num, ": ", self.cam.get(3), "X" ,self.cam.get(4))
    
    def start(self):
        ret, frame = self.cam.read()
        if(ret):
            thread = Thread(target=self.gen_frames, args=())
            thread.daemon = True
            thread.start()
        else:
            print("Camera Fail");
        return self
        
    def gen_frames(self):  
        while True:
            success, self.frame = self.cam.read()
            
            if (not success):
                break
            
            
    def loading(self):
        return (b'--frame\r\n'b'Content-Type: image/png\r\n\r\n' + self.loadimg + b'\r\n')
    
    def set_state(self, state):
        self.state = state
    
    def clear_q(self):
        while(not self.q.empty()):
            self.q.get()
        
    def get_q(self):
        return self.q.get()
