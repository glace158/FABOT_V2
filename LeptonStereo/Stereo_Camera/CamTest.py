import numpy as np
import cv2

def showVideo(num):
    try:
        cap= cv2.VideoCapture(num)
        
    except:
        print("fail")
        return
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error")
            break
        
        cv2.imshow('video', frame)
        
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        
    cap.release()
    cv2.destroyAllWindows()

showVideo(1 )
