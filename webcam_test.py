import cv2
import time

index = 0

while True: 
    time.sleep(1)
    try:
        cap = cv2.VideoCapture(index)
        print(index)
        cap.release()
    except:
        pass
    index+=1            
