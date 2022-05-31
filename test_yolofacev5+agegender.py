import numpy as np
import torch
import detect_face_cam
import cv2
import time
import onnxruntime
from levi_googlenet import AgeGenderClassifier

class FPS():
    def __init__(self):
        self.totframes = 0
        self.startime = time.time()

    def update(self):
        self.totframes = self.totframes+1

    def getfps(self, curtime):
        return self.totframes/(curtime-self.startime)

if __name__ == '__main__':

    cam = cv2.VideoCapture(1)

    fps = FPS()

    torch.onnx
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolov5_face = detect_face_cam.load_model("model_files/yolov5n-face.pt", device)
    agegender = AgeGenderClassifier()

    _, frame = cam.read()
    frame_w, frame_l = frame.shape[0], frame.shape[1]

    while True:
        check, frame = cam.read()
        frame, xyxy = detect_face_cam.detect_one(yolov5_face, frame, device)

        if xyxy is not None:
            frame = agegender.gender_age_detector(xyxy, frame)


        cv2.imshow('video', frame)

        fps.update()

        print(f"Current fps: {fps.getfps(time.time())}")

        key = cv2.waitKey(1)
        if key == 27:
            break
