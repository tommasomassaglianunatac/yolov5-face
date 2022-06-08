from utils.face_detector import YoloFaceV5
import numpy as np
import cv2
import torch

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cam = cv2.VideoCapture(0)

    model = YoloFaceV5('models\yolov5n_state_dict.pt', 'models\yolov5n.yaml', 'cpu')

    #frame_w, frame_l = frame.shape[0], frame.shape[1]

    while True:
        check, frame = cam.read()

        detections = model.detect_one(frame)

        print(detections)

        cv2.imshow('video', frame)

        key = cv2.waitKey(1)
        if key == 27:
            break