import torch
import detect_face_cam
import cv2
import time

class FPS():
    def __init__(self):
        self.totframes = 0
        self.startime = time.time()

    def update(self):
        self.totframes = self.totframes+1

    def getfps(self, curtime):
        return self.totframes/(curtime-self.startime)

if __name__ == '__main__':

    cam = cv2.VideoCapture(0)

    fps = FPS()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = detect_face_cam.load_model("yolov5n-face.pt", device)
    

    while True:
        check, frame = cam.read()
        frame = detect_face_cam.detect_one(model, frame, device)

        cv2.imshow('video', frame)

        fps.update()

        print(f"Current fps: {fps.getfps(time.time())}")

        key = cv2.waitKey(1)
        if key == 27:
            break
