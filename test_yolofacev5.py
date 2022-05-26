import torch
import detect_face_cam
import cv2
import time

class FPS():
    def __init__(self):
        self.totframes = 0
        self.tottime = 0

    def update(self, elapsedtime):
        self.totframes = self.totframes+1
        self.tottime = self.tottime+elapsedtime

    def getfps(self):
        return self.totframes/self.tottime

if __name__ == '__main__':

    cam = cv2.VideoCapture(0)

    fps = FPS()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = detect_face_cam.load_model("yolov5n-face.pt", device)
    

    while True:
        start = time.time()
        check, frame = cam.read()
        frame = detect_face_cam.detect_one(model, frame, device)

        cv2.imshow('video', frame)

        elapsed = time.time()-start

        fps.update(elapsed)
        print(f"Current fps: {fps.getfps()}")

        key = cv2.waitKey(1)
        if key == 27:
            break
