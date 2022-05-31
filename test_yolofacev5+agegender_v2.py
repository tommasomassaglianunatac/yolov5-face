import numpy as np
import torch
import detect_face_cam
import cv2
import time
import onnxruntime

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
    agegender_model = onnxruntime.InferenceSession('model_files/age_gender_model.onnx')

    #frame_w, frame_l = frame.shape[0], frame.shape[1]

    while True:
        check, frame = cam.read()

        frame, xyxy = detect_face_cam.detect_one(yolov5_face, frame, device)

        for face_rect in xyxy:
            face_rect = np.array(face_rect, np.int16)
            face = frame[face_rect[1]:face_rect[3],face_rect[0]:face_rect[2]]
            inputs = np.transpose(cv2.resize(face, (64, 64)), (2, 0, 1))
            inputs = np.expand_dims(inputs, 0).astype(np.float32) / 255.
            predictions = agegender_model.run(['output'], input_feed={'input': inputs})[0]
            print(predictions)
            gender = int(np.argmax(predictions[0, :2]))
            age = int(predictions[0, 2])
            cv2.putText(frame, '{}, {}'.format(['Male', 'Female'][gender], age), (face_rect[2], face_rect[1]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0))


        cv2.imshow('video', frame)

        fps.update()

        print(f"Current fps: {fps.getfps(time.time())}")

        key = cv2.waitKey(1)
        if key == 27:
            break
