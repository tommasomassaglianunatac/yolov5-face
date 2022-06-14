from utils.face_detector import YoloFaceV5
from multitask_agegender import agegender
import numpy as np
import cv2
import torch
import tensorflow as tf

def resizeframe(cropped_frame):
    resized = cv2.resize(cropped_frame, (224,224), interpolation = cv2.INTER_AREA)
    return resized

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cam = cv2.VideoCapture(0)

    model = YoloFaceV5('models\yolov5n_state_dict.pt', 'models\yolov5n.yaml', 'cpu')
    agegenderm = agegender.get_model()

    mf = ["M","F"]

    #frame_w, frame_l = frame.shape[0], frame.shape[1]

    while True:
        check, frame = cam.read()


        detections, frame = model.detect_one(frame, return_img=True)
        
        if len(detections) > 0:
            faces = tf.convert_to_tensor([resizeframe(frame[d[1]:d[3], d[0]:d[2]]) for d in detections], dtype=np.int16)    
            predicted_agegender = agegenderm.predict(faces, verbose=0)
            predicted_genders = [mf[np.argmax(gender)] for gender in predicted_agegender[0]]
            ages = np.arange(0, 101).reshape(101, 1)
            predicted_ages = predicted_agegender[1].dot(ages).flatten()

            for i in range(len(faces)):
                frame = cv2.putText(frame, f"{predicted_genders[i]}, {predicted_ages[i]:.0f}", (detections[i][0], detections[i][3]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        cv2.imshow('video', frame)

        key = cv2.waitKey(1)
        if key == 27:
            break