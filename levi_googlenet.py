# SPDX-License-Identifier: Apache-2.0

import cv2
import onnxruntime as ort
import argparse
import numpy as np
import sys
# sys.path.append('..')
# from ultraface.dependencies.box_utils import predict

class AgeGenderClassifier():
    def __init__(self) -> None:
        self.gender_classifier_onnx = "model_files/gender_googlenet.onnx"
        self.gender_classifier = ort.InferenceSession(self.gender_classifier_onnx)
        self.genderList=['Male','Female']
        pass

    # scale current rectangle to box
    def scale(self, box):
        width = box[2] - box[0]
        height = box[3] - box[1]
        maximum = max(width, height)
        dx = int((maximum - width)/2)
        dy = int((maximum - height)/2)

        bboxes = [box[0] - dx, box[1] - dy, box[2] + dx, box[3] + dy]
        return bboxes

    # crop image
    def cropImage(self, image, box):
        num = image[box[1]:box[3], box[0]:box[2]]
        return num



    # gender classification method
    def genderClassifier(self, orig_image):
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image_mean = np.array([104, 117, 123])
        image = image - image_mean
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)

        input_name = self.gender_classifier.get_inputs()[0].name
        genders = self.gender_classifier.run(None, {input_name: image})
        gender = self.genderList[genders[0].argmax()]
        return gender
    # ------------------------------------------------------------------------------------------------------------------------------------------------
    # Face age classification using GoogleNet onnx model
    age_classifier_onnx = "model_files/age_googlenet.onnx"

    # Start from ORT 1.10, ORT requires explicitly setting the providers parameter if you want to use execution providers
    # other than the default CPU provider (as opposed to the previous behavior of providers getting set/registered by default
    # based on the build flags) when instantiating InferenceSession.
    # For example, if NVIDIA GPU is available and ORT Python package is built with CUDA, then call API as following:
    # ort.InferenceSession(path/to/model, providers=['CUDAExecutionProvider'])
    age_classifier = ort.InferenceSession(age_classifier_onnx)
    ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

    # age classification method
    def ageClassifier(self, orig_image):
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image_mean = np.array([104, 117, 123])
        image = image - image_mean
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)

        input_name = self.age_classifier.get_inputs()[0].name
        ages = self.age_classifier.run(None, {input_name: image})
        age = self.ageList[ages[0].argmax()]
        return age

    def gender_age_detector(self, boxes, frame):

        for box in boxes:
            box = np.array(box, np.int32)
            box = self.scale(box)
            cropped = self.cropImage(frame, box)
            gender = self.genderClassifier(cropped)
            age = self.ageClassifier(cropped)

            cv2.putText(frame, f'{gender}, {age}', (box[0]+30, box[3]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
            
        return frame