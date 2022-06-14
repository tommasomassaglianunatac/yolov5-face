# -*- coding: UTF-8 -*-
import cv2
import torch
import copy
import numpy as np

from utils.common import letterbox
from utils.general import check_img_size, non_max_suppression_face, scale_coords
from utils.yolo import Model


class YoloFaceV5:
    def __init__(self, weights_name, config_name, device) -> None:
        self.model = self.load_model(weights_name, config_name, device)
        self.device = device


    def scale_coords_landmarks(self, img1_shape, coords, img0_shape, ratio_pad=None):
        # Rescale coords (xyxy) from img1_shape to img0_shape
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        coords[:, [0, 2, 4, 6, 8]] -= pad[0]  # x padding
        coords[:, [1, 3, 5, 7, 9]] -= pad[1]  # y padding
        coords[:, :10] /= gain
        #clip_coords(coords, img0_shape)
        coords[:, 0].clamp_(0, img0_shape[1])  # x1
        coords[:, 1].clamp_(0, img0_shape[0])  # y1
        coords[:, 2].clamp_(0, img0_shape[1])  # x2
        coords[:, 3].clamp_(0, img0_shape[0])  # y2
        coords[:, 4].clamp_(0, img0_shape[1])  # x3
        coords[:, 5].clamp_(0, img0_shape[0])  # y3
        coords[:, 6].clamp_(0, img0_shape[1])  # x4
        coords[:, 7].clamp_(0, img0_shape[0])  # y4
        coords[:, 8].clamp_(0, img0_shape[1])  # x5
        coords[:, 9].clamp_(0, img0_shape[0])  # y5
        return coords

    def show_results(self, img, xyxy, conf, landmarks, class_num):
        h,w,c = img.shape
        tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
        x1 = int(xyxy[0])
        y1 = int(xyxy[1])
        x2 = int(xyxy[2])
        y2 = int(xyxy[3])
        cv2.rectangle(img, (x1,y1), (x2, y2), (0,255,0), thickness=tl, lineType=cv2.LINE_AA)

        clors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]

        for i in range(5):
            point_x = int(landmarks[2 * i])
            point_y = int(landmarks[2 * i + 1])
            cv2.circle(img, (point_x, point_y), tl+1, clors[i], -1)

        tf = max(tl - 1, 1)  # font thickness
        label = str(conf)[:5]
        cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        return img

    def load_model(self, weights_name,config_name, device):
        state_dict = torch.load(weights_name)
        detector = Model(cfg=config_name)
        detector.load_state_dict(state_dict)
        detector = detector.to(device).float().eval()

        return detector



    def detect_one(self, frame, conf_threshold = 0.5, return_img = False):

        #TODO implement confidence threshold

        with torch.no_grad():
            # Load model
            img_size = 800
            conf_thres = 0.3
            iou_thres = 0.5

            orgimg = frame  # BGR
            img0 = copy.deepcopy(orgimg)

            h0, w0 = orgimg.shape[:2]  # orig hw
            r = img_size / max(h0, w0)  # resize image to img_size
            if r != 1:  # always resize down, only resize up if training with augmentation
                interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
                img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

            imgsz = check_img_size(img_size, s=self.model.stride.max())  # check img_size

            img = letterbox(img0, new_shape=imgsz)[0]

            # Convert
            img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416

            img = torch.from_numpy(img).to(self.device)
            img = img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            pred = self.model(img)[0]

            # Apply NMS
            pred = non_max_suppression_face(pred, conf_thres, iou_thres)

            detections = []

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orgimg.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class

                    det[:, 5:15] = self.scale_coords_landmarks(img.shape[2:], det[:, 5:15], orgimg.shape).round()

                    for j in range(det.size()[0]):
                        xyxy = np.array(det[j, :4].view(-1), dtype=np.int16).tolist()
                        conf = det[j, 4].cpu().numpy()
                        detections.append(xyxy)
                        landmarks = det[j, 5:15].view(-1).tolist()
                        class_num = det[j, 15].cpu().numpy()

                        if return_img:
                            orgimg = self.show_results(orgimg, xyxy, conf, landmarks, class_num)

        if return_img:
                return detections, orgimg
        return detections