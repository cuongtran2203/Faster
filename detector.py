import os
import cv2
import time
import argparse

import torch
from model.detector import Detector

from ultils.utils import *
import numpy as np
from time import perf_counter

LABEL_NAMES=["bus","car","person","trailer","truck","lane"]
class Detection():
    def __init__(self,model_path=None,data_path=None):
        self.cfg=load_datafile(data_path)
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Detector(self.cfg["classes"], self.cfg["anchor_num"], True,).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
    def detect(self, org_img:np.ndarray):
        res_img = cv2.resize(org_img, (self.cfg["width"], self.cfg["height"]), interpolation = cv2.INTER_LINEAR) 
        img = res_img.reshape(1, self.cfg["height"], self.cfg["width"], 3)
        
        img = torch.from_numpy(img.transpose(0,3, 1, 2))
        img = img.to(self.device).float() / 255.0
        start = perf_counter()
        preds = self.model(img)
        end = perf_counter()
        time = (end - start) * 1000.
        print("forward time:%fms"%time)
        output = handel_preds(preds, self.cfg, self.device)
        output_boxes = non_max_suppression(output, conf_thres = 0.3, iou_thres = 0.4)
        h, w= org_img.shape[:2]
        scale_h, scale_w = h / self.cfg["height"], w / self.cfg["width"]
        for box in output_boxes[0]:
            box = box.tolist()
        
            obj_score = box[4]
            category = LABEL_NAMES[int(box[5])]

            x1, y1 = int(box[0] * scale_w), int(box[1] * scale_h)
            x2, y2 = int(box[2] * scale_w), int(box[3] * scale_h)
            cv2.rectangle(org_img, (x1, y1), (x2, y2), (255, 255, 0), 2)
            
            cv2.putText(org_img, '%.2f' % obj_score, (x1, y1 - 5), 0, 0.7, (0, 255, 0), 2)	
            cv2.putText(org_img, category, (x1, y1 - 25), 0, 0.7, (0, 255, 0), 2)
        return org_img

if __name__ == '__main__':
    model=Detection(model_path="./model_best.pth",data_path="./coco.data")
    cap=cv2.VideoCapture("video6.avi")
    while True:
        _,frame=cap.read()
        frame=cv2.resize(frame,(1280,720))
        img=model.detect(frame)
        cv2.imshow("frame",img)
        key=cv2.waitKey(1)
        if key ==ord("q"):
            break


        