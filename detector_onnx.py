import torch
import numpy as np
import onnxruntime
import torch
import torch.nn as nn
from ultils.utils import *
import numpy as np
from time import perf_counter
LABEL_NAMES=["bus","car","person","trailer","truck","lane"]
# class Detector_onnx():
#     def __init__(self,model_path_fpn=None,model_path_shufflenetv2=None,data_path=None ) :
#         super(Detector_onnx, self).__init__()
#         out_depth=72
#         self.stage_out_channels = [-1, 24, 48, 96, 192]
#         self.providers = ['CPUExecutionProvider']
#         self.cfg=load_datafile(data_path)
#         self.session_backbone= onnxruntime.InferenceSession(model_path_shufflenetv2, providers=self.providers)
#         self.session_detect = onnxruntime.InferenceSession(model_path_fpn, providers=self.providers)
#         self.output_reg_layers = nn.Conv2d(out_depth, 4 * self.cfg["anchor_num"], 1, 1, 0, bias=True)
#         self.output_obj_layers = nn.Conv2d(out_depth, self.cfg["anchor_num"], 1, 1, 0, bias=True)
#         self.output_cls_layers = nn.Conv2d(out_depth, self.cfg["classes"], 1, 1, 0, bias=True)
#     def preprocess(self,ori_img):
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         res_img = cv2.resize(ori_img, (320,320), interpolation = cv2.INTER_LINEAR) 
#         img = res_img.reshape(1,320,320, 3)
#         img = torch.from_numpy(img.transpose(0,3, 1, 2))
#         img = img.to(device).float() / 255.0
#         return img
#     def check(self,x):
#         C2,C3=self.session_backbone.run(None, {self.session_backbone.get_inputs()[0].name:np.asarray(x)})
#         print(C2.shape,C3.shape)
#         cls_2, obj_2, reg_2, cls_3, obj_3, reg_3=self.session_detect.run(None, {self.session_detect.get_inputs()[0].name:np.asarray(np.stack(C2,C3))})
#         out_reg_2 = self.output_reg_layers(reg_2)
#         out_obj_2 = self.output_obj_layers(obj_2)
#         out_cls_2 = self.output_cls_layers(cls_2)
#         out_reg_3 = self.output_reg_layers(reg_3)
#         out_obj_3 = self.output_obj_layers(obj_3)
#         out_cls_3 = self.output_cls_layers(cls_3)
#         out_reg_2 = out_reg_2.sigmoid()
#         out_obj_2 = out_obj_2.sigmoid()
#         out_cls_2 = F.softmax(out_cls_2, dim = 1)
#         out_reg_3 = out_reg_3.sigmoid()
#         out_obj_3 = out_obj_3.sigmoid()
#         out_cls_3 = F.softmax(out_cls_3, dim = 1)
#         return torch.cat((out_reg_2, out_obj_2, out_cls_2), 1).permute(0, 2, 3, 1), \
#                 torch.cat((out_reg_3, out_obj_3, out_cls_3), 1).permute(0, 2, 3, 1)  
        
class Detect():
    def __init__(self, model_path='out.onnx',data_path=None) :
        self.providers = ['CPUExecutionProvider']
        self.cfg=load_datafile(data_path)
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.session = onnxruntime.InferenceSession(model_path, providers=self.providers)
    def detect(self,org_img:np.ndarray):
        res_img = cv2.resize(org_img, (self.cfg["width"], self.cfg["height"]), interpolation = cv2.INTER_LINEAR) 
        img = res_img.reshape(1, self.cfg["height"], self.cfg["width"], 3)
        img = torch.from_numpy(img.transpose(0,3, 1, 2))
        img = img.to(self.device).float() / 255.0
        start = perf_counter()
        C1,C2,C3= self.session.run(None, {self.session.get_inputs()[0].name:np.asarray(img)})
        print(C3.shape)
        end = perf_counter()
        time = (end - start) * 1000.
        print("forward time:%f ms"%time)
        output_data = torch.from_numpy(preds[0])
        output_data=torch.reshape(output_data,(1,21,40,40))
        output = handel_preds(output_data, self.cfg, self.device)
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
        print(output_data)
        
        
  
if __name__ == '__main__':
    model=Detect(model_path="./model_best.onnx",data_path="./coco.data")
    cap=cv2.VideoCapture("video6.avi")
    while True:
        _,frame=cap.read()
        frame=cv2.resize(frame,(1280,720))
        model.detect(frame)
        break
        # cv2.imshow("frame",img)
        # key=cv2.waitKey(1)
        # if key ==ord("q"):
        #     break
