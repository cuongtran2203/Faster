import cv2
import time

import torch
import torchvision
import torch.nn.functional as F

import os, time
import numpy as np
from tqdm import tqdm

#加载data
def load_datafile(data_path):
    #需要配置的超参数
    cfg = {"model_name":None,
    
           "epochs": None,
           "steps": None,           
           "batch_size": None,
           "subdivisions":None,
           "learning_rate": None,

           "pre_weights": None,        
           "classes": None,
           "width": None,
           "height": None,           
           "anchor_num": None,
           "anchors": None,

           "val": None,           
           "train": None,
           "names":None
        }

    assert os.path.exists(data_path), "请指定正确配置.data文件路径"

    #指定配置项的类型
    list_type_key = ["anchors", "steps"]
    str_type_key = ["model_name", "val", "train", "names", "pre_weights"]
    int_type_key = ["epochs", "batch_size", "classes", "width",
                   "height", "anchor_num", "subdivisions"]
    float_type_key = ["learning_rate"]
    
    #加载配置文件
    with open(data_path, 'r') as f:
        for line in f.readlines():
            if line == '\n' or line[0] == "[":
                continue
            else:
                data = line.strip().split("=")
                #配置项类型转换
                if data[0] in cfg:
                    if data[0] in int_type_key:
                       cfg[data[0]] = int(data[1])
                    elif data[0] in str_type_key:
                        cfg[data[0]] = data[1]
                    elif data[0] in float_type_key:
                        cfg[data[0]] = float(data[1])
                    elif data[0] in list_type_key:
                        cfg[data[0]] = [float(x) for x in data[1].split(",")]
                    else:
                        print("配置文件有错误的配置项")
                else:
                    print("%s配置文件里有无效配置项:%s"%(data_path, data))
    return cfg

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def non_max_suppression(prediction, conf_thres=0.3, iou_thres=0.45, classes=None):
    """Performs Non-Maximum Suppression (NMS) on inference results
    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """

    nc = prediction.shape[2] - 5  # number of classes

    # Settings
    # (pixels) minimum and maximum box width and height
    max_wh = 4096
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 1.0  # seconds to quit after
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [torch.zeros((0, 6), device="cpu")] * prediction.shape[0]

    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[x[..., 4] > conf_thres]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        conf, j = x[:, 5:].max(1, keepdim=True)
        x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            # sort by confidence
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # Batched NMS
        c = x[:, 5:6] * max_wh  # classes
        # boxes (offset by class), scores
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]

        output[xi] = x[i].detach().cpu()

        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded
    return output

def make_grid(h, w, cfg, device):
    hv, wv = torch.meshgrid([torch.arange(h), torch.arange(w)])
    return torch.stack((wv, hv), 2).repeat(1,1,3).reshape(h, w, cfg["anchor_num"], -1).to(device)

#特征图后处理
def handel_preds(preds, cfg, device):
    #加载anchor配置
    anchors = np.array(cfg["anchors"])
    anchors = torch.from_numpy(anchors.reshape(len(preds) // 3, cfg["anchor_num"], 2)).to(device)

    output_bboxes = []
    layer_index = [0, 0, 0, 1, 1, 1]

    for i in range(len(preds) // 3):
        bacth_bboxes = []
        reg_preds = preds[i * 3]
        obj_preds = preds[(i * 3) + 1]
        cls_preds = preds[(i * 3) + 2]

        for r, o, c in zip(reg_preds, obj_preds, cls_preds):
            r = r.permute(1, 2, 0)
            r = r.reshape(r.shape[0], r.shape[1], cfg["anchor_num"], -1)

            o = o.permute(1, 2, 0)
            o = o.reshape(o.shape[0], o.shape[1], cfg["anchor_num"], -1)

            c = c.permute(1, 2, 0)
            c = c.reshape(c.shape[0],c.shape[1], 1, c.shape[2])
            c = c.repeat(1, 1, 3, 1)

            anchor_boxes = torch.zeros(r.shape[0], r.shape[1], r.shape[2], r.shape[3] + c.shape[3] + 1)

            #计算anchor box的cx, cy
            grid = make_grid(r.shape[0], r.shape[1], cfg, device)
            stride = cfg["height"] /  r.shape[0]
            anchor_boxes[:, :, :, :2] = ((r[:, :, :, :2].sigmoid() * 2. - 0.5) + grid) * stride

            #计算anchor box的w, h
            anchors_cfg = anchors[i]
            anchor_boxes[:, :, :, 2:4] = (r[:, :, :, 2:4].sigmoid() * 2) ** 2 * anchors_cfg # wh

            #计算obj分数
            anchor_boxes[:, :, :, 4] = o[:, :, :, 0].sigmoid()

            #计算cls分数
            anchor_boxes[:, :, :, 5:] = F.softmax(c[:, :, :, :], dim = 3)

            #torch tensor 转为 numpy array
            anchor_boxes = anchor_boxes.cpu().detach().numpy() 
            bacth_bboxes.append(anchor_boxes)     

        #n, anchor num, h, w, box => n, (anchor num*h*w), box
        bacth_bboxes = torch.from_numpy(np.array(bacth_bboxes))
        bacth_bboxes = bacth_bboxes.view(bacth_bboxes.shape[0], -1, bacth_bboxes.shape[-1]) 

        output_bboxes.append(bacth_bboxes)    
    output = torch.cat(output_bboxes, 1)
            
    return output
