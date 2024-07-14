import torch
import cv2
from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.torch_utils import select_device
from utils.augmentations import letterbox
import numpy as np


# Const
weights = "./yolov5n.pt"
device = "cpu"
dnn = False
data = "data/coco128.yaml"
half = False
imgsz = (640, 640)
conf_thres = 0.5
iou_thres = 0.45
classes = None
agnostic_nms = False
max_det = 1000


# Recognition class
class Recognition():
    def __init__(self, parent):
        self.parent = parent

        # Load model
        self.device = select_device(device)
        self.model = DetectMultiBackend(
            weights, device=self.device, dnn=dnn, data=data, fp16=half
        )
        self.imgsz = check_img_size(imgsz, s=self.model.stride)
        self.det = None


    # Update recognition results
    def update(self):
        while True:
            if self.parent.curr_image is not None:
                # Run inference
                self.model.warmup(imgsz=(1, 3, *self.imgsz))
                
                im0s = self.parent.curr_image
                im = letterbox(
                    im0s, self.imgsz, stride=self.model.stride, auto=self.model.pt)[0]  # padded resize
                im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
                im = np.ascontiguousarray(im)  # contiguous

                im = torch.from_numpy(im).to(self.model.device)
                im = im.half() if self.model.fp16 else im.float()
                im /= 255
                if len(im.shape) == 3:
                    im = im[None]
                pred = self.model(im, augment=False, visualize=False)
                pred = non_max_suppression(
                    pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det
                )

                self.det = pred[0]
                self.img_tensor = im