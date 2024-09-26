from __future__ import print_function

import cv2
import numpy as np
import os
_GREEN = (18, 217, 15)
_RED = (15, 18, 217)

def vis_bbox(img, bbox, color=_GREEN, thick=1):
    '''Visualize a bounding box'''
    img = img.astype(np.uint8)
    (x0, y0, x1, y1) = bbox
    cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), color, thickness=thick)
    return img

def vis_one_image(img, boxes, color=_GREEN):
    for bbox in boxes:
        img = vis_bbox(img, (bbox[0], bbox[1], bbox[2], bbox[3]), color)
    return img

def save_addimg(img, img_path, args):
    img_folder = img_path.split('/')[-2]
    img_name = img_path.split('/')[-1]
    saving_folder = os.path.join(args.vis_path,img_folder)
    if not os.path.isdir(saving_folder):
        os.makedirs(saving_folder)
    save_path = os.path.join(saving_folder, img_name)
    cv2.imwrite(save_path,img)