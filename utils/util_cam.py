import pickle
from utils.ddt.ddt_func import *
from utils.ddt.ddt_vis import *
import cv2
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
def get_cam_ddt(model, image=None, args=None):
    """
        Return CAM tensor which shape is (batch, 1, h, w)
    """
    with torch.no_grad():
        if image is not None:
            _ = model(image)

        # Extract feature map
        if args.distributed:
            heatmap = model.module.get_ddt_obj()
        else:
            heatmap = model.get_ddt_obj()
        return heatmap

def get_cam_ddt_list(model, image=None, args=None):
    with torch.no_grad():
        if image is not None:
            _ = model(image)
        # Extract feature map
        if args.distributed:
            heatmap_list = model.module.get_ddt_obj_list()
            # heatmap_list,_ = model.module.get_fused_cam()
        else:
            heatmap_list = model.get_ddt_obj_list()
            # heatmap_list,_ = model.get_fused_cam()
        return heatmap_list

def resize_cam(cam, size=(224, 224)):
    cam = cv2.resize(cam, (size[0], size[1]), interpolation=cv2.INTER_CUBIC)
    cam = cam - cam.min()
    cam = cam / cam.max()
    return cam

def blend_cam(image, cam):

    mask = cam.repeat(3, 1, 1).permute(1, 2, 0).cpu().detach().numpy()
    mask = cv2.resize(mask, (224, 224))
    mask = cv2.applyColorMap(mask.astype(np.uint8), cv2.COLORMAP_JET)
    save_img = cv2.addWeighted(image, 0.5, mask, 0.5, 0.0)
    save_img = save_img.transpose(2, 1, 0)
    save_img = torch.from_numpy(save_img)
    return save_img

def intensity_to_gray(intensity, normalize=True, _sqrt=False):
    assert intensity.ndim == 2

    if _sqrt:
        intensity = np.sqrt(intensity)

    if normalize:
        intensity -= intensity.min()
        intensity /= intensity.max()

    intensity = np.uint8(255*intensity)
    return intensity
def intensity_to_rgb(intensity, cmap='cubehelix', normalize=False):
    assert intensity.ndim == 2, intensity.shape
    intensity = intensity.astype("float")

    if normalize:
        intensity -= intensity.min()
        intensity /= intensity.max()

    cmap = 'jet'
    cmap = plt.get_cmap(cmap)
    intensity = cmap(intensity)[..., :3]
    return intensity.astype('float32') * 255.0

def generate_pca(pca_map):
    pca_map = cv2.resize(pca_map, (256, 256),
                     interpolation=cv2.INTER_CUBIC)
    pca_map = intensity_to_gray(pca_map, normalize=True)
    return pca_map

def generate_psedo_box(cam):
    # image_height, image_width, _ = image.shape
    cam = cv2.resize(cam, (112, 112),
                     interpolation=cv2.INTER_CUBIC)
    gray_heatmap = intensity_to_gray(cam, normalize=True)

    return gray_heatmap

def generate_bbox(image, cam, gt_bbox, thr_val, args):
    '''
    image: single image with shape (h, w, 3)
    cam: single image with shape (h, w, 1), data type is numpy
    gt_bbox: [x, y, x + w, y + h]
    thr_val: float value (0~1)

    return estimated bounding box, blend image with boxes
    '''
    image_height, image_width, _ = image.shape
    # print("here get image shape",image_height,image_width)
    # print("but input atten shape",cam.shape)

    _gt_bbox = list()
    _gt_bbox.append(max(int(gt_bbox[0]), 0))
    _gt_bbox.append(max(int(gt_bbox[1]), 0))
    _gt_bbox.append(min(int(gt_bbox[2]), image_height - 1))
    _gt_bbox.append(min(int(gt_bbox[3]), image_width))

    cam = cv2.resize(cam, (image_height, image_width),
                     interpolation=cv2.INTER_CUBIC)
    heatmap = intensity_to_rgb(cam, normalize=True).astype('uint8')
    heatmap_BGR = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)

    # heatmap = cv2.applyColorMap(np.uint8(255 * cam_map_cls), cv2.COLORMAP_JET)

    blend = image * 0.3 + heatmap_BGR * 0.7
    # gray_heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2GRAY)
    gray_heatmap = intensity_to_gray(cam, normalize=True)
    #
    # temp1=thr_val * np.max(gray_heatmap)
    # temp2=filters.threshold_local(gray_heatmap,35)

    thr_val = thr_val * np.max(gray_heatmap)

    _, thr_gray_heatmap = cv2.threshold(gray_heatmap,
                                        int(thr_val), 255,
                                        cv2.THRESH_BINARY)

    # if args.bbox_mode == 'classical':
    #
    #     dt_gray_heatmap = thr_gray_heatmap
    #
    #     # Wayne added, try a distance transform here
    #     # thr_gray_heatmap = cv2.distanceTransform(thr_gray_heatmap, cv2.DIST_L2, 3)
    #     # thr_gray_heatmap = intensity_to_gray(
    #     #     thr_gray_heatmap, normalize=True, _sqrt=True)
    #     # _, dt_gray_heatmap = cv2.threshold(thr_gray_heatmap,
    #     #                                    10, 255,
    #     #                                    cv2.THRESH_BINARY)
    #     try:
    #         _, contours, _ = cv2.findContours(dt_gray_heatmap,
    #                                           cv2.RETR_TREE,
    #                                           cv2.CHAIN_APPROX_SIMPLE)
    #     except:
    #         contours, _ = cv2.findContours(dt_gray_heatmap,
    #                                        cv2.RETR_TREE,
    #                                        cv2.CHAIN_APPROX_SIMPLE)
    #
    #     _img_bbox = (image.copy()).astype('uint8')
    #
    #     blend_bbox = blend.copy()
    #     cv2.rectangle(blend_bbox,
    #                   (_gt_bbox[0], _gt_bbox[1]),
    #                   (_gt_bbox[2], _gt_bbox[3]),
    #                   (0, 0, 255), 2)
    #
    #     # may be here we can try another method to do
    #     # TODO
    #     # threshold all the box and then merge it
    #     # and then rank it,
    #     # finally merge it
    #     if len(contours) != 0:
    #         c = max(contours, key=cv2.contourArea)
    #
    #         x, y, w, h = cv2.boundingRect(c)
    #         estimated_bbox = [x, y, x + w, y + h]
    #         cv2.rectangle(blend_bbox,
    #                       (x, y),
    #                       (x + w, y + h),
    #                       (0, 255, 0), 2)
    #     # estimated_bboxs = []
    #     # if len(contours) != 0:
    #     #     for c in contours:
    #     #         x, y, w, h = cv2.boundingRect(c)
    #     #         estimated_bbox = [x, y, x + w, y + h]
    #     #         estimated_bboxs.append(estimated_bbox)
    #     else:
    #         estimated_bbox = [0, 0, 1, 1]
    #         # estimated_bboxs = [estimated_bbox]
    #
    #     # estimated_bboxs = sorted(estimated_bboxs, key=lambda x: (
    #     #     x[3]-x[1])*(x[2]-x[0]), reversed=True)
    #
    #     # estimated_bbox = estimated_bboxs[0]
    #     # for box in estimated_bboxs:
    #     #     t = 0.5*max((estimated_bbox[3]-estimated_bbox[1])*(estimated_bbox[2] -
    #     #                                                    estimated_bbox[0]), (box[3]-box[1])*(box[2]-box[0]))
    #     #     x_overlap = max(
    #     #         0, min(estimated_bbox[2], box[2])-max(estimated_bbox[0], box[0]))
    #     #     y_overlap = max(
    #     #         0, min(estimated_bbox[3], box[3])-max(estimated_bbox[1], box[1]))
    #     #     if x_overlap*y_overlap>t:
    #     #         do merge
    #
    #     return estimated_bbox, blend_bbox, blend

    # elif args.bbox_mode == 'DANet':  # mode is union
    def extract_bbox_from_map(boolen_map):
        assert boolen_map.ndim == 2, 'Invalid input shape'
        rows = np.any(boolen_map, axis=1)
        cols = np.any(boolen_map, axis=0)
        if rows.max() == False or cols.max() == False:
            return 0, 0, 0, 0
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        # here we modify a box to a list
        return [xmin, ymin, xmax, ymax]

    # thr_gray_map is a gray map
    estimated_bbox = extract_bbox_from_map(thr_gray_heatmap)
    blend_bbox = blend.copy()
    cv2.rectangle(blend_bbox,
                  (_gt_bbox[0], _gt_bbox[1]),
                  (_gt_bbox[2], _gt_bbox[3]),
                  (0, 0, 255), 2)
    cv2.rectangle(blend_bbox,
                  (estimated_bbox[0], estimated_bbox[1]),
                  (estimated_bbox[2], estimated_bbox[3]),
                  (0, 255, 0), 2)
    return estimated_bbox, blend_bbox, heatmap

def blend_ddt_cam(image_class, image_name, cam, args):

    if args.dataset == "CUB":
        image = cv2.resize(cv2.imread(os.path.join(args.data_root, image_class, image_name)), (224, 224))

    if args.dataset == "ILSVRC":
        # folder_name, name = image_name.split('/')
        # print(os.path.join(args.data_root, 'val',folder_name, name))
        # image = cv2.resize(cv2.imread(os.path.join(args.data_root, 'train',folder_name, name)), (224, 224))
        image = cv2.resize(cv2.imread(os.path.join(args.data_root, 'test', image_name)), (224, 224))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = cam.repeat(3, 1, 1).permute(1, 2, 0).cpu().detach().numpy()*255.
    mask = cv2.resize(mask, (224, 224))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    mask = cv2.applyColorMap(mask.astype(np.uint8), cv2.COLORMAP_JET)
    save_img = cv2.addWeighted(image, 0.5, mask, 0.5, 0.0)

    return save_img

def load_bbox(args):
    """ Load bounding box information """
    origin_bbox = {}
    image_sizes = {}
    resized_bbox = {}

    dataset_path = args.data_list
    resize_size = args.resize_size
    crop_size = args.crop_size
    if args.dataset == 'CUB':
        with open(os.path.join(dataset_path, 'bounding_boxes.txt')) as f:
            for each_line in f:
                file_info = each_line.strip().split()
                image_id = int(file_info[0])

                boxes = map(float, file_info[1:])

                origin_bbox[image_id] = list(boxes)

        with open(os.path.join(dataset_path, 'sizes.txt')) as f:
            for each_line in f:
                file_info = each_line.strip().split()
                image_id = int(file_info[0])
                image_width, image_height = map(float, file_info[1:])

                image_sizes[image_id] = [image_width, image_height]

        if args.VAL_CROP:
            shift_size = (resize_size - crop_size) // 2
        else:
            resize_size = crop_size
            shift_size = 0
        for i in origin_bbox.keys():
            num_boxes = len(origin_bbox[i]) // 4
            for j in range(num_boxes):
                x, y, bbox_width, bbox_height = origin_bbox[i][j:j+4]
                image_width, image_height = image_sizes[i]
                left_bottom_x = int(max(x / image_width * resize_size - shift_size, 0))
                left_bottom_y = int(max(y / image_height * resize_size - shift_size, 0))

                right_top_x = int(min((x + bbox_width) / image_width * resize_size - shift_size, crop_size - 1))
                right_top_y = int(min((y + bbox_height) / image_height * resize_size - shift_size, crop_size - 1))
                resized_bbox[i] = [[left_bottom_x, left_bottom_y, right_top_x, right_top_y]]

    elif args.dataset == 'ILSVRC':
        with open(os.path.join(dataset_path, 'gt_ImageNet.pickle'), 'rb') as f:
            info_imagenet = pickle.load(f)

        origin_bbox = info_imagenet['gt_bboxes']
        image_sizes = info_imagenet['image_sizes']

        if args.VAL_CROP:
            shift_size = (resize_size - crop_size) // 2
        else:
            resize_size = crop_size
            shift_size = 0
        for key in origin_bbox.keys():

            image_height, image_width = image_sizes[key]
            resized_bbox[key] = list()
            x_min, y_min, x_max, y_max = origin_bbox[key][0]
            left_bottom_x = int(max(x_min / image_width * resize_size - shift_size, 0))
            left_bottom_y = int(max(y_min / image_height * resize_size - shift_size, 0))
            right_top_x = int(min(x_max / image_width * resize_size - shift_size, crop_size - 1))
            right_top_y = int(min(y_max / image_height * resize_size - shift_size, crop_size - 1))

            resized_bbox[key].append([left_bottom_x, left_bottom_y, right_top_x, right_top_y])
    else:
        raise Exception("No dataset named {}".format(args.dataset))
    return resized_bbox


def get_bboxes(cam, cam_thr=0.5):
    """
    image: single image with shape (h, w, 3)
    cam: single image with shape (h, w, 1)
    gt_bbox: [x, y, x + w, y + h]
    thr_val: float value (0~1)
    return estimated bounding box, blend image with boxes
    """
    cam_ = cam.permute(1, 2, 0).cpu().numpy()
    cam_ = cv2.resize(cam_, (224, 224))
    cam = (cam_ * 255.).astype(np.uint8)

    map_thr = cam_thr * np.max(cam)
    _, thr_gray_heatmap = cv2.threshold(cam,
                                        int(map_thr), 255,
                                        cv2.THRESH_BINARY)

    try:
        _, contours, _ = cv2.findContours(thr_gray_heatmap,
                                          cv2.RETR_TREE,
                                          cv2.CHAIN_APPROX_SIMPLE)
    except:
        contours, _ = cv2.findContours(thr_gray_heatmap,
                                       cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 0:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        estimated_bbox = [x, y, x + w, y + h]
    else:
        estimated_bbox = [0, 0, 0, 0]

    return estimated_bbox

def get_ddt_compo(model, args=None):
    with torch.no_grad():

        if args.distributed:
            compo = model.module.get_ddt_compo()
        else:
            compo = model.get_ddt_compo()
        return compo