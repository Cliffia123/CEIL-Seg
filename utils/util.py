import os
import cv2
import torch
import shutil
import torchvision.utils as vutils
from numpy import *
from utils.ddt.ddt_func import *
from utils.ddt.ddt_vis import *

from skimage import measure, color

IMAGE_MEAN_VALUE = [0.485, 0.456, 0.406]
IMAGE_STD_VALUE = [0.229, 0.224, 0.225]


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def accuracy_pca(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def calculate_IOU(boxA_list, boxB):
    max_iou = 0
    for boxA in (boxA_list):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = (xB - xA + 1) * (yB - yA + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        if boxAArea + boxBArea!=interArea:
            iou = interArea / float(boxAArea + boxBArea - interArea)
        else:
            iou = 0
        max_iou = max(max_iou, iou)
    # return the intersection over union value
    return max_iou

def IoU(A, B):
    # xmin_a, ymin_a, xmax_a, ymax_a = A
    # xmin_b, ymin_b, xmax_b, ymax_b = B
    # weight = min(xmax_a, xmax_b) - max(xmin_a, xmin_b)
    # height = min(ymax_a, ymax_b) - max(ymin_a, ymin_b)
    # if weight<=0 or height<=0:
    #     return 0
    # sa    = (xmax_a-xmin_a)*(ymax_a-ymin_a)
    # sb    = (xmax_b-xmin_b)*(ymax_b-ymin_b)
    # inter = weight*height
    # union = sa+sb-inter
    # return inter/(union+1e-12)
    boxA = A
    boxB = B
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    if boxAArea + boxBArea!=interArea:
        iou = interArea / float(boxAArea + boxBArea - interArea)
    else:
        iou = 0
    return iou

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch != 0 and epoch % args.LR_decay == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
        print('LR is adjusted at {}/{}'.format(
            epoch, args.epochs
        ))

def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth.tar'):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    torch.save(state, os.path.join(save_dir, filename))
    if is_best:
        shutil.copyfile(os.path.join(save_dir, filename),
                        os.path.join(save_dir, 'model_ILSVRC_ACoL_eil.pth.tar'))


def load_model(model, optimizer, args):
    """ Loading pretrained / trained model. """
    if os.path.isfile(args.resume):
        if args.gpu == 0:
            print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume, map_location='cpu')
        try:
            args.start_epoch = checkpoint['epoch']
        except (TypeError, KeyError) as e:
            print("=> No 'epoch' keyword in checkpoint.")
        try:
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                # best_acc1 = best_acc1.to(args.gpu)
                pass
        except (TypeError, KeyError) as e:
            print("=> No 'best_acc1' keyword in checkpoint.")
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except (TypeError, KeyError, ValueError) as e:
            print("=> Fail to load 'optimizer' in checkpoint.")
        try:
            if args.gpu == 0:
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
        except (TypeError, KeyError) as e:
            if args.gpu == 0:
                print("=> No 'epoch' in checkpoint.")

        model.load_state_dict(checkpoint['state_dict'], strict=True)
    else:
        if args.gpu == 0:
            print("=> no checkpoint found at '{}'".format(args.resume))

    return model, optimizer


def draw_bbox(image, iou, gt_box, pred_box, is_top1=False):
    
    #(0, 0, 255)-->蓝色,  (0,255,0)绿色, 
    def draw_bbox(img, box1, box2, color1=(0, 0, 255), color2=(0, 255, 0)):
        for box in box1:
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color1, 2)
        # for box in box2:
        cv2.rectangle(img, (box2[0], box2[1]), (box2[2], box2[3]), color2, 2)
        return img
    def mark_target(img, text='target', pos=(25, 25), size=2):
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), size)
        return img

    boxed_image = image.copy()

    # draw bbox on image
    boxed_image = draw_bbox(boxed_image, gt_box, pred_box)

    # mark the iou
    mark_target(boxed_image, '%.1f' % (iou * 100), (150, 30), 2)

    if is_top1:
        mark_target(boxed_image, 'TOP1', pos=(15, 30))

    return boxed_image


def save_images(folder_name, epoch, i, blend_tensor, args):
    """ Save Tensor image in the folder. """
    saving_folder = os.path.join(args.log_folder, folder_name)
    if not os.path.isdir(saving_folder):
        os.makedirs(saving_folder)
    file_name = 'HEAT_TEST_{}_{}.jpg'.format(epoch+1, i)
    saving_path = os.path.join(saving_folder, file_name)
    if args.gpu == 0:
        vutils.save_image(blend_tensor, saving_path)

def save_erased_images(folder_name, epoch, i, images_list, args):
    """
       将image uint ndrray模式保存为cv2格式
       :param image_uint8要保存的tensor
       :param folder_name, image_name: 保存的文件名
    """
    saving_folder = os.path.join(args.log_folder, folder_name)
    file_name = 'HEAT_TEST_{}_{}.jpg'.format(epoch + 1, i)
    saving_path = os.path.join(saving_folder, file_name)

    if not os.path.isdir(saving_folder):
        os.makedirs(saving_folder)
    images_list = np.concatenate(images_list,1)
    cv2.imwrite(saving_path, images_list)

def get_ddt(output):

    feature_list = []
    if len(output.shape) == 3:
        output = np.expand_dims(output, 0)
    output = np.transpose(output, (0, 2, 3, 1))
    output_vec = output
    n, h, w, c = output.shape

    output = np.reshape(output, (n * h * w, c))
    feature_list.append(output)
    X = np.concatenate(feature_list, axis=0)

    mean_matrix = np.mean(X, 0)
    X = X - mean_matrix
    trans_matrix = sk_pca(X, 1)
    #output_heatmap.shape -->(batch_size/num_gpu, 28, 28, 1)
    #trans_matrix.shape --> ()
    output_vec = output_vec - mean_matrix
    output_heatmap = np.dot(output_vec, trans_matrix.T)

    return output_heatmap

def get_ddt_2(ddt_map, image_name, args):
    now_class_dict = {}
    feature_list = []
    box_list= []
    ddt_map = to_data(ddt_map)
    output = torch.squeeze(ddt_map).numpy()
    if len(output.shape) == 3:
        output = np.expand_dims(output, 0)
    output = np.transpose(output, (0, 2, 3, 1))
    n, h, w, c = output.shape
    for i in range(n):
        #path[i] --> img的位置和名字，output --> 图像
        img_root = os.path.join(args.data_root, image_name[i])
        now_class_dict[img_root] = output[i, :, :, :]
    output = np.reshape(output, (n * h * w, c))
    feature_list.append(output)
    X = np.concatenate(feature_list, axis=0)
    mean_matrix = np.mean(X, 0)
    X = X - mean_matrix
    trans_matrix = sk_pca(X, 1)
    for img_path, img_vet in now_class_dict.items():
        w = 14
        h = 14
        he = 448
        wi = 448
        v = img_vet - mean_matrix
        heatmap = np.dot(v, trans_matrix.T)  #28,28,1
        heatmap = cv2.resize(heatmap, (h, w),interpolation=cv2.INTER_NEAREST)

        heatmap = cv2.resize(heatmap, (he, wi), interpolation=cv2.INTER_NEAREST)
        highlight = np.zeros(heatmap.shape)
        highlight[heatmap > 0] = 1

        # max component
        all_labels = measure.label(highlight)
        highlight = np.zeros(highlight.shape)
        highlight[all_labels == count_max(all_labels.tolist())] = 1

        # 可视化
        dst = color.label2rgb(all_labels, bg_label=0)  # 根据不同的标记显示不同的颜色
        dst = np.uint8(np.interp(dst, (dst.min(), dst.max()), (0, 255)))

        # visualize heatmap

        # highlight_1 = cv2.resize(highlight, (he, wi), interpolation=cv2.INTER_NEAREST)
        # image = np.expand_dims(highlight_1, axis=2)

        ori_img = cv2.imread(img_path)
        ori_img = cv2.resize(ori_img, (448, 448))
        imgadd = cv2.addWeighted(ori_img, 0.5, dst, 0.5, 0)
        # imgadd = imgadd.transpose(2, 0, 1)
        # dst = dst.transpose(2, 0, 1)
        highlight = np.round(highlight * 255)
        highlight_big = cv2.resize(highlight, (he, wi), interpolation=cv2.INTER_NEAREST)
        # 将图像转化为二值化
        props = measure.regionprops(highlight_big.astype(int))

        if len(props) == 0:
            # print(highlight)
            bbox = [0, 0, wi, he]
        else:

            temp = props[0]['bbox']
            bbox = [temp[1], temp[0], temp[3], temp[2]]
        temp_bbox = [bbox[0], bbox[1], bbox[2], bbox[3]]
        cv2.rectangle(imgadd, (temp_bbox[0], temp_bbox[1]),
                      (temp_bbox[2], temp_bbox[3]), (255, 0, 0), 4)

        temp_bbox = [int(x / 2) for x in temp_bbox]
        box_list.append(temp_bbox)
        save_addimg(imgadd, img_path, args)

    return box_list




