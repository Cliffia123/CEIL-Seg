import os
import json
import torchvision.transforms as transforms
from torch.backends import cudnn
import torchvision.models as models
from tensorboardX import SummaryWriter
from numpy import *
from skimage import measure, color
# from scipy.misc import imresize
from PIL import Image
import cv2
import sys
sys.path.append("../..")
sys.path.append("..")
sys.path.append("../../..")
from utils.ddt.ddt_func import *
from utils.ddt.ddt_vis import *
from utils.ddt.ddt_IoU import *

from utils.util import *
from utils.ddt import *
from utils.ddt.ddt_IoU import *
from utils.util_cam import load_bbox
from utils.util_args import str2bool
import argparse
from utils.ddt.ddt_imagenet_dataset import DDTImageNetDataset


parser = argparse.ArgumentParser(description='Parameters for DDT generate box')
parser.add_argument('--input_size',default=448,dest='input_size')
parser.add_argument('data',metavar='DIR',help='path to imagenet dataset')
parser.add_argument('--gpu',help='which gpu to use',default='0,1,2',dest='gpu')
parser.add_argument('--output_path',default='/data0/caoxz/ACoL_EIL/VGG16-448',dest='output_path')
parser.add_argument('--batch_size',default=16,dest='batch_size')
parser.add_argument('--dataset', type=str, default='CUB')
parser.add_argument("--data-list", type=str, help="data list path")
parser.add_argument('--crop-size', type=int, default=224, help='validation crop size')
parser.add_argument('--resize-size', type=int, default=448, help='validation resize size')
parser.add_argument('--VAL-CROP', type=str2bool, nargs='?', const=True, default=False,
                        help='Evaluation method'
                             'If True, Evaluate on 256x256 resized and center cropped 224x224 map'
                             'If False, Evaluate on directly 224x224 resized map')

parser.add_argument('--vis_path',default='/data0/caoxz/ACoL_EIL/vis',dest='vis_path')

args = parser.parse_args()

if args.dataset == 'CUB':
   args.data_list = '/data0/caoxz/ACoL_EIL/datalist/CUB'
elif args.dataset == 'ILSVRC':
   args.data_list = '/data0/caoxz/ACoL_EIL/datalist/ILSVRC'


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
os.environ['OMP_NUM_THREADS'] = "10"
os.environ['MKL_NUM_THREADS'] = "10"

writer = SummaryWriter(log_dir='./log')


cudnn.benchmark = True
model_ft = models.vgg16(pretrained=True)
model = model_ft.features

train_on_gpu = torch.cuda.is_available()
if train_on_gpu:

   model = torch.nn.DataParallel(model).cuda()
   model.eval()
projdir = args.output_path
if not os.path.exists(projdir):
    os.makedirs(projdir)

transform = transforms.Compose([
    transforms.Resize((args.input_size,args.input_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])
batch_size = args.batch_size
train_set = os.path.join(args.data,'test')
a = DDTImageNetDataset(root=train_set,batch_size=args.batch_size, transforms=transform)
num_nan = 0

#加载gt
gt_box = load_bbox(args)

iou_corr =0
correct_1 = []
correct_5 = []
nu =0
IoU_avg = []

for class_ind in range(200):
    now_class_dict = {}
    feature_list = []
    ddt_bbox = {}
    with torch.no_grad():
        print("nu:", nu)
        for (input_img,path) in a[class_ind]:
            #input_img,path:30,30
            input_img = to_variable(input_img)
            output = model(input_img)
            #torch.Size([30, 512, 14, 14])
            output = to_data(output)
            output = torch.squeeze(output).numpy()
            if len(output.shape) == 3:
                output = np.expand_dims(output,0)
            output = np.transpose(output,(0,2,3,1))
            output_vec = output
            n,h,w,c = output.shape
            for i in range(n):
                now_class_dict[path[i]] = output[i,:,:,:]
            output = np.reshape(output,(n*h*w,c))
            feature_list.append(output)

        X = np.concatenate(feature_list,axis=0)
        mean_matrix = np.mean(X, 0)
        X = X - mean_matrix
        print("-------第%d个--------"%(num_nan+1))
        # print("Before PCA")
        trans_matrix = sk_pca(X, 1)
        print("AFTER PCA")
        cls = a.label_class_dict[class_ind]

        # print('trans_matrix shape is {}'.format(trans_matrix.shape))
        cnt = 0
        num_nan+=1
        #  k,v -- > /data0/caoxz/ACoL_EIL/train/001.Black_footed_Albatross/Black_Footed_Albatross_0040_796066.jpg, image
        #(30, 14, 14, 512)
        # output_vec = output_vec-mean_matrix
        #(30, 14, 14, 512) (512, 1)
        # output_heatmap = np.dot(output_vec, trans_matrix.T)
        # print(output_heatmap.shape)

        for k,v in now_class_dict.items():
            w = 14
            h = 14
            he = 448
            wi = 448
            v = v - mean_matrix
            heatmap = np.dot(v, trans_matrix.T)
            # exit()

            heatmap = np.reshape(heatmap, (h, w))
            heatmap= cv2.resize(heatmap, (he, wi), interpolation=cv2.INTER_NEAREST)
            highlight = np.zeros(heatmap.shape)
            highlight[heatmap > 0] = 1
            
            # max component
            all_labels = measure.label(highlight)
            highlight = np.zeros(highlight.shape)
            highlight[all_labels == count_max(all_labels.tolist())] = 1

            #可视化
            dst = color.label2rgb(all_labels,bg_label=0)  # 根据不同的标记显示不同的颜色
            dst = np.uint8(np.interp(dst, (dst.min(), dst.max()), (0, 255)))

            # visualize heatmap
            # show highlight in origin image
            #he, wi (448, 448)
            # highlight_1 = cv2.resize(highlight, (he, wi), interpolation=cv2.INTER_NEAREST)
            # image = np.expand_dims(highlight_1, axis=2)

            ori_img = cv2.imread(k)
            # ori_img = Image.open(k).convert("RGB")
            # print(ori_img.size)
            ori_img = cv2.resize(ori_img,(448, 448))
            # print(highlight_temp.shape,ori_img.shape)
            imgadd = cv2.addWeighted(ori_img, 0.7, dst, 0.3, 0)
            # imgadd = imgadd.transpose(2, 0, 1)
            dst = dst.transpose(2, 0, 1)

            # writer.add_image("highlight_1", imgadd, num_nan)
            # writer.add_image("dst", dst, num_nan)
            save_addimg(imgadd,k,args)
            highlight = np.round(highlight * 255)

            highlight_big = cv2.resize(highlight, (he, wi), interpolation=cv2.INTER_NEAREST)
            #将图像转化为二值化
            props = measure.regionprops(highlight_big.astype(int))
            if len(props) == 0:
                #print(highlight)
                bbox = [0, 0, wi, he]
            else:
                temp = props[0]['bbox']
                bbox = [temp[1], temp[0], temp[3], temp[2]]

            temp_bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
            temp_save_box = [x / 448 for x in temp_bbox]
            ddt_bbox[os.path.join(cls, k)] = temp_save_box

            highlight_big = np.expand_dims(np.asarray(highlight_big), 2)
            highlight_3 = np.concatenate((np.zeros((he, wi, 1)), np.zeros((he, wi, 1))), axis=2)
            highlight_3 = np.concatenate((highlight_3, highlight_big), axis=2)
            cnt +=1
            #加载原始的box
            
            # if cnt < 40:
            #     savepath = '/data0/caoxz/ACoL_EIL/DDT/%s' % cls
            #     if not os.path.exists(savepath):
            #         os.makedirs(savepath)
            #     raw_img = Image.open(k).convert("RGB")
            #     raw_img = raw_img.resize((448,448))
            #     raw_img = np.asarray(raw_img)
            #     raw_img = cv2.cvtColor(raw_img,cv2.COLOR_BGR2RGB)
            #     cv2.rectangle(raw_img, (temp_bbox[0], temp_bbox[1]),
            #                   (temp_bbox[2] + temp_bbox[0], temp_bbox[3] + temp_bbox[1]), (255, 0, 0), 4)
            #     save_name = k.split('/')[-1]
            #     # cv2.imwrite(os.path.join(savepath, save_name), np.asarray(raw_img))
            #     raw_img = torch.from_numpy(raw_img).permute(2,0,1)
            #     #计算iou
            box_predict = [temp_bbox[0], temp_bbox[1],
                           temp_bbox[2] + temp_bbox[0], temp_bbox[3] + temp_bbox[1]]
            box_predict = [x / 2 for x in box_predict]
            # print("gt坐标与预测坐标：",gt_box[a.image_ids[nu]][0], box_predict)

            iou = calculate_IOU(gt_box[a.image_ids[nu]][0], box_predict)

            # if iou >= 0.5:
                #统计top1的概率
            if(nu%32==0):
                correct_1.append(iou)
            if(nu%32<5):
                correct_5.append(iou)
            # iou_corr+=1
            nu += 1
                   # iou_corr +=1
    # print("IOU为:",(iou_corr/nu)%100)
    # IoU_avg.append((iou_corr/nu)%100)
    # print(mean(IoU_avg))
    x_1 = sum(i >= 0.5 for i in correct_1)
    x_5 = sum(i >= 0.5 for i in correct_5)
    print("Top_1:", x_1/len(correct_1))
    print("Top_5:", x_5/len(correct_5))
    print("GT:", iou_corr/nu)
