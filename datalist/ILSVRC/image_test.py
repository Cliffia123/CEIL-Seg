import cv2
import os
bounding_boxes = 'bounding_boxes_mix_order.txt'
image_path = '/data1/jiaming/data/imagenet/test'
val = 'val_mix.txt'
image_name = []
image_box = []
origin_bbox = {}
with open(val) as f:
    for line in f:
        info = line.strip().split()
        image_name.append(info[1])
        # img = cv2.imread(os.path.join(image_path, info[1]))

with open(bounding_boxes) as f:
    for each_line in f:
        file_info = each_line.strip().split()
        image_id = int(file_info[0])
        boxes = map(float, file_info[1:])
        origin_bbox[image_id] = list(boxes)
num=0
for i in range(len(image_name)):
    if num<100:
        img = cv2.imread(os.path.join(image_path, image_name[i]))
        print(origin_bbox[i])
        cv2.rectangle(img,(int(origin_bbox[i][0]),int(origin_bbox[i][1])),
                      (int(origin_bbox[i][2]),int(origin_bbox[i][3])),(0,0,255),2)
        image_str = str(i)+'.jpg'
        cv2.imwrite(os.path.join('image',image_str),img)
    num+=1

