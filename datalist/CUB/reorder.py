import os
from collections import defaultdict

image_ids = []
image_labels = []
image_name = []
datalist = '/data0/caoxz/EIL/ilsvrc_eil_geometric/datalist/CUB/train.txt'
# bounding_box = 'datalist/ILSVRC/bounding_boxes.txt'
#新建一个字典
class_first_compo = defaultdict(list)
with open(datalist) as f:
    for line in f:
        info = line.strip().split()
        image_id = info[-1]
        class_first_compo[image_id].append(info)
        # image_ids.append(info)
# print(class_first_compo[str(0)])
f = open('pca.txt', 'a')
for i in range(200):
    str_temp = ""
    # for k in range(25):
    k=0
    for j in class_first_compo[str(i)]:
        str_temp = j[0] + " "+j[1]+" "+j[2]
        print(str_temp,'\n')
        k+=1
        if k<26:
            f.writelines([str_temp,'\n'])
        else:
            continue
print("end")