# import os
from collections import defaultdict
image_ids = []
image_labels = []
image_name = []
datalist = 'train.txt'
class_first_compo = defaultdict(list)
with open(datalist) as f:
    str_temp = ""
    for line in f:
        info = line.strip().split()
        name = info[0]
        class_first_compo[info[1]].append(name)

f = open('pca_3.txt', 'a')
p = 0
h = 0
for i in range(1000):
    h = 0
    for k in class_first_compo[str(i)]:
        if h<25:
            str_temp = str(k) +" "+ str(i)
            f.writelines([str_temp, '\n'])
        h+=1

f.close()
print('end')
