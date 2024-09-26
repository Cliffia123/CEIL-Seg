from torchvision.datasets import ImageFolder
import torch.utils.data as data
from PIL import Image
import os
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')
class SubImageDataset(data.Dataset):
    def __init__(self, now_list, transforms=None):
        self.imgs = now_list
        self.transforms = transforms

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path,label = self.imgs[index]
        now_img = pil_loader(path)
        if self.transforms is not None:
            now_img = self.transforms(now_img)
        return now_img,path

    def __len__(self):
        return len(self.imgs)

    def __next__(self):
        pass
class DDTImageNetDataset(data.Dataset):
    def __init__(self, root='/data0/caoxz/ACoL_EIL/', transforms=None, batch_size=32,target_transform=None):
        datalist = os.path.join('/data0/caoxz/datasets/CUB_200_2011', 'test.txt')
        self.datalist = datalist
        image_ids = []
        image_names= []
        image_labels = []
        image_classes = []
        with open(self.datalist) as f:
            for line in f:
                info = line.strip().split()
                image_ids.append(int(info[0]))
                image_class , image_name = info[1].split('/')
                image_classes.append(image_class)
                image_names.append(image_name)
                image_labels.append(int(info[2]))
        self.image_ids = image_ids
        self.image_names = image_names
        self.image_labels = image_labels  #0-200
        self.image_classes = image_classes

        self.img_dataset = ImageFolder(root)
        self.label_class_dict = {}
        
        for idx in range(len(image_labels)):
            self.label_class_dict[image_labels[idx]] = self.image_classes[idx]
        
        from collections import defaultdict
        self.class_dict = defaultdict(list)
       
        for idx in range(len(image_ids)):
            image_label = self.image_labels[idx]
            class_img = self.image_classes[idx]
            self.class_dict[image_label].append((os.path.join(root,class_img,image_names[idx]),image_label))
        self.all_dataset = []

        for i in range(200):

            self.all_dataset.append(SubImageDataset(self.class_dict[i],transforms=transforms))

        self.batch_size = batch_size
        self.transforms = transforms
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        now_dataset = self.all_dataset[index]

        now_loader = data.DataLoader(now_dataset,batch_size=self.batch_size,shuffle=False,num_workers=8)
        for i,(img,path) in enumerate(now_loader):
            yield img,path
        pass

    def __len__(self):
        return len(self.class_dict)

    def __next__(self):
        pass
if __name__ == '__main__':
    import torchvision
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224,224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ]
    )

    a = DDTImageNetDataset(batch_size=32 ,transforms=transform)
    for i in a[0]:
        import sys
        sys.exit()
