import os
from PIL import Image
from torch.utils.data import Dataset


class VOCDataset(Dataset):
    def __init__(self, root=None, datalist=None, transform=None, is_train=True):

        self.root = root
        self.datalist = datalist
        self.transform = transform
        self.is_train = is_train
        image_class = []
        image_names= []
        image_labels_number = []
        image_id = []
        #1 2008_001982.jpg 14
        with open(self.datalist) as f:
            for line in f:
                info = line.strip().split()
                image_name = info[1]

                image_class.append(info[-1])
                image_names.append(image_name)
                image_id.append(int(info[0]))
                image_labels_number.append(int(info[2]))
        self.image_class = image_class
        self.image_names = image_names
        self.image_labels_number = image_labels_number
        self.image_id = image_id

    def __getitem__(self, idx):
        image_class = self.image_class[idx]
        image_name = self.image_names[idx]
        image_labels_number = self.image_labels_number[idx]
        image_id = self.image_id[idx]
        image = Image.open(os.path.join(self.root, image_name)).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, image_id, image_class, image_name, image_labels_number



    def __len__(self):
        return len(self.image_class)
