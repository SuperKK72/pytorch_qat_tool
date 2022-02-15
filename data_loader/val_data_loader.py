import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import cv2
import os
from PIL import Image


class valDataset(Dataset):
    def __init__(self, val_data_dir, images, labels, transform=None):
        self.val_data_dir = val_data_dir
        self.images = images
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.images)
    def __getitem__(self, index):
        image_name = self.images[index]
        image_path = os.path.join(self.val_data_dir, image_name)
        image = cv2.imread(image_path, -1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = image[np.newaxis, :, :, :]
        shape = image.shape
        if len(shape)==2:
            image = image[:,:,np.newaxis]
            image = np.concatenate((image, image, image), axis=-1)
        image = Image.fromarray(image)
        # label = torch.Tensor([int(self.labels[index])])
        label = torch.from_numpy(np.array([int(self.labels[index])]))
        if self.transform is not None:
            image = self.transform(image)
        return image, label

def get_images_and_labels(val_label_file):
    images = []
    labels = []
    f = open(val_label_file, 'r')
    lines = f.readlines()
    l_num = len(lines)
    print("valid dataset size: {}".format(l_num))
    for i in range(l_num):
        image, label = lines[i].split(' ')
        image = image.strip()
        label = label.strip()
        images.append(image)
        labels.append(label)
        # print("{} {}".format(image, label))
        # exit(0)
    f.close()
    return images, labels

if __name__ == "__main__":
    print("Hello SuperKK72.")
    val_data_dir = "/workspace/kli/ILSVRC2012/valid"
    val_label_file = "./val.txt"
    images, labels = get_images_and_labels(val_label_file)
    valDataset = valDataset(val_data_dir, images, labels)
    print("OK.")