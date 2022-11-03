import os
import numpy as np
import torch
from torch.utils import data
from PIL import Image
import torchvision.transforms as transforms


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


class AtlantisDataSet(data.Dataset):
    def __init__(self, rootdir, split, joint_transform=None, padding_size = 0):
        super(AtlantisDataSet, self).__init__()
        self.rootdir = rootdir
        self.split = split
        self.images_base = os.path.join(self.rootdir, "images", self.split)
        self.masks_base = os.path.join(self.rootdir, "masks", self.split)
        self.items_list = self.get_images_list(self.images_base, self.masks_base)

        self.joint_transform = joint_transform
        mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        train_input_transform = []
        train_input_transform += [transforms.ToTensor(),
                                  transforms.Normalize(*mean_std)]
        self.image_transform = transforms.Compose(train_input_transform)
        self.label_transform = MaskToTensor()

        if self.split == 'val' or 'test':
            self.padding_size = padding_size

    def get_images_list(self, images_base, masks_base):
        items_list = []
        for root, dirs, files in os.walk(images_base, topdown=True):
            mask_root = os.path.join(masks_base, os.path.split(root)[1])
            for name in files:
                if name.endswith(".jpg"):
                    mask_name = name.split(".")
                    mask_name = mask_name[0] + ".png"
                    img_file = os.path.join(root, name)
                    label_file = os.path.join(mask_root, mask_name)

                    items_list.append({
                        "img": img_file,
                        "label": label_file,
                        "name": name
                    })
        return items_list

    def __getitem__(self, index):
        image_path = self.items_list[index]["img"]
        label_path = self.items_list[index]["label"]
        name = self.items_list[index]["name"]
        image = Image.open(image_path).convert('RGB')
        width, height = image.size
        label = Image.open(label_path)

        if self.joint_transform:
            image, label = self.joint_transform(image, label)
        image = self.image_transform(image)
        label = self.label_transform(label)

        if self.split == 'val' or self.split == 'test':
            top_pad =  self.padding_size - image.shape[1]
            right_pad =  self.padding_size - image.shape[2]
            image = np.lib.pad(image, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)

        label_copy = label-1
        label_copy[label_copy == -1] = 255

        return image, label_copy, name, width, height

    def __len__(self):
        return len(self.items_list)
