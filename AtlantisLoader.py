#a
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
        # self.id_to_trainid = {22:  0, 23:  1, 24:  2, 25:  3, 26:  4, 27:  5, 28:  6, 29:  7, 30:  8, 31:  9,
        #                       32: 10, 33: 11, 34: 12, 35: 13, 36: 14, 37: 15, 38: 16, 39: 17, 40: 18, 41: 19,
        #                       42: 20, 43: 21, 44: 22, 45: 23, 46: 24, 47: 25, 48: 26, 49: 27, 50: 28, 51: 29,
        #                       52: 30, 53: 31, 54: 32, 55: 33, 56: 34}


        self.id_to_trainid = { 3:  0,  4:  1,  7:  2,  9:  3, 10:  4, 11:  5, 12:  6, 13:  7, 16:  8, 17:  9,
                              18: 10, 19: 11, 20: 12, 21: 13, 22: 14, 23: 15, 24: 16, 26: 17, 29: 18, 30: 19,
                              32: 20, 33: 21, 34: 22, 35: 23, 36: 24, 38: 25, 39: 26, 40: 27, 43: 28, 44: 29,
                              45: 30, 53: 31, 54: 32, 55: 33, 56: 34}

        if self.split == 'val' or 'test':
            self.padding_size = padding_size

    def get_images_list(self, images_base, masks_base):
        items_list = []
        for root, dirs, files in os.walk(images_base, topdown=True):
            mask_root = os.path.join(masks_base, root.split("/")[-1])
            for name in files:
                if name.endswith(".jpg"):
                    # print(name)
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

        label_copy = 255 * np.ones(label.shape, dtype=np.uint8)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v

        return image, label_copy, name, width, height

    def __len__(self):
        return len(self.items_list)

class Atlantis36DataSet(data.Dataset):
    def __init__(self, rootdir, split, joint_transform=None, padding_size = 0):
        super(Atlantis36DataSet, self).__init__()
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
        # self.id_to_trainid = {22:  0, 23:  1, 24:  2, 25:  3, 26:  4, 27:  5, 28:  6, 29:  7, 30:  8, 31:  9,
        #                       32: 10, 33: 11, 34: 12, 35: 13, 36: 14, 37: 15, 38: 16, 39: 17, 40: 18, 41: 19,
        #                       42: 20, 43: 21, 44: 22, 45: 23, 46: 24, 47: 25, 48: 26, 49: 27, 50: 28, 51: 29,
        #                       52: 30, 53: 31, 54: 32, 55: 33, 56: 34}


        self.id_to_trainid = { 3:  0,  4:  1,  7:  2,  9:  3, 10:  4, 11:  5, 12:  6, 13:  7, 16:  8, 17:  9,
                              18: 10, 19: 11, 20: 12, 21: 13, 22: 14, 23: 15, 24: 16, 26: 17, 29: 18, 30: 19,
                              32: 20, 33: 21, 34: 22, 35: 23, 36: 24, 38: 25, 39: 26, 40: 27, 42: 28, 43: 29,
                              44: 30, 45: 31, 53: 32, 54: 33, 55: 34, 56: 35}

        if self.split == 'val' or 'test':
            self.padding_size = padding_size

    def get_images_list(self, images_base, masks_base):
        items_list = []
        for root, dirs, files in os.walk(images_base, topdown=True):
            mask_root = os.path.join(masks_base, root.split("/")[-1])
            for name in files:
                if name.endswith(".jpg"):
                    # print(name)
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

        label_copy = 255 * np.ones(label.shape, dtype=np.uint8)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v

        return image, label_copy, name, width, height

    def __len__(self):
        return len(self.items_list)


class Atlantis56DataSet(data.Dataset):
    def __init__(self, rootdir, split, joint_transform=None, padding_size = 0):
        super(Atlantis56DataSet, self).__init__()
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
        # self.id_to_trainid = {22:  0, 23:  1, 24:  2, 25:  3, 26:  4, 27:  5, 28:  6, 29:  7, 30:  8, 31:  9,
        #                       32: 10, 33: 11, 34: 12, 35: 13, 36: 14, 37: 15, 38: 16, 39: 17, 40: 18, 41: 19,
        #                       42: 20, 43: 21, 44: 22, 45: 23, 46: 24, 47: 25, 48: 26, 49: 27, 50: 28, 51: 29,
        #                       52: 30, 53: 31, 54: 32, 55: 33, 56: 34}


        self.id_to_trainid = { 3:  0,  4:  1,  7:  2,  9:  3, 10:  4, 11:  5, 12:  6, 13:  7, 16:  8, 17:  9,
                              18: 10, 19: 11, 20: 12, 21: 13, 22: 14, 23: 15, 24: 16, 26: 17, 29: 18, 30: 19,
                              32: 20, 33: 21, 34: 22, 35: 23, 36: 24, 38: 25, 39: 26, 40: 27, 43: 28, 44: 29,
                              45: 30, 53: 31, 54: 32, 55: 33, 56: 34}

        if self.split == 'val' or 'test':
            self.padding_size = padding_size

    def get_images_list(self, images_base, masks_base):
        items_list = []
        for root, dirs, files in os.walk(images_base, topdown=True):
            mask_root = os.path.join(masks_base, root.split("/")[-1])
            for name in files:
                if name.endswith(".jpg"):
                    # print(name)
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


class AWODataSet(data.Dataset):
    def __init__(self, rootdir, split, joint_transform=None, padding_size = 0):
        super(AWODataSet, self).__init__()
        self.rootdir = rootdir
        self.split = split
        self.images_base = os.path.join(self.rootdir, self.split)
        self.items_list = self.get_images_list(self.images_base)

        self.joint_transform = joint_transform
        mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        train_input_transform = []
        train_input_transform += [transforms.ToTensor(),
                                  transforms.Normalize(*mean_std)]
        self.image_transform = transforms.Compose(train_input_transform)

        if self.split == 'val' or 'test':
            self.padding_size = padding_size


        self.label_name_to_id = {'canal': 0, 'ditch':  1,  'estuary':  2,  'fjord':  3, 'flood':  4, 'glaciers':  5,
                                 'hot_spring':  6, 'lake':  7, 'puddle':  8, 'rapids':  9, 'reservoir': 10, 'river': 11,
                                 'river_delta': 12, 'sea': 13, 'snow': 14, 'swamp': 15, 'swimming_pool': 16, 'waterfall': 17, 'wetland': 18}

    def get_images_list(self, images_base):
        items_list = []
        for root, dirs, files in os.walk(images_base, topdown=True):
            for name in files:
                if name.endswith(".jpg"):
                    img_file = os.path.join(root, name)
                    label_name = root.split("/")[-1]
                    items_list.append({
                        "img": img_file,
                        "label": label_name,
                        "name": name
                    })
        return items_list

    def __getitem__(self, index):
        image_path = self.items_list[index]["img"]
        label = self.items_list[index]["label"]
        name = self.items_list[index]["name"]
        image = Image.open(image_path).convert('RGB')
        image_grey = Image.open(image_path).convert('L')
        width, height = image.size

        if self.joint_transform:
            image, image_grey = self.joint_transform(image, image_grey)
        image = self.image_transform(image)

        if self.split == 'val' or self.split == 'test':
            top_pad =  self.padding_size - image.shape[1]
            right_pad =  self.padding_size - image.shape[2]
            image = np.lib.pad(image, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)

        label_copy = 255
        for k, v in self.label_name_to_id.items():
            if label == k:
                label_copy = v

        return image, label_copy, name, width, height

    def __len__(self):
        return len(self.items_list)


class ATEXDataSet(data.Dataset):
    def __init__(self, rootdir, split, joint_transform=None, padding_size = 0):
        super(ATEXDataSet, self).__init__()
        self.rootdir = rootdir
        self.split = split
        self.images_base = os.path.join(self.rootdir, self.split)
        self.items_list = self.get_images_list(self.images_base)

        self.joint_transform = joint_transform
        mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        train_input_transform = []
        train_input_transform += [transforms.ToTensor(),
                                  transforms.Normalize(*mean_std)]
        self.image_transform = transforms.Compose(train_input_transform)

        if self.split == 'val' or 'test':
            self.padding_size = padding_size


        self.label_name_to_id = {'delta': 0, 'estuary':  1,  'flood':  2, 'glaciers':  3,
                                 'hot_spring':  4, 'lake':  5, 'pool':  6, 'puddle':  7, 'rapids': 8, 'river': 9,
                                 'sea': 10, 'snow': 11, 'swamp': 12, 'waterfall': 13, 'wetland': 14}

    def get_images_list(self, images_base):
        items_list = []
        for root, dirs, files in os.walk(images_base, topdown=True):
            for name in files:
                if name.endswith(".jpg"):
                    img_file = os.path.join(root, name)
                    label_name = root.split("/")[-1]
                    items_list.append({
                        "img": img_file,
                        "label": label_name,
                        "name": name
                    })
        return items_list

    def __getitem__(self, index):
        image_path = self.items_list[index]["img"]
        label = self.items_list[index]["label"]
        name = self.items_list[index]["name"]
        image = Image.open(image_path).convert('RGB')
        image_grey = Image.open(image_path).convert('L')
        width, height = image.size

        if self.joint_transform:
            image, image_grey = self.joint_transform(image, image_grey)
        image = self.image_transform(image)

        if self.split == 'val' or self.split == 'test':
            top_pad =  self.padding_size - image.shape[1]
            right_pad =  self.padding_size - image.shape[2]
            image = np.lib.pad(image, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)

        label_copy = 255
        for k, v in self.label_name_to_id.items():
            if label == k:
                label_copy = v

        return image, label_copy, name, width, height

    def __len__(self):
        return len(self.items_list)

class AtlantisvisDataSet(data.Dataset):
    def __init__(self, rootdir, split, joint_transform=None, padding_size = 0):
        super(AtlantisvisDataSet, self).__init__()
        self.rootdir = rootdir
        self.split = split
        self.images_base = os.path.join(self.rootdir, "images", self.split)
        self.masks_base = os.path.join(self.rootdir, "masks", self.split)
        self.items_list = self.get_images_list(self.images_base, self.masks_base)

        self.joint_transform = joint_transform
        mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        train_input_transform = []
        train_input_transform += [transforms.ToTensor()]
        self.image_transform = transforms.Compose(train_input_transform)
        self.label_transform = MaskToTensor()
        # self.id_to_trainid = {22:  0, 23:  1, 24:  2, 25:  3, 26:  4, 27:  5, 28:  6, 29:  7, 30:  8, 31:  9,
        #                       32: 10, 33: 11, 34: 12, 35: 13, 36: 14, 37: 15, 38: 16, 39: 17, 40: 18, 41: 19,
        #                       42: 20, 43: 21, 44: 22, 45: 23, 46: 24, 47: 25, 48: 26, 49: 27, 50: 28, 51: 29,
        #                       52: 30, 53: 31, 54: 32, 55: 33, 56: 34}


        self.id_to_trainid = { 3:  0,  4:  1,  7:  2,  9:  3, 10:  4, 11:  5, 12:  6, 13:  7, 16:  8, 17:  9,
                              18: 10, 19: 11, 20: 12, 21: 13, 22: 14, 23: 15, 24: 16, 26: 17, 29: 18, 30: 19,
                              32: 20, 33: 21, 34: 22, 35: 23, 36: 24, 38: 25, 39: 26, 40: 27, 43: 28, 44: 29,
                              45: 30, 53: 31, 54: 32, 55: 33, 56: 34}

        if self.split == 'val' or 'test':
            self.padding_size = padding_size

    def get_images_list(self, images_base, masks_base):
        items_list = []
        for root, dirs, files in os.walk(images_base, topdown=True):
            mask_root = os.path.join(masks_base, root.split("/")[-1])
            for name in files:
                if name.endswith(".jpg"):
                    # print(name)
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
        image = self.image_transform(image)
        label = self.label_transform(label)

        label_copy = label-1
        label_copy[label_copy == -1] = 255

        return image, label_copy, name, width, height

    def __len__(self):
        return len(self.items_list)
