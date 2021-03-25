import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision import models
# from torchsummary import summary
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import os
import copy

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# RGB codes Based on Class Color IDs  
palette = [0,0,0,0,64,64,64,64,192,23,46,206,224,128,0,32,64,0,32,64,128,48,141,46,32,128,192,4,125,181,20,56,144,12,51,91,125,209,109,127,196,133,32,
    224,128,224,224,0,73,218,110,82,8,150,56,86,248,23,113,200,32,55,235,42,174,171,160,128,96,254,232,244,28,192,189,128,160,160,118,164,54,192,
    32,224,192,96,96,221,88,16,169,86,134,96,160,160,92,244,86,190,154,26,58,83,207,160,96,224,138,200,187,32,224,224,80,128,128,118,158,248,254,
    54,89,16,128,64,80,128,64,16,64,64,24,139,177,176,0,64,240,64,64,80,32,128,80,160,128,144,96,128,208,96,128,144,32,192,208,32,192,214,141,3,
    254,23,113,39,24,186,84,91,147]

# zero_pad = 256 * 3 - len(palette)
# for i in range(zero_pad):
#     palette.append(0)
#     # name id  color

class Atlantis(Dataset):

    def __init__(self, rootdir="./data/atlantis/", split="train", transform=None):
        super(Atlantis, self).__init__()
        self.rootdir = rootdir
        self.split = split
        self.transform = transform
        self.images_base = os.path.join(self.rootdir, "images", self.split)
        self.masks_base = os.path.join(self.rootdir, "masks", self.split)
        self.items_list = self.get_images_list()

    def get_images_list(self):
        items_list = []
        for root, dirs, files in os.walk(self.images_base, topdown=True):
            mask_root = os.path.join(self.masks_base, root.split("/")[-1])
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

    def image_resize(self, img, new_size=512):
        alpha = random.uniform(1.0, 2.0)
        width, height = img.size
        w = width
        h = height
        if width < new_size:
            w = new_size
            h = height * (w / width)
            if h < new_size:
                w = w * (new_size / h)
                h = new_size
        elif height < new_size:
            h = new_size
            w = width * (h / height)
            if w < new_size:
                h = h * (new_size / w)
                w = new_size
        return int(alpha * w), int(alpha * h)

    def image_padding(self, img, new_size=700):
        w, h = img.size
        right = new_size - w
        bottom = new_size - h
        # left, top, right, bottom = (20, 30, 40, 50)
        dimg = ImageOps.expand(img, border=(0, 0, right, bottom), fill=0)
        return dimg

    def __getitem__(self, index):
        image_path = self.items_list[index]["img"]
        mask_path = self.items_list[index]["label"]
        name = self.items_list[index]["name"]
        image = Image.open(image_path).convert('RGB')
        self.width, self.height = image.size
        mask = Image.open(mask_path)

        if (self.split == "val" or self.split == "test"):
            image = self.image_padding(image)
            mask = self.image_padding(mask)

        elif self.split == "train":
            resize = transforms.Resize(self.image_resize(image), interpolation=Image.NEAREST)
            image = resize(image)
            mask = resize(mask)

            # we used this instead of TF.random_crop becasue the cropping parameters
            # should be the same for both image and mask
            i, j, h, w = transforms.RandomCrop.get_params(
                image, output_size=(512, 512))
            image = TF.crop(image, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)

            # Random horizontal flipping
            if random.random() > 0.3:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # Random vertical flipping
            # if random.random() > 0.1:
            #     image = TF.vflip(image)
            #     mask = TF.vflip(mask)

        # Transform to tensor
        image = TF.to_tensor(image)
        # mask = F.to_tensor(mask)
        mask = torch.from_numpy(np.array(mask))

        if self.transform:
            image = self.transform(image)

        return image, mask, name, self.width, self.height

    def __len__(self):
        return len(self.items_list)


data_transforms = {
    'train': transforms.Compose([
        # transforms.RandomCrop((300,300)),
        # Image.NEAREST, Image.BILINEAR, Image.BICUBIC and Image.ANTIALIAS
        # transforms.Resize((500, 500), interpolation=Image.NEAREST),
        # transforms.RandomHorizontalFlip(),
        # transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # torchvision.transforms.Normalize(mean, std, inplace=False)
    ]),
    'val': transforms.Compose([
        # transforms.Resize((500, 500)),
        # transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]), 
   'test': transforms.Compose([
        # transforms.Resize((500, 500)),
        # transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Hyper-parameters
num_workers = 2
batch_size = 2
num_epochs = 100
learning_rate = 1e-6

image_datasets = {x: Atlantis(split=x, transform=data_transforms[x])
                  for x in ['train', 'val', 'test']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, drop_last=False) for x in ['train', 'val', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
# CHECKING THE IMAGES AND MASKS

def imshow(inp, title="Image"):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = 0.229 * inp + 0.485
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.title("title")
    plt.show()

def colorize_mask(mask):
    mask = mask.numpy()
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask

# print(f"length of the data: {len(dataloaders['val']) * batch_size}")
# dataiter = iter(dataloaders['val'])
# images, labels, names, w, h = dataiter.next()
# print("images: (batch size, channels, Height, Width)")
# print(f"image feeder: {images.size(), images.dtype}")
# # print(images.max(), images.min())
# print(names)
# print(w, h)
# imshow(images[0])

# print("masks: (batch size, Height, Width)")
# print(f"mask feeder: {labels.size(), labels.dtype}")
# # print(labels.max(), labels.min())
# # print(np.unique(labels.numpy())) # set batch_size to 1
# # print(labels)

# # imshow(images[0])
# colorize_mask(labels[0]).show()
# exit()

# TRAINING
def csv_writer(log_list, fieldnames, model_name):
    import csv
    with open(f"./models/{model_name}/output.csv", 'w', newline='') as filehandler:
        fh_writer = csv.DictWriter(filehandler, fieldnames=fieldnames)

        fh_writer.writeheader()
        for item in log_list:
            fh_writer.writerow(item)

def one_hot(index, classes):
    size = index.size()[:1] + (classes,)
    view = index.size()[:1] + (1,)
    mask = torch.Tensor(size).fill_(0).cuda()
    index = index.view(view)
    ones = 1.
    return mask.scatter_(1, index, ones)

class Focalloss(nn.Module):
    def __init__(self, num_classes=19, gamma=0, eps=1e-7, size_average=True, one_hot=True, ignore_index=255, weight=None):
        super(Focalloss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.classs = num_classes
        self.size_average = size_average
        self.num_classes = num_classes
        self.one_hot = one_hot
        self.ignore_index = ignore_index
        self.weights = weight


    def forward(self, input, target, eps=1e-5):
        B, C, H, W = input.size()
        input = input.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C

        target = target.view(-1)
        if self.ignore_index is not None:
            valid = (target != self.ignore_index)
            input = input[valid]
            target = target[valid]

        target_onehot = one_hot(target, input.size(1))

        probs = F.softmax(input, dim=1)
        if self.weights != None:
            probs = (self.weights * probs * target_onehot).sum(1)
        else:
            probs = (probs * target_onehot).sum(1)
        probs = probs.clamp(self.eps, 1. - self.eps)
        log_p = probs.log()

        batch_loss = -(torch.pow((1 - probs), self.gamma)) * log_p
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


def train_model(model, model_name, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    log_list = []

    fieldnames = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]

    for epoch in range(num_epochs):
        log_dict = {}
        log_dict["epoch"] = epoch

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels, _, _, _ in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels - 1
                labels[labels == -1] = 255
                labels = labels.long()
                labels = labels.to(device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs) 
                    # output: OrderedDict([('out', tensor()), ('aux', tensor())])
                    _, preds = torch.max(outputs['out'], 1)
                    # print(outputs['out'].size())
                    # exit()
                    loss = criterion(outputs['out'], labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)

            # if phase == 'train':
            #     scheduler.step()

            # taking average over the whole dataset for each epoch  
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / (dataset_sizes[phase] * labels.size(1) * labels.size(2))

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'train':
                log_dict["train_loss"] = epoch_loss
                log_dict["train_acc"] = epoch_acc.item()

            # deep copy the model
            if phase == 'val':
                log_dict["val_loss"] = epoch_loss
                log_dict["val_acc"] = epoch_acc.item()
                log_list.append(log_dict)

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

                    state = {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        # "scheduler_state": scheduler.state_dict(),
                        "best_acc": epoch_acc,
                    }
                    save_path = f"./models/{model_name}/model.pth"
                    torch.save(state, save_path)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    csv_writer(log_list, fieldnames, model_name)
    return model


from torchvision.models.segmentation.deeplabv3 import DeepLabHead

model_name = "deeplabv3_resnet50_ndata"

import os
try:
    os.makedirs(os.path.join("./models", model_name))
except FileExistsError:
    pass

model = models.segmentation.deeplabv3_resnet50(pretrained=True, progress=True)
model.classifier = DeepLabHead(2048, 56)

# FILE = "./models/deeplabv3_resnet50/model.pth"
# checkpoint = torch.load(FILE)
# model.load_state_dict(checkpoint['model_state'])

model.to(device)

# criterion = nn.CrossEntropyLoss(ignore_index=255)
criterion = Focalloss(num_classes=56, gamma=2, ignore_index=255)

# Observe that all parameters are being optimized
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

# model = train_model(model, model_name, criterion, optimizer,
#                     scheduler=None, num_epochs=num_epochs)

############################################ TESTING PART ###################################
phase = 'val'
FILE = f"./models/{model_name}/model.pth"
# it takes the loaded dictionary, not the path file itself
checkpoint = torch.load(FILE)
model.load_state_dict(checkpoint['model_state'])
# optimizer.load_state_dict(checkpoint['optimizer_state'])
# scheduler.load_state_dict(checkpoint['scheduler_state'])
epoch = checkpoint['epoch']

model.to(device)
model.eval()

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true > 0) & (label_true < n_class)
    hist = np.bincount(n_class * label_true[mask].astype(
        int) + label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def label_accuracy_score(label_trues, label_preds, n_class=56):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist) / (
            hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        )
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return (acc, acc_cls, mean_iu, fwavacc)

import operator
from itertools import starmap

def imsave(ibatch, names, title="Image"):
    for indx, image in enumerate(ibatch):
        image = image.numpy().transpose((1, 2, 0))
        image = np.array([[[0.229, 0.224, 0.225]]]) * image + np.array([[[0.485, 0.456, 0.406]]])
        image = np.clip(image, 0, 1)
        plt.imshow(image)
        plt.savefig(names[indx])

def save_mask(mbatch, names, width, height, flag="gt"):
    for indx, mask in enumerate(mbatch):
        mask = mask.numpy()
        if flag == "gt":
            mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
            mask.putpalette(palette)
            name = names[indx].split(".")[0] + "_gt.png"    
        else:
            mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
            mask.putpalette(palette)
            # mask = mask.resize((width[indx].item(), height[indx].item()), resample=Image.NEAREST)
            mask = mask.crop((0, 0, width[indx].item(), height[indx].item()))
            name = names[indx].split(".")[0] + "_pred.png"
        path = f"./models/{model_name}/predictions"
        print(f"{path}/{name}")  
        mask.save(f"{path}/{name}")

acc = 0
acc_cls = 0
mean_iu = 0
fwavacc = 0
running_corrects = 0
with torch.no_grad():
    for images, labels, names, w, h in dataloaders[phase]:
        
        # imshow(images[0])
        # imsave(images, names)
        # save_mask(labels, names, w, h)

        labels = labels - 1
        labels [labels == -1] = 255
        images = images.to(device)
        # labels = labels.to(device)

        outputs = model(images)
        # max returns (value ,index)
        _, preds = torch.max(outputs['out'], 1)
        preds = preds.to('cpu')
        # save_mask(preds, names, w, h, flag="pred")

        running_corrects += torch.sum(preds == labels)
        labels = labels.numpy()
        preds = preds.numpy()

        acc, acc_cls, mean_iu, fwavacc = starmap(operator.add, zip((acc, acc_cls, mean_iu, fwavacc),
            label_accuracy_score(labels, preds)))

    epoch_acc = running_corrects.double() / (dataset_sizes[phase] * labels.shape[1] * labels.shape[2])
    print(epoch_acc)
    print(f"acc: {100 * acc / len(dataloaders[phase]):.4f}")
    print(f"acc_cls: {100 * acc_cls / len(dataloaders[phase]):.4f}")
    print(f"mean_iu: {100 * mean_iu / len(dataloaders[phase]):.4f}")
    print(f"fwavacc: {100 * fwavacc / len(dataloaders[phase]):.4f}")

    # dataiter = iter(dataloaders[phase])
    # images, labels, names, w, h = dataiter.next()
    
    # imshow(images[0])
    # colorize_mask(labels[0]).show()
    # labels = labels - 1
    # labels [labels == -1] = 255
    # images = images.to(device)
    # # labels = labels.to(device)

    # outputs = model(images)
    # _, preds = torch.max(outputs['out'], 1)
    # preds = preds.to('cpu')
    # colorize_mask(preds[0]).show()
    # print(np.unique(labels[0].numpy()), np.unique(preds[0].numpy()))