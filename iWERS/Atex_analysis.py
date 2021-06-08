import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy


def csv_writer(log_list, fieldnames, model_name):
    import csv
    with open(f"./models/{model_name}/results_acc.csv", 'w', newline='') as filehandler:
        fh_writer = csv.DictWriter(filehandler, fieldnames=fieldnames)

        fh_writer.writeheader()
        for item in log_list:
            fh_writer.writerow(item)


"""
CLASS torchvision.transforms.ToTensor:
Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to 
a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
"""
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


class ToHSV:
    def __call__(self, sample):
        # inputs = sample
        return sample.convert('HSV')


data_transforms = {
    'train': transforms.Compose([
        # transforms.RandomResizedCrop(224),
        # ToHSV(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        # ToHSV(),
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

data_dir = "./data/atex"

image_datasets = {x: datasets.ImageFolder(os.path.join(
    data_dir, x), data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(
    image_datasets[x], batch_size=64, shuffle=True, num_workers=2) for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# a = image_datasets["val"][5]
# print(a[0].shape)
# print(type(image_datasets["val"]))

# batch1 = iter(dataloaders["val"])
# images, labels = batch1.next()

# print(dataset_sizes["train"], dataset_sizes["val"])


# def imshow(inp, title):
#     """Imshow for Tensor."""
#     inp = inp.numpy().transpose((1, 2, 0))
#     inp = std * inp + mean
#     inp = np.clip(inp, 0, 1)
#     plt.imshow(inp)
#     plt.title(title)
#     plt.show()


# # Get a batch of training data
# inputs, classes = next(iter(dataloaders['train']))

# # Make a grid from batch
# out = torchvision.utils.make_grid(inputs)
# imshow(out, title=[class_names[x] for x in classes])

def train_model(model, model_name, criterion, optimizer, scheduler=None, num_epochs=30):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    log_list = []
    fieldnames = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]

    for epoch in range(num_epochs):

        log_dic = {}
        log_dic["epoch"] = epoch

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
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # if phase == 'train':
                # scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'train':
                log_dic["train_loss"] = epoch_loss
                log_dic["train_acc"] = epoch_acc.item()

            # deep copy the model
            if phase == 'val':
                log_dic["val_loss"] = epoch_loss
                log_dic["val_acc"] = epoch_acc.item()

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

        log_list.append(log_dic)
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    csv_writer(log_list, fieldnames, model_name)
    return model


#### Finetuning the convnet ####
# Load a pretrained model and reset final fully connected layer.
from torchsummary import summary
from efficientnet_pytorch import EfficientNet
# import pretrainedmodels

model_name = "shufflenet_v2_x1_0_t4"

import os
try:
    os.makedirs(os.path.join("./models", model_name))
except FileExistsError:
    pass


# model = EfficientNet.from_pretrained('efficientnet-b7')
# num_ftrs = model._fc.in_features
# model._fc = nn.Linear(num_ftrs, 15)

model = models.shufflenet_v2_x1_0(pretrained=True)

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 15)

# num_ftrs = model.classifier[6].in_features
# model.classifier[6] = nn.Linear(num_ftrs, 15)

# squeezenet1_0
# model.classifier[1] = nn.Conv2d(512, 15, kernel_size=(1, 1), stride=(1, 1))
# model.num_classes = 15


model = model.to(device)
# summary(model, (3, 32, 32))

# print(model)
# exit()

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer = optim.SGD(model.parameters(), lr=2.5e-4,
                      momentum=0.9, weight_decay=0.0001)
# optimizer = torch.optim.Adam(model.parameters(), lr=2.5e-4)

# step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

model = train_model(model, model_name, criterion, optimizer, num_epochs=30)
exit()

data_transforms = transforms.Compose([
    # ToHSV(),
    # transforms.Resize(256),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

data_dir = "./data/atex"
batch_size = 1

test_dataset = datasets.ImageFolder(
    os.path.join(data_dir, 'test'), data_transforms)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

test_dataset_size = len(test_dataset)
class_names = image_datasets['train'].classes

# batch1 = iter(test_loader)
# images, labels = batch1.next()
# print(images)

FILE = f"./models/{model_name}/model.pth"

checkpoint = torch.load(FILE)
model.load_state_dict(checkpoint['model_state'])
optimizer.load_state_dict(checkpoint['optimizer_state'])
# scheduler.load_state_dict(checkpoint['scheduler_state'])
epoch = checkpoint['epoch']

# print(epoch)
# exit()

model.to(device)
model.eval()

log_dic = {}

y_true = []
y_pred = []

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(15)]
    n_class_samples = [0 for i in range(15)]
    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        y_true.append(labels.item())
        y_pred.append(predicted.item())

from sklearn.metrics import classification_report
final_report = classification_report(y_true, y_pred, target_names=class_names)

with open("./cool_dogs.txt", "a") as cool_dogs_file:
    cool_dogs_file.write(final_report)
print("finish")
#         for i in range(labels.size(0)):
#             label = labels[i]
#             pred = predicted[i]
#             if (label == pred):
#                 n_class_correct[label] += 1
#             n_class_samples[label] += 1

#     acc = 100.0 * n_correct / n_samples
#     # print(f"Accuracy of the network: {acc} %")
#     log_dic['network'] = acc

#     for i in range(15):
#         acc = 100.0 * n_class_correct[i] / n_class_samples[i]
#         log_dic[class_names[i]] = acc
#         # print(f"Accuracy of {class_names[i]}: {acc:.2f} %")

# import csv
# with open(f'./models/{model_name}/acc_results.csv', 'w') as f:
#     w = csv.writer(f)
#     w.writerows(log_dic.items())
#     # w.writerow(log_dic.values())
# print(log_dic)
exit()

from PIL import Image
# pic = Image.open('./data/atex/val/delta/28981184021.x000.y352.jpg')
# pic = pic.resize((256, 256), resample=Image.BICUBIC)
# pic.show()
# exit()

tensor = model.conv1.weight.clone()
tensor = tensor.cpu()

# tensor1 = tensor[6].detach().numpy()
# tensor1 = tensor1.transpose((1, 2, 0))
# mean = np.mean(tensor1, axis=tuple(range(tensor1.ndim - 1)))
# std = np.std(tensor1, axis=tuple(range(tensor1.ndim - 1)))
# print(mean)
# print(std)
# tensor1 = (tensor1 - mean) / std
# # print(tensor1)
# filter1 = Image.fromarray(tensor1.astype(np.uint8))

# # PIL.Image.NEAREST (use nearest neighbour)
# # PIL.Image.BILINEAR (linear interpolation)
# # PIL.Image.BICUBIC (cubic spline interpolation)
# filter1 = filter1.resize((28, 28), resample=Image.BICUBIC)
# filter1.show()

# exit()


def visTensor(tensor, ch=0, allkernels=False, nrow=8, padding=1):
    n, c, w, h = tensor.shape

    if allkernels:
        tensor = tensor.view(n * c, -1, w, h)
    elif c != 3:
        tensor = tensor[:, ch, :, :].unsqueeze(dim=1)

    rows = np.min((tensor.shape[0] // nrow + 1, 64))
    grid = torchvision.utils.make_grid(
        tensor, nrow=nrow, normalize=True, padding=padding)
    plt.figure(figsize=(nrow, rows))
    plt.imshow(grid.detach().numpy().transpose((1, 2, 0)))

    plt.axis('off')
    plt.ioff()
    plt.show()

# if __name__ == "__main__":
#     filter = model_conv.layer1[0].conv1.weight.clone()
#     print(filter.shape)
#     visTensor(filter.cpu(), ch=0, allkernels=False)

#     plt.axis('off')
#     plt.ioff()
#     plt.show()


visTensor(tensor.cpu())
