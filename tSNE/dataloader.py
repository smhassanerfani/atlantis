import os
import torch
from torch import optim, nn
import numpy as np
from torchvision import models, datasets, transforms

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


class ToHSV:
    def __call__(self, sample):
        # inputs = sample
        return sample.convert('HSV')


data_transforms = {
    'train': transforms.Compose([
        # ToHSV(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        # ToHSV(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

data_dir = "/home/serfani/Documents/Microsoft_project/iWERS/data/atex"


image_datasets = {x: datasets.ImageFolder(os.path.join(
    data_dir, x), data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(
    image_datasets[x], batch_size=1, shuffle=True, num_workers=2) for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        # Extract VGG-16 Feature Layers
        self.features = list(model.features)
        self.features = nn.Sequential(*self.features)
        # Extract VGG-16 Average Pooling Layer
        self.pooling = model.avgpool
        # Convert the image into one-dimensional vector
        self.flatten = nn.Flatten()
        # Extract the first part of fully-connected layer from VGG16
        self.fc = model.classifier[0]

    def forward(self, x):
        # It will take the input 'x' until it returns the feature vector called 'out'
        out = self.features(x)
        out = self.pooling(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out


# Initialize the model
model = models.vgg16(pretrained=True)

num_ftrs = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_ftrs, 15)

FILE = "/home/serfani/Documents/Microsoft_project/iWERS/models/vgg16/model.pth"

checkpoint = torch.load(FILE)
model.load_state_dict(checkpoint['model_state'])
# optimizer.load_state_dict(checkpoint['optimizer_state'])
# scheduler.load_state_dict(checkpoint['scheduler_state'])
# epoch = checkpoint['epoch']

# print(epoch)
# exit()

model.to(device)
model.eval()


new_model = FeatureExtractor(model)


features = []
_labels = []
# Change the device to GPU
new_model = new_model.to(device)

for inputs, labels in dataloaders['val']:
    # print(inputs.shape)
    inputs = inputs.to(device)
    labels = labels.to(device)
    with torch.no_grad():
        feature = new_model(inputs)
        features.append(feature.cpu().detach().numpy().reshape(-1))
        _labels.append(labels.cpu().numpy())


features = np.asarray(features)
_labels = np.asarray(_labels).reshape(-1)
# print(_labels)
# exit()
# print(features.shape)


classes = ['pool', 'flood', 'hot_spring', 'waterfall', 'lake', 'snow', 'rapids',
           'river', 'glaciers', 'puddle', 'sea', 'delta', 'estuary', 'wetland', 'swamp']


def tsne_plot(Y, labels, classes=classes):
    NUM_COLORS = len(classes)
    cm = plt.get_cmap('gist_rainbow')
    cidx = 0
    fig, ax = plt.subplots()
    markers = ["o", "x", "*", "+", 'd', "o", "x",
               "*", "+", 'd', "o", "x", "*", "+", 'd']
    ax.set_prop_cycle(color=[cm(1. * i / NUM_COLORS)
                             for i in range(NUM_COLORS)])
    for idx_, class_ in enumerate(classes):
        idx = np.sum(idx_ == labels)
        cidx += idx
        iidx = cidx - idx
        # print(iidx, cidx)
        ax.scatter(Y[iidx: cidx, 0],
                   Y[iidx:cidx:, 1], label=class_, marker=markers[idx_])
    ax.legend()
    ax.grid(True)

    plt.show()


def tsne_3dplot(Y, labels, classes=classes):
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib import colors

    NUM_COLORS = len(classes)
    cm = plt.get_cmap('gist_rainbow')
    cidx = 0

    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")

    markers = ["o", "x", "*", "+", 'd', "o", "x",
               "*", "+", 'd', "o", "x", "*", "+", 'd']
    axis.set_prop_cycle(color=[cm(1. * i / NUM_COLORS)
                               for i in range(NUM_COLORS)])
    for idx_, class_ in enumerate(classes):
        idx = np.sum(idx_ == labels)
        cidx += idx
        iidx = cidx - idx
        # print(iidx, cidx)
        axis.scatter(Y[iidx: cidx, 0],
                     Y[iidx:cidx:, 1], Y[iidx:cidx:, 2], label=class_, marker=markers[idx_])

    axis.set_xlabel(r"$1^{st}$ dim")
    axis.set_ylabel(r"$2^{nd}$ dim")
    axis.set_zlabel(r"$3^{rd}$ dim")
    axis.legend()
    axis.grid(True)

    plt.show()


# import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.ticker import NullFormatter
from sklearn import manifold
# from time import time


n_components = 3
# (fig, subplots) = plt.subplots(1, 5, figsize=(15, 8))
perplexities = [5, 10, 20, 30]


for i, perplexity in enumerate(perplexities):
    # ax = subplots[0][i + 1]

    t0 = time.time()
    tsne = manifold.TSNE(n_components=n_components, n_iter=2000, method='exact', init='random',
                         random_state=0, perplexity=perplexity, learning_rate=200, verbose=1)
    Y = tsne.fit_transform(features)
    t1 = time.time()
    print("perplexity=%d in %.2g sec" % (perplexity, t1 - t0))
    # ax.set_title("Perplexity=%d" % perplexity)
    tsne_3dplot(Y, _labels, classes=classes)
    # ax.scatter(Y[red, 0], Y[red, 1], c="r")
    # ax.scatter(Y[green, 0], Y[green, 1], c="g")
    # ax.xaxis.set_major_formatter(NullFormatter())
    # ax.yaxis.set_major_formatter(NullFormatter())
    # ax.axis('tight')
