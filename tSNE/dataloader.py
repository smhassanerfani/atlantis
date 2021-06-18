import os
import torch
from torch import optim, nn
import numpy as np
from torchvision import models, datasets, transforms
import matplotlib.pyplot as plt

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
    image_datasets[x], batch_size=1, shuffle=False, num_workers=2) for x in ['train', 'val']}

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


# # Initialize the model
model = models.vgg16(pretrained=True)

num_ftrs = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_ftrs, 15)

FILE = "/home/serfani/Documents/Microsoft_project/iWERS/models/vgg16/model.pth"

checkpoint = torch.load(FILE)
model.load_state_dict(checkpoint['model_state'])
# optimizer.load_state_dict(checkpoint['optimizer_state'])
# scheduler.load_state_dict(checkpoint['scheduler_state'])
# epoch = checkpoint['epoch']

model.to(device)
model.eval()

new_model = FeatureExtractor(model)

print(new_model)
exit()

features = []
_labels = []
# Change the device to GPU
new_model = new_model.to(device)

from tqdm import tqdm

for inputs, labels in tqdm(dataloaders['train']):

    inputs = inputs.to(device)
    _labels.append(labels.item())
    with torch.no_grad():
        # feature = new_model(inputs)
        # features.append(feature.cpu().detach().numpy().reshape(-1))
        features.append(inputs.cpu().detach().numpy().reshape(-1))

features = np.asarray(features)
_labels = np.asarray(_labels)

print("Output of VGG ftrs:")
print(features.shape, _labels.shape)


def tsne_plot(Y, labels, classes=class_names):
    NUM_COLORS = len(classes)

    # plt.rcParams['image.cmap'] = 'nipy_spectral'
    cm = plt.get_cmap('tab20', lut=NUM_COLORS)
    cidx = 0
    markers = ["o", ".", "D", "x", "X", "d", "*",
               "<", "s", "+", "^", "P", "p", "4", ">"]
    fig, ax = plt.subplots()

    fig.set_figheight(10)
    fig.set_figwidth(10)
    ax.set_prop_cycle(color=[cm(i) for i in range(NUM_COLORS)])
    for idx_, class_ in enumerate(classes):
        idx = np.sum(idx_ == labels)
        cidx += idx
        iidx = cidx - idx
        ax.scatter(Y[iidx: cidx, 0],
                   Y[iidx:cidx:, 1], label=class_, marker=markers[idx_])
    ax.legend()
    ax.grid(True)

    plt.show()

# def tsne_plot(features, labels, classes=list()):

#     import matplotlib.patches as mpatches

#     cmap = plt.get_cmap('tab20c', lut=len(classes))

#     fig, ax = plt.subplots()

#     fig.set_figheight(10)
#     fig.set_figwidth(10)
#     patches = [mpatches.Patch(color=cmap(idx), label=name)
#                for idx, name in enumerate(classes)]

#     ax.scatter(features[:, 0], features[:, 1], 50, labels, cmap=cmap)
#     ax.legend(handles=patches, loc='upper right')
#     plt.show()


# PCA to visualize first two eigenvectors of train data
from sklearn.decomposition import PCA
pca = PCA(n_components=500, random_state=88)
pca.fit(features)
features = pca.transform(features)

print(features.shape)


from sklearn import manifold
from time import time

n_components = 2
perplexities = [20]


for i, perplexity in enumerate(perplexities):

    t0 = time()
    tsne = manifold.TSNE(n_components=n_components, n_iter=10000, method='exact', init='random',
                         random_state=0, perplexity=perplexity, learning_rate=200, verbose=2)  # method= 'barnes_hut'
    Y = tsne.fit_transform(features)
    t1 = time()
    print("perplexity=%d in %.2g sec" % (perplexity, t1 - t0))
    np.savetxt('./train_tsne_ftrs.txt', Y, delimiter=',')
    # np.savetxt('./train_labels.txt', _labels, delimiter=',')

    tsne_plot(Y, _labels, classes=class_names)
exit()
