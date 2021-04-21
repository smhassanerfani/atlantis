import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io

from scipy import ndimage as ndi
from skimage.filters import gabor_kernel
from skimage.feature import local_binary_pattern

# from skimage.util import img_as_float
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from classifiers import KNearestNeighbor


plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def image_normalization(image):
    # Normalize images separately across the three color channels.
    mean = np.mean(image, axis=tuple(range(image.ndim - 1)))
    std = np.std(image, axis=tuple(range(image.ndim - 1)))
    return (image - mean) / std


def kernel3C(gkernel):
    arrays = [kernel for _ in range(3)]
    return np.stack(arrays, axis=2)


def power(image, kernel, norm=False, as_gray=True):

    if norm:
        image = image_normalization(image)
    if as_gray:
        real_feature = ndi.convolve(image, np.real(kernel), mode='wrap')
        imag_feature = ndi.convolve(image, np.imag(kernel), mode='wrap')
    else:
        kernel = kernel3C(kernel)
        real_feature = ndi.convolve(
            image, np.real(kernel), mode='wrap')[:, :, 1]
        imag_feature = ndi.convolve(
            image, np.imag(kernel), mode='wrap')[:, :, 1]

    return np.sqrt(real_feature**2 + imag_feature**2)


def dataloader(dataset, as_gray=True, rootdir="./data/atex/"):

    dataset = {
        "train": {"data": [], "target": []},
        "test": {"data": [], "target": []},
        "val": {"data": [], "target": []}
    }

    data_sets = ["train", "val", "test"]
    for set_ in data_sets:
        datadir = os.path.join(rootdir, set_)
        counter = 0
        idx = 0
        for root, dirs, files in os.walk(datadir, topdown=True):
            for image in files:
                if image.endswith(".jpg"):
                    dataset[set_]["data"].append(
                        io.imread(os.path.join(root, image), as_gray=as_gray))
                    dataset[set_]["target"].append(counter - 1)
                    idx += 1
            counter += 1
        dataset[set_]["data"] = np.asarray(dataset[set_]["data"])
        dataset[set_]["target"] = np.asarray(dataset[set_]["target"])

    return dataset


############################# ANALYSIS #############################
# visualization of patches
# atex = dataloader("atex", as_gray=False)

# classes = ['waterfall', 'river', 'sea', 'wetland', 'delta', 'pool', 'puddle',
#            'swamp', 'glaciers', 'lake', 'rapids', 'snow', 'estuary', 'flood', 'hot_spring']

# num_classes = len(classes)

# samples_per_class = 7
# for y, cls in enumerate(classes):
#     idxs = np.flatnonzero(atex["train"]["target"] == y)
#     idxs = np.random.choice(idxs, samples_per_class, replace=False)
#     for i, idx in enumerate(idxs):
#         plt_idx = i * num_classes + y + 1
#         plt.subplot(samples_per_class, num_classes, plt_idx)
#         plt.imshow(atex["train"]["data"][idx].astype('uint8'))
#         plt.axis('off')
#         if i == 0:
#             plt.title(cls)
# plt.show()


# # PCA to visualize first two eigenvectors of train data
# atex = dataloader("atex", as_gray=True)
# data = atex["train"]["data"].reshape(atex["train"]["data"].shape[0], -1)

# pca = PCA(n_components=100, random_state=88)
# pca.fit(data)
# x = pca.transform(data)

# fig, axes = plt.subplots(nrows=1, ncols=1)
# axes.scatter(x[:, 0], x[:, 1], c=atex["train"]["target"])
# plt.show()

# exit()

# # KMeans to evaluate the inetria  # change the metric!
# inertia_list = []
# for k in range(2, 16):
#     kmn = KMeans(n_clusters=k, random_state=88)
#     kmn.fit(x)
#     pred = kmn.labels_
#     inertia_list.append(kmn.inertia_)

# fig, axes = plt.subplots(nrows=1, ncols=1)
# axes.plot(np.arange(2, 16), inertia_list, 'o--')
# plt.grid(True)
# plt.show()

# exit()

# # KNN analysis
# # loading data
as_gray = True
norm = False
atex = dataloader("atex", as_gray=as_gray)
# as_gray=True --> results are around 20%
# as_gray=False --> results are around 10%
# preprocessing works! TODO: check the results with normalization
X_train = atex["train"]["data"]
y_train = atex["train"]["target"]
X_val = atex["val"]["data"]
y_val = atex["val"]["target"]


X_train = X_train.reshape(X_train.shape[0], -1)
X_val = X_val.reshape(X_val.shape[0], -1)

# # t-SNE
from tsne import tsne
import pylab
import time

since = time.time()
Y = tsne(X_train, 2, 50, 20.0)
np.savetxt('tsne.txt', Y, delimiter=',')
time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))

pylab.scatter(Y[:, 0], Y[:, 1], 20, y_train)
pylab.show()

exit()
# # PCA
# pca = PCA(n_components=100, random_state=88)
# X_train = pca.fit_transform(X_train)
# X_val = pca.fit_transform(X_val)


# # gabor analysis
sigma = 1
theta = (1 / 4.) * np.pi
frequency = 0.1
kernel = gabor_kernel(frequency, theta=theta, sigma_x=sigma, sigma_y=sigma)

X_train = map(lambda x: power(x, kernel, norm=norm, as_gray=as_gray), X_train)
X_train = np.asarray(list(X_train))

X_val = map(lambda x: power(x, kernel, norm=norm, as_gray=as_gray), X_val)
X_val = np.asarray(list(X_val))

# # lbp analysis
# METHOD = 'uniform'
# radius = 3
# n_points = 8 * radius

# X_train = map(lambda x: local_binary_pattern(
#     x, n_points, radius, METHOD), X_train)
# X_train = np.asarray(list(X_train))

# X_val = map(lambda x: local_binary_pattern(
#     x, n_points, radius, METHOD), X_val)
# X_val = np.asarray(list(X_val))

####################################
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_val = np.reshape(X_val, (X_val.shape[0], -1))

classifier = KNearestNeighbor()
k_choices = [1, 3, 5, 8, 15, 50, 70, 100, 200, 300, 500]
k_to_accuracies = {}

for k in k_choices:
    k_to_accuracies[k] = []

    # use of k-nearest-neighbor algorithm
    classifier.train(X_train, y_train)
    y_pred = classifier.predict(X_val, k=k, method=3)

    # Compute the fraction of correctly predicted examples
    num_correct = np.sum(y_pred == y_val)
    accuracy = float(num_correct) / X_val.shape[0]
    k_to_accuracies[k].append(accuracy)

# Print out the computed accuracies
for k in sorted(k_to_accuracies):
    for accuracy in k_to_accuracies[k]:
        # print('k = %d, accuracy = %f' % (k, accuracy))
        print('%d, %f' % (k, accuracy))
