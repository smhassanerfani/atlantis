import os
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage import io

from scipy import ndimage as ndi
from skimage.filters import gabor_kernel
from skimage.feature import local_binary_pattern
from skimage.color import rgb2hsv
from skimage.measure import block_reduce

# from skimage.util import img_as_float
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from classifiers import KNearestNeighbor
from tsne import tsne

plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
# plt.rcParams['image.cmap'] = 'gray'


def image_normalization(dataset, as_gray=False):
    # # Normalize images separately across the three color channels.
    # mean = np.mean(image, axis=tuple(range(image.ndim - 1)))
    # std = np.std(image, axis=tuple(range(image.ndim - 1)))
    shape = dataset.shape
    dataset = dataset.reshape(dataset.shape[0], -1)
    dataset = dataset.astype(np.float)
    dataset -= np.mean(dataset, axis=0)
    dataset /= np.std(dataset, axis=0)
    if as_gray:
        return dataset.reshape(dataset.shape[0], shape[1], shape[2])
    return dataset.reshape(dataset.shape[0], shape[1], shape[2], shape[3])


def kernel3C(gkernel):
    arrays = [kernel for _ in range(3)]
    return np.stack(arrays, axis=2)


def power(image, kernel, norm=False, as_gray=False):

    if norm:
        image = image_normalization(image, as_gray=as_gray)
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


def dataloader(dataset, as_gray=False, norm=False, hsv=False, rootdir="./data/atex/"):

    dataset = {
        "train": {"data": [], "target": []},
        "test": {"data": [], "target": []},
        "val": {"data": [], "target": []}
    }

    # data_sets = ["train", "val", "test"]
    for set_ in tqdm(dataset.keys()):
        datadir = os.path.join(rootdir, set_)
        counter = 0
        # idx = 0
        for root, dirs, files in os.walk(datadir, topdown=True):
            for image in files:
                if image.endswith(".jpg"):
                    dataset[set_]["data"].append(
                        io.imread(os.path.join(root, image), as_gray=as_gray))
                    dataset[set_]["target"].append(counter - 1)
                    # idx += 1
            counter += 1
        dataset[set_]["data"] = np.asarray(dataset[set_]["data"])
        dataset[set_]["target"] = np.asarray(dataset[set_]["target"])
        if norm:
            dataset[set_]["data"] = image_normalization(
                dataset[set_]["data"], as_gray=as_gray)
        if hsv:
            dataset[set_]["data"] = rgb2hsv(dataset[set_]["data"])

    return dataset


############################# ANALYSIS #############################
# visualization of patches
# atex = dataloader("atex", as_gray=False, norm=False, hsv=False)

classes = ['pool', 'flood', 'hot_spring', 'waterfall', 'lake', 'snow', 'rapids',
           'river', 'glaciers', 'puddle', 'sea', 'delta', 'estuary', 'wetland', 'swamp']

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

# exit()
# # PCA to visualize first two eigenvectors of train data
# atex = dataloader("atex", as_gray=False, norm=False, hsv=False)
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
as_gray = False
norm = False
atex = dataloader("atex", as_gray=as_gray, norm=norm, hsv=True)

X_train = atex["train"]["data"]
y_train = atex["train"]["target"]
X_val = atex["val"]["data"]
y_val = atex["val"]["target"]

# # t-SNE

X_train = X_train.reshape(X_train.shape[0], -1)
X_val = X_val.reshape(X_val.shape[0], -1)


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
        ax.scatter(Y[iidx:cidx, 0],
                   Y[iidx:cidx:, 1], label=class_, marker=markers[idx_])
    ax.legend()
    ax.grid(True)

    plt.show()


# path = "./tsne3D_hsv_val_1000.txt"
# data = np.loadtxt(path, delimiter=',')
# print(data.shape)

since = time.time()
Y = tsne(X_train, 3, 50, 20.0)
np.savetxt('./tsne3D_hsv_train_1000.txt', Y, delimiter=',')
time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))


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
        axis.scatter(Y[iidx:cidx, 0],
                     Y[iidx:cidx:, 1], Y[iidx:cidx:, 2], label=class_, marker=markers[idx_])

    axis.set_xlabel(r"$1^{st}$ dim")
    axis.set_ylabel(r"$2^{nd}$ dim")
    axis.set_zlabel(r"$3^{rd}$ dim")
    axis.legend()
    axis.grid(True)

    plt.show()


tsne_3dplot(Y, y_train)
# Y = np.loadtxt("./models/tsne/tsne_hsv_train_1000.txt", delimiter=',')
# tsne_plot(Y, y_val)


exit()
# # PCA
# pca = PCA(n_components=100, random_state=88)
# X_train = pca.fit_transform(X_train)
# X_val = pca.fit_transform(X_val)


# # gabor analysis
# gjet_train = []
# gjet_val = []
# sigma = np.pi
# for mu in tqdm([0, 1, 2, 3, 4, 5, 6, 7]):
#     for nu in tqdm([0, 1, 2, 3, 4]):

#         theta = (mu / 8.) * np.pi
#         frequency = (np.pi / 2) / (np.sqrt(2)) ** nu
#         kernel = gabor_kernel(frequency, theta=theta,
#                               sigma_x=sigma, sigma_y=sigma)

#         X_train = map(lambda x: power(
#             x, kernel, norm=norm, as_gray=as_gray), X_train_)
#         X_train = map(lambda x: block_reduce(x, (2, 2), func=np.max), X_train)
#         X_train = np.asarray(list(X_train))
#         gjet_train.append(X_train)

#         X_val = map(lambda x: power(
#             x, kernel, norm=norm, as_gray=as_gray), X_val_)
#         X_val = map(lambda x: block_reduce(x, (2, 2), func=np.max), X_val)
#         X_val = np.asarray(list(X_val))
#         gjet_val.append(X_val)
#         # print(f"number of kernel: {len(gjet_val)}")

# gjet_train = np.array(gjet_train)
# gjet_train = gjet_train.transpose(1, 2, 3, 0)
# print(gjet_train.shape)

# gjet_val = np.array(gjet_val)
# gjet_val = gjet_val.transpose(1, 2, 3, 0)
# print(gjet_val.shape)

# X_train = gjet_train.reshape(gjet_train.shape[0], -1)
# np.savetxt('gjet_hsv_train.txt', X_train, delimiter=',')
# print(X_train.shape)

# X_val = gjet_val.reshape(gjet_val.shape[0], -1)
# np.savetxt('gjet_hsv_val.txt', X_val, delimiter=',')
# print(X_val.shape)

# pca = PCA(n_components=500, random_state=88)
# X_train = pca.fit_transform(X_train)
# np.savetxt('gjet_hsv_train_pca_500.txt', X_train, delimiter=',')
# print(X_train.shape)
# X_val = pca.fit_transform(X_val)
# np.savetxt('gjet_hsv_val_pca_500.txt', X_val, delimiter=',')
# print(X_val.shape)


# # lbp analysis
METHOD = 'uniform'
radius = 1
n_points = 8 * radius

X_train = map(lambda x: local_binary_pattern(
    x, n_points, radius, METHOD), X_train)
X_train = np.asarray(list(X_train))
print(X_train.shape)

X_val = map(lambda x: local_binary_pattern(
    x, n_points, radius, METHOD), X_val)
X_val = np.asarray(list(X_val))

# ####################################
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_val = np.reshape(X_val, (X_val.shape[0], -1))

# X_train = np.loadtxt("./models/tsne/gjet_train_pca_500.txt", delimiter=',')
# X_val = np.loadtxt("./models/tsne/gjet_val_pca_500.txt", delimiter=',')

classifier = KNearestNeighbor()
k_choices = [1, 3, 5, 8, 15, 50, 70, 100, 200, 300, 500]
k_to_accuracies = {}

for k in tqdm(k_choices):
    k_to_accuracies[k] = []

    # use of k-nearest-neighbor algorithm
    classifier.train(X_train, y_train)
    y_pred = classifier.predict(X_val, k=k, method=2)

    # Compute the fraction of correctly predicted examples
    num_correct = np.sum(y_pred == y_val)
    accuracy = float(num_correct) / X_val.shape[0]
    k_to_accuracies[k].append(accuracy)

# Print out the computed accuracies
for k in sorted(k_to_accuracies):
    for accuracy in k_to_accuracies[k]:
        # print('k = %d, accuracy = %f' % (k, accuracy))
        print('%d, %f' % (k, accuracy))
