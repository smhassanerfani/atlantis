import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.feature import local_binary_pattern
import os

plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


rootdir = "./data/atex/"
xatex = {}
yatex = {}
lbp = {}

xatex["train"] = np.zeros([8753, 32, 32])
xatex["val"] = np.zeros([1252, 32, 32])
xatex["test"] = np.zeros([2498, 32, 32])

lbp["train"] = np.zeros([8753, 32, 32])
lbp["val"] = np.zeros([1252, 32, 32])
lbp["test"] = np.zeros([2498, 32, 32])

yatex["train"] = np.zeros(8753)
yatex["val"] = np.zeros(1252)
yatex["test"] = np.zeros(2498)


METHOD = 'uniform'
radius = 3
n_points = 8 * radius

atex_sets = ["train", "val", "test"]

for path in atex_sets:
    atexdir = os.path.join(rootdir, path)
    counter = 0
    idx = 0
    for root, dirs, files in os.walk(atexdir, topdown=True):
        # print(counter)
        for image in files:
            # print(idx)
            if image.endswith(".jpg"):
                xatex[path][idx] = io.imread(
                    os.path.join(root, image), as_gray=True)
                lbp[path][idx] = local_binary_pattern(
                    xatex[path][idx], n_points, radius, METHOD)
                yatex[path][idx] = counter
                idx += 1
        counter += 1


X_train = lbp["train"]
y_train = yatex["train"].astype(int) - 1
X_val = lbp["val"]
y_val = yatex["val"].astype(int) - 1
X_test = lbp["test"]
y_test = yatex["test"].astype(int) - 1

classes = ['pool', 'flood', 'hot_spring', 'waterfall', 'lake', 'snow', 'rapids',
           'river', 'glaciers', 'puddle', 'sea', 'delta', 'estuary', 'wetland', 'swamp']
num_classes = len(classes)
# samples_per_class = 7
# for y, cls in enumerate(classes):
#     idxs = np.flatnonzero(y_train == y)
#     idxs = np.random.choice(idxs, samples_per_class, replace=False)
#     for i, idx in enumerate(idxs):
#         plt_idx = i * num_classes + y + 1
#         plt.subplot(samples_per_class, num_classes, plt_idx)
#         plt.imshow(X_train[idx])
#         plt.axis('off')
#         if i == 0:
#             plt.title(cls)
# plt.show()


X_test = np.reshape(X_test, (X_test.shape[0], -1))
X_val = np.reshape(X_val, (X_val.shape[0], -1))
# print(X_train.shape, X_val.shape)


def kullback_leibler_divergence(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    filt = np.logical_and(p != 0, q != 0)
    return np.sum(p[filt] * np.log2(p[filt] / q[filt]))


def kullback_leibler_divergencev(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    return np.sum(p * np.log2(p / q))


n_bins = int(X_test.max() + 1)

X = np.apply_along_axis(lambda x: np.histogram(
    x, density=True, bins=n_bins, range=(0, n_bins))[0], 1, X_test)

import time

dists = np.zeros(X.shape[0])
since = time.time()
for i in range(X.shape[0]):
    dists[i] = kullback_leibler_divergence(X[0], X[i])
print(time.time() - since)
print(dists.shape)
print(dists[:20])

X += 0.000001
X = np.asarray(X)

since = time.time()
dists2 = np.sum(X[0] * np.log2(X[0] / X), axis=1)
print(time.time() - since)
print(dists2[:20])

exit()
from classifiers import KNearestNeighbor

# Create a kNN classifier instance.
# Remember that training a kNN classifier is a noop:
# the Classifier simply remembers the data and does no further processing
classifier = KNearestNeighbor()
classifier.train(X_test, y_test)

# dists = classifier.compute_distances_no_loops(X_val)
# dists = classifier.compute_distances_two_loops(X_val)


# num_folds = 5
k_choices = [1, 3, 5, 8, 10, 15, 20]

# X_train_folds = []
# y_train_folds = []
################################################################################
# TODO:                                                                        #
# Split up the training data into folds. After splitting, X_train_folds and    #
# y_train_folds should each be lists of length num_folds, where                #
# y_train_folds[i] is the label vector for the points in X_train_folds[i].     #
# Hint: Look up the numpy array_split function.                                #
################################################################################
# X_train_folds = np.array_split(X_train, num_folds)
# y_train_folds = np.array_split(y_train, num_folds)
################################################################################
#                                 END OF YOUR CODE                             #
################################################################################

# A dictionary holding the accuracies for different values of k that we find
# when running cross-validation. After running cross-validation,
# k_to_accuracies[k] should be a list of length num_folds giving the different
# accuracy values that we found when using that value of k.
k_to_accuracies = {}


################################################################################
# TODO:                                                                        #
# Perform k-fold cross validation to find the best value of k. For each        #
# possible value of k, run the k-nearest-neighbor algorithm num_folds times,   #
# where in each case you use all but one of the folds as training data and the #
# last fold as a validation set. Store the accuracies for all fold and all     #
# values of k in the k_to_accuracies dictionary.                               #
################################################################################
for k in k_choices:
    k_to_accuracies[k] = []
    # for i in range(num_folds):
    # prepare training data for the current fold
    # X_train_fold = np.concatenate([ fold for j, fold in enumerate(X_train_folds) if i != j ])
    # y_train_fold = np.concatenate([ fold for j, fold in enumerate(y_train_folds) if i != j ])

    # use of k-nearest-neighbor algorithm
    classifier.train(X_test, y_test)
    y_pred = classifier.predict(X_val, k=k, num_loops=2)

    # Compute the fraction of correctly predicted examples
    num_correct = np.sum(y_pred == y_val)
    accuracy = float(num_correct) / X_val.shape[0]
    k_to_accuracies[k].append(accuracy)

################################################################################
#                                 END OF YOUR CODE                             #
################################################################################

# Print out the computed accuracies
for k in sorted(k_to_accuracies):
    for accuracy in k_to_accuracies[k]:
        print('k = %d, accuracy = %f' % (k, accuracy))
