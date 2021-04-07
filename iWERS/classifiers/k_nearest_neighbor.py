import numpy as np


class KNearestNeighbor(object):
  """ a kNN classifier with L2 distance """

  def __init__(self):
    pass

  def train(self, X, y):
    """
    Train the classifier. For k-nearest neighbors this is just
    memorizing the training data.

    Inputs:
    - X: A numpy array of shape (num_train, D) containing the training data
      consisting of num_train samples each of dimension D.
    - y: A numpy array of shape (N,) containing the training labels, where
         y[i] is the label for X[i].
    """
    self.X_train = X
    self.y_train = y

  def predict(self, X, k=1, method=0):
    """
    Predict labels for test data using this classifier.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data consisting
         of num_test samples each of dimension D.
    - k: The number of nearest neighbors that vote for the predicted labels.
    - method: Determines which implementation to use to compute distances
      between training points and testing points.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].
    """
    if (method == "kld" or method == 0):
      dists = self.compute_distances_kld(X)
    elif (method == "chi2" or method == 1):
      dists = self.compute_distances_chi2(X)
    elif (method == "llv" or method == 2):
      dists = self.compute_distances_llv(X)
    elif (method == "l2norm" or method == 3):
      dists = self.compute_distances_l2norm(X)
    else:
      raise ValueError('Invalid value %d for method' % method)

    return self.predict_labels(dists, k=k)

  # def kullback_leibler_divergence(self, p, q):
  #   p = np.asarray(p)
  #   q = np.asarray(q)
  #   # filt = np.logical_and(p != 0, q != 0)
  #   return np.sum(p * np.log2(p / q), axis=1)

  # def chi_square_statistics(self, p, q):
  #   p = np.asarray(p)
  #   q = np.asarray(q)
  #   return np.sum(((p - q) ** 2) / (p + q), axis=1)

  # def log_likelihood_statistics(self, p, q):
  #   p = np.asarray(p)
  #   q = np.asarray(q)
  #   return - np.sum((p * np.log2(q)), axis=1)

  def compute_distances_kld(self, X):
    """
    kullback_leibler_divergence
    Compute the distance between each test point in X and each training point
    in self.X_train using a nested loop over both the training data and the 
    test data.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data.

    Returns:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      is the Euclidean distance between the ith test point and the jth training
      point.
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]

    dists = np.zeros((num_test, num_train))

    n_bins = int(X.max() + 1)

    X = np.apply_along_axis(lambda x: np.histogram(
        x, density=True, bins=n_bins, range=(0, n_bins))[0], 1, X)
    X_train = np.apply_along_axis(lambda x: np.histogram(
        x, density=True, bins=n_bins, range=(0, n_bins))[0], 1, self.X_train)

    X += 0.000001
    X_train += 0.000001

    X = np.asarray(X)
    X_train = np.asarray(X_train)

    for i in range(num_test):
      dists[i, :] = np.sum(X[i] * np.log2(X[i] / X_train), axis=1)

    return dists

  def compute_distances_chi2(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a single loop over the test data.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]

    dists = np.zeros((num_test, num_train))

    n_bins = int(X.max() + 1)

    X = np.apply_along_axis(lambda x: np.histogram(
        x, density=True, bins=n_bins, range=(0, n_bins))[0], 1, X)
    X_train = np.apply_along_axis(lambda x: np.histogram(
        x, density=True, bins=n_bins, range=(0, n_bins))[0], 1, self.X_train)

    X += 0.000001
    X_train += 0.000001

    X = np.asarray(X)
    X_train = np.asarray(X_train)

    for i in range(num_test):
      dists[i, :] = np.sum(((X[i] - X_train) ** 2) / (X[i] + X_train), axis=1)

    return dists

  def compute_distances_llv(self, X):
    """
    log_likelihood_statistics
    Compute the distance between each test point in X and each training point
    in self.X_train using a nested loop over both the training data and the 
    test data.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data.

    Returns:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      is the Euclidean distance between the ith test point and the jth training
      point.
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]

    dists = np.zeros((num_test, num_train))

    n_bins = int(X.max() + 1)

    X = np.apply_along_axis(lambda x: np.histogram(
        x, density=True, bins=n_bins, range=(0, n_bins))[0], 1, X)
    X_train = np.apply_along_axis(lambda x: np.histogram(
        x, density=True, bins=n_bins, range=(0, n_bins))[0], 1, self.X_train)

    X += 0.000001
    X_train += 0.000001

    X = np.asarray(X)
    X_train = np.asarray(X_train)

    for i in range(num_test):
      dists[i, :] = - np.sum((X[i] * np.log2(X_train)), axis=1)

    return dists

  def compute_distances_l2norm(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using no explicit loops.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    # expand equation (x-y)^2 = x^2 + y^2 - 2xy
    dists = np.sum(X**2, axis=1, keepdims=True) + np.sum(self.X_train**2, axis=1) \
        - 2 * np.matmul(X, self.X_train.T)
    dists = np.sqrt(dists)

    return dists

  def predict_labels(self, dists, k=1):
    """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.

    Inputs:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      gives the distance betwen the ith test point and the jth training point.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in range(num_test):
      # A list of length k storing the labels of the k nearest neighbors to
      # the ith test point.
      closest_y = []
      closest_y = self.y_train[np.argsort(dists[i])][0:k]
      y_pred[i] = np.bincount(closest_y).argmax()

    return y_pred
