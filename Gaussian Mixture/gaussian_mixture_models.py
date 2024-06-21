import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture

# Generate a toy 2D dataset
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

plt.scatter(X[:, 0], X[:, 1], s=40, cmap='viridis')

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# Apply Gaussian Mixture Model clustering
gmm = GaussianMixture(n_components=4, random_state=0).fit(X)
labels = gmm.predict(X)

# Plotting the clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')

# Optionally, plot the center of each cluster
centers = gmm.means_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)

plt.title("GMM Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

gmm.converged_

gmm.n_iter_

gmm.covariances_

gmm.means_

gmm.weights_

X.shape

gmm.predict(X)
# hard clustering

gmm.predict_proba(X)
# soft clustering

gmm.sample(10)
# density estimation
# generative model -> ChatGPT

from scipy.stats import multivariate_normal

# Plotting the clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')


# Define the grid for plotting
x = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
y = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 100)
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T

# Plotting the contour plot of each Gaussian component
colors = ['red', 'yellow', 'blue', 'black']
for i, (mean, covar, color) in enumerate(zip(gmm.means_, gmm.covariances_, colors)):
    rv = multivariate_normal(mean, covar)
    Z = rv.pdf(XX)
    Z = Z.reshape(X.shape)
    plt.contour(X, Y, Z, levels=np.linspace(Z.min(), Z.max(), 7), colors=color)

plt.title("GMM Clustering with Gaussian Contours")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# Generating differently shaped blobs
np.random.seed(0)
n_samples = 300

# First blob: circular
X1 = np.random.normal(0, 1, (n_samples // 3, 2))

# Second blob: elongated
X2 = np.random.normal(5, 2, (n_samples // 3, 2))
X2[:, 1] *= 2

# Third blob: larger circular
X3 = np.random.normal(10, 2, (n_samples // 3, 2))

# Combine the blobs to form the dataset
X = np.vstack([X1, X2, X3])

# Apply Gaussian Mixture Model clustering
gmm = GaussianMixture(n_components=3, random_state=0, covariance_type='spherical').fit(X)
labels = gmm.predict(X)

# Plotting the clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')

# Plot the center of each cluster
centers = gmm.means_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)

# Define the grid for plotting
x = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
y = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 100)
X_grid, Y_grid = np.meshgrid(x, y)
XX = np.array([X_grid.ravel(), Y_grid.ravel()]).T

# Plotting the contour plot of each Gaussian component
colors = ['red', 'green', 'blue']
for i, (mean, covar, color) in enumerate(zip(gmm.means_, gmm.covariances_, colors)):
    rv = multivariate_normal(mean, covar)
    Z = rv.pdf(XX)
    Z = Z.reshape(X_grid.shape)
    plt.contour(X_grid, Y_grid, Z, levels=np.linspace(Z.min(), Z.max(), 7), colors=color)

plt.title("GMM Clustering with Gaussian Contours")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# Try GMMs with 1 to N components
N = 10
aics = []
bics = []
for i in range(1, N+1):
    gmm = GaussianMixture(n_components=i).fit(X)
    aics.append(gmm.aic(X))
    bics.append(gmm.bic(X))

# Plotting the AIC and BIC
plt.plot(range(1, N+1), aics, label='AIC')
plt.plot(range(1, N+1), bics, label='BIC')
plt.legend()
plt.xlabel('Number of Components')
plt.ylabel('Information Criterion')
plt.show()

# Author: Ron Weiss <ronweiss@gmail.com>, Gael Varoquaux
# Modified by Thierry Guillemot <thierry.guillemot.work@gmail.com>
# License: BSD 3 clause

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold

colors = ["navy", "turquoise", "darkorange"]


def make_ellipses(gmm, ax):
    for n, color in enumerate(colors):
        if gmm.covariance_type == "full":
            covariances = gmm.covariances_[n][:2, :2]
        elif gmm.covariance_type == "tied":
            covariances = gmm.covariances_[:2, :2]
        elif gmm.covariance_type == "diag":
            covariances = np.diag(gmm.covariances_[n][:2])
        elif gmm.covariance_type == "spherical":
            covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        ell = mpl.patches.Ellipse(
            gmm.means_[n, :2], v[0], v[1], angle=180 + angle, color=color
        )
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)
        ax.set_aspect("equal", "datalim")


iris = datasets.load_iris()

# Break up the dataset into non-overlapping training (75%) and testing
# (25%) sets.
skf = StratifiedKFold(n_splits=4)
# Only take the first fold.
train_index, test_index = next(iter(skf.split(iris.data, iris.target)))


X_train = iris.data[train_index]
y_train = iris.target[train_index]
X_test = iris.data[test_index]
y_test = iris.target[test_index]

n_classes = len(np.unique(y_train))

# Try GMMs using different types of covariances.
estimators = {
    cov_type: GaussianMixture(
        n_components=n_classes, covariance_type=cov_type, max_iter=20, random_state=0
    )
    for cov_type in ["spherical", "diag", "tied", "full"]
}

n_estimators = len(estimators)

plt.figure(figsize=(3 * n_estimators // 2, 6))
plt.subplots_adjust(
    bottom=0.01, top=0.95, hspace=0.15, wspace=0.05, left=0.01, right=0.99
)


for index, (name, estimator) in enumerate(estimators.items()):
    # Since we have class labels for the training data, we can
    # initialize the GMM parameters in a supervised manner.
    estimator.means_init = np.array(
        [X_train[y_train == i].mean(axis=0) for i in range(n_classes)]
    )

    # Train the other parameters using the EM algorithm.
    estimator.fit(X_train)

    h = plt.subplot(2, n_estimators // 2, index + 1)
    make_ellipses(estimator, h)

    for n, color in enumerate(colors):
        data = iris.data[iris.target == n]
        plt.scatter(
            data[:, 0], data[:, 1], s=0.8, color=color, label=iris.target_names[n]
        )
    # Plot the test data with crosses
    for n, color in enumerate(colors):
        data = X_test[y_test == n]
        plt.scatter(data[:, 0], data[:, 1], marker="x", color=color)

    y_train_pred = estimator.predict(X_train)
    train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
    plt.text(0.05, 0.9, "Train accuracy: %.1f" % train_accuracy, transform=h.transAxes)

    y_test_pred = estimator.predict(X_test)
    test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
    plt.text(0.05, 0.8, "Test accuracy: %.1f" % test_accuracy, transform=h.transAxes)

    plt.xticks(())
    plt.yticks(())
    plt.title(name)

plt.legend(scatterpoints=1, loc="lower right", prop=dict(size=12))


plt.show()

