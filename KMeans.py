from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def dist(a, b):
    """
    Compute the Euclidean distance between two points

    :param a: type: ndarray
    :param b: type: ndarray
    :return: Euclidean distance computed as ||a-b||
    :raise ValueError: If the two arrays have different shapes
    """
    if a.shape != b.shape:
        raise ValueError
    return np.sqrt(np.sum(np.square(a - b)))


def get_cluster(arr, centroids):
    """
    Assign arr to one of the clusters based on
    least distance to cluster centroid
    :param arr: type: (d,) ndarray
    :param centroids: type: k x d ndarray
    Cluster centroids. Each centroid must have the same shape as arr
    :return: the index of the cluster that arr should be assigned to
    """
    min_dist = float('inf')
    assigned = None
    for idx, c in enumerate(centroids):
        d = dist(arr, c)
        if d < min_dist:
            min_dist = d
            assigned = idx

    return assigned


def recompute_centroids(X, c, centroids):
    """
    Recompute the centroids of each cluster to be the mean of the data points
    in the cluster
    :param X: data points
    :param c: cluster assignments
    :param centroids: matrix of centroids
    :return: new matrix of centroids
    """
    new_centroids = []
    for i in range(len(centroids)):
        mask = c == i
        mask = mask.astype(np.float32).reshape(1, c.shape[0])
        c_n = np.count_nonzero(mask)
        new_c = np.sum(np.matmul(mask, X), axis=0) / c_n
        new_centroids.append(new_c)

    return np.asarray(new_centroids, dtype=np.float32)


def check_converging(c, new_c, eps):
    """
    Returns True if any of the centroids have not moved much
    :param c: previous centroid
    :param new_c: new centroid
    :param eps: expected margin between the previous and new centroid
    to consider them to have not moved
    :return: type: bool
    """
    diff = np.sqrt(np.sum(np.square(c - new_c), 1))
    return not np.all(diff <= eps)


def initialize_centroids(X, k):
    """
    Initialize k centroids from the given X
    :param X: Data points
    :param k: number of clusters
    :return: Matrix of centroids
    """
    centroids = []
    # Choose initial centroid randomly
    np.random.seed(int(datetime.now().timestamp()))
    c_idx = np.random.randint(X.shape[0])
    centroids.append(X[c_idx])

    # Calculate the square of Euclidean distance between each data point
    # and the chosen initial centroid. This would be used as a weighted
    # probability distribution to choose the subsequent centroids. This approach
    # ensures that the chosen initial centroids are far away from each other.
    D_x = np.sum(np.square(X - centroids[0]), 1).reshape(X.shape[0])
    D_x = D_x / np.sum(D_x)
    for _ in range(1, k):
        c_idx = np.random.choice(X.shape[0], p=D_x)
        centroids.append(X[c_idx])

    centroids = np.asarray(centroids, dtype=np.float32)

    return centroids


def do_clustering(X, k):
    """
    Do K-Means Clustering to assign each data point in X to one of k clusters
    :param X: Data points
    :param k: number of clusters expected
    :return: the centroids for each cluster and the assignments for the data pts
    """
    eps = 0.0003
    centroids = initialize_centroids(X, k)

    iteration = 0
    centroids_not_converged = True
    cluster_assignment = np.full(X.shape[0], np.nan, dtype=np.int8)
    while centroids_not_converged:
        for idx, x in enumerate(X):
            cluster_assignment[idx] = get_cluster(x, centroids)

        new_centroids = recompute_centroids(X, cluster_assignment, centroids)
        centroids_not_converged = check_converging(centroids, new_centroids,
                                                      eps)
        centroids = new_centroids
        iteration += 1

    print("Clustering outcome:")
    print(np.unique(cluster_assignment, return_counts=True))
    visualize_clustering(X, cluster_assignment)

    return centroids, cluster_assignment


def visualize_clustering(X, c):
    """
    Visualize the cluster assignment of the data points in X after running
    K-Means clustering
    :param X: N x D matrix that contains N D-dimensional data points
    :param c: N dimensional vector that contains the cluster assignment of each data point
    """
    fig = plt.figure()
    ax = Axes3D(fig)

    for i, config in enumerate([('r', 'o'), ('b', '^')]):
        mask = c == i
        pts = X[mask,:]
        ax.scatter(pts[:, -3], pts[:, -2], pts[:, -1], c=config[0], marker=config[1])

    ax.set_xlabel('L')
    ax.set_xlabel('a')
    ax.set_xlabel('b')

    plt.show()
