import numpy as np
import cv2
from KMeans import do_clustering
from functions.io_data import read_data
from scipy.stats import multivariate_normal
from sklearn.preprocessing import scale


def expectation_step(X, mu_k, sigma_k, pi_k):
    N = X.shape[0]
    K = mu_k.shape[0]
    r = np.empty((N, K))

    for i in range(K):
        mu = mu_k[i]
        sigma = sigma_k[i]
        rv = multivariate_normal(mean=mu, cov=sigma)
        r[:, i] = pi_k[i] * rv.pdf(X)

    r = r / np.sum(r, axis=1)[:, None]

    return r


def maximization_step(X, G):
    """
    Re estimate the parameters using the current responsibilities
    :param X: N x D dimensional matrix containing data points
    :param G: N x K dimensional matrix
    :return: a tuple of the new mean, covariances and coefficients
    """
    N, D = X.shape
    K = G.shape[1]
    N_k = np.sum(G, axis=0)

    # calculate new coefficients
    pi_k = N_k / N

    # calculate new mean
    mu_k = np.empty((K, D))
    for i in range(K):
        mu_k[i, :] = np.sum(X * G[:, i].reshape((N, 1)), axis=0) / N_k[i]

    # calculate new covariances
    sigma_k = np.empty((K, D, D))
    for i in range(K):
        dev = X - mu_k[None, i]
        sum_cov = np.zeros((D, D))
        for j in range(N):
            cov = np.matmul(dev[j].reshape(D, 1), dev[j].reshape(1, D))
            sum_cov += cov * G[j, i]
        sigma_k[i, :, :] = sum_cov / N_k[i]

    return mu_k, sigma_k, pi_k


def log_likelihood(X, mu_k, sigma_k, pi_k):
    """
    Evaluate the log likelihood for all data points
    :param X: N x D dimensional matrix containing data points
    :param mu_k: K x D dimensional matrix containing each cluster's centroid
    :param sigma_k: K x D x D dimensional array containing each cluster's cov
    :param pi_k: K dimensional vector containing each cluster's coefficient
    :return: accumulated log likelihood of all data points
    """
    N = X.shape[0]
    K = mu_k.shape[0]

    rvs = []
    for i in range(K):
        rvs.append(multivariate_normal(mean=mu_k[i], cov=sigma_k[i]))

    output = 0
    for i in range(N):
        mixture = 0
        for j in range(K):
            mixture += pi_k[j] * rvs[j].pdf(X[i])
        output += np.log(mixture)

    return output


def check_convergence(old_p, new_p, eps):
    """
    Return True if the difference between the new log-likelihood and the old one
    is less than eps
    :param old_p: old log-likelihood
    :param new_p: new log-likelihood
    :param eps: the significance of the difference that would indicate convergence
    :return: type: bool
    """
    return np.square(new_p - old_p) < eps


def initialize_covariance(X, c, mu_k):
    """
    Initialize the covariance of each cluster based on the standard deviation
    of each dimension
    :param X: matrix of D-dimensional data points
    :param c: D-dimensional vector indicating cluster assignment of each data
    :param mu_k: centroids of each cluster
    :return: K x D x D array containing covariance matrix of each cluster
    """
    K = mu_k.shape[0]
    D = mu_k.shape[1]

    sigma = np.empty((K, D, D), dtype=np.float32)

    for i in range(K):
        pts = X[c == i, :]
        n = pts.shape[0]
        mu = mu_k[i]
        stddev = np.sqrt(np.sum(np.square(pts - mu[None, :]), axis=0) / n)
        cov = np.diag(stddev)
        sigma[i, :, :] = cov

    return sigma


def run_EM(filename):
    data, image = read_data("a2/"+filename, False)
    rows = image.shape[0]
    cols = image.shape[1]
    # X = image.reshape((image.shape[0] * image.shape[1], image.shape[2]))
    X = np.copy(data)
    X = scale(X)

    EPS = 1E-8
    N = X.shape[0]  # number of observations
    K = 2  # number of clusters to segment into
    D = 3  # dimension of each data point

    # run K-means to initialize mu_k
    print("Running initialization...")
    mu_k, cluster_assignment = do_clustering(X, K)

    # initialize covariances to std dev of dimension within each cluster
    sigma_k = initialize_covariance(X, cluster_assignment, mu_k)

    # initialize pi_k to be uniformly distributed
    _, counts = np.unique(cluster_assignment, return_counts=True)
    pi_k = counts / N

    old_p = 0.0
    while True:
        # calculating responsibilities at expectation step
        print("Expectation step...")
        r = expectation_step(X, mu_k, sigma_k, pi_k)

        # re-estimate parameters using current responsibilities
        print("Maximization step...")
        mu_k, sigma_k, pi_k = maximization_step(X, r)

        # evaluate the log likelihood
        p = log_likelihood(X, mu_k, sigma_k, pi_k)
        print("Log likelihood:", p)

        if check_convergence(old_p, p, EPS):
            r = expectation_step(X, mu_k, sigma_k, pi_k)
            break
        else:
            old_p = p

    cluster_assignment = np.argmax(r, axis=1)

    k_bg = 0
    pixel_mask = np.zeros((N, D), dtype=np.float32)
    pixel_mask[cluster_assignment == k_bg] = [100.0, 0.0, 0.0]
    mask_image = np.reshape(pixel_mask, (cols, rows, D)).transpose((1, 0, 2))
    output_filename = "{}_mask.jpg".format(filename)
    cv2.imwrite(output_filename,
                (cv2.cvtColor(mask_image, cv2.COLOR_Lab2BGR) * 255).astype(
                    np.uint8))


if __name__ == '__main__':
    testfiles = ['cow.txt', 'owl.txt', 'zebra.txt', 'fox.txt']
    for file in testfiles:
        print("Running file:", file)
        run_EM(file)
