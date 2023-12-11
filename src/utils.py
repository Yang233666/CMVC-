import numpy as np
from scipy.special import softmax


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def cos_sim(a, b):
    a = np.array(a)
    b = np.array(b)
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    # cos_theta = float(np.dot(a, b) / (a_norm * b_norm))
    cos_theta = float(np.sum(a * b) / (a_norm * b_norm))
    cos_theta = 0.5 + 0.5 * cos_theta
    return cos_theta


def cosine_distance(a, b):
    a = np.array(a)
    b = np.array(b)
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    cos_theta = float(np.dot(a, b) / (a_norm * b_norm))
    cos_distance = 1 - cos_theta
    # cos_distance = 0.5 - 0.5 * cos_theta
    return cos_distance


def CE(X, Y):
    X_s = softmax(X, axis=-1)
    Y_s = softmax(Y, axis=-1)
    return -np.sum(np.where(Y_s != 0.0, X_s * np.log(Y_s), 0))


def BhattacharyyaDistance(X, Y):
    X_s = softmax(X, axis=-1)
    Y_s = softmax(Y, axis=-1)
    bc = np.sum(np.sqrt(X_s * Y_s))
    return -np.log(bc)


def KL(X, Y):
    Y = np.where(Y == 0, np.finfo(float).eps, Y)
    return np.sum(np.where(X != 0.0, X * np.log(X / Y), 0))


def F_diver(X, Y):
    Y = np.where(Y == 0, np.finfo(float).eps, Y)
    t = X / Y
    return np.sum(np.where(t != 0, X * t * np.log(t), 0))


def HellingerDistance(X, Y):
    X_s = softmax(X, axis=-1)
    Y_s = softmax(Y, axis=-1)

    X_s = np.where(X > 0, X, 0)
    Y_s = np.where(Y > 0, Y, 0)
    return 1 / np.sqrt(2) * np.linalg.norm(np.sqrt(X_s) - np.sqrt(Y_s))


def JensenShannonDivergence(X, Y):
    M = (X + Y) / 2
    M = np.where(M == 0, np.finfo(float).eps, M)
    return 0.5 * np.sum(X * np.log(np.where(X / M > 0, X / M, 1))) + \
           0.5 * np.sum(Y * np.log(np.where(Y / M > 0, Y / M, 1)))

    X_s = softmax(X, axis=-1)
    Y_s = softmax(Y, axis=-1)
    M = (X_s + Y_s) / 2
    M = np.where(M == 0, np.finfo(float).eps, M)
    return 0.5 * np.sum(X_s * np.log(np.where(X_s != 0, X_s / M, 1))) + \
           0.5 * np.sum(Y_s * np.log(np.where(Y_s != 0, Y_s / M, 1)))


def PearsonCorrelation(X, Y):
    X_ = X - np.mean(X)
    Y_ = Y - np.mean(Y)
    return np.sum(np.multiply(X_, Y_)) / (np.linalg.norm(X_) * np.linalg.norm(Y_))


def BrayCuritisDistance(X, Y):
    return np.sum(np.abs(X - Y)) / (np.sum(X) + np.sum(Y))


def CanberrraDistance(X, Y):
    f = abs(X) + abs(Y)
    return np.sum(np.where(f != 0, np.abs(X - Y) / f, 0))


def ChiSquare(X, Y):
    Y = np.where(Y == 0, np.finfo(float).eps, Y)
    return np.sum(np.square(X - Y) / Y)
