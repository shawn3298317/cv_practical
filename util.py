import random
import math
from shapely.geometry.point import Point

def split_k_folds(X, Y, k, K):

    assert(X.shape[0] == Y.shape[0])

    shuffled_ind = list(range(X.shape[0]))
    random.shuffle(shuffled_ind)

    N = math.floor(X.shape[0]/K)

    train_ind = shuffled_ind[:k*N] + shuffled_ind[(k+1)*N:]
    eval_ind = shuffled_ind[k*N: (k+1)*N]

    return X[train_ind], Y[train_ind], X[eval_ind], Y[eval_ind]

