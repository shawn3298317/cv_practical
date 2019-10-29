import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from shapely.geometry.point import Point
from skimage.measure import block_reduce
from skimage.draw import circle_perimeter_aa

from train_model import Model


def draw_circle(img, row, col, rad):
    rr, cc, val = circle_perimeter_aa(row, col, rad)
    valid = (
        (rr >= 0) &
        (rr < img.shape[0]) &
        (cc >= 0) &
        (cc < img.shape[1])
    )
    img[rr[valid], cc[valid]] = val[valid]


def noisy_circle(size, radius, noise):
    img = np.zeros((size, size), dtype=np.float)

    # Circle
    row = np.random.randint(size)
    col = np.random.randint(size)
    rad = np.random.randint(10, max(10, radius))
    draw_circle(img, row, col, rad)

    # Noise
    img += noise * np.random.rand(*img.shape)
    return (row, col, rad), img


def find_circle(img, model=None):
    if not model:
        return (0,0,0)
    x = np.array([img[:, :, None]])
    y_pred = model.predict(x, 1)
    # Fill in this function
    y_pred_2 = y_pred * 200
    #print(y_pred, y_pred_2)
    return int(y_pred_2[0][0]), int(y_pred_2[0][1]), int(y_pred_2[0][2])


def iou(params0, params1):
    row0, col0, rad0 = params0
    row1, col1, rad1 = params1

    shape0 = Point(row0, col0).buffer(rad0)
    shape1 = Point(row1, col1).buffer(rad1)

    return (
        shape0.intersection(shape1).area /
        shape0.union(shape1).area
    )

def load_model(fn):
    m = Model(filter_sizes = [3, 3, 2, 2],
              kernel_nums = [10, 10, 10, 10],
              strides = [1, 1, 1, 1],
              batch_size = 50,
              training_mode = False)
    m.build_model()
    m.model.load_weights(fn)
    return m

def threshold(img):
    img = img/3
    idx = img[:, :] < 0.7
    img2 = img.copy()
    img2[idx] = 0
    img2[~idx] = 1
    return img2

def downsample(img):
    return block_reduce(img, block_size=(2, 2), func = np.max)


def generate_train_data():
    results = []
    Xs = []
    Ys = []

    for _ in range(50000):
        params, img = noisy_circle(200, 50, 2)
        #print(img.shape, params, img)
        Xs.append(downsample(threshold(img)))
        Ys.append(params)
        detected = find_circle(img)
        results.append(iou(params, detected))
    results = np.array(results)
    print((results > 0.7).mean())

    Xs = np.asarray(Xs)
    Ys = np.asarray(Ys)
    print(Xs.shape)
    print(Ys.shape)
    np.save("X_train_large.npy", Xs)
    np.save("Y_train_large.npy", Ys)

def main():
    results = []
    #model = load_model("ckpt/model-0020.ckpt")
    model = load_model("ckpt/model-0150.ckpt")
    for _ in range(1000):
        params, img = noisy_circle(200, 50, 2)
        detected = find_circle(downsample(threshold(img)), model)
        #print(params, detected, iou(params, detected))
        results.append(iou(params, detected))
    results = np.array(results)
    print((results > 0.7).mean())

if __name__ == "__main__":
    #generate_train_data()
    main()
