import glob
import time

import cv2
import numpy as np

import torch


def test_loading_speed():

    ### loading with numpy
    x = []
    then = time.time()
    x = np.load('data/gebaerdenlernen/testdata/npy/null.npy')
    now = time.time()
    print(f"Shape of np array {x.shape}")
    print(f"Numpy single npy took: {now - then}")

    x = []
    then = time.time()
    x = [np.load(image) for image in glob.glob('data/gebaerdenlernen/testdata/npy/null/*.npy')]
    x = np.array(x)
    now = time.time()
    print(f"Shape of np array {x.shape}")
    print(f"Numpy several npy took: {now - then}")

    x = []
    then = time.time()
    x = np.load('data/gebaerdenlernen/testdata/npy/null_npz.npz')
    x = x['arr_0.npy']
    now = time.time()
    print(f"Shape of np array {x.shape}")
    print(f"Numpy single npz took: {now - then}")

    x = []
    then = time.time()
    x = np.load('data/gebaerdenlernen/testdata/npy/null_compressed_npz.npz')
    x = x['arr_0.npy']
    now = time.time()
    print(f"Shape of np array {x.shape}")
    print(f"Numpy single compressed npz took: {now - then}")

    ### loading with opencv
    x = []
    then = time.time()
    cap = cv2.VideoCapture('data/gebaerdenlernen/testdata/mp4/null.mp4')
    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        x.append(frame)
    x = np.array(x)
    now = time.time()
    print(f"Shape of np array {x.shape}")
    print(f"OpenCV videocapture took: {now - then}")

    x = []
    then = time.time()
    x = [cv2.imread(image) for image in glob.glob('data/gebaerdenlernen/testdata/png/null/*.png')]
    x = np.array(x)
    now = time.time()
    print(f"Shape of np array {x.shape}")
    print(f"OpenCV several files took: {now - then}")

    ### loading with opencv to pytorch tensor
    x = []
    then = time.time()
    cap = cv2.VideoCapture('data/gebaerdenlernen/testdata/mp4/null.mp4')
    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        x.append(frame)
    x = torch.tensor(np.array(x))
    now = time.time()
    print(f"Size of tensor {x.size()}")
    print(f"OpenCV videocapture took: {now - then}")

    ### loading with pytorch to pytorch tensor
    x = []
    then = time.time()
    x = torch.load('data/gebaerdenlernen/testdata/pt/null.pt')
    now = time.time()
    print(f"Size of tensor {x.size()}")
    print(f"Pytorch single file took: {now - then}")


if __name__ == "__main__":
    test_loading_speed()

### TEST RESULTS ###

# Shape of np array (138, 480, 640, 3)                  File-size: 127,2mb     <- Fastest
# Numpy single npy took: 0.06524491310119629
# Shape of np array (138, 480, 640, 3)                  File-size: 127,2mb
# Numpy several npy took: 0.14262604713439941
# Shape of np array (138, 480, 640, 3)                  File-size: 127,2mb
# Numpy single npz took: 0.15310215950012207
# Shape of np array (138, 480, 640, 3)                  File-size:  37,3mb
# Numpy single compressed npz took: 0.451735258102417
# Shape of np array (138, 480, 640, 3)                  File-size:   0,7mb     <- Winner
# OpenCV videocapture took: 0.15002799034118652
# Shape of np array (138, 480, 640, 3)                  File-size:  42,2mb
# OpenCV several files took: 0.7739019393920898

# Size of tensor torch.Size([138, 480, 640, 3])         File-size:   0,7mb
# OpenCV videocapture took: 0.24478530883789062
# Size of tensor torch.Size([138, 480, 640, 3])         File-size: 127,2mb
# Pytorch single file took: 0.04524421691894531
