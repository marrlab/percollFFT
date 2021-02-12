import numpy as np
import cv2
import pickle
import pickletools
from fouriersignal import fourierSignal

"""Data Loader for training data"""


# Data loader
class percollDataLoaderMultiClass:
    def __init__(self, indexFold, train, augmented=True):
        if augmented:
            if train:
                name = "/storage/scratch/users/ario.sadafi/percoll-new/foldTrain" + str(indexFold) + ".pkl"
                with open(name, "rb") as f:
                    self.imageList = pickle.load(f)
            else:
                name = "/storage/scratch/users/ario.sadafi/percoll-new/foldTest" + str(indexFold) + ".pkl"
                with open(name, "rb") as f:
                    self.imageList = pickle.load(f)
        else:
            name = "/storage/scratch/users/ario.sadafi/percoll-new/foldTest" + str(indexFold) + ".pkl"
            with open(name, "rb") as f:
                imList = pickle.load(f)
            self.imageList = []
            for i, l in enumerate(imList):
                if i % 6 == 0:
                    self.imageList.append(l)


    def __len__(self):
        return len(self.imageList)

    def __getitem__(self, index):
        image = np.array(self.imageList[index]['Img'], dtype="float64")
        image *= 1.0 / image.max()
        if image.shape[1] != 500 or image.shape[2] != 500:
            image = np.array([cv2.resize(image[0], (500, 500)),
                              cv2.resize(image[1], (500, 500)),
                              cv2.resize(image[2], (500, 500))])

        label = np.array([self.imageList[index]['label'] == c
                          for c in [0, 1, 2, 3]],
                         dtype=np.uint8)  # change this to fit with the 8 labels

        _, _, _, fourierColors, _ = fourierSignal(image)
        return image, label, fourierColors


if __name__ == "__main__":
    PDL = percollDataLoaderMultiClass(0, train=True, augmented=True)
    print(PDL[4])
    for data in PDL:
        print(data[0].shape)
