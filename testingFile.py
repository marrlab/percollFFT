import torch
import torch.utils.data as data_utils
import os
import cv2
import sys
from torchvision import models
import torch.optim as optim
import numpy as np
import torch.nn as nn
import pickle
import pickletools
from skimage.transform import rotate
from skimage.util import random_noise
from skimage.filters import gaussian
import sklearn.metrics as metrics
from percollDataLoaderMultiClass import percollDataLoaderMultiClass
from percollDataLoader import percollDataLoader


def test(modelname, model, i, run, fusion):
    # Initializing Data Loader
    cuda = torch.cuda.is_available()
    PDL = percollDataLoaderMultiClass(i, train=False, augmented=False)

    # Instantiate the model
    model.eval()
    if cuda:
        model = model.cuda()

    # Test the model
    testList = []
    with torch.no_grad():
        for index, (image, label, fourierColors) in enumerate(PDL):
            image, label, fourierColors = torch.Tensor(image).unsqueeze(0), \
                                          torch.Tensor(label).unsqueeze(0), \
                                          torch.Tensor(fourierColors).unsqueeze(0)
        if cuda:
            image = image.cuda()
            label = label.cuda()
            fourierColors = fourierColors.cuda()

        if fusion == "2":
            outputs = model(image)
        else:
            outputs = model(image, fourierColors)

        testList.append({"Image": index, "label": label.cpu(), "prediction": outputs.cpu().data})

    name = "outputs/" + modelname + "-" + fusion + "-" + str(run) + "-" + str(i) + ".pkl"
    with open(name, "wb") as f:
        pickle.dump(testList, f)
    return testList

k = 3

for fusion in ["0", "1", "2"]:
    for modelname in ["alexnet", "vgg16"]:
        for run in range(5):
            for i in range(k):
                modelName = "Models/" + modelname + "-" + fusion + "-" + str(i) + "-" + str(run) + ".pth"
                model = torch.load(modelName)
                testList = test(modelname, model, i, run, fusion)
