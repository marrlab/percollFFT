import os
import sys
import cv2
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pickle
import torch
import torch.utils.data as data_utils

from percollDataLoaderMultiClass import percollDataLoaderMultiClass
from model import alexnetmodel, vgg_create, alex_early_create, vgg_create_early, alexnet_normal, vgg16_normal

run = sys.argv[1]
modelname = sys.argv[2]
fusion = sys.argv[3]  # 1 latefusion #0 early fusion # 2 normal


def train(modelname, i, run):
    # Initializing Data Loader
    cuda = torch.cuda.is_available()
    loader_kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}

    PDL = percollDataLoaderMultiClass(i, train=True, augmented=True)
    train_loader = data_utils.DataLoader(PDL, batch_size=1, shuffle=True, **loader_kwargs)
    PDLT = percollDataLoaderMultiClass(i, train=False, augmented=False)
    test_loader = data_utils.DataLoader(PDLT, batch_size=1, shuffle=True, **loader_kwargs)

    # Instantiate the model
    if modelname == "alexnet":
        num_epochs = 30

        if fusion == "0":
            model = alex_early_create()
        elif fusion == "1":
            model = alexnetmodel()
        elif fusion == "2":
            model = alexnet_normal()

    elif modelname == "vgg16":
        num_epochs = 15

        if fusion == "0":
            model = vgg_create_early()
        elif fusion == "1":
            model = vgg_create()
        elif fusion == "2":
            model = alexnet_normal()

    if cuda:
        model = model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

    # Training
    for epoch in range(num_epochs):
        model.train()
        accLoss = 0
        for batch_idx, (image, label, fourierColors) in enumerate(train_loader):
            optimizer.zero_grad()
            image, label, fourierColors = image.float(), label.float(), fourierColors.float()

            if cuda:
                image, label, fourierColors = image.cuda(), label.cuda(), fourierColors.cuda()

            if fusion == "2":
                probab = model(image)
            else:
                probab = model(image, fourierColors)
            # loss = nn.CrossEntropyLoss()(probab, torch.argmax(label, 1))
            loss = nn.BCELoss()(probab, label)
            accLoss = loss + accLoss
            loss.backward()
            optimizer.step()

        accLoss = accLoss / len(train_loader)

        print('Epoch: {}, accLoss: {:.4f}'.format(epoch, accLoss.cpu()))

        if (epoch + 1) % 10 == 0:
            model.eval()
            acctestLoss = 0.
            for image, label, fourierColors in test_loader:
                image, label, fourierColors = image.float(), label.float(), fourierColors.float()

                if cuda:
                    image, label, fourierColors = image.cuda(), label.cuda(), fourierColors.cuda()

                if fusion == "2":
                    probab = model(image)
                else:
                    probab = model(image, fourierColors)

                tloss = nn.BCELoss()(probab, label)
                acctestLoss = tloss + acctestLoss
            acctestLoss = acctestLoss / len(test_loader)
            print('Epoch: {}, test_Loss: {:.4f}'.format(epoch, acctestLoss.cpu()))

    name = "Models/" + modelname + "-" + fusion + "-" + str(i) + "-" + run + ".pth"
    torch.save(model, name)
    return model


k = 3
for i in range(k):
    model = train(modelname, i, run)
