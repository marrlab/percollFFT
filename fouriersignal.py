from scipy.fft import fft
from math import floor as floor
import numpy as np
import matplotlib.pyplot as plt
import cv2


def fourierSignal(image):
    y = 2
    indexNumber = 0
    signalChannels = np.zeros((3, floor(image.shape[2]/5)))
    signal = np.zeros(floor(image.shape[2]/5))
    x = floor(image.shape[1]/2)
    while(y<image.shape[2]):
        channelSum = np.zeros(3)
        for i in range(3):
            for xcoord in [-2, -1, 0, 1, 2]:
                for ycoord in [-2, -1, 0, 1, 2]:
                    channelSum[i] = channelSum[i] + image[i][y + ycoord][x + xcoord]
            channelSum[i] = channelSum[i]/25
            signalChannels[i][indexNumber] = channelSum[i]
        signal[indexNumber] = signalChannels[0][indexNumber] + signalChannels[1][indexNumber] + signalChannels[2][indexNumber]
        indexNumber = indexNumber + 1
        y = y + 5
    fourierColors = np.zeros((3,100))
    for i in range(3):
        fourierColors[i][:] = 2.0 / 100 * np.abs(fft(signalChannels[i])[0:100])

    fourier = fft(signal)
    img_bgr = []
    image = np.transpose(image, (1,2,0))
    b,g,r = cv2.split(image)
    img_bgr = cv2.merge([b,g,r])
    img_bgr = cv2.resize(img_bgr, (200,800))
    for i in range(100):
        startpoint = (0 + i*2 , 0 )
        endpoint = (10 + i*2, 40)
        color = (floor(signalChannels[0][i]), floor(signalChannels[1][i]), floor(signalChannels[2][i]))
        imageCreated = cv2.rectangle(img_bgr, startpoint, endpoint, color, -1)
    return signalChannels, signal, fourier, fourierColors, img_bgr
