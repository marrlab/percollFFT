import numpy as np
import pickle
from skimage.transform import rotate
from skimage.util import random_noise
from skimage.filters import gaussian


def augment(imageList0):
  imageList = []
  for index in range(len(imageList0)):
    imageList.append({'Img': imageList0[index]['Img'], 'label': imageList0[index]['label']})
    imageList.append({'Img': np.fliplr(imageList0[index]['Img']), 'label': imageList0[index]['label']})
    imageList.append({'Img': random_noise(imageList0[index]['Img'],var=0.2**2), 'label': imageList0[index]['label']})
    imageList.append({'Img': imageList0[index]['Img'][:,5:495,5:495], 'label': imageList0[index]['label']})
    vectorinit=imageList0[index]['Img'][:,0:500,0:10]
    vectorend=imageList0[index]['Img'][:,0:500,10:500]
    imagis=np.concatenate((vectorend,vectorinit), axis = 2)
    imageList.append({'Img': imagis, 'label': imageList0[index]['label']})

  return imageList

with open("folds.pkl", "rb") as f:
        folds = pickle.load(f)

k = 3
for indexFold in range(k):
  for train in [True, False]:
    imageList = []
    if train:
      for i in range(len(folds[indexFold][0])):
        imageList.append({'Img': folds[indexFold][0][i]['Img'], 'label': folds[indexFold][0][i]['label']})
      imageList = augment(imageList)
      name = "foldTrain" + str(indexFold) + ".pkl"
      with open(name,"wb") as f:
        pickle.dump(imageList,f)
    else:
      for i in range(len(folds[indexFold][1])):
        imageList.append({'Img':folds[indexFold][1][i]['Img'], 'label': folds[indexFold][1][i]['label']})
      imageList = augment(imageList)
      name = "foldTest" + str(indexFold) + ".pkl"
      with open(name,"wb") as f:
        pickle.dump(imageList,f)
print("finished")
