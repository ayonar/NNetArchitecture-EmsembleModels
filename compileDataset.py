# Compile Dataset into pickle file

from PIL import Image
import numpy as np
import os
import glob
import pickle


# load triangles, tetragons and pentagons
def load_dataset(impath) :
    '''Gets path for image locations and returns list of images.'''
    data = []
    for fname in os.listdir(impath):
        pathname = os.path.join(impath, fname)
        with Image.open(pathname) as img:
            img2 = np.array(img)
            data.append(img2)
    return data

# dataset paths
dataset_dir = os.path.abspath('./data/PolygonImagesV7')
mypaths=[os.path.dirname(dataset_dir + "/3/image_0.png"),
         os.path.dirname(dataset_dir + "/4/image_20000.png"),
         os.path.dirname(dataset_dir + "/5/image_40000.png")]
dataset = []
for mypath in mypaths:
    tempdata = load_dataset(mypath)
    data = np.array(tempdata)
    np.random.shuffle(data)
    dataset.append(data)


# Split the dataset into train, validatin and test sets.
train_x = np.concatenate((dataset[0][:12000], dataset[1][:12000], dataset[2][:12000]), axis=0)
val_x = np.concatenate((dataset[0][12000:16000], dataset[1][12000:16000], dataset[2][12000:16000]), axis=0)
test_x = np.concatenate((dataset[0][16000:20000], dataset[1][16000:20000], dataset[2][16000:20000]), axis=0)
train_y = np.concatenate(( np.zeros(int(len(train_x)/len(mypaths))) , np.ones(int(len(train_x)/len(mypaths))) , 2*np.ones(int(len(train_x)/len(mypaths))) ) , axis=0)
val_y = np.concatenate(( np.zeros(int(len(val_x)/len(mypaths))) , np.ones(int(len(val_x)/len(mypaths))) , 2*np.ones(int(len(val_x)/len(mypaths))) ) , axis=0)
test_y = np.concatenate(( np.zeros(int(len(test_x)/len(mypaths))) , np.ones(int(len(test_x)/len(mypaths))) , 2*np.ones(int(len(test_x)/len(mypaths))) ) , axis=0)

import dill
filename = 'PolygonImagesV7.pkl' #edit file name
dill.dump_session(filename)