# We import all our dependencies.
from n2v.models import N2VConfig, N2V
import numpy as np
from csbdeep.utils import plot_history
from n2v.utils.n2v_utils import manipulate_val_data
from n2v.internals.N2V_DataGenerator import N2V_DataGenerator
from matplotlib import pyplot as plt
import urllib
import os
import zipfile

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['KMP_DUPLICATE_LIB_OK']='True'

## create a DataGenerator object
datagen = N2V_DataGenerator()

## Read all .tif files from path. Note that the shape of the tif file must be ZYX.
imgs = datagen.load_imgs_from_directory(directory = "data/train/", dims='ZYX')

## Displays the shape of the first image.
## The function 'datagen.load_imgs_from_directory' automatically adds two additional dimensions to the image. 'ZYX' -> 'SZYXC'
print(imgs[0].shape)

## extract patches for training and validation.
patch_shape = (32,32,32)
patches = datagen.generate_patches_from_list(imgs, num_patches_per_img=100, shape=patch_shape,augment=True)
print('Generated all patches:', patches.shape)

## The patches were divided into training set and validation set at the ratio of 80% and 20%
X = patches[:5120]
X_val = patches[5120:]

## Create an N2VConfig object.
## The network training parameters can be modified here.
config = N2VConfig(X,
                   unet_residual=True, #whether to use residual connection
                   unet_n_depth=2, #depth of unet network
                   unet_kern_size=3, #kernel size
                   unet_n_first=16, #initial number of filters
                   unet_last_activation='linear',
                   train_loss='mae',
                   train_epochs=100,
                   train_steps_per_epoch=int(X.shape[0] / 64),
                   train_learning_rate=0.003,
                   train_batch_size=4,
                   train_tensorboard=True,
                   train_checkpoint='weights_best.h5',
                   train_reduce_lr={'factor': 0.5, 'patience': 10},
                   batch_norm=True,
                   n2v_perc_pix=0.5,
                   n2v_patch_shape=(32,32,32),
                   n2v_manipulator='uniform_withCP',
                   n2v_neighborhood_radius=5,
                   single_net_per_channel=True)

vars(config)

## Naming the model
model_name = 'model2'
## Model path
basedir = 'model/'
## create a n2v network model
model = N2V(config=config, name=model_name, basedir=basedir)

history = model.train(X, X_val)

print(sorted(list(history.history.keys())))
plt.figure(figsize=(16,5))
plot_history(history,['loss','val_loss'])
