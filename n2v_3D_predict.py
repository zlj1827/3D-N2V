
# We import all our dependencies.
from n2v.models import N2V
import numpy as np
from matplotlib import pyplot as plt
from tifffile import imread
from csbdeep.io import save_tiff_imagej_compatible
import os

## Load the trained model by model path and name.
model_name = 'model1'
basedir = 'model'
model = N2V(config=None, name=model_name, basedir=basedir)
model.load_weights('weights_best.h5') # can choose 'weights_best.h5' or 'weights_last.h5'

## load test data and predict
img = imread('data/test/Simu_data21_5s_Gau0.4.tif')
pred = model.predict(img, axes='ZYX', n_tiles=(2,4,4))

## show the denoise result
plt.figure(figsize=(30,30))
plt.subplot(1,2,1)
plt.imshow(np.max(img,axis=0),
           cmap='magma',
           vmin=np.percentile(img,0.1),
           vmax=np.percentile(img,99.9)
          )
plt.title('Input')
plt.subplot(1,2,2)
plt.imshow(np.max(pred,axis=0),
           cmap='magma',
           vmin=np.percentile(pred,0.1),
           vmax=np.percentile(pred,99.9)
          )
plt.title('prediction')

## save the denoise result
save_tiff_imagej_compatible('Simu_data21_5s_Gau0.4_bestModel.tif', pred, 'ZYX')