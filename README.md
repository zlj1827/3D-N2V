## 3D-N2V
An example of a noise2void network that can be used for 3D data denoising

# Environment
Python 3.9 and TensorFlow 2.10.

# User guide
1. Train the n2v model with 'n2v_3D_train.py';
2. Use the trained model to denoise the data with 'n2v_3D_predict.py';
For example, the example data (data/test/'Simu_data21_5s_Gau0.4.tif') is denoised using the example model (model/model1/' weights_best-.h5 ').
