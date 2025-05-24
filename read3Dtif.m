clc; clear; close all;

%%
file_path = 'Simu_data21_5s_Gau0.4_bestModel.tif';
info = imfinfo(file_path);
num_layers = numel(info); % Get the number of image layers

% Allocate space for an array that stores 3D data
width = info(1).Width;
height = info(1).Height;
depth = num_layers; 
data = zeros(height, width, depth);

% The image data of each layer is read in a loop and stored in a 3D array
for layer = 1:num_layers
    data(:,:,layer) = imread(file_path, layer);
end

%%
save( 'Simu_data21_5s_Gau0.4_bestModel.mat','data')
