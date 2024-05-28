import torch
import numpy as np


#PATH = "/root/datasets/s3dis/Area_1/office_3.pth"
PATH = "/root/datasets/NYU_Depth_V2/dataset/train/000027.pth"
#PATH = '/root/datasets/NYU_Depth_V2/dataset/000001.pth'
model = torch.load(PATH)
print(model.keys())[]

coords = model['coord']

print(type(coords[0][0]))
print(coords.shape)

# finding min max
x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
z_min, z_max = coords[:, 2].min(), coords[:, 2].max()

print(f"X Channel: min = {x_min}, max = {x_max}")
print(f"Y Channel: min = {y_min}, max = {y_max}")
print(f"Z Channel: min = {z_min}, max = {z_max}")




"""
#print(model[wiw1].shape)
print('########################')
PATH = "/root/datasets/NYU_Depth_V2/dataset/test/000001.pth"
model = torch.load(PATH)

wiw1 = 'coord'
wiw2 = 'semantic_gt'

print(np.unique(model[wiw2][:, 0]))
"""




"""
import imageio

# Load the image file
IMG_PATH = "/root/datasets/NYU_Depth_V2/label40/000001.png"
image = imageio.imread(IMG_PATH)

# Convert the image to a numpy array
image_array = np.array(image)

# Print the shape of the array
print("Shape of the image array:", image_array.shape)
print(type(image_array[0, 0]))
"""