import torch
import numpy as np

PATH = "/root/datasets/s3dis/Area_1/conferenceRoom_1.pth"
model = torch.load(PATH)
# print(model.keys())
wiw1 = 'coord'
wiw2 = 'semantic_gt'



print(type(model[wiw1][0, 0]))
print(model[wiw1].shape)

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