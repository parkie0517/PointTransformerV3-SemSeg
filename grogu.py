import torch
import numpy as np
PATH = "/root/datasets/s3dis/Area_1/conferenceRoom_1.pth"
model = torch.load(PATH)
print(model.keys())
wiw = 'normal'
print(type(model[wiw]))
print(model[wiw].shape)
print(model[wiw][0:10])