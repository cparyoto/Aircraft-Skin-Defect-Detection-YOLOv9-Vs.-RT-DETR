########## import libraries ##########

from ultralytics import RTDETR
import os


########## change working directory ##########

os.chdir('/home/users/industry/hpcic/hpcic32/scratch/')


########## run code ##########

## define the model
model = RTDETR('rtdetr-x.pt')

## train the model
model.train(data="datasets/data.yaml", device=[0,1], \
	amp=False, optimizer='SGD')
