########## import libraries ###################################################
from ultralytics import RTDETR
import os
from skopt import load


########## set working directory ##############################################
os.chdir('/home/users/industry/hpcic/hpcic32/scratch/')


########## define objective function (model training) #########################

def obj_func(x):

    ## define the model
    model = RTDETR('rtdetr-x.pt')
    
    ## train the model
    train_result = \
        model.train(data="datasets_2/data.yaml", epochs=150, device=[0,1], \
            patience=100, close_mosaic=0, \
            batch=int(x[0]*4), \
            lr0=x[1], \
            lrf=x[2], \
            momentum=x[3], \
            weight_decay=x[4], \
            warmup_epochs=x[5], \
            warmup_momentum=x[6], \
            warmup_bias_lr=x[7], \
            box=x[8], \
            cls=x[9], \
            dfl=x[10], \
            pose=x[11], \
            kobj=x[12], \
            nbs=int(x[13]*4), \
            hsv_h=x[14], \
            hsv_s=x[15], \
            hsv_v=x[16], \
            degrees=x[17], \
            translate=x[18], \
            scale=x[19], \
            shear=x[20], \
            perspective=x[21], \
            flipud=x[22], \
            fliplr=x[23], \
            bgr=x[24], \
            mosaic=x[25], \
            mixup=x[26], \
            copy_paste=x[27], \
            erasing=x[28] \
            )


########## load data from tuning ##############################################
data = load('RT_DETR_150epochs_dataset_2_Bayesian_tuning_100ncalls.pkl')
x = data['x']


## train the model
obj_func(x)

