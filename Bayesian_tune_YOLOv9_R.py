########## import libraries ###################################################
from ultralytics import YOLO
import os
import numpy as np
from skopt import gp_minimize, dump, load


########## set working directory ##############################################
os.chdir('/home/users/industry/hpcic/hpcic32/scratch/')


########## define objective function (model training) #########################

def obj_func(x):

    ## define the model
    model = YOLO("yolov9e.pt")
    
    ## train the model
    train_result = \
        model.train(data="datasets/data.yaml", epochs=150, device=[0], \
            patience=100, plots=False, close_mosaic=0, \
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
    
    ## get model statistic
    stat = list(train_result.results_dict.values())
    
    ## calculate the objective score
    return -stat[1] # return the negative score


########## perform tuning #####################################################

# initial point
# x0 = \
#     [[4, 1e-2, 1e-2, 0.937, 5e-4, 3.0, 0.8, 0.1, 7.5, \
#       0.5, 1.5, 12.0, 2.0, 16, 0.015, 0.7, 0.4, 0.0, \
#       0.1, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.4], \
#       [1, 1e-10, 1e-10, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, \
#       0.0, 1.0, 10.0, 0.0, 1, 0.0, 0.0, 0.0, -180.0, \
#       0.0, 0.0, -180.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], \
#       [4, 0.1, 0.1, 1.0, 0.01, 10.0, 1.0, 0.5, 10.0, \
#       1.0, 2.0, 15.0, 5.0, 16, 1.0, 1.0, 1.0, 180.0, \
#       1.0, 1.0, 180.0, 0.001, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], \
#       [4, 0.00621, 0.01483, 0.98, 0.00041, 2.16096, 0.75328, 0.1, 6.57252, \
#       0.74545, 1.14894, 12.0, 2.0, 16, 0.01445, 0.62457, 0.29451, 0.0, \
#       0.1095, 0.22324, 0.0, 0.0, 0.0, 0.29843, 0.0, 0.5157, 0.0, 0.0, 0.4], \
#       [4, 0.00518, 0.01501, 0.97841, 0.00044, 5.0, 0.91582, 0.1, 5.36039, \
#       0.58737, 1.86033, 12.0, 2.0, 16, 0.01296, 0.4661, 0.37721, 0.0, \
#       0.16806, 0.17542, 0.0, 0.0, 0.0, 0.26778, 0.0, 0.79816, 0.0, 0.0, 0.4]]

data = load('yolov9_Bayesian_tuning_R_70ncalls.pkl')
x0 = data['x_iters']
y0 = data['func_vals']


res = gp_minimize(obj_func, [\
        (1, 8), \
        (1e-10, 0.1), \
        (1e-10, 0.1), \
        (0.0, 1.0), \
        (0.0, 1e-2), \
        (1.0, 10.0), \
        (0.0, 1.0), \
        (0.0, 0.5), \
        (1.0, 20.0), \
        (0.0, 1.0), \
        (1.0, 2.0), \
        (10.0, 15.0), \
        (0.0, 5.0), \
        (1, 16), \
        (0.0, 1.0), \
        (0.0, 1.0), \
        (0.0, 1.0), \
        (-180.0, 180.0), \
        (0.0, 1.0), \
        (0.0, 1.0), \
        (-180.0, 180.0), \
        (0.0, 1e-3), \
        (0.0, 1.0), \
        (0.0, 1.0), \
        (0.0, 1.0), \
        (0.0, 1.0), \
        (0.0, 1.0), \
        (0.0, 1.0), \
        (0.0, 1.0)], \
      x0=x0, y0=y0, n_calls=30, n_initial_points=0)

dump(res, 'yolov9_Bayesian_tuning_R.pkl', store_objective=False)

