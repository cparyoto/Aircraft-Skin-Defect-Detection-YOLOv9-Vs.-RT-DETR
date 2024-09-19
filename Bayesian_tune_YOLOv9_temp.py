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
	    amp=False, optimizer='SGD', \
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
    w = [0.3, 0.4, 0.3, 0.0] # scoring weight
    return -np.nansum(np.array(stat[0:4])*w) # return the negative score


########## perform tuning #####################################################

# initial point
x0 = \
    [[4, 1e-2, 1e-2, 0.937, 5e-4, 3.0, 0.8, 0.1, 7.5, \
      0.5, 1.5, 12.0, 2.0, 16, 0.015, 0.7, 0.4, 0.0, \
      0.1, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.4], \
      [1, 1e-10, 1e-10, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, \
      0.0, 1.0, 10.0, 0.0, 1, 0.0, 0.0, 0.0, -180.0, \
      0.0, 0.0, -180.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], \
      [4, 0.1, 0.1, 1.0, 0.01, 10.0, 1.0, 0.5, 10.0, \
      1.0, 2.0, 15.0, 5.0, 16, 1.0, 1.0, 1.0, 180.0, \
      1.0, 1.0, 180.0, 0.001, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]

# data = load('yolov9_dataset_2_Bayesian_tuning_52ncalls.pkl')
# x0 = data['x_iters']
# y0 = data['func_vals']

res = gp_minimize(obj_func, [\
        (1, 4), \
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
      x0=x0, n_calls=15, n_initial_points=5)

dump(res, 'yolov9_temp_Bayesian_tuning.pkl', store_objective=False)

