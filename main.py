
"""
@author: Zhang Xi

CASPER model Learning, prediction and visulization of a single twitter cascade.
"""

import json
import time

from functions import learn, predict, visulization_plot


theta_min = {'gamma': 1e-6, 'beta':1e-6, 'kappa': 1e-6} 
theta_max = {'gamma': 10.0, 'beta':100.0, 'kappa': 1.0 } 
theta_init = {'gamma': 0.1, 'beta': 0.1, 'kappa':0.1}
        

with open('tweet_cascade_2.json', 'r') as f:
    cascade = json.load(f)

# following the format of pd.Timedelta. For example, '30min', '1h'
censoring_time = '1h'

#---------Specify prediction intervals--------------------------
# following the format of pd.Timedelta. For example, '10min', '1h'
# can be a single prediction interval vale, for example: delta_times = '1h'
# Or a list of prediction intervals 
# delta_times = '4d'
delta_times =  ['6h', '12h', '1d', '3d', '6d'] 

#--------------------Learn-------------------------------------------------    
# To learn the model parameters
print('Start model learning')
st = time.time()
theta_learned = learn(censoring_time, cascade, theta_min, theta_max, theta_init)
time_used = time.time()-st
print('Finish model learning, time used: {} mins'.format(time_used/60))

#--------------------Predict-----------------------------------------------
if theta_learned['gamma'] >=1:
    print('the cascade will be viral')
else:     
    pred_results = predict(delta_times, cascade, censoring_time, theta_learned, get_bounds=False)
    print('the predict count in {} are {}'.format(delta_times, pred_results))
    #--------------------Visulization------------------------------------------
    s_df = visulization_plot('example_figure.png', cascade, censoring_time, theta_learned, last_pred_time ='12h', get_bounds=False)











