import time
import numpy as np 

def f_writetime(time):
    hours = np.int8(np.floor(time/3600))
    minutes = np.int8(np.floor((time - hours*3600)/60))
    seconds = np.floor(time - hours*3600 - minutes*60)
    
    return str(hours) + " hour(s), " + str(minutes) + " minute(s), and " + str(seconds) + " seconds"