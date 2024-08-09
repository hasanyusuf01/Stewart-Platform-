from robot import *
import numpy as np
import time 

rbt = Robot()
rbt.show_graphics()

i = 0.
while i<100:
    i += 0.01
    rbt.g_move_abs(r=Rotation.from_euler('xyz', [5*np.sin(i), 5*np.cos(i), 0], degrees=True))
    #rbt.g_move_abs(p=[0,0,home_H+(i*0.1)])
    time.sleep(0.005)
