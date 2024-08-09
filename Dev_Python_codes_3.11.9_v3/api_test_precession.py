from robot import *
import numpy as np
import time 

rbt = Robot()

rbt.connect('COM11')

rbt.move_abs(z=100)

time.sleep(2)

i=0

while i<100:
    i += 1
    rbt.move_abs(0,0,100,10.0*np.sin(i),10.0*np.cos(i),0)
    #rbt.move_cycloidal(0,0,10,1.0*np.sin(i),1.0*np.cos(i),0)
    time.sleep(0.2)

time.sleep(2)
rbt.move_abs(z=100)
#rbt.move_cycloidal(z=10)


