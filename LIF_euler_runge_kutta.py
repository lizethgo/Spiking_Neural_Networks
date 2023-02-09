'''Leaky Integrate and Fire neuron
source: https://neuronaldynamics.epfl.ch/online/Ch1.S3.html
'''

import math 
import matplotlib.pyplot as plt
import numpy as np
import time as t
def plot(neuron, time, dt, I, method):

    # build the v and u vector
    steps = math.ceil(time/dt)
    v = np.zeros(steps)

    v[0] = neuron.v
 
    for i in range(steps): 
        neuron.step(dt, I[i], method)
        v[i] = neuron.v 

    vTime = np.arange(0, time, dt, dtype=None)
    plt.plot(vTime, v, color='b', label="potential")
    plt.plot(vTime, I, color='r', label="current")
    plt.title("Single neuron stimulation")
    plt.xlabel("Time [ms]")
    plt.ylabel("Voltage [mv]")
    plt.show()
    
    
class LIF:
    def __init__(self):
        self.R = 3000
        self.C = 0.006
        self.uR = -10
        self.thrs = 15
        self.maxV = 30
        self.v = -65

    def step(self, I, dt, method=0): 
        
        if self.v >= self.thrs:
            self.v = self.uR
        else: 
            if method == 1: 
                self.solve_rk4(dt, I)
            elif method == 0: 
                self.solve_euler(dt, I)

            if self.v >= self.thrs: 
                self.v = self.maxV 

    def solve_euler(self, I, dt):        
        dv = self.fu(self.v, I) * dt 
        self.v += dv 

    def solve_rk4(self, I, dt):
        dv1 = self.fu(self.v, I) * dt
        dv2 = self.fu(self.v + dv1 * 0.5, I) * dt
        dv3 = self.fu(self.v + dv2 * 0.5, I) * dt
        dv4 = self.fu(self.v + dv3, I) * dt
        dv = 1 / 6 * (dv1 + dv2 * 2 + dv3 * 2 + dv4)
        self.v += dv


    def fu(self, v, I):
        return (-v + self.R * I) / (self.R * self.C)

neuron = LIF() 
time = 500
dt = 0.01
steps = math.ceil(time / dt)
I = [0 if 200/dt <= i <= 300/dt  else 0.01 for i in range(steps)]

# Comment the test you do not wish to do 

# TEST : EULER
plot(neuron, time, dt, I, 0)

#TEST : RK4
plot(neuron, time, dt, I, 1)
#plot(neuron)


# build the v and u vector
steps = math.ceil(time/dt)
v = np.zeros(steps)

v[0] = neuron.v
 
T = t.time()
for i in range(steps): 

    neuron.step(dt, I[i], method=1)
    v[i] = neuron.v 
     #to count the time needed for calculations
elapsed = t.time() - T
print("solve_neuron calculation time: %.15fs" %elapsed)


vTime = np.arange(0, time, dt, dtype=None)

plt.plot(vTime, v, color='b', label="potential")
plt.plot(vTime, I, color='r', label="current")
plt.title("Single neuron stimulation")
plt.xlabel("Time [ms]")
plt.ylabel("Voltage [mv]")
plt.show()



