import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import glob
import sys


def read_data(path):
    file = open(path, 'r')
    return file.read().splitlines()

# =*==*=*=*=*=*=*=START=*==*=*=*=*=*=*=

# Swarm
path = 'cost_measure/swarm/*.dat'
files = glob.glob(path)

sw_x = []
sw_y = []
maj = []
for name in files:
    file = open(name, 'r')
    measurment = file.read().split("\n")
    measurment.pop() # Remove empty last line
    measurment_length = len(measurment)

    for meas in measurment:
        meas = meas.split(" ")
        sw_x.append(float(meas[0]))
        sw_y.append(float(meas[1]))


# Neural
path = 'cost_measure/neural/*.dat'
files = glob.glob(path)

neur_x = []
neur_y = []
for name in files:
    file = open(name, 'r')
    measurment = file.read().split("\n")
    measurment.pop() # Remove empty last line
    measurment_length = len(measurment)

    for meas in measurment:
        meas = meas.split(" ")
        neur_x.append(float(meas[0]))
        neur_y.append(float(meas[1]))

# Genetic
path = 'cost_measure/genetic/*.dat'
files = glob.glob(path)

measurment_length = None
gen_x = []
gen_y = []
for name in files:
    file = open(name, 'r')
    measurment = file.read().split("\n")
    measurment.pop() # Remove empty last line
    measurment_length = len(measurment)

    for meas in measurment:
        meas = meas.split(" ")
        gen_x.append(float(meas[0]))
        gen_y.append(float(meas[1]))

fig = plt.figure(1)
plt.ylabel('Cost')
plt.xlabel('Time (s)')
plt.xlim((0, 10))
# plt.ylim((0, 300))

# plt.scatter(sw_x, sw_y, color='red')
# red_patch = mpatches.Patch(color='red', label='Swarm')

plt.scatter(neur_x, neur_y, color='blue')
blue_patch = mpatches.Patch(color='blue', label='Neural')

# plt.scatter(gen_x, gen_y, color='purple')
# purple_patch = mpatches.Patch(color='purple', label='Genetic')

# plt.legend(handles=[red_patch, blue_patch, purple_patch])
plt.legend(handles=[blue_patch])

plt.show()

sys.exit(1)
