import math
import numpy as np
import uuid
import random
import sys
import network as neur_net
import time

def read_file(filename):
    file = open(path, 'r')
    return file.read().splitlines()


def write_file(lines_output, path):
    file = open(path, 'w')
    file.write(lines_output)


def format_input_variables(lines):
    formated_lines = []
    for line in lines:
        formated_line = line.split(" ")
        formated_lines.append(list(map(float, formated_line))) # Converts from string to int
    return formated_lines


def create_random_vars(count, min, max):
    random_vars = []
    for i in range(count):
        random_vars.append(random.uniform(min, max))

    return random_vars

# =*==*=*=*=*=*=*=START=*==*=*=*=*=*=*=
# Training data
path = 'forestfires.txt'
lines = read_file(path)
lines.pop(0) # Removes inputfile header

network_data = format_input_variables(lines)
network_data = np.array(network_data)

X = network_data[:,0:12]
Y = network_data[:,12]

# Normalize X and Y
# X -= np.mean(X, axis = 0) # Better if I don't normalize X?
# X /= np.std(X, axis = 0)

max_y = max(Y) # Area is > 0
Y = Y / max_y

network_data = []
for x, y in zip(X, Y):
    network_data.append((x,y))

random.shuffle(network_data) # Shuffle data to get different test_data
test_data = network_data[:100] # Test data is 50 first entries
training_data = network_data[101:] # Training data is the rest

training_X = []
training_Y = []
for line in training_data:
    training_X.append(line[0])
    training_Y.append(line[1])

test_X = []
test_Y = []
for line in test_data:
    test_X.append(line[0])
    test_Y.append(line[1])


nbr_input = 12
nbr_hidden = 10
nbr_output = 1
network = neur_net.Neural_network(nbr_input, nbr_hidden, nbr_output)


epochs = 10000
max_time = 100 # it will stop after 100 seconds
mini_batch_size = 3
learn_rate = 0.1
measurment_list = []

start_time = time.time()
for i in range(epochs):

    a1_list, z2_list, a2_list, a3_array, z3_list = network.feed_forward(len(test_X), test_X)
    error = a3_array - test_Y
    error *= max_y

    network.SGD(training_X, training_Y, learn_rate)
    # print("epoch done")

    # Least square sqrt (f(x) - y)^2
    diff_pow_2 = np.power(error, 2)
    rows = len(error)
    sum = np.sum(diff_pow_2) / rows
    cost = math.sqrt(sum)

    print("Cost is " + str(cost))

    elapsed_time = time.time() - start_time
    if  elapsed_time > max_time:
        break
    measurment_list.append([elapsed_time, cost])


print("******** Finished ********")


output_lines = ""
for measurment in measurment_list:
    output_lines += "{:s} {:s}\n".format(str(measurment[0]), str(measurment[1]))

hash = uuid.uuid4().hex
output_path = "cost_measure/neural/{:s}.dat".format(hash)
write_file(output_lines, output_path)


