import random
import math
import operator
from numpy.random import choice
import copy
import time
import uuid
import numpy as np
import sys

def measure_calc_cost(popu, nbr_loops):
    for i in range(nbr_loops):
        for indiv in popu.individuals:
            indiv.calculate_cost()


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


def create_random_vars(count):
    random_vars = []
    for i in range(count):
        random_vars.append(random.uniform(-1000, 1000))

    return random_vars


def measure_cost(iterations, popu):
    popu.sort()
    best_cost = popu.individuals[0].cost
    cost_eval = Population.nbr_cost_eval
    return (cost_eval, best_cost)



class Population:
    nbr_cost_eval = 0

    def __init__(self, nbr_indiv, training_data):
        self.nbr_indiv = nbr_indiv
        self.nbr_vars = len(training_data[0])
        self.individuals = self.init_individuals(nbr_indiv, training_data)
        self.next_generation = []

    def init_individuals(self, nbr_indiv, training_data):
        solutions = []
        for line in training_data:
            solutions.append(line.pop())
        training_sol = solutions

        individuals = []
        for i in range(nbr_indiv):
            individuals.append(Individual(training_data, training_sol))

        return individuals


    def sort(self):
        self.individuals.sort(key=operator.attrgetter('cost'))

    def print_best(self, number):
        self.sort()
        for i in range(number):
            print(self.individuals[i].cost)

    def do_selection(self):
        strongest = self.individuals.pop(0)

        population_cost = 0
        for indiv in self.individuals:
            population_cost += indiv.cost

        surv_inter = 0
        weights = []
        for indiv in self.individuals:
            surv_chance = population_cost / indiv.cost
            weights.append(surv_chance)

        norm_weights = [float(i)/sum(weights) for i in weights]

        self.next_generation = list(choice(self.individuals, self.nbr_indiv / 2 - 1, p=norm_weights, replace=False))
        self.next_generation.append(strongest) # Elitism: Strongest indivivual survives

        self.individuals = copy.deepcopy(self.next_generation) # Creates children


    def do_crossover(self):
        # Change two random creates genes
        # Remove those and repeat a given time
        random.shuffle(self.individuals)
        indivs = self.individuals

        for i in range(int(len(indivs) / 2)):
            start_i = random.randrange(self.nbr_vars)
            length = random.randrange(1, self.nbr_vars - 1)

            for j in range(length):
                temp_var = indivs[i].variables[start_i]
                indivs[i].variables[start_i] = indivs[i+1].variables[start_i]
                indivs[i+1].variables[start_i] = temp_var

                start_i += 1
                start_i = start_i % (self.nbr_vars)


    def do_crossover2(self):
        random.shuffle(self.individuals)
        indivs = self.individuals

        for i in range(0, int(len(indivs) / 2), 2): # Step 2 size

            for j in range(self.nbr_vars):
                if random.random() < 0.5:
                    temp_var = indivs[i].variables[j]
                    indivs[i].variables[j] = indivs[i+1].variables[j]
                    indivs[i+1].variables[j] = temp_var




    def do_mutation(self):
        for indiv in self.individuals:
            swap_i = random.randrange(self.nbr_vars)
            multiplier = random.uniform(-1.5, 1.5)
            indiv.variables[swap_i] = indiv.variables[swap_i] * multiplier


    def do_mix_mutation(self): # TODO evaluate method. Good bad?
        for indiv in self.individuals:
            swap_i = random.randrange(self.nbr_vars)
            parent = list(choice(self.next_generation, 1))
            parent = parent[0]
            indiv.variables[swap_i] = (indiv.variables[swap_i] + parent.variables[swap_i]) / 2


class Individual:
    def __init__(self, training_input, training_sol):
        self.variables = create_random_vars(13)
        self.training_input = training_input
        self.training_sol = training_sol
        self.cost = None
        self.calculate_cost2()

    def calculate_cost2(self):
        sum = 0 # sum of (h_x - y)^2
        for i, line in enumerate(self.training_input):
            line_sol = self.training_sol[i]

            x_sum = self.variables[0] # One more var than Xi in line
            for j, x in enumerate(line):
                x_sum += self.variables[j+1] * x

            diff = math.pow(x_sum - line_sol, 2)
            sum += diff

        self.cost = math.sqrt(sum / len(self.training_input))
        Population.nbr_cost_eval += 1

    def calculate_cost(self): # Matrix implementation of least square
        X = np.matrix(self.training_input)
        Y = np.matrix(self.training_sol)
        rows = len(X)
        X = np.c_[np.ones(rows), X] # Add col of ones in the beginning

        x_sum = np.dot(X, self.variables) # x0*var0 + x1*var1...
        diff = x_sum - Y
        diff_pow_2 = np.power(diff, 2)

        sum = np.sum(diff_pow_2) / rows
        self.cost = math.sqrt(sum)
        Population.nbr_cost_eval += 1



# =*==*=*=*=*=*=*=START=*==*=*=*=*=*=*=
path = 'forestfires.txt'
lines = read_file(path)
lines.pop(0) # Removes inputfile header
training_data = format_input_variables(lines)

popu = Population(100, training_data)

print("******** Finished. Generation 0's best are ********")
popu.print_best(5)

measurment_list = []
iterations = 4000
max_time = 100

start_time = time.time()
popu.sort()
for x in range(iterations):


    popu.do_selection()
    popu.do_crossover2()
    popu.do_mutation()


    for i in range(4):
        popu.do_mutation()
        # popu.do_mix_mutation()

    for indiv in popu.individuals:
            indiv.calculate_cost()

    popu.next_generation.extend(popu.individuals)

    popu.individuals = popu.next_generation
    popu.sort()

    elapsed_time = time.time() - start_time
    if  elapsed_time > max_time:
        break
    measurment_list.append([elapsed_time, popu.individuals[0].cost])


print("******** Finished. Best ones are ********")
print(popu.print_best(5))

output_lines = ""
for measurment in measurment_list:
    output_lines += "{:s} {:s}\n".format(str(measurment[0]), str(measurment[1]))

hash = uuid.uuid4().hex
output_path = "cost_measure/genetic/{:s}.dat".format(hash)
write_file(output_lines, output_path)


##### TIME TEST #####

# start_time = time.clock()

# measure_calc_cost(popu, 10) # Seems to be slow
# # measure_sort(popu, 10) # Seems to be very fast

# stop_time = time.clock()
# elap_time = stop_time - start_time
# print(elap_time)


