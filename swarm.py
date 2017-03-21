import random
import math
import operator
from numpy.random import choice
import copy
import time
import uuid


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

class Swarm:
    nbr_cost_eval = 0

    def __init__(self, nbr_indiv, training_data):
        pass
        self.individuals = self.init_individuals(nbr_indiv, training_data)
        # best individual

    def init_individuals(self, nbr_indiv, training_data):
        solutions = []
        for line in training_data:
            solutions.append(line.pop())
        training_sol = solutions

        individuals = []
        for i in range(nbr_indiv):
            individuals.append(Individual(training_data, training_sol))

        return individuals

    def move(self):
        for indiv in self.individuals:
            indiv.move()

    def sort(self):
        self.individuals.sort(key=operator.attrgetter('cost'))

    def print_best(self, number):
        self.sort()
        for i in range(number):
            print(self.individuals[i].cost)

    def print_glob_min(self):
        print(self.individuals[0].best_global_cost)

    def print_vector_length(self):
        vec_length = float(0)
        for indiv in self.individuals:
            sum = 0
            for x in indiv.velocity:
                sum += math.pow(x, 2)
            vec_length += math.sqrt(sum)
        print("Vec length is {:f}".format(vec_length / len(self.individuals)))



class Individual:
    best_global_pos = None
    best_global_cost = float("inf")

    def __init__(self, training_input, training_sol):
        self.training_input = training_input
        self.training_sol = training_sol
        self.position = create_random_vars(13, -500, 500)
        self.velocity = create_random_vars(13, -300, 300) # TODO what value?
        self.best_local_pos = self.position
        self.best_local_cost = float("inf")
        self.cost = None
        self.calculate_cost()


    def calculate_cost(self):
        sum = 0 # sum of (h_x - y)^2
        for i, line in enumerate(self.training_input):
            line_sol = self.training_sol[i]

            x_sum = self.position[0] # One more var than Xi in line
            for j, x in enumerate(line):
                x_sum += self.position[j+1] * x

            diff = math.pow(x_sum - line_sol, 2)
            sum += diff

        self.cost = math.sqrt(sum / len(self.training_input))

        if self.cost < self.best_local_cost:
            self.best_local_cost = self.cost
            self.best_local_pos = self.position
        if self.cost < Individual.best_global_cost:
            Individual.best_global_cost = self.cost
            Individual.best_global_pos = self.position

        Swarm.nbr_cost_eval += 1


    def move(self):
        # w = 0.7
        # pp = 1.5
        # pg = 0.8
        w = 0.7
        pp = 1.7
        pg = 0.7
        a = random.random()
        b = random.random()

        t1 = Vector(self.velocity).mul_scalar(w)

        t2 = Vector(self.best_local_pos)
        t2 = t2.subtr_vector(Vector(self.position))
        t2 = t2.mul_scalar(pp * a)

        g = Vector(Individual.best_global_pos)
        t3 = g.subtr_vector(Vector(self.position))
        t3 = t3.mul_scalar(pg * b)

        sum_of_vectors = t1.add_vector(t2.add_vector(t3))

        self.position = Vector(self.position).add_vector(sum_of_vectors).get_vars()
        self.velocity = sum_of_vectors.get_vars()

        self.calculate_cost()




class Vector:
    def __init__(self, variables):
        self.variables = variables

    def get_vars(self):
        return self.variables

    def mul_scalar(self, scalar):
        return Vector([x * scalar for x in self.variables])

    def add_vector(self, other_vector):
        temp_vec = []
        for i in range(len(self.variables)):
            temp_vec.append(self.variables[i] + other_vector.get_vars()[i])
        return Vector(temp_vec)

    def subtr_vector(self, other_vector):
        temp_vec = []
        for i in range(len(self.variables)):
            temp_vec.append(self.variables[i] - other_vector.get_vars()[i])
        return Vector(temp_vec)


    def length(self):
        vec_length = 0
        for x in self.variables:
            vec_length += math.pow(x, 2)
        return math.sqrt(vec_length)


    def print(self):
        print(self.variables)



# =*==*=*=*=*=*=*=START=*==*=*=*=*=*=*=
path = 'forestfires.txt'
lines = read_file(path)
lines.pop(0) # Removes inputfile header

training_data = format_input_variables(lines)
nbr_indiv = 100
swarm = Swarm(nbr_indiv, training_data)
swarm.print_best(1)

measurment_list = []
min_cost = float("inf")
nbr_iterations = 8500
max_time = 100


start_time = time.time()
for x in range(nbr_iterations):
    swarm.move()

    elapsed_time = time.time() - start_time
    if  elapsed_time > max_time:
        break
    measurment_list.append([elapsed_time, Individual.best_global_cost])

print("******** Finished ********")
print(swarm.individuals[0].position)
swarm.print_best(1)
swarm.print_glob_min()


output_lines = ""
for measurment in measurment_list:
    output_lines += "{:s} {:s}\n".format(str(measurment[0]), str(measurment[1]))

hash = uuid.uuid4().hex
output_path = "cost_measure/swarm/{:s}.dat".format(hash)
write_file(output_lines, output_path)
