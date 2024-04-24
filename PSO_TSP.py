import csv
from itertools import zip_longest
import logging
import math
import multiprocessing
import os
import random
import threading
import numpy as np
import datetime
import time
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


file_path = 'input.txt' 
df = pd.read_csv(file_path, delimiter='\s+')  
print(df.info())
df['CUST_NO.'] = df.index
df.to_csv('solomon_data.txt', index=False)

with open('solomon_data.txt', 'r') as file:
    lines = file.readlines()

# Initialize the customers dictionary
customers = {}
num_customers = 50
end_point_data = num_customers + 2
num_particles = 100
num_customers = 5
max_iterations = 400
w = 0.5
c1 = 1.5
c2 = 1.5
cord_data = []
best_val = []
# Process each line and create the dictionary entries
for line in lines[1:end_point_data]:  # Skip the header line
    list_cord = []
    data = line.strip().split(',')
    cust_no = int(data[0])
    xcoord = int(data[1])
    ycoord = int(data[2])
    demand = int(data[3])
    ready_time = int(data[4])
    due_date = int(data[5])
    service_time = int(data[6])
    list_cord.append(xcoord)
    list_cord.append(ycoord)
    cord_data.append(list_cord)
    customers[cust_no] = (cust_no,xcoord, ycoord, demand, ready_time, due_date, service_time)

class Particle:
    def __init__(self, num_customers):
        self.position = np.random.uniform(-5, 5, size=num_customers)
        self.velocity = np.random.uniform(-1, 1, size=num_customers)
        self.pbest_position = self.position.copy()
        self.pbest_fitness = float('inf')

def calculate_distance(coord1, coord2):
    x_diff = coord1[1] - coord2[1]
    y_diff = coord1[2] - coord2[2]
    distance = math.sqrt(x_diff ** 2 + y_diff ** 2)
    return distance

def convert_cus(solution):
    temp_list = solution.copy()
    sorted_solution = sorted(temp_list)
    for i in range(len(temp_list)):
        temp_list[i] = sorted_solution.index(temp_list[i]) + 1
    temp_list = temp_list.astype(int)
    return temp_list

def fitness(solution):
    temp_list = convert_cus(solution)
    fn_cost = 0
    total_cost = 0
    current_location = 0
    end_point = 0
    for cust in temp_list:
        total_cost += calculate_distance(customers[current_location],customers[cust])
        current_location = cust
    total_cost += calculate_distance(customers[current_location], customers[end_point])
    fn_cost+=total_cost
    return fn_cost

def update_velocity(particle, gbest_position, w, c1, c2):
    inertia_term = w * particle.velocity
    cognitive_term = c1 * np.random.rand() * (particle.pbest_position - particle.position)
    social_term = c2 * np.random.rand() * (gbest_position - particle.position)
    return inertia_term + cognitive_term + social_term

def update_position(particle):
    particle.position += particle.velocity

def pso(num_particles, num_customers, max_iterations):
    particles = [Particle(num_customers) for _ in range(num_particles)]
    for particle in particles:
        particle.pbest_fitness = fitness(particle.position)
    gbest = min(particles, key=lambda x:x.pbest_fitness)
    gbest_fitness = gbest.pbest_fitness
    best_val.append(gbest_fitness)
    gbest_position = gbest.position
    # for particle in particles:
    #     print(convert_cus(particle.position))
    # print("+++++++++++++++++++++++++")
    for _ in range(max_iterations):
        for i, particle in enumerate(particles):
            fitness_val = fitness(particle.position)
            if fitness_val < particle.pbest_fitness:
                particle.pbest_fitness = fitness_val
                particle.pbest_position = particle.position.copy()

            if fitness_val < gbest_fitness:
                gbest_fitness = fitness_val
                best_val.append(gbest_fitness)
                gbest_position = particle.position.copy()
            particle.velocity = update_velocity(particle, gbest_position, w, c1, c2)
            update_position(particle)
        #     print(convert_cus(particle.position))
        # print("+++++++++++++++++++++++++")
        # time.sleep(1)
    return { "best route" : convert_cus(gbest_position),
            "best_cost" : gbest_fitness}

# Example usage
solution = pso(num_particles, num_customers, max_iterations)
print(f"best route: {solution['best route']}")
print(f"best_cost: {solution['best_cost']}")