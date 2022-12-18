import sys
import numpy as np
from random import SystemRandom
import time

K = 0.5
crossover_probability = 0.75

global obj_func_1
global obj_func_2
obj_func_1 = []
obj_func_2 = []

# Upper and lower bounds for variables
class set_limits(object):
    def __init__(self, lims):
        self.io_min = lims[0]
        self.io_max = lims[1]
        self.latent_min = lims[2]
        self.latent_max = lims[3]

# Initializes the fixed mapping to map floats to integers
# Divides the range (0-1) into fixed number of bins
def initialize_mapping(n_inputs):
    global float_nums
    # Precision of each value is 2
    float_nums = np.array([float("{:.2f}".format(i*(1/(2*n_inputs+1)))) for i in range(2*n_inputs+1)])
    global mapping
    mapping = {float_nums[i]: i for i in range(2*n_inputs+1)}

# Cadidate: An autoencoder neural network
# Structure: Input layers: 1, 2, 3, 4, Latent layer, Output layers: 5, 6, 7, 8
# [layer1_nodes, layer2_nodes, layer3_nodes, layer4_nodes, layer5_nodes, layer6_nodes, layer7_nodes, layer8_nodes, latent_layer_nodes]
def initialize_candidates(cand_size, pop_size, limits):
    initial_vectors = []
    cand = []
    for i in range(pop_size):
        cand = np.array([SystemRandom().uniform(0, 1) for j in range(cand_size)])
        initial_vectors.append(cand)
    return initial_vectors

def get_F(_min=-2, _max=2):
    return SystemRandom().uniform(_min, _max + sys.float_info.epsilon)

# Picks 3 random candidates from parents pool and performs mutation
def mutation(parents, index, K, F, candidate):
    parents_minus_i = parents[:]
    parents_minus_i.pop(index)
    r1,r2,r3 = SystemRandom().sample(parents_minus_i, 3)
    mutant = candidate + K*(r1 - candidate) + F*(r2 - r3)
    return mutant

# Return true if each value (io layer nodes or latent nodes) in the candidate is within the bounds
def check_constraints(candidate, limits):
    # print(candidate)
    for x in candidate[:-1]:
        if not(limits.io_min <= x <= limits.io_max):
            return False
    if not(limits.latent_min <= candidate[-1] <= limits.latent_max):
        return False
    return True

# Perform crossover between 2 vectors and generate trial vector
def crossover(mutant, candidate):
    rand = SystemRandom().random()
    trial = [mutant[i] if rand <= crossover_probability else candidate[i] for i in range(len(candidate))]
    return trial

# Mapping of floating values in candidate to integer values using fixed mapping
def float_to_int_mapping(vector):
    mapped = []
    for x in vector:
        closest_id = (np.abs(float_nums - x)).argmin()
        mapped.append(float_nums[closest_id])
    int_vector = []
    for i in mapped:
        int_vector.append(mapping[i])
    return int_vector

# Checking if one candidate pareto domainates another based on obj function values
def dominate(cand1_index, cand2_index):
    flag = 0
    for func in [obj_func_1, obj_func_2]:
        c1_fitness = func[cand1_index]
        c2_fitness = func[cand2_index]
        # Minimization problem
        if c1_fitness > c2_fitness: return False
        if c1_fitness < c2_fitness: flag = 1
    if flag == 1: return True
    return False

def get_front(population, mapped_pop):
    # Initialize n and S
    pop_size = len(population)
    n = [0] * pop_size
    S = []

    # Pareto dominance to choose front members
    # Computing ni and Si for each member
    for index1 in range(pop_size):
        Si = []
        for index2 in range(pop_size):
            if (dominate(index1, index2)):
                Si.append(index1)
            elif dominate(index2, index1): 
                n[index1] += 1
        S.append(Si)

    # Find front members. If ni = 0, add to front
    front = []
    mapped_front = []
    front_indices = []
    for index, ni_value in enumerate(n):
        if (ni_value==0):
            front.append(population[index])
            mapped_front.append(mapped_pop[index])
            front_indices.append(index)
    
    # Reduce nj by 1 for each member j belonging to Si of a front member i
    for index, member in enumerate(front):
        # j represents indices of members that this member dominates
        for j in S[front_indices[index]]:
            n[j] -= 1
    
    # Remove the front members from the population
    # new_population = [member for member in population if member not in front]
    new_population = [member for i, member in enumerate(population) if i not in front_indices]
    new_mapped_pop = [member for i, member in enumerate(mapped_pop) if i not in front_indices]

    # If front is empty, there are no non-dominating members in population
    if (len(front) == 0):
        exit()
    return front, mapped_front, front_indices, new_population, new_mapped_pop

def crowding_distance(population, indices, unfilled_spots):
    # Array of Di for each member i of front k
    crowd_dist = [sys.maxsize] * len(population)

    # Convert population to a list
    population_list = [member.tolist() for member in population]

    # For each function
    for func_values in [obj_func_1, obj_func_2]:
        # Sort population in ascending order of objective function values
        sorted_pop = [np.array(member) for _, member in sorted(zip(func_values, population_list))]

        # Sort the function values in ascending order
        sorted_func_values = sorted(func_values)
    
        # For each member in sorted population, find crowding distance di
        # Add computed di to Di
        for index in range(1, len(sorted_pop)-1):
            f_prev = sorted_func_values[index-1]
            f_next = sorted_func_values[index+1]
            f_first = sorted_func_values[0]
            f_last = sorted_func_values[-1]
            crowd_dist[index] += (np.abs(f_prev - f_next)/(np.abs(f_first - f_last)+sys.float_info.epsilon))

    # Sort population in descending order of crowding distances
    final_sorted_pop = [member for dist, member in sorted(zip(crowd_dist, sorted_pop), key=lambda x: x[0], reverse=True)]
    final_sorted_indices = [index for dist, index in sorted(zip(crowd_dist, indices), key=lambda x: x[0], reverse=True)]
    return final_sorted_pop[:unfilled_spots], final_sorted_indices[:unfilled_spots]

# To find the next generation of candidates from the parents+trials pool
def nsde(population, functions, gen, f1, f2):

    print("NSDE Algo")
    pop_size = len(population)
    next_gen_size = int(pop_size/2) # Size of new generation = N
    
    next_gen = []
    # best_accuracy = 0

    # Map all candidates from float to integer space
    mapped_pop = []
    for vector in population:
        mapped_pop.append(float_to_int_mapping(vector))
    # NOTE: First half of mapped_pop has parent vectors, second half has trial vectors

    # Compute and store obj function values for all candidates
    # obj_func_1 = training loss
    # obj_func_2 = no of latent nodes 
    # For the first generation, calculate all values
    print("Training model...")

    global obj_func_1
    global obj_func_2

    # Create pool of processes
    # pool = multiprocessing.Pool()

    if (gen == 0):
        # Initialize 
        obj_func_1 = [0] * pop_size
        obj_func_2 = [0] * pop_size

        for i in range(0, pop_size):
            obj_func_1[i] = functions[0](mapped_pop[i])
            # if accuracy > best_accuracy: best_accuracy = accuracy
            obj_func_2[i] = functions[1](mapped_pop[i])

    else:
        # We only recalculate the trial vector function values, since parent vector function values
        # were computed in previous generation
        for i in range(next_gen_size, pop_size):
            obj_func_1[i] = functions[0](mapped_pop[i])
            # if accuracy > best_accuracy: best_accuracy = accuracy
            obj_func_2[i] = functions[1](mapped_pop[i])
            
    unfilled_spots = next_gen_size # initially all spots are empty
    while(len(next_gen) < next_gen_size):
        # Generate a front
        front, mapped_front, front_indices, new_population, new_mapped_pop = get_front(population, mapped_pop)

        # If size of front is smaller than the remaining spots, add all front members to next gen
        if (len(front) <= unfilled_spots):
            for member in front:
                next_gen.append(member)
            # Write obj functions of front to output file
            for index in front_indices:
                f1.write(str(obj_func_1[index]))
                f1.write(",")
                f2.write(str(obj_func_2[index]))
                f2.write(",")
            unfilled_spots -= len(front)

        # If size of front is greater than the remaining spots, choose members using crowding distance algo
        else:
            chosen_members, chosen_indices = crowding_distance(front, front_indices, int(unfilled_spots))
            for member in chosen_members:
                next_gen.append(member)
            for index in chosen_indices:
                f1.write(str(obj_func_1[index]))
                f1.write(",")
                f2.write(str(obj_func_2[index]))
                f2.write(",")
            unfilled_spots -= len(chosen_members)

        # Update population
        population = new_population[:]
        mapped_pop = new_mapped_pop[:]
    f1.write("\n")
    f2.write("\n")
    # return next_gen, best_accuracy

# Replace every negative value with a random value between 0 and 1
def approx_trial(trial):
    return [SystemRandom().uniform(0, 1) if (val < 0 or val > 1) else val for val in trial]

def moode(pop_size, cand_size, n_inputs, gens, functions):
    # Create the random number mapping to map from float to int space
    initialize_mapping(n_inputs)
    # best_accuracies = []

    # Set the constraints for each gene
    for key, value in mapping.items():
        if value == n_inputs: 
            latent_upper_lim = key
            break
    limits = set_limits([0, 1, 0, latent_upper_lim])
    

    # Open file to write output
    f1 = open("outputx.csv", "a")
    f2 = open("outputy.csv", "a")
    # Clear contents
    f1.seek(0)
    f1.truncate()
    f2.seek(0)
    f2.truncate()

    # Create initial population of parents
    parents = initialize_candidates(cand_size, pop_size, limits)
    for g in range(gens):
        start_time = time.time()

        print('Generation ', g+1, ':\n')

        trials = []
        F = get_F()
        for index, candidate in enumerate(parents):
            retry_count = 0
            while(True):
                # Perform mutation and crossover to get trial and check constraints on trial
                mutant = mutation(parents, index, K, F, candidate)
                trial = crossover(mutant, candidate)
                if (check_constraints(trial, limits) == True): break
                retry_count += 1
                # If too many retries, replace negative values with random values in trial vector
                if (retry_count == 5000):
                    trial = approx_trial(trial)
                    break
            trials.append(np.array(trial))

        # NSDE selection
        population = parents + trials
        next_gen = nsde(population, functions, g, f1, f2)
        # best_accuracies.append(best_accuracy)
        parents = next_gen[:]
        print("--- %s seconds ---" % (time.time() - start_time))

        # Force write buffer contents to file
        f1.flush()
        f2.flush()
    
    # print("Best accuracies across epochs: ", best_accuracies)
    f1.close()
    f2.close()
    return next_gen


    # With threshold = 1000, around 8-10% vectors are approximated, 34 secs
    # With threshold = 1500, around 4-5% vectors are approximated, 25 secs
    # With threshold = 2000, around 4-5% vectors are approximated, 48 secs