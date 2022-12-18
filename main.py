from moode_funcs import moode
from autoencoder import train_model, latent_count
from config import *
import numpy as np
import time

if __name__ == "__main__":
    # func = Chanking_Haimes()
    # func_name = 'Autoencoder'
    functions = [train_model, latent_count]

    start_time = time.time()
    # next_gen = moode(population_size, candidate_size, no_of_inputs, num_of_gens, [func.f1, func.f2])
    next_gen = moode(population_size, candidate_size, no_of_inputs, num_of_gens, functions)
    print("--- %s seconds ---" % (time.time() - start_time))

    # Plotting
    # plot_2_obj(next_gen, [train_model, latent_count])