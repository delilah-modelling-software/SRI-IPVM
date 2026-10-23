#
# Streamspawner and three subclasses: Clusterspawner, Betaspawner, and Uniformspawner
#

import numpy as np
import scipy as sp
import scipy.stats as ss

# from collections import deque
# import multimodal_network.view as mnv


class Streamspawner:
    def __init__(self):
        return

    def generate_spawning_times(self, T_simulation, T_step, n_agents):
        return []

#
# @todo: Add a T_period to Clusterspawner and Betaspawner, after which the spawning times will be calculated
# in two stages: determine (with uniform distribution) within which period of the simulation each spawning event
# will occur and then determine (using appropriate GPDF) when within that period the spawning even will occur.
#

#
# Generalised PDF (probability distribution function) is the Dirac delta function. All pedestrians spawn at once.
# Suitable for people who have arrived in an enclosed box, e.g. elevators, buses, etc.
#
class Clusterspawner(Streamspawner):
    def __init__(self, timeshift):
        super(Clusterspawner, self).__init__(self)
        self.T_shift = timeshift
        return

    def generate_spawning_times(self, T_simulation, T_step, n_agents):
        for k in range(int(n_agents)):
            t_sample = np.fmod(sample_times[k] + self.T_shift, T_simulation)
            step = int(sample_times[k] / T_step)
            spawning_list.append(step)
            # print("Spawning pedestrians for ({}, {}) @ time-step {}".format(i, j, step0))
        spawning_list.sort()
        return spawning_list

#
# PDF (probability distribution function) is the uniform distribution. Pedestrians
# spawn continuously at a uniform rate. Suitable for people who have arrived without
# passing through traffic signalised crossings, e.g. footbridges and pedestrian tunnels.
#
class Uniformspawner(Streamspawner):
    def __init__(self):
        super(Uniformspawner, self).__init__()
        return
    
    def generate_spawning_times(self, T_simulation, T_step, n_agents):
        sample_times = ss.uniform.rvs(loc = 0, scale = T_simulation, size = n_agents)
        spawning_list = []
        for k in range(int(n_agents)):
            t_sample = sample_times[k]
            step     = int(t_sample / T_step)
            spawning_list.append(step)
            # print("Spawning agents for ({}, {}) @ time-step {}".format(i, j, step0))
        spawning_list.sort()
        return spawning_list

#
# PDF (probability distribution function) is a beta distribution. Pedestrians
# spawn continuously with one or two modal times. Suitable for people who have arrived by
# passing through timed bottlenecks, e.g. signalised crossings.
#
class Betaspawner(Streamspawner):
    def __init__(self, timeshift, a, b):
        super(Betaspawner, self).__init__()
        self.T_shift = timeshift
        self.a = a
        self.b = b
        return

    def generate_spawning_times(self, T_simulation, T_step, n_agents):
        sample_times  = ss.beta.rvs(self.a, self.b, loc = 0, scale = T_simulation, size = n_agents)
        spawning_list = []
        for k in range(int(n_agents)):
            t_sample = np.fmod(sample_times[k] + self.T_shift, T_simulation)
            step     = int(t_sample / T_step)
            spawning_list.append(step)
            # print("Spawning pedestrians for ({}, {}) @ time-step {}".format(i, j, step0))
        spawning_list.sort()
        return spawning_list
