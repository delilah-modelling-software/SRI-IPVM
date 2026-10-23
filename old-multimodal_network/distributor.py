#
# Streamspawner and three subclasses: Clusterspawner, Betaspawner, and Uniformspawner
#

import numpy as np
import scipy as sp
import scipy.stats as ss

# from collections import deque
# import multimodal_network.view as mnv


class Distributor:
    def __init__(self):
        return

    def generate_values(self, T_extent, T_step, n_agents):
        return []

#
# @todo: Add a T_period to Clusterspawner and Betaspawner, after which the spawning times will be calculated
# in two stages: determine (with uniform distribution) within which period of the simulation each spawning event
# will occur and then determine (using appropriate GPDF) when within that period the spawning even will occur.
#

class DistributorPeriodic(Distributor):
    def __init__(self, T_period, child_distributor):
        super(DistributorPeriodic, self).__init__(self)
        self.t_period = T_period
        return
    
    def generate_values(self, T_extent, T_step, n_agents):
        assert(T_extent % self.T_period == 0)
        n_periods      = np.ceil(T_extent / T_period)
        uniform_values = ss.uniform.rvs(loc = 0, scale = T_extent, size = n_agents)

        periods = np.floor(uniform_values / self.period)
        steps   = child_distributor.generate_values(self.T_period, T_step, n_agents)
        steps  += periods
        steps.sort()

#
# Generalised PDF (probability distribution function) is the Dirac delta function. All pedestrians spawn at once.
# Suitable for people who have arrived in an enclosed box, e.g. elevators, buses, etc.
#
class DistributorCluster(Distributor):
    def __init__(self, timeshift):
        super(DistributorCluster, self).__init__(self)
        self.T_shift = timeshift
        return

    def generate_values(self, T_simulation, T_step, n_agents):
        values = np.zeros(n_agents)
        shifted_values = np.floor(np.fmod(values + self.T_shift, T_simulation)).astype(int)
        shifted_values.sort()
        return shifted_values

#
# PDF (probability distribution function) is the uniform distribution. Pedestrians
# spawn continuously at a uniform rate. Suitable for people who have arrived without
# passing through traffic signalised crossings, e.g. footbridges and pedestrian tunnels.
#
class DistributorUniform(Distributor):
    def __init__(self):
        super(DistributorUniform, self).__init__()
        return
    
    def generate_values(self, T_simulation, T_step, n_agents):
        values = ss.uniform.rvs(loc = 0, scale = T_simulation, size = n_agents)
        steps  = np.floor(values / T_step).astype(int)
        steps.sort()
        return steps

#
# PDF (probability distribution function) is a beta distribution. Pedestrians
# spawn continuously with one or two modal times. Suitable for people who have arrived by
# passing through timed bottlenecks, e.g. signalised crossings.
#
class DistributorBeta(Distributor):
    def __init__(self, timeshift, a, b):
        super(DistributorBeta, self).__init__()
        self.T_shift = timeshift
        self.a = a
        self.b = b
        return

    def generate_values(self, T_simulation, T_step, n_agents):
        values         = ss.beta.rvs(self.a, self.b, loc = 0, scale = T_simulation, size = n_agents)
        shifted_values = np.floor(np.fmod(values + self.T_shift, T_simulation)).astype(int)
        shifted_values.sort()
        return shifted_values
