#
# Agent and two subclasses, Pedestrian and Vehicle
#
# @todo: Move some of the Vehicle functionality into a Car subclass
# @todo: Add other vehicle subclasses, e.g. Bicycle 
#

import numpy as np
import model.configuration as smc

#
# Agent Types
#
from enum import IntEnum
class AgentType(IntEnum):
    PEDESTRIAN = 0
    BICYCLE    = 1
    MOTORCYCLE = 2
    CAR        = 3
    TRUCK      = 4
    BUS        = 5
    TRAM       = 6
    N_TYPES    = 7

d_audio = {AgentType.PEDESTRIAN:2, AgentType.CAR:6}

theta_max = {}
theta_max[AgentType.PEDESTRIAN] = smc.T_step * smc.omega_max_PTN
theta_max[AgentType.CAR]        = smc.T_step * smc.omega_max_CAR

cos_theta_max = {}
cos_theta_max[AgentType.PEDESTRIAN] = np.cos(smc.T_step * smc.omega_max_PTN)
cos_theta_max[AgentType.CAR]        = np.cos(smc.T_step * smc.omega_max_CAR)

sin_theta_max = {}
sin_theta_max[AgentType.PEDESTRIAN] = np.sin(smc.T_step * smc.omega_max_PTN)
sin_theta_max[AgentType.CAR]        = np.sin(smc.T_step * smc.omega_max_CAR)


#
# Agent Identifiers
#
AGENT_ID_BITLENGTH = 13

next_id = 0
def get_new_id():
    global next_id
    id = next_id
    next_id += 1
    assert(next_id <= (1 << AGENT_ID_BITLENGTH))
    return id

#
# Centroids will reserve blocks of consecutive ids, so this function supports that.
# @todo: Allow this function to return fewer ids if need be? This would require an
#     actual length to be returned.
#
def reserve_id_block(length):
    global next_id
    id = next_id
    next_id += length
    assert(next_id <= (1 << AGENT_ID_BITLENGTH))
    return id

