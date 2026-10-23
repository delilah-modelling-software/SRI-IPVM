# from enum import Enum

# import numpy as np
# import scipy as sp
# import scipy.stats as ss

# import model.defs as smd
# import model.agency.defs as smad
# import model.numerical.defs as smnd
# import model.zone.defs as smzd
# import model.zone.zone as smzz

#
# interaction-codec functions.
#

class DemandInfo:
    def __init__(self, activity_probability, activity_popularities):
        self.activity_probability  = activity_probability
        self.activity_popularities = activity_popularities

class ActivityArea:
    def __init__(self, vestibule_shape, foci, barriers):
        self.vestibule_shape = vestibule_shape
        self.foci            = foci
        self.barriers        = barriers
        return

    # def register_activity_area(self, model):
        # vestibule = 
