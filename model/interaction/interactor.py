#
# Agent and two subclasses, Pedestrian and Vehicle
#
# @todo: Move some of the Vehicle functionality into a Car subclass
# @todo: Add other vehicle subclasses, e.g. Bicycle 
#

# import numpy as np
# import scipy as sp
# import scipy.stats as ss

# from collections import deque

# import view.defs as svd
# import view.view as svv
# import model.agency.defs as smad
# import model.interaction.codec as smic
# import model.interaction.interaction as smii
# import model.agency.state as smas
# import model.geometry.shape as smgs
# import model.geometry.grid as smgg
# import model.defs as smd
# import model.force as smf
# import model.numerical.defs as smnd
# import model.topology.approachnode as smtApp
# import model.topology.activitynode as smtAct

class Interactor:
    #
    # Done(was @todo): Pass route instead of src, dst.
    #   cur_src_idx = true_src_idx = route[0]
    #   cur_dst_idx  = route[1]
    #   true_dst_idx = route[-1]
    #
    def __init__(self, id, physical):
        self.id              = id
        self.physical        = physical

        return

    def register_model(self, model):
        self.model = model
        return

    def is_physical(self):
        return physical

    def get_current_shape(self, u):
        return None

