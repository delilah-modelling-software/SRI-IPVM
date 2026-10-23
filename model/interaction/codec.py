from enum import Enum

# import numpy as np
# import scipy as sp
# import scipy.stats as ss

import model.defs as smd
import model.agency.defs as smad
# import model.numerical.defs as smnd
import model.zone.defs as smzd
# import model.zone.zone as smzz

#
# interaction-codec functions.
#

class Codec:
    def __init__(self, id1_bitlength, id2_bitlength):
        self.id1_bitlength = id1_bitlength
        self.id2_bitlength = id2_bitlength
        self.id1_bitmask   = (1 << id1_bitlength) - 1
        self.id2_bitmask   = (1 << id2_bitlength) - 1
        return

    def encode(self, id1, id2):
        smd.dbgprint("InteractionCodec::encode_interaction between agents {} and {}".format(id1, id2))
        smd.dbgassert(id1 < id2)
        # return id1 * self.max_n_agents + id2
        return id1 << self.id2_bitlength | id2

    def decode(self, intrctn_key):
        # id1 = intrctn_key / self.max_n_agents
        # id2 = intrctn_key % self.max_n_agents
        id1 = (intrctn_key >> self.id2_bitlength) & self.id1_bitmask
        id2 = intrctn_key & self.id2_bitmask
        return (id1, id2)

interagent_codec = Codec(smad.AGENT_ID_BITLENGTH, smad.AGENT_ID_BITLENGTH)
contrazone_codec = Codec(smad.AGENT_ID_BITLENGTH, smzd.ZONE_ID_BITLENGTH)

