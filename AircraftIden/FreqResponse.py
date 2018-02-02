import numpy as np
import scipy


class FreqResponse(object):
    def __init__(self, freq, Hs,cohers = None,trims = None):
        self.freq = freq
        self.Hs = Hs
        self.coherens = cohers
        self.trims = trims
        pass
