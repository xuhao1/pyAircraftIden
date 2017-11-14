import sympy
class StateSpaceIdenSIMO(object):
    def __init__(self, freq, H, coheren, nw=20, enable_debug_plot=False):
        self.freq = freq
        self.H  = H
        self.wg = 1.0
        self.wp = 0.01745
        self.est_omg_ptr_list = []
        self.enable_debug_plot = enable_debug_plot
        self.coheren = coheren
        self.nw = nw

    #def estimate(self):