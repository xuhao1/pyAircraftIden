import sympy as sp
import numpy as np
import math
import random


class StateSpaceModel(object):
    def __init__(self, M: sp.Matrix, F: sp.Matrix, G: sp.Matrix, H0: sp.Matrix, H1: sp.Matrix, syms: list):
        # M \dot x = F x + G u
        # y = H0 X + H1 \dot x
        # Matrix M F G H0 H1 include symbolic item, all is list in syms
        self.M = M
        self.F = F
        self.G = G
        self.H0 = H0
        self.H1 = H1
        self.syms = syms

        self.check_dims()

        self.check_syms()
        self.T = None
        pass

    def check_dims(self):
        # 0 - - -n
        # |
        # |
        # m
        self.dims, n = self.M.shape
        assert self.dims == n, "Shape of M must equal"
        m, n = self.F.shape
        assert m == n == self.dims, 'Error on F shape needs {0}x{0} got {1}x{2}'.format(self.dims, m, n)

        m, n = self.G.shape
        self.u_dims = n
        assert m == self.dims, 'Error on G shape needs {0}x{2} got {1}x{2}'.format(self.dims, m, n)

        m, n = self.H0.shape
        self.y_dims = m
        assert n == self.dims, 'Error on H0 shape needs {1}x{0} got {1}x{2}'.format(self.dims, m, n)

        m, n = self.H1.shape
        assert n == self.dims and m == self.y_dims, 'Error on H0 shape needs {1}x{0} got {2}x{3}'.format(self.dims,
                                                                                                         self.y_dims, m,
                                                                                                         n)

    def check_syms(self):
        pass

    def calucate_transfer_matrix(self, sym_subs):
        # sym_subs = dict()
        M_num = self.M.evalf(subs=sym_subs)
        F_num = self.F.evalf(subs=sym_subs)
        G_num = self.G.evalf(subs=sym_subs)
        H0_num = self.H0.evalf(subs=sym_subs)
        H1_num = self.H1.evalf(subs=sym_subs)
        Minv = M_num ** -1
        s = sp.symbols('s')
        Tpart2 = ((s * sp.eye(self.dims) - Minv * F_num) ** -1) * Minv * G_num
        self.T = (H0_num + s * H1_num) * Tpart2

    def get_transfer_func(self, y_index, u_index):
        # Must be run after cal
        assert self.T is not None, "Must run calucate_transfer_matrix first"
        return self.T[u_index, y_index]


class StateSpaceIdenSIMO(object):
    def __init__(self, freq, Hs, coheren, nw=20, enable_debug_plot=False):
        self.freq = freq
        self.H = Hs
        self.wg = 1.0
        self.wp = 0.01745
        self.est_omg_ptr_list = []
        self.enable_debug_plot = enable_debug_plot
        self.coheren = coheren
        self.nw = nw

    def channel_cost_func(self):
        pass

    def estimate(self, state_space_model, syms, constant_defines=None):
        if constant_defines is None:
            constant_defines = dict()

        pass


def lat_dyn_example():
    # X = [u,w,q,th]
    Xwdot, Zwdot, Mwdot = sp.symbols('Xwdot Zwdot Mwdot')

    M = sp.Matrix([[1, -Xwdot, 0, 0],
                   [0, 1 - Zwdot, 0, 0],
                   [0, -Mwdot, 1, 0],
                   [0, 0, 0, 1]])

    g = 9.78
    Xu, Xw, Xq, W0, th0 = sp.symbols('Xu Xw Xq W0 th0')
    Zu, Zw, Zq, U0 = sp.symbols('Zu Zw Zq U0')
    Mu, Mw, Mq = sp.symbols('Mu Mw Mq')

    F = sp.Matrix([[Xu, Xw, Xq - W0, -g * sp.cos(th0)],
                   [Zu, Zw, Zq + U0, -g * sp.sin(th0)],
                   [Mu, Mw, Mq, 0],
                   [0, 0, 1, 0]])

    Xele, Zele, Mele = sp.symbols('Xele,Zele,Mele')
    G = sp.Matrix([[Xele],
                   [Zele],
                   [Mele],
                   [0]])

    # direct using u w q th for y
    H0 = sp.Matrix([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

    H1 = sp.Matrix.zeros(4, 4)

    syms = [Xwdot, Zwdot, Mwdot,
            Xu, Xw, Xq, W0, th0,
            Zu, Zw, Zq, U0,
            Mu, Mw, Mq,
            Xele, Zele, Mele]
    lat_dyn_state_space = StateSpaceModel(M, F, G, H0, H1, syms)

    subs = dict()
    for key in syms:
        subs[key] = random.random()
    print("using subs {}".format(subs))
    lat_dyn_state_space.calucate_transfer_matrix(subs)
    ele2u = lat_dyn_state_space.get_transfer_func(0, 0)
    pass


if __name__ == "__main__":
    lat_dyn_example()
