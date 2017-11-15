import sympy as sp
import numpy as np
import math
import random
from AircraftIden import FreqIdenSIMO
import matplotlib.pyplot as plt
from scipy.optimize import minimize


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
        self.s = sp.symbols('s')
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

    def calcuate_symbolic_trans_matrix(self):
        print("Try to calc symbolic transfer matrix")
        M = self.M
        F = self.F
        G = self.G
        H0 = self.H0
        H1 = self.H1
        Minv = M ** -1
        s = self.s
        Tpart2 = ((s * sp.eye(self.dims) - Minv * F) ** -1) * Minv * G
        self.T = (H0 + s * H1) * Tpart2
        print("Symbolic transfer matrix {}".format(self.T))

    def calucate_transfer_matrix(self, sym_subs):
        # sym_subs = dict()
        M_num = self.M.evalf(subs=sym_subs)
        F_num = self.F.evalf(subs=sym_subs)
        G_num = self.G.evalf(subs=sym_subs)
        H0_num = self.H0.evalf(subs=sym_subs)
        H1_num = self.H1.evalf(subs=sym_subs)
        Minv = M_num ** -1
        s = self.s
        Tpart2 = ((s * sp.eye(self.dims) - Minv * F_num) ** -1) * Minv * G_num
        self.T = (H0_num + s * H1_num) * Tpart2

    def calucate_transfer_matrix_at_s(self, sym_subs, s):
        # sym_subs = dict()
        M_num = self.M.evalf(subs=sym_subs)
        F_num = self.F.evalf(subs=sym_subs)
        G_num = self.G.evalf(subs=sym_subs)
        H0_num = self.H0.evalf(subs=sym_subs)
        H1_num = self.H1.evalf(subs=sym_subs)
        Minv = M_num ** -1
        Tpart2 = ((s * sp.eye(self.dims) - Minv * F_num) ** -1) * Minv * G_num
        self.Tnum = (H0_num + s * H1_num) * Tpart2

    def get_transfer_func(self, y_index, u_index):
        # Must be run after cal
        assert self.T is not None, "Must run calucate_transfer_matrix first"
        return self.T[y_index, u_index]

    def get_amp_pha_from_matrix(self, u_index, y_index):
        h = self.Tnum[y_index, u_index]
        h = complex(h)
        amp = 20 * np.log10(np.absolute(h))
        pha = np.arctan2(h.imag, h.real) * 180 / math.pi
        return amp, pha


class StateSpaceIdenSIMO(object):
    def __init__(self, freq, Hs, coherens, nw=20, enable_debug_plot=False):
        self.freq = freq
        self.Hs = Hs
        self.wg = 1.0
        self.wp = 0.01745
        self.est_omg_ptr_list = []
        self.enable_debug_plot = enable_debug_plot
        self.coherens = coherens
        self.nw = nw

    def cost_func(self, ssm: StateSpaceModel, x):
        sym_sub = dict()
        assert len(x) == len(self.x_syms), 'State length must be equal with x syms'
        # setup state x
        for i in range(len(x)):
            sym_sub[self.x_syms[i]] = x[i]

        # init transfer matrix
        sym_sub.update(self.constant_defines)

        # print(sym_sub)
        # ssm.calucate_transfer_matrix(sym_sub)

        def cost_func_at_omg_ptr_chn(omg_ptr, y_index):
            # amp, pha = ssm.get_amp_pha_from_trans(trans, omg)
            amp, pha = ssm.get_amp_pha_from_matrix(0, y_index)
            h = self.Hs[y_index][omg_ptr]
            h_amp = 20 * np.log10(np.absolute(h))
            h_pha = np.arctan2(h.imag, h.real) * 180 / math.pi
            pha_err = h_pha - pha
            if pha_err > 180:
                pha_err = pha_err - 360
            if pha_err < -180:
                pha_err = pha_err + 360
            J = self.wg * pow(h_amp - amp, 2) + self.wp * pow(pha_err, 2)

            gama2 = self.coherens[y_index][omg_ptr]

            wgamma = 1.58 * (1 - math.exp(-gama2 * gama2))
            wgamma = wgamma * wgamma
            return J * wgamma

        def cost_func_at_omg_ptr(omg_ptr):
            omg = self.freq[omg_ptr]
            ssm.calucate_transfer_matrix_at_s(sym_sub, omg * 1j)
            chn_cost_func = lambda chn: cost_func_at_omg_ptr_chn(omg_ptr, chn)
            chn_cost_func = np.vectorize(chn_cost_func)
            J_arr = chn_cost_func(range(ssm.y_dims))
            J = np.average(J_arr)
            return J

        omg_ptr_cost_func = np.vectorize(cost_func_at_omg_ptr)
        J = np.average(omg_ptr_cost_func(self.est_omg_ptr_list)) * 20
        print(x, J)
        return J

    def estimate(self, ssm: StateSpaceModel, syms, omg_min=None, omg_max=None, constant_defines=None):
        if constant_defines is None:
            constant_defines = dict()
        self.constant_defines = constant_defines
        self.init_omg_list(omg_min, omg_max)
        self.syms = syms
        self.x_syms = []

        for sym in syms:
            if sym in constant_defines.keys():
                print("Known param {}:{}".format(sym, constant_defines[sym]))
            else:
                self.x_syms.append(sym)
        self.x_dims = len(self.x_syms)
        print("Will estimate num {} {}".format(self.x_syms.__len__(), self.x_syms))

        self.solve(lambda x: self.cost_func(ssm, x))

    def solve(self, f):
        x0 = self.setup_initvals()
        bounds = [(None, None) for i in range(len(x0))]
        # bounds[-1] = (0, 0.1)
        ret = minimize(f, x0, options={'maxiter': 10000000, 'disp': False}, bounds=bounds, tol=1e-15)
        x = ret.x.copy()
        J = ret.fun
        return x, J

    def setup_initvals(self):
        x0 = np.random.rand(self.x_dims)
        return x0

    def init_omg_list(self, omg_min, omg_max):
        if omg_min is None:
            omg_min = self.freq[0]

        if omg_max is None:
            omg_max = self.freq[-1]

        omg_list = np.linspace(np.log(omg_min), np.log(omg_max), self.nw)
        omg_list = np.exp(omg_list)
        # print("omg list {}".format(omg_list))

        omg_ptr = 0
        self.est_omg_ptr_list = []
        for i in range(self.freq.__len__()):
            freq = self.freq[i]
            if freq > omg_list[omg_ptr]:
                self.est_omg_ptr_list.append(i)
                omg_ptr = omg_ptr + 1
            elif omg_ptr < omg_list.__len__() and i == self.freq.__len__() - 1:
                self.est_omg_ptr_list.append(i)
                omg_ptr = omg_ptr + 1


def lat_dyn_example():
    # save_data_list = ["running_time", "yoke_pitch", "theta", "airspeed", "q", "aoa", "VVI", "alt"]
    arr = np.load("../data/sweep_data_2017_10_18_14_07.npy")
    time_seq_source = arr[:, 0]
    ele_seq_source = arr[:, 1]
    q_seq_source = arr[:, 4]
    vvi_seq_source = arr[:, 6]
    theta_seq_source = arr[:, 2]
    airspeed_seq_source = arr[:, 3]

    simo_iden = FreqIdenSIMO(time_seq_source, 0.5, 50, ele_seq_source, airspeed_seq_source, vvi_seq_source,
                             q_seq_source, theta_seq_source, win_num=32)

    plt.figure("Ele->Airspeed")
    simo_iden.plt_bode_plot(0)
    plt.figure("Ele->VVI")
    simo_iden.plt_bode_plot(1)
    plt.figure("Ele->Q")
    simo_iden.plt_bode_plot(2)
    plt.figure("Ele->Th")
    simo_iden.plt_bode_plot(3)

    plt.pause(1)

    freq, Hs, coherens = simo_iden.get_all_idens()
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
    # lat_dyn_state_space.calucate_transfer_matrix(subs)
    # ele2u = lat_dyn_state_space.get_transfer_func(0, 0)
    ssm_iden = StateSpaceIdenSIMO(freq, Hs, coherens)
    # def estimate(self, ssm: StateSpaceModel, syms, omg_min=None, omg_max=None, constant_defines=None):
    U_trim = airspeed_seq_source[0]
    W_trim = vvi_seq_source[0]
    th_trim = theta_seq_source[0]

    print("using trim U {} W {} th {}".format(U_trim, W_trim, th_trim))
    ssm_iden.estimate(lat_dyn_state_space, syms, constant_defines={U0: U_trim, W0: W_trim, th0: th_trim})


def test_pure_symbloic():
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
    syms = [Xwdot, Zwdot, Mwdot,
            Xu, Xw, Xq, W0, th0,
            Zu, Zw, Zq, U0,
            Mu, Mw, Mq,
            Xele, Zele, Mele]
    subs = dict()
    for key in syms:
        subs[key] = random.random()
    lat_dyn_state_space.calucate_transfer_matrix_at_s(subs, 36J)
    print(lat_dyn_state_space.T)


if __name__ == "__main__":
    lat_dyn_example()
    # test_pure_symbloic()
