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

        self.pro_calc()

        self.new_params_raw_defines = dict()
        self.new_params_raw_pos = dict()
        self.new_params_list = list()
        self.new_params_defs_list = list()
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

    def pro_calc(self):
        M_inv = self.M ** -1
        self.A = M_inv * self.F
        self.B = M_inv * self.G
        # print("A {} B {}".format(self.A,self.B))

    def load_constant_defines(self, constant_syms):
        # print(constant_syms)
        # print(self.A)
        A = self.A.evalf(subs=constant_syms)
        print(A)
        B = self.B.evalf(subs=constant_syms)
        H0 = self.H0.evalf(subs=constant_syms)
        H1 = self.H1.evalf(subs=constant_syms)

        self.A_converted = A.copy()
        self.B_converted = B.copy()
        self.H0_converted = H0.copy()
        self.H1_converted = H1.copy()

        self.A_numeric = self.determine_unknown_from_mat(self.A_converted, 'A')
        self.B_numeric = self.determine_unknown_from_mat(self.B_converted, 'B')
        self.H0_numeric = self.determine_unknown_from_mat(self.H0_converted, 'H0')
        self.H1_numeric = self.determine_unknown_from_mat(self.H1_converted, 'H1')

        print("new A {} B {}".format(self.A_converted, self.B_converted))
        print("new H0 {} H1 {}".format(self.H0_converted, self.H1_converted))

        # self.solve_params_from_newparams()

    def solve_params_from_newparams(self,x):
        equs = []
        assert self.new_params_defs_list.__len__() == self.new_params_list.__len__()
        for i in range(self.new_params_defs_list.__len__()):
            equs.append(self.new_params_defs_list[i] - x[i])
        print("Solving equs {}".format(equs))
        print("Unknown {}".format(self.syms))
        solvs = sp.solve(equs, tuple(self.syms))
        print("solves",solvs)
        return solvs

    def get_new_params(self):
        return self.new_params_list

    def determine_unknown_from_mat(self, mat, matname):
        m, n = mat.shape
        matnew = np.zeros([m, n])
        for i in range(m):
            for j in range(n):
                element = mat[i, j]
                if not element.is_number:
                    print("{} {} {} unkown : {}".format(matname, i, j, element))
                    new_param_name = "{}_{}_{}".format(matname, i, j)
                    new_param = sp.symbols(new_param_name)
                    self.new_params_raw_defines[new_param] = element
                    self.new_params_list.append(new_param)
                    self.new_params_defs_list.append(element)
                    self.new_params_raw_pos[new_param] = (matname, i, j)
                    mat[i, j] = new_param
                else:
                    matnew[i, j] = element
        return matnew

    def calucate_transfer_matrix(self, sym_subs):
        # sym_subs = dict()
        A_num = self.A.evalf(subs=sym_subs)
        B_num = self.B.evalf(subs=sym_subs)
        H0_num = self.H0.evalf(subs=sym_subs)
        H1_num = self.H1.evalf(subs=sym_subs)

        s = self.s

        Tpart2 = (s * np.eye(self.dims) - A_num) ** -1 * B_num
        self.T = (H0_num + s * H1_num) * Tpart2

    def calucate_transfer_matrix_at_s(self, sym_subs, s, using_converted=False):
        # sym_subs = dict()
        if not using_converted:
            A_num = sp.matrix2numpy(self.A.evalf(subs=sym_subs), dtype=np.complex)
            B_num = sp.matrix2numpy(self.B.evalf(subs=sym_subs), dtype=np.complex)
            H0_num = sp.matrix2numpy(self.H0.evalf(subs=sym_subs), dtype=np.complex)
            H1_num = sp.matrix2numpy(self.H1.evalf(subs=sym_subs), dtype=np.complex)
        else:
            for sym in sym_subs:
                mat_process = None
                v = sym_subs[sym]
                (mn, i, j) = self.new_params_raw_pos[sym]
                if mn == "A":
                    mat_process = self.A_numeric
                elif mn == "B":
                    mat_process = self.B_numeric
                elif mn == "H0":
                    mat_process = self.H0_numeric
                elif mn == "H1":
                    mat_process = self.H1_numeric
                assert mat_process is not None, "Mat name {} illegal".format(mn)
                mat_process[i][j] = v
        A_num = self.A_numeric
        B_num = self.B_numeric
        H0_num = self.H0_numeric
        H1_num = self.H1_numeric

        TT = np.linalg.inv((s * np.eye(self.dims) - A_num))
        Tpart2 = np.dot(TT, B_num)
        self.Tnum = np.dot((H0_num + s * H1_num), Tpart2)

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
    def __init__(self, freq, Hs, coherens, nw=20, enable_debug_plot=False, max_sample_time=100):
        self.freq = freq
        self.Hs = Hs
        self.wg = 1.0
        self.wp = 0.01745
        self.est_omg_ptr_list = []
        self.enable_debug_plot = enable_debug_plot
        self.coherens = coherens
        self.nw = nw
        self.max_sample_time = max_sample_time
        self.accept_J = 100

    def cost_func(self, ssm: StateSpaceModel, x):
        sym_sub = dict()
        assert len(x) == len(self.x_syms), 'State length must be equal with x syms'
        # setup state x
        for i in range(len(x)):
            sym_sub[self.x_syms[i]] = x[i]

        def cost_func_at_omg_ptr(omg_ptr):
            omg = self.freq[omg_ptr]
            ssm.calucate_transfer_matrix_at_s(sym_sub, omg * 1j, using_converted=True)

            def chn_cost_func(y_index):
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

            chn_cost_func = np.vectorize(chn_cost_func)
            J_arr = chn_cost_func(range(ssm.y_dims))
            J = np.average(J_arr)
            return J

        omg_ptr_cost_func = np.vectorize(cost_func_at_omg_ptr)
        J = np.average(omg_ptr_cost_func(self.est_omg_ptr_list)) * 20
        return J

    def estimate(self, ssm: StateSpaceModel, syms, omg_min=None, omg_max=None, constant_defines=None):
        if constant_defines is None:
            constant_defines = dict()
        self.init_omg_list(omg_min, omg_max)

        self.syms = syms
        ssm.load_constant_defines(constant_defines)
        self.x_syms = list(ssm.get_new_params())
        self.x_dims = len(self.x_syms)
        print("Will estimate num {} {}".format(self.x_syms.__len__(), self.x_syms))

        J_min = 1000000
        x = None
        for i in range(self.max_sample_time):
            x_tmp, J = self.solve(ssm)
            if J < J_min:
                print("Found new better res J:{} x {} sampled NUM {}".format(J, x_tmp, i))
                x = x_tmp.copy()
                J_min = J
                if J_min < self.accept_J:
                    break
        ssm.solve_params_from_newparams(x)
        return J_min, x

    # def solving_x_to_params(self,x):
    #     self.
    #     pass

    def solve(self, ssm):
        f = lambda x: self.cost_func(ssm, x)
        x0 = self.setup_initvals(ssm)
        bounds = [(None, None) for i in range(len(x0))]
        # bounds[-1] = (0, 0.1)
        ret = minimize(f, x0, options={'maxiter': 10000000, 'disp': False}, bounds=bounds, tol=1e-15)
        x = ret.x.copy()
        J = ret.fun
        return x, J

    def setup_initvals(self, ssm):
        source_syms = ssm.syms
        source_syms_dims = ssm.syms.__len__()
        source_syms_init_vals = np.random.rand(source_syms_dims) * 2 - 1
        subs = dict(zip(source_syms, source_syms_init_vals))

        x0 = np.zeros(self.x_dims)
        for i in range(self.x_dims):
            sym = self.x_syms[i]
            # Eval sym value from ssm
            sym_def = ssm.new_params_raw_defines[sym]
            v = sym_def.evalf(subs=subs)
            # print("new sym {} symdef {} vinit {}".format(sym, sym_def, v))
            x0[i] = v
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


def lat_dyn_uw():
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


def lat_dyn_example(iter):
    # save_data_list = ["running_time", "yoke_pitch", "theta", "airspeed", "q", "aoa", "VVI", "alt"]
    arr = np.load("../data/sweep_data_2017_10_18_14_07.npy")
    time_seq_source = arr[:, 0]
    ele_seq_source = arr[:, 1]
    q_seq_source = arr[:, 4]
    vvi_seq_source = arr[:, 6]
    theta_seq_source = arr[:, 2] / 180 * math.pi
    airspeed_seq_source = arr[:, 3]
    aoa_seq_source = arr[:, 5] / 180 * math.pi

    # X = [V,aoa,the,q]
    plt.figure("source")
    plt.plot(time_seq_source, q_seq_source, label='q')
    plt.plot(time_seq_source, theta_seq_source, label='theta')
    plt.plot(time_seq_source, aoa_seq_source, label='aoa')
    plt.legend()
    # plt.show()
    simo_iden = FreqIdenSIMO(time_seq_source, 0.5, 50, ele_seq_source, airspeed_seq_source, aoa_seq_source,
                             theta_seq_source, q_seq_source, win_num=32)

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
    Zaldot, Maldot, V0 = sp.symbols('Zaldot Maldot V0')

    V0 = 52.7

    M = sp.Matrix([[1, 0, 0, 0],
                   [0, V0 - Zaldot, 0, 0],
                   [0, 0, 1, 0],
                   [0, -Maldot, 0, 1]])

    g = 9.78
    Xv, Xtv, Xa, al0 = sp.symbols('Xv Xtv Xa al0')
    Zv, Zv, Zq, Za = sp.symbols("Zv Zv Zq Za")
    Mv, Ma, Mq = sp.symbols("Mv Ma Mq")

    al0 = 0.015464836009954524
    F = sp.Matrix([[Xv + Xtv * sp.cos(al0), Xa, -g, 0],
                   [Zv - Xtv * sp.sin(al0), Za, 0, V0 + Zq],
                   [0, 0, 0, 1],
                   [Mv, Ma, 0, Mq]])

    Xele,  Mele = sp.symbols('Xele,Mele')
    G = sp.Matrix([[Xele * sp.cos(al0)],
                   [0],
                   [0],
                   [Mele]])

    # direct using u w q th for y
    H0 = sp.Matrix([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

    H1 = sp.Matrix.zeros(4, 4)

    syms = [Zaldot, Maldot,
            Xv, Xtv, Xa,
            Zv, Zq, Za,
            Mv, Ma, Mq,
            Xele, Mele]
            # V0, al0]
    lat_dyn_state_space = StateSpaceModel(M, F, G, H0, H1, syms)

    ssm_iden = StateSpaceIdenSIMO(freq, Hs, coherens, max_sample_time=iter)
    V_trim = airspeed_seq_source[0]
    th_trim = theta_seq_source[0]
    alpha_trim = aoa_seq_source[0]
    print("using trim V {} aoa {}".format(V_trim, alpha_trim))
    ssm_iden.estimate(lat_dyn_state_space, syms, constant_defines={V0: V_trim, al0: alpha_trim})


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
    constant_defines = {U0: 15, W0: 0, th0: 0.8}
    lat_dyn_state_space.load_constant_defines(constant_defines)
    new_unknown = lat_dyn_state_space.get_new_params()
    print(new_unknown)
    lat_dyn_state_space.calucate_transfer_matrix_at_s(subs, 36J)
    print(lat_dyn_state_space.Tnum)

    syms = lat_dyn_state_space.get_new_params()
    for key in syms:
        subs[key] = random.random()
    lat_dyn_state_space.calucate_transfer_matrix_at_s(subs, 36J, using_converted=True)
    print(lat_dyn_state_space.Tnum)


if __name__ == "__main__":
    lat_dyn_example(10)
    # test_pure_symbloic()
