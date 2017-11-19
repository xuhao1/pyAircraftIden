import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import math


class StateSpaceParamModel(object):
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

    def show_formula(self):
        a_mat = sp.latex(self.A)
        fig = plt.figure("Formula")
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, self.A)

        plt.pause(0.1)
        pass

    def load_constant_defines(self, constant_syms):
        A = self.A.evalf(subs=constant_syms)
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

        self.show_formula()
        # print("new A {} B {}".format(self.A_converted, self.B_converted))
        # print("new H0 {} H1 {}".format(self.H0_converted, self.H1_converted))

    def solve_params_from_newparams(self, x):
        equs = []
        assert self.new_params_defs_list.__len__() == self.new_params_list.__len__()
        for i in range(self.new_params_defs_list.__len__()):
            equs.append(self.new_params_defs_list[i] - x[i])
        # print("Solving equs {}".format(equs))
        # print("Unknown {}".format(self.syms))
        solvs = sp.solve(equs, tuple(self.syms))
        assert solvs[0].__len__() == self.syms.__len__(), "solvs {} cannot recover syms {}".format(solvs, self.syms)
        return dict(zip(self.syms, solvs[0]))

    def get_new_params(self):
        return self.new_params_list

    def determine_unknown_from_mat(self, mat, matname):
        m, n = mat.shape
        matnew = np.zeros([m, n])
        for i in range(m):
            for j in range(n):
                element = mat[i, j]
                if not element.is_number:
                    # print("{} {} {} unkown : {}".format(matname, i, j, element))
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

    def get_transfer_func(self, y_index, u_index):
        # Must be run after cal
        assert self.T is not None, "Must run calucate_transfer_matrix first"
        return self.T[y_index, u_index]

    @staticmethod
    def get_amp_pha_from_matrix(Tnum, u_index, y_index):
        h = Tnum[y_index, u_index]
        amp = 20 * np.log10(np.absolute(h))
        pha = np.arctan2(h.imag, h.real) * 180 / math.pi
        return amp, pha

    def get_ssm_by_syms(self, sym_subs, using_converted=False):
        A_num = self.A_numeric.copy()
        B_num = self.B_numeric.copy()
        H0_num = self.H0_numeric.copy()
        H1_num = self.H1_numeric.copy()
        if not using_converted:
            A_num = sp.matrix2numpy(self.A.evalf(subs=sym_subs), dtype=np.complex)
            B_num = sp.matrix2numpy(self.B.evalf(subs=sym_subs), dtype=np.complex)
            H0_num = sp.matrix2numpy(self.H0.evalf(subs=sym_subs), dtype=np.complex)
            H1_num = sp.matrix2numpy(self.H1.evalf(subs=sym_subs), dtype=np.complex)
        else:
            for sym, v in sym_subs.items():
                mat_process = None
                # v = sym_subs[sym]
                (mn, i, j) = self.new_params_raw_pos[sym]
                if mn == "A":
                    mat_process = A_num
                elif mn == "B":
                    mat_process = B_num
                elif mn == "H0":
                    mat_process = H0_num
                elif mn == "H1":
                    mat_process = H1_num
                assert mat_process is not None, "Mat name {} illegal".format(mn)
                mat_process[i][j] = v
        return StateSpaceModel(A_num, B_num, H0_num, H1_num)


class StateSpaceModel():
    def __init__(self, A_num, B_num, H0_num, H1_num=None):
        if H1_num is None:
            H1_num = np.zeros(H0_num.shape)
        self.A = A_num
        self.B = B_num
        self.H0 = H0_num
        self.H1 = H1_num
        self.check_dims()

    def calucate_transfer_matrix_at_omg(self, omg):
        s = omg * 1J
        TT = np.linalg.inv((s * np.eye(self.dims) - self.A))
        Tpart2 = np.dot(TT, self.B)
        Tnum = np.dot((self.H0 + s * self.H1), Tpart2)
        return Tnum

    def check_dims(self):
        # 0 - - -n
        # |
        # |
        # m
        self.dims, n = self.A.shape
        assert self.dims == n, "Shape of M must equal"
        m, n = self.A.shape
        assert m == n == self.dims, 'Error on F shape needs {0}x{0} got {1}x{2}'.format(self.dims, m, n)

        m, n = self.B.shape
        self.u_dims = n
        assert m == self.dims, 'Error on G shape needs {0}x{2} got {1}x{2}'.format(self.dims, m, n)

        m, n = self.H0.shape
        self.y_dims = m
        assert n == self.dims, 'Error on H0 shape needs {1}x{0} got {1}x{2}'.format(self.dims, m, n)

        m, n = self.H1.shape
        assert n == self.dims and m == self.y_dims, 'Error on H0 shape needs {1}x{0} got {2}x{3}'.format(self.dims,
                                                                                                         self.y_dims, m,
                                                                                                         n)
