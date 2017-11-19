import sympy as sp
import numpy as np
import math
import random
from AircraftIden import FreqIdenSIMO
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import copy
import multiprocessing
from AircraftIden.StateSpaceParamModel import StateSpaceParamModel, StateSpaceModel


class StateSpaceIdenSIMO(object):
    def __init__(self, freq, Hs, coherens, nw=20, enable_debug_plot=False, max_sample_times=10, accept_J=50,
                 y_names=None):
        self.freq = freq
        self.Hs = Hs
        self.wg = 1.0
        self.wp = 0.01745
        self.est_omg_ptr_list = []
        self.enable_debug_plot = enable_debug_plot
        self.coherens = coherens
        self.nw = nw
        self.max_sample_times = max_sample_times
        self.accept_J = accept_J
        self.x_dims = 0
        self.x_syms = []
        self.y_dims = len(Hs)
        self.y_names = y_names

        self.x_best = None
        self.J_min = -1

        self.fig = None

    def cost_func(self, sspm: StateSpaceParamModel, x):
        sym_sub = dict()
        assert len(x) == len(self.x_syms), 'State length must be equal with x syms'
        # setup state x
        sym_sub = dict(zip(self.x_syms, x))
        ssm = sspm.get_ssm_by_syms(sym_sub, using_converted=True)

        def cost_func_at_omg_ptr(omg_ptr):
            omg = self.freq[omg_ptr]
            Tnum = ssm.calucate_transfer_matrix_at_omg(omg)

            def chn_cost_func(y_index):
                # amp, pha = sspm.get_amp_pha_from_trans(trans, omg)
                amp, pha = StateSpaceModel.get_amp_pha_from_matrix(Tnum, 0, y_index)
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
            J_arr = chn_cost_func(range(sspm.y_dims))
            J = np.average(J_arr)
            return J

        omg_ptr_cost_func = np.vectorize(cost_func_at_omg_ptr)
        J = np.average(omg_ptr_cost_func(self.est_omg_ptr_list)) * 20
        return J

    def estimate(self, sspm: StateSpaceParamModel, syms, omg_min=None, omg_max=None, constant_defines=None):
        assert self.y_dims == sspm.y_dims, "StateSpaceModel dim : {} need to iden must have same dims with Hs {}".format(
            sspm.y_dims, self.y_dims)
        if constant_defines is None:
            constant_defines = dict()
        self.init_omg_list(omg_min, omg_max)

        self.syms = syms
        sspm.load_constant_defines(constant_defines)
        self.x_syms = list(sspm.get_new_params())
        self.x_dims = len(self.x_syms)
        print("Will estimate num {} {}".format(self.x_syms.__len__(), self.x_syms))

        J, x = self.parallel_solve(sspm)
        x_syms = sspm.solve_params_from_newparams(x)
        print("J : {} syms {}".format(J, x_syms))


        self.x_best = x
        self.J_min = J

        if self.enable_debug_plot:
            self.draw_freq_res(sspm, x)
            plt.show()


        return self.J_min, self.get_best_ssm()

    def parallel_solve(self, sspm):
        self.sspm = sspm
        pool = multiprocessing.Pool()
        result = pool.map(self.solve, range(self.max_sample_times))
        J_min = 100000
        x = None
        for J, xtmp in result:
            if J < J_min:
                J_min = J
                x = xtmp
        return J_min, x

    def solve(self, id=0):
        sspm = copy.deepcopy(self.sspm)
        print("Solve id {}".format(id))
        f = lambda x: self.cost_func(sspm, x)
        x0 = self.setup_initvals(sspm)
        # return x0, 0
        ret = minimize(f, x0)
        x = ret.x.copy()
        J = ret.fun

        print("exit id {} J:{}".format(id, J))
        return J, x

    def get_H_from_s_trans(self, trans):
        trans = sp.simplify(trans)
        omg_to_h = np.vectorize(lambda omg: complex(trans.evalf(subs={sp.symbols("s"): omg * 1J})))
        return omg_to_h(self.freq)

    def get_best_ssm(self) -> StateSpaceModel:
        assert self.x_best is not None, "You must estimate first"
        sym_sub = dict(zip(self.x_syms, self.x_best))
        return self.sspm.get_ssm_by_syms(sym_sub, using_converted=True)

    def draw_freq_res(self, sspm: StateSpaceParamModel, x):
        if self.fig is not None:
            plt.close(self.fig)

        self.fig, self.axs = plt.subplots(self.y_dims, 1, sharey=True)
        fig, axs = self.fig, self.axs
        fig.set_size_inches(25, 15)
        fig.canvas.set_window_title('FreqRes vs est')
        fig.tight_layout()
        fig.subplots_adjust(right=0.9)
        Hest = copy.deepcopy(self.Hs)

        ssm = self.get_best_ssm()

        for omg_ptr in range(self.freq.__len__()):
            u_index = 0
            omg = self.freq[omg_ptr]
            Tnum = ssm.calucate_transfer_matrix_at_omg(omg)
            for y_index in range(self.y_dims):
                h = Tnum[y_index, u_index]
                h = complex(h)
                Hest[y_index][omg_ptr] = h

        for y_index in range(self.y_dims):
            # trans = sspm.get_transfer_func(y_index, 0)
            amp0, pha0 = FreqIdenSIMO.get_amp_pha_from_h(self.Hs[y_index])
            amp1, pha1 = FreqIdenSIMO.get_amp_pha_from_h(Hest[y_index])
            # amp1, pha1 = amp0, pha0
            ax1 = axs[y_index]
            if self.y_names is not None:
                ax1.title.set_text(self.y_names[y_index])

            p1, = ax1.semilogx(self.freq, amp0, '.', color='tab:blue', label="Hs")
            p2, = ax1.semilogx(self.freq, amp1, '', color='tab:blue', label="Hest")
            ax1.set_ylabel('db', color='tab:blue')
            ax1.grid(which="both")

            ax2 = axs[y_index].twinx()
            ax2.set_ylabel('deg', color='tab:orange')
            ax2.tick_params('y', colors='tab:orange')

            p3, = ax2.semilogx(self.freq, pha0, '.', color='tab:orange', label="pha")
            p4, = ax2.semilogx(self.freq, pha1, color='tab:orange', label="phaest")
            # ax2.grid(which="both")

            ax3 = ax1.twinx()
            # ax3.grid(which="both")
            p5, = ax3.semilogx(self.freq, self.coherens[y_index], color='tab:gray', label="Coherence")

            ax3.spines["right"].set_position(("axes", 1.05))
            # ax2.set_ylabel('coherence', color='tab:gray')
            lines = [p1, p2, p3, p4]

            ax1.legend(lines, [l.get_label() for l in lines])

    def setup_initvals(self, sspm):
        source_syms = sspm.syms
        source_syms_dims = sspm.syms.__len__()
        source_syms_init_vals = np.random.rand(source_syms_dims) * 2 - 1
        subs = dict(zip(source_syms, source_syms_init_vals))

        x0 = np.zeros(self.x_dims)
        for i in range(self.x_dims):
            sym = self.x_syms[i]
            # Eval sym value from sspm
            sym_def = sspm.new_params_raw_defines[sym]
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
