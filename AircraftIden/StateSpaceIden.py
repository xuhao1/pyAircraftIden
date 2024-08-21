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
import time
import sys


class StateSpaceIdenSIMO(object):
    def __init__(self, freqres, nw=20, enable_debug_plot=False, max_sample_times=20, accept_J=5,
                 y_names=None, reg = 1.0, cpu_use = None, iter_callback = None, con_str = None):
        self.freq = freqres.freq
        self.Hs = freqres.Hs
        self.wg = 1.0
        self.wp = 0.01745
        self.est_omg_ptr_list = []
        self.enable_debug_plot = enable_debug_plot
        self.coherens = freqres.coherens
        self.nw = nw
        self.max_sample_times = max_sample_times
        self.accept_J = accept_J
        self.x_dims = 0
        self.x_syms = []
        self.y_dims = len(self.Hs)
        self.y_names = y_names

        self.x_best = None
        self.J_min = -1
        
        self.reg = reg

        self.fig = None

        self.cpu_use = cpu_use
        self.con_str = con_str
        self.iter_callback= iter_callback
        self.param1_index = []
        self.param2_index = []
        self.is_negative = []
    
    def print_res(self):
        assert self.x_best is not None, "You must estimate first"
        x_syms = self.sspm.solve_params_from_newparams(self.x_best)
        print(x_syms)
        sym_sub = dict(zip(self.x_syms, self.x_best))
        ssm = self.sspm.get_ssm_by_syms(sym_sub, using_converted=True)
        print("A")
        print(ssm.A)
        print("B")
        print(ssm.B)
    
    def getMatrices(self):
        assert self.x_best is not None, "You must estimate first"
        x_syms = self.sspm.solve_params_from_newparams(self.x_best)
        sym_sub = dict(zip(self.x_syms, self.x_best))
        ssm = self.sspm.get_ssm_by_syms(sym_sub, using_converted=True)
        return ssm.A, ssm.B
    
    def user_constrain_index(self):
        print("in constraint index function")
        if self.con_str:
            print("in first if")
            for constraints in self.con_str:
                print(constraints[0], " : " ,constraints[1])
                param1 = constraints[0]
                param2 = constraints[1]
                is_negative = param2.startswith('-')
                param2 = param2.lstrip('-')
                try:
                    param1_index = self.x_syms.index(sp.symbols(param1))
                    print("index 1 constraint: ",param1_index)
                    self.param1_index.append(param1_index)
                except ValueError as e:
                    raise ValueError(f"Error: Symbol {param1} not found in self.x_syms. Cannot continue.") from e
                
                try:
                    param2_index = self.x_syms.index(sp.Symbol(param2))
                    print("index 2 constraint: ",param2_index)
                    self.param2_index.append(param2_index)
                except ValueError as e:
                    raise ValueError(f"Error: Symbol {param2} not found in self.x_syms. Cannot continue.") from e
                
                self.is_negative.append(is_negative)

    def estimate(self, sspm: StateSpaceParamModel, syms, omg_min=None, omg_max=None, constant_defines=None, rand_init_max = 1, bounds=None):
        assert self.y_dims == sspm.y_dims, "StateSpaceModel dim : {} need to iden must have same dims with Hs {}".format(
            sspm.y_dims, self.y_dims)

        if constant_defines is None:
            constant_defines = dict()
        self.init_omg_list(omg_min, omg_max)

        if bounds is None:
            self.lower_bnd=None
            self.upper_bnd=None
            self.rand_init_max = rand_init_max
        else:
            self.lower_bnd=bounds[0]
            self.upper_bnd=bounds[1]
        self.syms = syms
        sspm.load_constant_defines(constant_defines)
        self.x_syms = list(sspm.get_new_params())
        self.x_dims = len(self.x_syms)
        assert self.x_dims == len(self.syms), "Every unknown param must be provide in syms!"
        print("Will estimate num {} {}".format(self.x_syms.__len__(), self.x_syms))
        self.user_constrain_index()

        if self.max_sample_times > 1:
            print("in parallel solve (estimate)")
            J, x = self.parallel_solve(sspm)
        else:
            self.sspm = sspm
            J_min = 100000
            sspm = copy.deepcopy(self.sspm)
            x0 = self.setup_initvals(sspm)
            G_prev = np.zeros((len(x0)))
            x_prev = x0
            for i in range(50):
                print("iter: ",i)
                J, x_tmp, G = self.solve(0, x0 = x0)
                if J < J_min:
                    # x0 = x
                    # J_min = J
                    # self.x_best = x
                    # self.J_min = J
                    self.J_min = J
                    J_min = J
                    if np.any(G == G_prev):
                        update = G
                    else:
                        update = G*(x_tmp - x_prev)/(G-G_prev)
                    x0 = x_tmp - 0.01*update #learning rate of 0.01
                    self.x_best = x_tmp
                    x_prev = x_tmp
                    G_prev = G
                    print("Found New Better cost:", J)
                    if J < self.accept_J:
                        x_syms = sspm.solve_params_from_newparams(self.x_best)
                        return self.J_min, self.get_best_ssm()
                else:
                    print("solution: ",J)
                    if J < 1.5*J_min:
                        if np.any(G == G_prev):
                            update = G
                        else:
                            update = G*(x_tmp - x_prev)/(G-G_prev)

                        x0 = x_tmp - 0.01*update #learning rate of 0.01
                        x_prev = x_tmp
                        G_prev = G
                    
                    else:
                        x0 = self.setup_initvals(sspm)                  

        x_syms = sspm.solve_params_from_newparams(self.x_best)

        if self.enable_debug_plot:
            self.draw_freq_res()
            plt.show()

        return self.J_min, self.get_best_ssm()

    def parallel_solve(self, sspm):
        self.sspm = sspm
        print("in parallel solve")
        if self.cpu_use is None:
            cpu_use = multiprocessing.cpu_count() - 1
        else:
            cpu_use = self.cpu_use

        if cpu_use < 1:
            cpu_use = 1

        if cpu_use > self.max_sample_times:
            cpu_use = self.max_sample_times

        pool = multiprocessing.Pool(cpu_use)
        # result = pool.map_async(self.solve, range(self.max_sample_times))
        results = []
        for i in range(self.max_sample_times):
            result = pool.apply_async(self.solve, (i,))
            results.append(result)

        self.J_min = 100000
        self.x_best = None
        should_exit_pool = False
        while not should_exit_pool:
            if results.__len__() == 0:
                print("All in pool finish")
                print("Using J {} x {}".format(J, x_tmp))
                break
            for i in range(results.__len__()):
                thr = results[i]
                if thr.ready() and thr.successful():
                    J, x_tmp = thr.get()
                    if J < self.J_min:
                        self.J_min = J
                        self.x_best = x_tmp
                        print("results {}".format(x_tmp))
                        print("Found new better {}".format(J))

                        if self.enable_debug_plot:
                            pass

                    if J < self.accept_J:
                        # print("Terminate pool")
                        pool.terminate()
                        print("Using J {} x {}".format(self.J_min, self.x_best))
                        return self.J_min, self.x_best
                    
                    del results[i]
                    break

            time.sleep(0.01)
        pool.terminate()
        # print("Using J {} x {}".format(self.J_min, self.x_best))
        return self.J_min, self.x_best

    def solve_callback(self, x, x_state):
        print(x)
        print(x_state)
        sys.stdout.flush()

    def setup_initvals(self,sspm):
        x0 = np.zeros(self.x_dims)
        if self.lower_bnd:
            for k in range(self.x_dims):
                lbnd = self.lower_bnd[k]
                ubnd = self.upper_bnd[k]
                ret = lbnd + np.random.rand()*(ubnd - lbnd)
                x0[k] = ret
        else:
            source_syms = sspm.syms
            source_syms_dims = sspm.syms.__len__()
            source_syms_init_vals = (np.random.rand(source_syms_dims) * 2 - 1) * self.rand_init_max
            subs = dict(zip(source_syms, source_syms_init_vals))
            for i in range(self.x_dims):
                sym = self.x_syms[i]
                sym_def = sspm.new_params_raw_defines[sym]
                v = sym_def.evalf(subs=subs)
                x0[i] = v
        return x0
                
    def solve(self, id=0, x0 = None):
        sspm = copy.deepcopy(self.sspm)
        f = lambda x: self.cost_func(sspm, x)

        con = {'type': 'ineq', 'fun': lambda x: self.constrain_func(sspm,x)}
        opts = {'maxiter':10000}

        #print("{} using init {}".format(id, x0))
        sys.stdout.flush()
        if x0 is None or len(x0) == 0:
            x0 = self.setup_initvals(sspm)
        bnds = None
        bnds = []
        if self.lower_bnd:
            for k in range(self.lower_bnd.__len__()):
                bnds.append((self.lower_bnd[k],self.upper_bnd[k]))

            ret = minimize(f, x0, constraints=con,options=opts,bounds=bnds)
        else:
            ret = minimize(f, x0, constraints=con,options=opts)

        x = ret.x.copy()
        J = ret.fun
        G = ret.jac
        return J, x, G


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

                pha_err = (pha_err + 180) % 360 - 180

                J = self.wg * pow(h_amp - amp, 2) + self.wp * pow(pha_err, 2)

                gama2 = self.coherens[y_index][omg_ptr]
                if gama2 > 0:
                    wgamma = 1.58 * (1 - math.exp(-gama2 * gama2))
                    wgamma = wgamma * wgamma
                else:
                    wgamma = 0
                return J * wgamma

            chn_cost_func = np.vectorize(chn_cost_func)
            J_arr = chn_cost_func(range(sspm.y_dims))
            J = np.average(J_arr)
            return J

        omg_ptr_cost_func = np.vectorize(cost_func_at_omg_ptr)
        J = np.average(omg_ptr_cost_func(self.est_omg_ptr_list)) * 20 + self.reg * np.linalg.norm(x,2)
        return J

    def constrain_func(self, sspm: StateSpaceParamModel, x):
        sym_sub = dict()
        assert len(x) == len(self.x_syms), 'State length must be equal with x syms'
        if self.con_str:        
            for i in range(len(self.is_negative)):
                if self.is_negative[i]:
                    x[self.param1_index[i]] = -x[self.param2_index[i]]
                else:
                    x[self.param1_index[i]] = x[self.param2_index[i]]

        sym_sub = dict(zip(self.x_syms, x))
        ssm = sspm.get_ssm_by_syms(sym_sub, using_converted=True)
        Amat = ssm.A
        eigs = np.linalg.eigvals(Amat)
        #print("eigs {} ret {}".format(eigs,-np.max(eigs)))
        return - np.max(np.real(eigs))


    def get_H_from_s_trans(self, trans):
        trans = sp.simplify(trans)
        omg_to_h = np.vectorize(lambda omg: complex(trans.evalf(subs={sp.symbols("s"): omg * 1J})))
        return omg_to_h(self.freq)

    def get_best_ssm(self) -> StateSpaceModel:
        assert self.x_best is not None, "You must estimate first"
        sym_sub = dict(zip(self.x_syms, self.x_best))
        return self.sspm.get_ssm_by_syms(sym_sub, using_converted=True)

    def draw_freq_res(self):
        if self.fig is not None:
            plt.close(self.fig)

        # Create subplots with shared y-axis
        self.fig, self.axs = plt.subplots(self.y_dims, 1, sharey=True)
        fig, axs = self.fig, self.axs
        fig.set_size_inches(15, 7)
        fig.tight_layout()
        fig.subplots_adjust(right=0.9)
        Hest = copy.deepcopy(self.Hs)

        ssm = self.get_best_ssm()

        for omg_ptr in range(len(self.freq)):
            u_index = 0
            omg = self.freq[omg_ptr]
            Tnum = ssm.calucate_transfer_matrix_at_omg(omg)
            for y_index in range(self.y_dims):
                h = Tnum[y_index, u_index]
                h = complex(h)  # Replace np.complex with built-in complex
                Hest[y_index][omg_ptr] = h

        for y_index in range(self.y_dims):
            amp0, pha0 = FreqIdenSIMO.get_amp_pha_from_h(self.Hs[y_index])
            amp1, pha1 = FreqIdenSIMO.get_amp_pha_from_h(Hest[y_index])

            ax1 = axs if self.y_dims == 1 else axs[y_index]  # Handle case when y_dims is 1

            if self.y_names is not None:
                ax1.set_title(self.y_names[y_index])

            p1, = ax1.semilogx(self.freq, amp0, '.', color='tab:blue', label="Hs")
            p2, = ax1.semilogx(self.freq, amp1, '', color='tab:blue', label="Hest")
            ax1.set_ylabel('db', color='tab:blue')
            ax1.grid(which="both")

            ax2 = ax1.twinx()  # Create twin y-axis for phase

            ax2.set_ylabel('deg', color='tab:orange')
            ax2.tick_params(axis='y', colors='tab:orange')

            p3, = ax2.semilogx(self.freq, pha0, '.', color='tab:orange', label="pha")
            p4, = ax2.semilogx(self.freq, pha1, color='tab:orange', label="phaest")

            ax3 = ax1.twinx()  # Create another twin y-axis for coherence
            ax3.spines["right"].set_position(("axes", 1.05))
            p5, = ax3.semilogx(self.freq, self.coherens[y_index], color='tab:gray', label="Coherence")

            lines = [p1, p2, p3, p4]
            ax1.legend(lines, [l.get_label() for l in lines])

        plt.show()

    def get_freq_res_data(self):
        # Prepare the data structures to hold the arrays to be returned
        data = {
            "freq": self.freq,
            "Hs_amp": [],
            "Hs_pha": [],
            "Hest_amp": [],
            "Hest_pha": [],
            "coherence": []
        }

        Hest = copy.deepcopy(self.Hs)
        ssm = self.get_best_ssm()

        # Calculate the estimated transfer function values
        for omg_ptr in range(len(self.freq)):
            u_index = 0
            omg = self.freq[omg_ptr]
            Tnum = ssm.calucate_transfer_matrix_at_omg(omg)
            for y_index in range(self.y_dims):
                h = Tnum[y_index, u_index]
                h = complex(h)  # Replace np.complex with built-in complex
                Hest[y_index][omg_ptr] = h

        # Calculate amplitude and phase for each dimension
        for y_index in range(self.y_dims):
            amp0, pha0 = FreqIdenSIMO.get_amp_pha_from_h(self.Hs[y_index])
            amp1, pha1 = FreqIdenSIMO.get_amp_pha_from_h(Hest[y_index])

            data["Hs_amp"].append(amp0)
            data["Hs_pha"].append(pha0)
            data["Hest_amp"].append(amp1)
            data["Hest_pha"].append(pha1)
            data["coherence"].append(self.coherens[y_index])

        return data


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