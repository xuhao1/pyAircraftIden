import math
import numpy as np
from AircraftIden import FreqIdenSIMO
import matplotlib.pyplot as plt
from scipy.optimize import minimize, basinhopping
import scipy.signal as signal
import time
import sympy as sp
from sympy import poly, latex
import multiprocessing
import random
from numpy import linalg as LA
import numbers

def poly_latex(poly,cha = "s"):
    ret_str = ""
    ords = len(poly) - 1
    for i in range(ords+1):
        ordn = ords - i
        if ordn == 0:
            ret_str = ret_str + "{:4.2f}".format(poly[i])
        else:
            if (poly[i] == 1):
                ret_str = ret_str + "s^{:d}+".format(ordn) 
            else:
                ret_str = ret_str + "{:4.2f} s^{:d}+".format(poly[i],ordn)
    return ret_str

class TransferFunctionModel(object):
    #TransferFunction Model, num and den is coefficent
    def __init__(self,num,den,tau = 0):
        self.num = num
        self.den = den
        self.tau = tau

        pass
    
    def freqres(self, w, unwarp = False):
        
        b = self.num
        a = self.den
        tau = self.tau
        s = 1j * w
        # print(s)
        # print(type(tau))
        h = np.polyval(b, s) * np.exp(-tau * s) / np.polyval(a, s)
        # print(np.absolute(h))
        h = np.complex64(h)
        amp = 20 * np.log10(np.absolute(h))
        if unwarp:
            pha = np.unwrap(np.arctan2(h.imag, h.real)) * 180 / math.pi
        else:
            pha = np.arctan2(h.imag, h.real) * 180 / math.pi
        return amp, pha
    
    
    def plot(self, freq = None):
        if freq is None:
            freq = np.linspace(1.0,10,10)
        # print(freq)
        amp,pha = self.freqres(freq,True)
        plt.semilogx(freq,amp,label="Amp")
        plt.semilogx(freq,pha,label="Pha")
        plt.legend()

    def latex(self):
        self.num = np.array(self.num) / self.den[0]
        self.den = np.array(self.den) / self.den[0]

        num = self.num
        den = self.den
        tau = self.tau
        return r"$\frac{" + poly_latex(num) + "}{" + poly_latex(den) +"}" + "e^{-" + "{:4.3f}".format(tau) + "t}$"

    

class TransferFunctionParamModel(object):
    #A TransferFunction with unknown parameters
    
    def __init__(self,num,den,tau = 0):
        self.num = num
        self.den = den
        self.tau = tau
        self.s = sp.symbols('s')


    def transfer_function_by_dict(self, sym_dict):
        s = self.s

        if isinstance(self.num, numbers.Number):
            num  = self.num
        else:
            num_formula =  self.num.subs(sym_dict)
            num = poly(num_formula,s).all_coeffs()

        if isinstance(self.den, numbers.Number):
            den = self.den
        else:
            den_formula =  self.den.subs(sym_dict)
            den = poly(den_formula,s).all_coeffs()

        AlltoFloat = lambda x: float(x.evalf())
        num = list(map(AlltoFloat,num))
        den = list(map(AlltoFloat,den))

        if not isinstance(self.tau, numbers.Number):
            tau = self.tau.subs(sym_dict)
            tau = float(tau.evalf())
        else:
            tau = self.tau

        return TransferFunctionModel(num,den,tau)

    def get_unknown_param_list(self):
        syms = set()
        if not isinstance(self.num, numbers.Number):
            syms.update(self.num.atoms(sp.Symbol))

        if not isinstance(self.den, numbers.Number):
            syms.update(self.den.atoms(sp.Symbol))

        if not isinstance(self.tau, numbers.Number):
            syms.update(self.tau.atoms(sp.Symbol))

        syms.remove(self.s)
        return list(syms)

    def symbol_expr(self):
        return self.num / self.den * sp.exp(-self.tau)

    def latex(self,sub = None):
        if sub is None:
            return latex(self.num / self.den * sp.exp(-self.tau))

        frac_part = r"$\frac{" + latex(self.num.subs(sub)) + "}{" + latex(self.den.subs(sub)) + "}"
        tau_part =  "" if self.tau == 0 else ("e^{-" + "{:4.3f}".format(self.tau.subs(sub)) + "t}$")
        return frac_part + tau_part

class TransferFunctionFit(object):
    def __init__(self, freq, H, coheren, tfpm, nw=20,
                 iter_times = 100,has_addition_integral = False,reg = 0.1):
        self.nw = nw
        self.source_freq = freq
        self.source_H = H
        self.source_coheren = coheren
        self.wg = 1.0
        self.wp = 0.01745
        self.iter_times = iter_times

        self.tfpm = tfpm
        self.unknown_param_list = self.tfpm.get_unknown_param_list()
        print("Uknown number {} {}".format(len(self.unknown_param_list),self.unknown_param_list))

        self.tau_index = None
        if tfpm.tau != 0:
            for i in range(len(self.unknown_param_list)):
                if self.unknown_param_list[i] == tfpm.tau:
                    self.tau_index = i
                    # print("Has unknown tau {}".format(self.tau_index))
                    break

        self.est_omg_ptr_list = []
        self.enable_debug_plot = False

        self.reg = 0.1
        
        self.has_addition_integral = has_addition_integral

        self.den_max_ord_ptr = 0
        self.x = None


    def get_transfer_function_by_x(self,x):
        syms = dict(zip(self.unknown_param_list,x))
        tf = self.tfpm.transfer_function_by_dict(syms)
        return tf

    def latex(self, sspm = False):
        if sspm or self.x is None:
            return self.tfpm.latex()
        else:
            return self.tf.latex()

    def cost_func_at_omg_ptr(self, tf, omg_ptr):
        omg = self.source_freq[omg_ptr]

        # print("tfdict:",tf.__dict__)
        amp, pha = tf.freqres(omg)

        h = self.source_H[omg_ptr]
        h_amp = 20 * np.log10(np.absolute(h))
        h_pha = np.arctan2(h.imag, h.real) * 180 / math.pi
        pha_err = h_pha - pha
        if pha_err > 180:
            pha_err = pha_err - 360
        if pha_err < -180:
            pha_err = pha_err + 360
        J = self.wg * pow(h_amp - amp, 2) + self.wp * pow(pha_err, 2)

        gama2 = self.source_coheren[omg_ptr]

        wgamma = 1.58 * (1 - math.exp(-gama2 * gama2))
        wgamma = wgamma * wgamma
        return J * wgamma

    def cost_func(self, x):
        tf = self.get_transfer_function_by_x(x)
        cost_func_at_omg = lambda omg_ptr: self.cost_func_at_omg_ptr(tf, omg_ptr)
        arr_func = np.vectorize(cost_func_at_omg)
        cost_arr = arr_func(self.est_omg_ptr_list)
        return np.sum(cost_arr) * 20 / self.nw + self.reg * LA.norm(x)

    def init_omg_list(self, omg_min, omg_max):
        if omg_min is None:
            omg_min = self.source_freq[0]

        if omg_max is None:
            omg_max = self.source_freq[-1]

        omg_list = np.linspace(np.log(omg_min), np.log(omg_max), self.nw)
        omg_list = np.exp(omg_list)
        # print("omg list {}".format(omg_list))

        omg_ptr = 0
        self.est_omg_ptr_list = []
        for i in range(self.source_freq.__len__()):
            freq = self.source_freq[i]
            if freq > omg_list[omg_ptr]:
                self.est_omg_ptr_list.append(i)
                omg_ptr = omg_ptr + 1
            elif omg_ptr < omg_list.__len__() and i == self.source_freq.__len__() - 1:
                self.est_omg_ptr_list.append(i)
                omg_ptr = omg_ptr + 1


    def estimate(self, omg_min=None, omg_max=None,accept_J=10):
        self.init_omg_list(omg_min, omg_max)

        J_min_max = 1000000
        J_min = J_min_max

        cpu_use = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(cpu_use)

        results = []
        for i in range(self.iter_times):
            result = pool.apply_async(self.solve)
            results.append(result)

        should_exit_pool = False
        print("Starting Estimate",end="")
        while not should_exit_pool:
            if results.__len__() == 0:
                print("All in pool finish")
                break
            for i in range(results.__len__()):
                thr = results[i]
                if thr.ready() and thr.successful():
                    x_tmp,J  = thr.get()
                    if J < J_min:
                        J_min = J
                        self.x = x_tmp
                        self.setup_transferfunc(x_tmp)
                        print("\r",end="")
                        print("Found new better {}".format(J), end="")

                    if J < accept_J:
                        print("")
                        pool.terminate()
                        return self.tf

                    del results[i]
                    break
            time.sleep(0.01)
        pool.terminate()
        return self.tf

    def setup_transferfunc(self, x):
        self.tf = self.get_transfer_function_by_x(x)

    def solve(self):
        f = self.cost_func
        x0 = self.setup_initvals()
        bounds = [(None, None) for i in range(len(x0))]
        bounds[-1] = (0, 0.1)
        ret = minimize(f, x0, options={'maxiter': 10000000, 'disp': False}, bounds=bounds, tol=1e-15)
        x = ret.x.copy()# / ret.x[self.den_max_ord_ptr]
        J = ret.fun
        return x, J

    def setup_initvals(self):
        x0 = np.random.rand(len(self.unknown_param_list)) - 0.5
        x0 = x0 * 2
        if self.tau_index != None:
            x0[self.tau_index] = 0
        return x0

    def plot(self, name = ""):
        H = self.source_H
        freq = self.source_freq

        mag, phase = self.tf.freqres(freq,unwarp=True)
        h_amp, h_phase = FreqIdenSIMO.get_amp_pha_from_h(H)

        plt.subplot(311)
        plt.semilogx(freq, h_amp, label='source')
        plt.semilogx(freq, mag, label='fit')
        plt.title(name + "H Amp")
        plt.grid(which='both')
        plt.legend()

        plt.subplot(312)
        plt.semilogx(freq, h_phase, label='source')
        plt.semilogx(freq, phase, label='fit')
        plt.title(name + "H Phase")
        plt.grid(which='both')
        plt.legend()

        plt.subplot(313)
        plt.semilogx(freq, self.source_coheren, label="coherence of xy")
        plt.legend()
        plt.title(name + "gamma2")
        plt.grid(which='both')

        pass


def siso_freq_iden():
    # save_data_list = ["running_time", "yoke_pitch", "theta", "airspeed", "q", "aoa", "VVI", "alt"]
    arr = np.load("../../XPlaneResearch/data/sweep_data_2017_12_10_19_05.npy")
    time_seq_source = arr[:, 0]
    ele_seq_source = arr[:, 1]
    q_seq_source = arr[:, 4]*math.pi / 180
    vvi_seq_source = arr[:, 6]

    simo_iden = FreqIdenSIMO(time_seq_source, 1, 30, ele_seq_source, q_seq_source, vvi_seq_source)

    #plt.figure(0)
    #simo_iden.plt_bode_plot(0)
    #
    freq, H, gamma2, gxx, gxy, gyy = simo_iden.get_freq_iden(0)

    fitter = TransferFunctionFit(freq, H, gamma2, 2, 4, nw=20, enable_debug_plot=True)
    fitter.estimate()

    fitter.plot()
    plt.show()


def test_freq_res():
    b = [1]
    a = [1, 1]
    w = np.linspace(0.01, 10, 100)
    res_amp = []
    res_pha = []

    for wi in w:
        amp_tmp, pha_tmp = freqres(b, a, wi)
        res_amp.append(amp_tmp)
        res_pha.append(pha_tmp)

    plt.semilogx(w, res_amp, label='amp')
    plt.semilogx(w, res_pha, label='pha')
    plt.grid(which='both')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    siso_freq_iden()
