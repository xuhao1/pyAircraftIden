import math
import numpy as np
import matplotlib.pyplot as plt
from AircraftIden import FreqIdenSIMO
from scipy.optimize import minimize, basinhopping
import scipy.signal as signal
import time


def freqres(b, a, tau, w):
    s = 1j * w
    h = np.polyval(b, s) * np.exp(-tau * s) / np.polyval(a, s)
    # amp, pha = 20 * np.log10(np.absolute(H)), np.arctan2(H.imag, H.real) * 180 / math.pi
    amp = 20 * np.log10(np.absolute(h))
    pha = np.arctan2(h.imag, h.real) * 180 / math.pi
    return amp, pha


def bodeplot(b, a, tau, ws):
    s = 1j * ws
    h = np.polyval(b, s) * np.exp(-tau * s) / np.polyval(a, s)
    # amp, pha = 20 * np.log10(np.absolute(H)), np.arctan2(H.imag, H.real) * 180 / math.pi
    amp = 20 * np.log10(np.absolute(h))
    pha = np.arctan2(h.imag, h.real) * 180 / math.pi
    return ws, amp, pha


class TransferFunctionFit(object):
    def __init__(self, freq, H, coheren, num_ord, den_ord, nw=20,enable_debug_plot = False):
        # num/den
        self.num_ord = num_ord
        self.den_ord = den_ord
        self.nw = nw
        self.source_freq = freq
        self.source_H = H
        self.source_coheren = coheren
        self.wg = 1.0
        self.wp = 0.01745
        self.iter_times = 100

        self.accept_J = 10

        self.est_omg_ptr_list = []
        self.enable_debug_plot = enable_debug_plot

    def cost_func_at_omg_ptr(self, num, den, tau, omg_ptr):
        omg = self.source_freq[omg_ptr]
        amp, pha = freqres(num, den, tau, omg)

        h = self.source_H[omg_ptr]
        h_amp = 20 * np.log10(np.absolute(h))
        h_pha = np.arctan2(h.imag, h.real) * 180 / math.pi
        J = self.wg * pow(h_amp - amp, 2) + self.wp * pow(h_pha - pha, 2)

        gama2 = self.source_coheren[omg_ptr]

        wgamma = 1.58 * (1 - math.exp(-gama2 * gama2))
        wgamma = wgamma * wgamma
        return J * wgamma

    def cost_func(self, num, den, tau):
        cost_func_at_omg = lambda omg_ptr: self.cost_func_at_omg_ptr(num, den, tau, omg_ptr)
        arr_func = np.vectorize(cost_func_at_omg)
        cost_arr = arr_func(self.est_omg_ptr_list)
        return np.sum(cost_arr) * 20 / self.nw

    def estimate(self, omg_min=None, omg_max=None):
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
        # print("omg ptr list",self.est_omg_ptr_list)
        # print("Will fit from {} rad/s to {} rad/s".format(omg_min, omg_max))

        def cost_func_x(x):
            num = x[0:self.num_ord]
            den = x[self.num_ord:self.num_ord + self.den_ord]
            tau = x[-1]
            return self.cost_func(num, den, tau)

        J_min_max = 1000000
        J_min = J_min_max
        x = [None for i in range(self.num_ord + self.den_ord + 1)]
        # plt.figure("resolving..")
        # plt.ion()
        for i in range(self.iter_times):
            x_tmp, J = self.solve(cost_func_x)
            if J < J_min:
                print("Found new better res J:{}".format(J))
                x = x_tmp.copy()
                self.setup_transferfunc(x)
                if self.enable_debug_plot:
                    plt.ion()
                    plt.figure("resolving..")
                    plt.clf()
                    self.plot()
                    if J_min == J_min_max:
                        plt.pause(0.1)
                    plt.pause(0.01)
                J_min = J
                if J_min <self.accept_J:
                    break
        plt.ioff()

        print("J {} num {} den {} tau {}".format(J, self.num, self.den, self.tau))
        return self.num, self.den, self.tau

    def setup_transferfunc(self, x):
        num = x[0:self.num_ord]
        den = x[self.num_ord:self.num_ord + self.den_ord]
        tau = x[-1]
        self.num = num
        self.den = den
        self.tau = tau

    def solve(self, f):
        x0 = self.setup_initvals()
        bounds = [(None, None) for i in range(len(x0))]
        bounds[-1] = (0, 0.1)
        ret = minimize(f, x0, options={'maxiter': 10000000, 'disp': False}, bounds=bounds, tol=1e-15)
        x = ret.x.copy() / ret.x[0]
        x[-1] = ret.x[-1]
        J = ret.fun
        return x, J

    def setup_initvals(self):
        x0 = np.random.rand(self.den_ord + self.num_ord + 1)
        x0[-1] = 0.00
        return x0

    def plot(self):
        H = self.source_H
        freq = self.source_freq
        num = self.num
        den = self.den
        tau = self.tau

        w, mag, phase = bodeplot(num, den, tau, self.source_freq)
        h_amp, h_phase = FreqIdenSIMO.get_amp_pha_from_h(H)

        plt.subplot(311)
        plt.semilogx(freq, h_amp, label='source')
        plt.semilogx(w, mag, label='fit')
        plt.title("H Amp")
        plt.grid(which='both')
        plt.legend()

        plt.subplot(312)
        plt.semilogx(freq, h_phase, label='source')
        plt.semilogx(w, phase, label='fit')
        plt.title("H Phase")
        plt.grid(which='both')
        plt.legend()

        plt.subplot(313)
        plt.semilogx(freq, self.source_coheren, label="coherence of xy")
        plt.legend()
        plt.title("gamma2")
        plt.grid(which='both')

        pass


def siso_freq_iden():
    # save_data_list = ["running_time", "yoke_pitch", "theta", "airspeed", "q", "aoa", "VVI", "alt"]
    arr = np.load("../data/sweep_data_2017_10_18_14_07.npy")
    time_seq_source = arr[:, 0]
    ele_seq_source = arr[:, 1]
    q_seq_source = arr[:, 4]
    vvi_seq_source = arr[:, 6]

    simo_iden = FreqIdenSIMO(time_seq_source, 0.2, 100, ele_seq_source, q_seq_source, vvi_seq_source, win_num=32)

    # plt.figure(0)
    # simo_iden.plt_bode_plot(0)
    #
    freq, H, gamma2, gxx, gxy, gyy = simo_iden.get_freq_iden(0)

    fitter = TransferFunctionFit(freq, H, gamma2, 2, 4, nw=20,enable_debug_plot=True)
    fitter.estimate()

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
