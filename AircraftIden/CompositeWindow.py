import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import math
from AircraftIden.SpectrumAnalyse import MultiSignalSpectrum
from scipy.optimize import minimize


class CompositeWindow(object):
    def __init__(self, x_seq, y_seq, sample_rate, omg_min, omg_max, win_num_lists):
        self.sample_rate = sample_rate
        self.time_seq = np.linspace(0, len(x_seq) / sample_rate, len(x_seq))
        self.x_seq = x_seq
        self.y_seq = y_seq
        self.win_min = win_num_lists
        self.win_max = win_num_lists

        self.specturmAnallist = [MultiSignalSpectrum(self.sample_rate, omg_min, omg_max, [y_seq, x_seq], win_num) for
                                 win_num in win_num_lists]
        self.win_slice_num = win_num_lists.__len__()
        self.get_freq_points()

        self.calc_specturms()

        self.gxx, self.gyy, self.gxy = self.estimate()

    def estimate(self):
        gxx_c = []
        gxy_c = []
        gyy_c = []

        gxxi = [self.get_inteploated_source_gxx(i) for i in range(self.win_slice_num)]
        gxyi = [self.get_inteploated_source_gxy(i) for i in range(self.win_slice_num)]
        gyyi = [self.get_inteploated_source_gyy(i) for i in range(self.win_slice_num)]
        cohereni = [self.get_inteploated_source_coherence(i) for i in range(self.win_slice_num)]

        def process_freq(freq_ptr):
            gxx_stackrel = self.gxx_stackrel[freq_ptr]
            gyy_stackrel = self.gyy_stackrel[freq_ptr]
            gxy_stackrel = self.gxy_stackrel[freq_ptr]
            coheren_stackrel = self.coheren_stackrel[freq_ptr]

            def Jfunc(x):
                Gxxc = x[0]
                Gyyc = x[1]
                GxyRe = x[2]
                GxyIm = x[3]
                J = 0
                for i in range(self.win_slice_num):
                    coh_c = pow(np.absolute(GxyRe + GxyIm * 1J), 2) / (np.absolute(Gxxc) * np.absolute(Gyyc))
                    ji = pow((Gxxc - gxxi[i][freq_ptr]) / gxx_stackrel, 2) + \
                         pow((Gyyc - gyyi[i][freq_ptr]) / gyy_stackrel, 2) + \
                         pow((GxyRe - gxyi[i][freq_ptr].real) / gxy_stackrel.real, 2) + \
                         pow((GxyIm - gxyi[i][freq_ptr].imag) / gxy_stackrel.imag, 2) + \
                         5.0 * pow((coh_c - cohereni[i][freq_ptr]) / coheren_stackrel, 2)
                    J = ji * self.Ws[i][freq_ptr]
                return J

            x0 = [gxx_stackrel,
                  gyy_stackrel,
                  gxy_stackrel.real,
                  gxy_stackrel.imag]
            ret = minimize(Jfunc, np.array(x0))

            return ret.x[0], ret.x[1], ret.x[2] + ret.x[3] * 1J

        for freq_ptr in range(self.freq.__len__()):
            gxx_ci, gyy_ci, gxy_ci = process_freq(freq_ptr)
            gxx_c.append(gxx_ci)
            gyy_c.append(gyy_ci)
            gxy_c.append(gxy_ci)
        gxx_c = np.array(gxx_c)
        gxy_c = np.array(gxy_c)
        gyy_c = np.array(gyy_c)
        plt.figure(233)
        plt.loglog(self.freq, gxx_c, label="gxx")
        plt.loglog(self.freq, gyy_c, label="gyy")
        plt.loglog(self.freq, np.absolute(gxy_c), label="gxy")
        plt.figure(0)
        # plt.show()
        return gxx_c, gyy_c, gxy_c

    def calc_specturms(self):
        # self.coherence_list = [self.calc_inteploated_source_coherence(i) for i in range(self.win_slice_num)]
        error_min_arr = []
        error_s = [self.calc_inteploated_source_error(i) for i in range(self.win_slice_num)]
        for freq_ptr in range(self.freq.__len__()):
            error_min = 100000
            for slice_ptr in range(self.win_slice_num):
                if error_s[slice_ptr][freq_ptr] < error_min:
                    error_min = error_s[slice_ptr][freq_ptr]
            error_min_arr.append(error_min)

        error_min_arr = np.array(error_min_arr)

        self.Ws = []
        for slice_ptr in range(self.win_slice_num):
            W_arr = []
            for freq_ptr in range(self.freq.__len__()):
                W_arr.append(math.pow(error_s[slice_ptr][freq_ptr] / error_min_arr[freq_ptr], -4))
            self.Ws.append(np.array(W_arr))

        self.gxx_stackrel = self.get_stackrel(self.get_inteploated_source_gxx)
        self.gxy_stackrel = self.get_stackrel(self.get_inteploated_source_gxy)
        self.gyy_stackrel = self.get_stackrel(self.get_inteploated_source_gyy)
        self.coheren_stackrel = np.absolute(self.gxy_stackrel) * np.absolute(self.gxy_stackrel) / (
                np.absolute(self.gxx_stackrel) * np.absolute(self.gyy_stackrel))

    def get_stackrel(self, func):
        Gstackrel = None
        for i in range(self.win_slice_num):
            if Gstackrel is None:
                Gstackrel = self.Ws[i] * self.Ws[i] * func(i)
            else:
                Gstackrel = Gstackrel + self.Ws[i] * self.Ws[i] * func(i)

        W2_sum = []
        for freq_ptr in range(self.freq.__len__()):
            Wii = 0
            for ptr in range(self.win_slice_num):
                Wii = Wii + pow(self.Ws[ptr][freq_ptr], 2)
            W2_sum.append(Wii)

        Gstackrel = Gstackrel / W2_sum
        return Gstackrel

    def get_inteploated_source_gxx(self, win_num_ptr):
        spean = self.specturmAnallist[win_num_ptr]
        freq, gxx = spean.get_gxx_by_index(-1)
        inte_func = interp1d(freq, gxx)
        return inte_func(self.freq)

    def get_inteploated_source_gxy(self, win_num_ptr):
        spean = self.specturmAnallist[win_num_ptr]
        freq, gxy = spean.get_gxy_by_index(-1, 0)
        inte_func = interp1d(freq, gxy)
        return inte_func(self.freq)

    def get_inteploated_source_gyy(self, win_num_ptr):
        spean = self.specturmAnallist[win_num_ptr]
        freq, gyy = spean.get_gxx_by_index(00)
        inte_func = interp1d(freq, gyy)
        return inte_func(self.freq)

    def calc_inteploated_source_error(self, win_num_ptr):
        Cerror = 0.7416198
        coherence = self.get_inteploated_source_coherence(win_num_ptr)
        gamma = np.sqrt(coherence)
        nd = self.specturmAnallist[win_num_ptr].win_num
        source_error = Cerror * np.sqrt(1 - coherence) / (gamma * math.sqrt(2 * nd))
        return source_error

    def get_inteploated_source_coherence(self, win_num_ptr):
        spean = self.specturmAnallist[win_num_ptr]
        freq, gxx = spean.get_gxx_by_index(-1)
        freq, gyy = spean.get_gxx_by_index(0)
        freq, gxy = spean.get_gxy_by_index(-1, 0)
        coherence = np.absolute(gxy) * np.absolute(gxy) / (np.absolute(gxx) * np.absolute(gyy))
        inte_func = interp1d(freq, coherence)
        coherence = inte_func(self.freq)
        return coherence

    def get_freq_points(self):
        self.freq = self.specturmAnallist[-1].fft_freq
        return self.specturmAnallist[-1].fft_freq

    @staticmethod
    def suggest_win_range(total_time, omg_max, time_ranges_num=3):
        win_num_min = 2
        win_num_max = int(total_time * omg_max / (20 * math.pi) - 1)

        if time_ranges_num < 3:
            win_num_min = 5
        assert win_num_max > win_num_min
        return win_num_min, win_num_max
