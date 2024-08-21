import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import math
from AircraftIden.SpectrumAnalyse import MultiSignalSpectrum
from scipy.optimize import minimize
#from multiprocessing import Pool,cpu_count


class CompositeWindow(object):
    def __init__(self, x_seq, y_seq, sample_rate, omg_min, omg_max, win_num_lists = None):
        print(f"Setting-up composite win, sample rate {sample_rate} from angluar frequency {omg_min}rad/s to {omg_max}rad/s")
        self.sample_rate = sample_rate
        self.total_time = len(x_seq) /sample_rate
        self.time_seq = np.linspace(0, self.total_time, len(x_seq))
        self.x_seq = x_seq
        self.y_seq = y_seq

        if win_num_lists is None:
            win_num_lists = self.win_num_lists = CompositeWindow.suggest_win_slices(self.total_time, omg_max)
            print("Composite using {}".format(self.win_num_lists))
        else:
            self.win_num_lists = win_num_lists

        self.specturmAnallist = [MultiSignalSpectrum(self.sample_rate, omg_min, omg_max, [y_seq, x_seq], win_num) for
                                 win_num in win_num_lists]
        self.win_slice_num = win_num_lists.__len__()
        self.get_freq_points()

        self.calc_specturms()

        self.gxx, self.gyy, self.gxy = self.estimate()

    def process_freq(self, freq_ptr):
        gxx_stackrel = self.gxx_stackrel[freq_ptr]
        gyy_stackrel = self.gyy_stackrel[freq_ptr]
        gxy_stackrel = self.gxy_stackrel[freq_ptr]
        coheren_stackrel = self.coheren_stackrel[freq_ptr]
        gxxi = self.gxxi
        gxyi = self.gxyi
        gyyi = self.gyyi
        cohereni = self.cohereni
        def Jfunc(x):
            Gxxc = x[0]
            Gyyc = x[1]
            GxyRe = x[2]
            GxyIm = x[3]
            coh_c = (GxyRe * GxyRe + GxyIm * GxyIm) / (math.fabs(Gxxc) * math.fabs(Gyyc))
            def Ji(i):
                ji = pow((Gxxc - gxxi[i][freq_ptr]) / gxx_stackrel, 2) + \
                     pow((Gyyc - gyyi[i][freq_ptr]) / gyy_stackrel, 2) + \
                     pow((GxyRe - gxyi[i][freq_ptr].real) / gxy_stackrel.real, 2) + \
                     pow((GxyIm - gxyi[i][freq_ptr].imag) / gxy_stackrel.imag, 2) + \
                     5.0 * pow((coh_c - cohereni[i][freq_ptr]) / coheren_stackrel, 2)
                return ji * self.Ws[i][freq_ptr]

            Jarr = (np.vectorize(Ji))(range(self.win_slice_num))
            return np.sum(Jarr)

        def Jder(x):
            Gxxc = x[0]
            Gyyc = x[1]
            GxyRe = x[2]
            GxyIm = x[3]

            gxx_stackrel = self.gxx_stackrel[freq_ptr]
            gyy_stackrel = self.gyy_stackrel[freq_ptr]
            gxy_stackrel = self.gxy_stackrel[freq_ptr]
            coheren_stackrel = self.coheren_stackrel[freq_ptr]
            coh_c = (GxyRe * GxyRe + GxyIm * GxyIm) / (math.fabs(Gxxc) * math.fabs(Gyyc))
            Gxynorm2 = GxyRe*GxyRe + GxyIm * GxyIm

            def Jderi(i):
                eta = 20 * (coh_c - cohereni[i][freq_ptr])/ coheren_stackrel
                Wi = self.Ws[i][freq_ptr]
                Jder = [
                    2*(Gxxc-gxxi[i][freq_ptr])/(gxx_stackrel*gxx_stackrel) + \
                        eta * - Gxynorm2/(Gxxc*Gxxc*Gyyc),
                    2*(Gyyc-gyyi[i][freq_ptr])/(gyy_stackrel*gyy_stackrel) + \
                        eta * - Gxynorm2/(Gxxc*Gyyc*Gyyc),
                    2*(GxyRe-gxyi[i][freq_ptr].real)/(gxy_stackrel.real*gxy_stackrel.real) + \
                        eta * (2*GxyRe)/(Gxxc*Gyyc),
                    2*(GxyIm-gxyi[i][freq_ptr].imag)/(gxy_stackrel.imag*gxy_stackrel.imag) + \
                        eta * (2*GxyIm)/(Gxxc*Gyyc),
                ]
                return np.array(Jder) * Wi

            ret = np.array([0, 0, 0, 0])
            for i in range(self.win_slice_num):
                ret = Jderi(i) + ret
            return ret

        x0 = [gxx_stackrel,
              gyy_stackrel,
              gxy_stackrel.real,
              gxy_stackrel.imag]
        ret = minimize(Jfunc, np.array(x0), jac=Jder, method="BFGS")

        return np.absolute(ret.x[0]), np.absolute(ret.x[1]), ret.x[2] + ret.x[3] * 1J

    def estimate(self):
        gxx_c = []
        gxy_c = []
        gyy_c = []

        self.gxxi = [self.get_inteploated_source_gxx(i) for i in range(self.win_slice_num)]
        self.gxyi = [self.get_inteploated_source_gxy(i) for i in range(self.win_slice_num)]
        self.gyyi = [self.get_inteploated_source_gyy(i) for i in range(self.win_slice_num)]
        self.cohereni = [self.get_inteploated_source_coherence(i) for i in range(self.win_slice_num)]

        #         print("Process to ptr {}/{} freq {} rad/s".format(freq_ptr, self.freq.__len__(), self.freq[freq_ptr]))
        for i in range(self.freq.__len__()):
            ret = [self.process_freq(i)]
            for gxx_ci, gyy_ci, gxy_ci in ret:
                gxx_c.append(gxx_ci.copy())
                gyy_c.append(gyy_ci.copy())
                gxy_c.append(gxy_ci.copy())
        gxx_c = np.array(gxx_c)
        gxy_c = np.array(gxy_c)
        gyy_c = np.array(gyy_c)

        return gxx_c, gyy_c, gxy_c

    def calc_specturms(self):
        # self.coherence_list = [self.calc_inteploated_source_coherence(i) for i in range(self.win_slice_num)]
        error_min_arr = []
        error_s = [self.calc_inteploated_source_error(i) for i in range(self.win_slice_num)]
        for freq_ptr in range(self.freq.__len__()):
            error_min = float('inf')
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
        freq, gyy = spean.get_gxx_by_index(0)
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
        self.freq = self.specturmAnallist[0].fft_freq
        return self.freq

    @staticmethod
    def suggest_win_range(total_time, omg_max, time_ranges_num=3):
        win_num_min = 5
        win_num_max = int(total_time * omg_max / (20 * math.pi) - 1)
        assert win_num_max > win_num_min
        return win_num_min, win_num_max

    @staticmethod
    def suggest_win_slices(total_time, omg_max):
        win_num_min, win_num_max = CompositeWindow.suggest_win_range(total_time, omg_max, time_ranges_num=3)
        slices = np.linspace(win_num_min,win_num_max,5)
        res = []
        for val in slices:
            res.append(int(val))
        return res