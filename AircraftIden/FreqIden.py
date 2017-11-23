import matplotlib

matplotlib.use("Qt5Agg")

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import math
from AircraftIden.SpectrumAnalyse import MultiSignalSpectrum
import copy
from AircraftIden.CompositeWindow import CompositeWindow


def remove_seq_average_and_drift(x_seq):
    x_seq = x_seq - np.average(x_seq)
    drift = x_seq[-1] - x_seq[0]
    start_v = x_seq[0]
    for i in range(len(x_seq)):
        x_seq[i] = x_seq[i] - drift * i / len(x_seq) - start_v
    return x_seq


def time_seq_preprocess(time_seq, *x_seqs, enable_resample=True, remove_drift_and_avg=True):
    tnew = time_seq
    if enable_resample:
        tnew = np.linspace(time_seq[0], time_seq[-1], num=len(time_seq), endpoint=True)

    sample_rate = len(time_seq) / (time_seq[-1] - time_seq[0])
    print("Sample rate is {0}".format(sample_rate))
    resampled_datas = [tnew]
    for x_seq in x_seqs:
        assert len(x_seq) == len(tnew), "Length of data seq must be euqal to time seq"
        if remove_drift_and_avg:
            x_seq = remove_seq_average_and_drift(x_seq)
        data = x_seq
        if enable_resample:
            inte_func = interp1d(time_seq, x_seq)
            data = inte_func(tnew)
        resampled_datas.append(data)
    return tuple(resampled_datas)


class FreqIdenSIMO:
    def __init__(self, time_seq, omg_min, omg_max, x_seq, *y_seqs, win_num=16, uniform_input=False, assit_input=None):

        self.time_seq, self.x_seq = time_seq_preprocess(time_seq, x_seq, remove_drift_and_avg=True,
                                                        enable_resample=not (uniform_input))
        _, *y_seqs = time_seq_preprocess(time_seq, *y_seqs, remove_drift_and_avg=True,
                                         enable_resample=not (uniform_input))
        self.y_seqs = list(y_seqs)

        self.enable_assit_input = False

        if assit_input is not None:
            _, self.x2_seq = time_seq_preprocess(time_seq, assit_input, remove_drift_and_avg=True,
                                                 enable_resample=not (uniform_input))
            self.enable_assit_input = True

        self.time_len = self.time_seq[-1] - self.time_seq[0]
        self.sample_rate = len(self.time_seq) / self.time_len
        self.omg_min = omg_min
        self.omg_max = omg_max

        datas = copy.deepcopy(self.y_seqs)
        if self.enable_assit_input:
            datas.append(self.x2_seq)
        datas.append(self.x_seq.copy())
        print("Start calc spectrum for data: totalTime{} sample rate {}".format(self.time_len, self.sample_rate))
        self.spectrumAnal = MultiSignalSpectrum(self.sample_rate, omg_min, omg_max, datas, win_num)

        print(CompositeWindow.suggest_win_range(self.time_len, omg_max, 3))
        win_num_lists = [2,5, 20, 32, 64]
        # win_num_lists = [64]
        self.compose = CompositeWindow(self.x_seq, self.y_seqs[0], self.sample_rate, omg_min, omg_max,
                                       win_num_lists)

    def get_cross_coherence(self, index1, index2):
        # Get cross coherence only works when there is a assit input
        # we treat x2 as a
        if self.enable_assit_input:
            freq, gxx = self.spectrumAnal.get_gxx_by_index(index1)
            _, gaa = self.spectrumAnal.get_gxx_by_index(index2)
            _, gxa = self.spectrumAnal.get_gxy_by_index(index1, index2)
            gxa2 = np.absolute(gxa) * np.absolute(gxa)
            return gxa2 / (gxx * gaa)
        else:
            return 1

    def get_assit_xx_norm(self):
        if self.enable_assit_input:
            return 1 - self.get_cross_coherence(-1, -2)
        else:
            return 1

    def get_assit_yy_norm(self, y_index):
        if self.enable_assit_input:
            return 1 - self.get_cross_coherence(-2, y_index)
        else:
            return 1

    def get_assit_xy_norm(self, y_index=0):
        if self.enable_assit_input:
            _, gaa = self.spectrumAnal.get_gxx_by_index(-2)
            _, gxa = self.spectrumAnal.get_gxy_by_index(-1, -2)
            _, gay = self.spectrumAnal.get_gxy_by_index(-2, y_index)
            _, gxy = self.spectrumAnal.get_gxy_by_index(-1, y_index)

            return 1 - (gxa * gay) / (gaa * gxy)
        else:
            return 1

    def compute_gxx_gxy_using_compose_window(self):
        pass

    def get_freq_iden(self, y_index=0):
        freq, gxx = self.spectrumAnal.get_gxx_by_index(-1)

        if self.enable_assit_input:
            gxx = gxx * self.get_assit_xx_norm()
        _, gxy = self.spectrumAnal.get_gxy_by_index(-1, y_index)

        if self.enable_assit_input:
            gxy = gxy * self.get_assit_xy_norm(y_index)

        _, gyy = self.spectrumAnal.get_gxx_by_index(y_index)

        if y_index == 0:
            gxx = self.compose.gxx
            gxy = self.compose.gxy
            gyy = self.compose.gyy

        # if self.enable_assit_input:
        #     gyy = gyy * self.get_assit_yy_norm(y_index)
        H = FreqIdenSIMO.get_h_from_gxy_gxx(gxy, gxx)
        gamma2 = FreqIdenSIMO.get_coherence(gxx, gxy, gyy)

        return freq, H, gamma2, gxx, gxy, gyy

    def get_all_idens(self):
        Hs = []
        coheres = []
        freq = None

        for i in range(self.y_seqs.__len__()):
            freq, h, co, _, _, _ = self.get_freq_iden(i)
            Hs.append(h)
            coheres.append(co)
        return freq, Hs, coheres

    def plt_bode_plot(self, index=0):
        # f, ax = plt.subplots()

        freq, H, gamma2, gxx, gxy, gyy = self.get_freq_iden(index)
        h_amp, h_phase = FreqIdenSIMO.get_amp_pha_from_h(H)

        plt.subplot(411)
        plt.grid(which='both')
        plt.semilogx(freq, 20 * np.log10(gxx), label='gxx')
        plt.semilogx(freq, 20 * np.log10(gyy), label='gyy')
        plt.semilogx(freq, 20 * np.log10(np.absolute(gxy)), label='gxy')
        plt.title("Gxx & Gyy Tilde of ele and theta")
        plt.legend()

        plt.subplot(412)
        plt.semilogx(freq, h_amp)
        plt.title("H Amp")
        plt.grid(which='both')
        plt.subplot(413)
        plt.semilogx(freq, h_phase)
        plt.title("H Phase")
        plt.grid(which='both')

        plt.subplot(414)
        plt.semilogx(freq, gamma2, label="coherence of xy")
        if self.enable_assit_input:
            plt.semilogx(freq, self.get_cross_coherence(-1, -2), label='coherece of x and assit input')
        plt.legend()
        plt.title("gamma2")
        plt.grid(which='both')

        pass

    @staticmethod
    def get_h_from_gxy_gxx(Gxy_tilde, Gxx_tilde):
        H = Gxy_tilde / Gxx_tilde
        return H

    @staticmethod
    def get_h_from_gyy_gxy(Gyy_tilde, Gxy_tilde):
        H = Gyy_tilde / Gxy_tilde
        return H

    @staticmethod
    def get_amp_pha_from_h(H):
        amp, pha = 20 * np.log10(np.absolute(H)), np.arctan2(H.imag, H.real) * 180 / math.pi
        pha = np.unwrap(pha)
        return amp, pha

    @staticmethod
    def get_coherence(gxx, gxy, gyy):
        # coherence
        return np.absolute(gxy) * np.absolute(gxy) / (np.absolute(gxx) * np.absolute(gyy))


def basic_test():
    arr = np.load("../data/sweep_data_2017_10_18_14_07.npy")
    time_seq_source = arr[:, 0]
    ele_seq_source = arr[:, 1]
    q_seq_source = arr[:, 4]
    airspeed_seq = arr[:, 3]
    theta_seq = arr[:, 2] / 180 * math.pi


    simo_iden = FreqIdenSIMO(time_seq_source, 0.1, 100, ele_seq_source,theta_seq, win_num=64)
    freq, H, gamma2, gxx, gxy, gyy = simo_iden.get_freq_iden(0)
    h_amp, h_phase = FreqIdenSIMO.get_amp_pha_from_h(H)

    plt.subplot(411)
    plt.grid()
    plt.semilogx(freq, 20 * np.log10(gxx), freq, 20 * np.log10(gyy), freq, 20 * np.log10(np.absolute(gxy)))
    # plt.semilogx(freq, 10 * np.log10(gxx), freq, 10 * np.log10(gyy))
    plt.title("Gxx & Gyy Tilde of ele and theta")

    plt.subplot(412)
    plt.semilogx(freq, h_amp)
    plt.title("H Amp")
    plt.grid()
    plt.subplot(413)
    plt.semilogx(freq, h_phase)
    plt.title("H Phase")
    plt.grid()

    plt.subplot(414)
    plt.semilogx(freq, gamma2)
    plt.title("gamma2")
    plt.grid()

    plt.show()


if __name__ == "__main__":
    basic_test()
