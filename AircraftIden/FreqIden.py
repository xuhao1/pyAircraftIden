import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import math
from AircraftIden.SpectrumAnalyse import MultiSignalSpectrum
import copy
from AircraftIden.CompositeWindow import CompositeWindow
from AircraftIden.FreqResponse import FreqResponse


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
    print("Sample rate is {0:3.1f}hz".format(sample_rate))
    resampled_datas = [tnew]
    for x_seq in x_seqs:
        assert len(x_seq) == len(tnew), "Length of data seq must be euqal to time seq"
        x_seq = copy.deepcopy(x_seq)
        if remove_drift_and_avg:
            x_seq = remove_seq_average_and_drift(x_seq)
        data = x_seq
        if enable_resample:
            inte_func = interp1d(time_seq, x_seq)
            data = inte_func(tnew)
        resampled_datas.append(data)
    return tuple(resampled_datas)


class FreqIdenSIMO:
    def __init__(self, time_seq, omg_min, omg_max, x_seq, *y_seqs, win_num=None, uniform_input=False, assit_input=None):

        self.time_seq, self.x_seq = time_seq_preprocess(time_seq, x_seq, remove_drift_and_avg=True,
                                                        enable_resample=not (uniform_input))
        self.trims = []
        for y_seq in y_seqs:
            self.trims.append(y_seq[0])
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

        if win_num is None:
            self.using_composite = True
        else:
            self.using_composite = False

        try:
            if self.using_composite:
                self.composes = [CompositeWindow(self.x_seq, y_seq, self.sample_rate, omg_min, omg_max)
                                for y_seq in self.y_seqs]
            else:
                datas = copy.deepcopy(self.y_seqs)
                if self.enable_assit_input:
                    datas.append(self.x2_seq)
                datas.append(self.x_seq.copy())
                print("Start calc spectrum for data: totalTime{} sample rate {}".format(self.time_len, self.sample_rate))

                self.spectrumAnal = MultiSignalSpectrum(self.sample_rate, omg_min, omg_max, datas, win_num)
        except KeyboardInterrupt:
            raise

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

    def get_freq_iden(self, y_index=0):
        if not self.using_composite:
            freq, gxx = self.spectrumAnal.get_gxx_by_index(-1)

            if self.enable_assit_input:
                gxx = gxx * self.get_assit_xx_norm()
            _, gxy = self.spectrumAnal.get_gxy_by_index(-1, y_index)

            if self.enable_assit_input:
                gxy = gxy * self.get_assit_xy_norm(y_index)

            _, gyy = self.spectrumAnal.get_gxx_by_index(y_index)
        else:
            freq = self.composes[y_index].freq
            gxx = self.composes[y_index].gxx
            gxy = self.composes[y_index].gxy
            gyy = self.composes[y_index].gyy

        H = FreqIdenSIMO.get_h_from_gxy_gxx(gxy, gxx)
        gamma2 = FreqIdenSIMO.get_coherence(gxx, gxy, gyy)

        return freq, H, gamma2, gxx, gxy, gyy

    def get_freqres(self, indexs = None):
        Hs = []
        coheres = []
        freq = None
        if indexs is None:
            indexs = range(self.y_seqs.__len__())
        for i in indexs:
            freq, h, co, _, _, _ = self.get_freq_iden(i)
            Hs.append(h)
            coheres.append(co)
        return FreqResponse(freq, Hs, coheres,self.trims)
    
    def save_to_csv(self, index, path):
        freq, H, gamma2, gxx, gxy, gyy = self.get_freq_iden(index)
        exdata = np.array([freq, np.real(H), np.imag(H)]).transpose()
        np.savetxt(path, exdata, delimiter=",")
    
    def return_plotting_data(self):
        freq, H, gamma2, gxx, gxy, gyy = self.get_freq_iden(0)
        h_amp, h_phase = FreqIdenSIMO.get_amp_pha_from_h(H)
        return freq, h_amp, h_phase, gamma2, gxx, gxy, gyy, H


    def plt_bode_plot(self, index=0, label="", xmin=None, xmax=None):
        # f, ax = plt.subplots()
        
        
        freq, H, gamma2, gxx, gxy, gyy = self.get_freq_iden(index)
        h_amp, h_phase = FreqIdenSIMO.get_amp_pha_from_h(H)
        ax1 = plt.subplot(411)
        ax1.semilogx(freq, 20 * np.log10(gxx), label=label+'gxx')
        ax1.semilogx(freq, 20 * np.log10(gyy), label=label+'gyy')
        ax1.semilogx(freq, 20 * np.log10(np.absolute(gxy)), label=label+'gxy')
        ax1.set_title("Gxx & Gyy Tilde of ele and theta")
        ax1.legend()
        ax1.grid(which='both')

        ax2 = plt.subplot(412)
        ax2.semilogx(freq, h_amp, label=label)
        ax2.set_title("H Amp")
        ax2.legend()
        ax2.grid(which='both')
        ax2 = plt.gca()
        ax2.set_xlim([xmin,xmax])
        # ax2.set_ylim([-10,10])

        ax3 = plt.subplot(413)
        ax3.semilogx(freq, h_phase, label=label)
        ax3.set_title("H Phase")
        ax3.legend()
        ax3.grid(which='both')
        ax3.set_xlim([xmin,xmax])
        # ax3.set_ylim([ -30, 30])

        ax4 = plt.subplot(414)
        ax4.semilogx(freq, gamma2, label=label+"coherence")
        if self.enable_assit_input:
            ax4.semilogx(freq, self.get_cross_coherence(-1, -2), label='coherece of x and assit input')
        ax4.legend()
        ax4.set_title("gamma2")
        ax4.grid(which='both')


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
    # arr = np.load("../../XPlaneResearch/data/sweep_data_2017_11_18_17_19.npy")
    time_seq_source = arr[:, 0]
    ele_seq_source = arr[:, 1]
    q_seq_source = arr[:, 4]
    airspeed_seq = arr[:, 3]
    theta_seq = arr[:, 2] / 180 * math.pi

    simo_iden = FreqIdenSIMO(time_seq_source, 0.1, 100, ele_seq_source, q_seq_source, win_num=64)
    freq, H, gamma2, gxx, gxy, gyy = simo_iden.get_freq_iden(0)
    h_amp, h_phase = FreqIdenSIMO.get_amp_pha_from_h(H)

    print(freq.__len__())
    plt.subplot(411)
    plt.semilogx(freq, 20 * np.log10(gxx), label="gxx")
    plt.semilogx(freq, 20 * np.log10(gyy), label="gyy")
    plt.semilogx(freq, 20 * np.log10(np.absolute(gxy)), label="gxy")
    plt.grid()
    plt.legend()
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
