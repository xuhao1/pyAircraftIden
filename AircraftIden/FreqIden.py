import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import math
from scipy import signal
from aircraft_iden.SpectrumAnalyse import MultiSignalSpectrum


def remove_seq_average_and_drift(x_seq):
    drift = x_seq[-1] - x_seq[0]
    for i in range(len(x_seq)):
        x_seq[i] = x_seq[i] - drift * i / len(x_seq)
    x_seq = x_seq - np.average(x_seq)
    return x_seq


def time_seq_resample(time_seq, filter_omg, *x_seqs, remove_drift_and_avg=True):
    tnew = np.linspace(time_seq[0], time_seq[-1], num=len(time_seq), endpoint=True)
    sample_rate = len(time_seq) / (time_seq[-1] - time_seq[0])
    print("Sample rate is {0}".format(sample_rate))
    resampled_datas = [tnew]
    filter_freq = filter_omg / math.pi / 2
    if filter_freq > 1:
        filter_freq = 1
    b, a = signal.butter(1, 2 * filter_freq / sample_rate)
    for x_seq in x_seqs:
        assert len(x_seq) == len(tnew), "Length of data seq must be euqal to time seq"
        if remove_drift_and_avg:
            x_seq = remove_seq_average_and_drift(x_seq)

        inte_func = interp1d(time_seq, x_seq)
        data = inte_func(tnew)
        # zi = signal.lfilter_zi(b, a)
        # z, _ = signal.lfilter(b, a, data, zi=zi * data[0])
        resampled_datas.append(data)
    return tuple(resampled_datas)


class FreqIdenSIMO:
    def __init__(self, time_seq, omg_min, omg_max, x_seq, *y_seqs, win_num=16):
        self.time_seq, self.x_seq = time_seq_resample(time_seq, omg_max * 5, x_seq, remove_drift_and_avg=True)
        _, *y_seqs = time_seq_resample(time_seq, omg_max * 5, *y_seqs, remove_drift_and_avg=True)
        self.y_seqs = list(y_seqs)

        self.time_len = self.time_seq[-1] - self.time_seq[0]
        self.sample_rate = len(self.time_seq) / self.time_len
        self.omg_min = omg_min
        self.omg_max = omg_max

        datas = self.y_seqs.copy()
        datas.append(self.x_seq)
        print("Start calc spectrum for data: totalTime{} sample rate {}".format(self.time_len, self.sample_rate))
        self.spectrumAnal = MultiSignalSpectrum(self.sample_rate, omg_min, omg_max, datas, win_num)

    def get_freq_iden(self, y_index):
        freq, gxx = self.spectrumAnal.get_gxx_by_index(-1)
        _, gxy = self.spectrumAnal.get_gxy_by_index(-1, y_index)
        _, gyy = self.spectrumAnal.get_gxx_by_index(y_index)
        H = FreqIdenSIMO.get_h_from_gxy_gxx(gxy, gxx)
        gamma2 = FreqIdenSIMO.get_gamma_sqr(gxx, gxy, gyy)
        return freq, H, gamma2, gxx, gxy, gyy

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
        return 20 * np.log10(np.absolute(H)), np.arctan2(H.imag, H.real) * 180 / math.pi

    @staticmethod
    def get_gamma_sqr(gxx, gxy, gyy):
        return np.absolute(gxy) * np.absolute(gxy) / (np.absolute(gxx) * np.absolute(gyy))


if __name__ == "__main__":
    arr = np.load("../data/sweep_data_2017_10_18_14_07.npy")
    time_seq_source = arr[:, 0]
    ele_seq_source = arr[:, 1]
    q_seq_source = arr[:, 4]
    simo_iden = FreqIdenSIMO(time_seq_source,0.1, 100, ele_seq_source, q_seq_source, win_num=32)
    freq, H, gamma2, gxx, gxy, gyy = simo_iden.get_freq_iden(0)
    h_amp, h_phase = FreqIdenSIMO.get_amp_pha_from_h(H)


    plt.subplot(411)
    plt.grid()
    plt.semilogx(freq, 20 * np.log10(gxx), freq, 20 * np.log10(gyy), freq, 20 * np.log10(np.absolute(gxy)))
    #plt.semilogx(freq, 10 * np.log10(gxx), freq, 10 * np.log10(gyy))
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