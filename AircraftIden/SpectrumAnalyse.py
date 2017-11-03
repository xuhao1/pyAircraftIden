import numpy as np
from aircraft_iden.czt import zoomfft
import math


def czt_seq(totaltime, omg_min, omg_max, *xseqs):
    freq = np.linspace(omg_min, omg_max, num=len(xseqs[0]))
    res = [freq]
    for x_seq in xseqs:
        assert len(xseqs[0]) == len(x_seq), "Length of data must be equal!"
        fft_s = zoomfft(x_seq, omg_min / math.pi / 2, omg_max / math.pi / 2, Fs=len(xseqs[0]) / totaltime)
        res.append(fft_s)
    return tuple(res)


class MultiSignalSpectrum:
    # Analyse Multi Signal Spectrum with default win_num
    def __init__(self, sample_rate, omg_min, omg_max, x_seqs, win_num=16):
        self.sample_rate = sample_rate
        self.x_seqs = x_seqs
        self.win_num = win_num
        self.per_win_time = 0
        self.data_windows = []
        self.data_num = len(x_seqs)
        self.omg_min = omg_min
        self.omg_max = omg_max
        self.fft_array = []
        self.fft_freq = []
        self.total_time = len(self.x_seqs[0]) / sample_rate

        self.calc_fft_for_seqs()

    def get_gxx_by_index(self, seq_index):
        win_num = self.win_num
        gxx_array = []
        for i in range(win_num):
            x_fft = self.fft_array[seq_index][i]
            gxx_array.append(MultiSignalSpectrum.get_gxx(x_fft, self.per_win_time))
        gxx = np.average(gxx_array, axis=0) / 0.612
        return self.fft_freq, gxx

    def get_gxy_by_index(self, x_index, y_index):
        gxy_array = []
        for i in range(self.win_num):
            x_fft = self.fft_array[x_index][i]
            y_fft = self.fft_array[y_index][i]
            gxy_array.append(MultiSignalSpectrum.get_gxy(x_fft, y_fft, self.per_win_time))
        gxy = np.average(gxy_array, axis=0) / 0.612
        return self.fft_freq, gxy

    def calc_fft_for_seqs(self):
        win_num = self.win_num
        self.cut_datas_to_windows()
        print("Win Time Len {0} s".format(self.per_win_time))
        for j in range(self.data_num):
            x_wins = self.data_windows[j]
            x_fft_wins = []
            for i in range(win_num):
                win_x_seq = x_wins[i]
                self.fft_freq, x_fft = czt_seq(self.per_win_time, self.omg_min, self.omg_max, win_x_seq)
                x_fft_wins.append(x_fft)
            self.fft_array.append(x_fft_wins)
        return self.fft_freq, self.fft_array

    def cut_datas_to_windows(self):
        # Using Hanning window
        winnum = self.win_num
        datas = self.x_seqs
        per_win_length = 2 * (len(datas[0]) // (winnum + 1))
        delta_win = per_win_length // 2
        print("len {0} perwinlen {1} delta {2} use data{3}".format(len(datas[0]), per_win_length, delta_win,
                                                                   delta_win * (winnum + 1)))
        res = []
        for data in datas:
            assert len(data) == len(datas[0]), "the length of input data seqs must be equal"
            data = MultiSignalSpectrum.cut_data_seq_to_windows(winnum, delta_win, per_win_length, data)
            res.append(data)

        self.data_windows = res
        self.per_win_time = per_win_length / self.sample_rate
        return res

    @staticmethod
    def get_gxx(x_fft, T):
        return np.absolute(x_fft) * np.absolute(x_fft) * 2 / T

    @staticmethod
    def get_gxy(x_fft: np.ndarray, y_fft: np.ndarray, T):
        return x_fft.conj() * y_fft * 2 / T

    @staticmethod
    def cut_data_seq_to_windows(winnum, delta, perwinlen, data):
        assert winnum * delta < len(data), "no enought data"
        windows = []
        for i in range(winnum):
            win_data = data[i * delta:i * delta + perwinlen]
            # print("num {} start ptr {} end {} to {}".format(i, i * delta, i * delta + perwinlen, len(win_data), len(data)))
            for j in range(perwinlen):
                t = 2 * math.pi * j / perwinlen
                wt = (1 - math.cos(t)) * 0.5
                # wt = 1
                # print("j {} winlen {} t {} wt {}".format(j, perwinlen, t, wt))
                win_data[j] = win_data[j] * wt
            windows.append(win_data)
        return windows
