import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import math
from scipy import signal
from scipy.signal import freqz
from aircraft_iden.FreqIden import *
import os


def draw_gxx(fn, omg_min=0.1, omg_max=10):
    arr = np.load(fn)
    time_seq_source = arr[:, 0]
    ele_seq_source = arr[:, 1]
    q_seq_source = arr[:, 4]
    total_time = time_seq_source[-1] - time_seq_source[0]
    sample_rate = len(time_seq_source) / total_time

    tnew, ele_seq, q_seq = time_seq_resample(time_seq_source, omg_max * 5, ele_seq_source, q_seq_source)
    freq_new, gxx, gxy, gyy = get_avg_win_g(total_time, omg_min, omg_max, ele_seq, q_seq, 64)
    f, Pxx_den = signal.welch(ele_seq, sample_rate, nperseg=1024, scaling="spectrum")
    f = f * math.pi * 2
    plt.semilogx(freq_new, 10 * np.log10(gxx))
    plt.semilogx(f, 10 * np.log10(Pxx_den) + 50)
    plt.grid()
    plt.show()
    pass


def process_data(fn, win_num=5, omg_min=0.1, omg_max=20, enable_total=False):
    arr = np.load(fn)
    np.savetxt("{}.csv".format(fn), arr, delimiter=",", newline=os.linesep)
    time_seq_source = arr[:, 0]
    ele_seq_source = arr[:, 1]
    q_seq_source = arr[:, 4]
    total_time = time_seq_source[-1] - time_seq_source[0]
    sample_rate = len(time_seq_source) / total_time

    tnew, ele_seq, q_seq = time_seq_resample(time_seq_source, omg_max * 5, ele_seq_source, q_seq_source)
    plt.figure(0)
    plt.plot(time_seq_source, ele_seq_source, time_seq_source, q_seq_source, tnew, ele_seq, tnew, q_seq)
    plt.grid()
    plt.title("Ele & Q source")


    if enable_total:
        # cut_data_seq_to_windows()
        # print(ele_wins, q_wins)
        freq, Gxx_tilde, Gxy_tilde, Gyy_tilde = get_g_by_time_seq(total_time, omg_min, omg_max, ele_seq, q_seq)
        H = get_h_from_gxy_gxx(Gxy_tilde, Gxx_tilde, freq, omg_min, omg_max)
        gamma_sqr = get_gamma_sqr(Gxx_tilde, Gxy_tilde, Gyy_tilde)
        h_amp, h_phase = get_amp_pha_from_h(H)

        plt.subplot(233)
        plt.semilogx(freq, 20 * np.log10(Gxx_tilde), freq, 20 * np.log10(Gyy_tilde))
        plt.title("Gxx & Gyy Tilde of ele and theta")
        plt.subplot(234)
        plt.semilogx(freq, h_amp)
        plt.grid()
        plt.title("H Amp")

        plt.subplot(235)
        plt.semilogx(freq, h_phase)
        plt.grid()
        plt.title("H phase")

        plt.subplot(236)
        plt.semilogx(freq, gamma_sqr)
        plt.grid()
        plt.title("GammaSqr")

    plt.figure(1)

    freq_new, H_new, gamma_sqr_new, gxx, gxy, gyy = get_h_gamma_by_data(total_time, omg_min, omg_max, ele_seq,
                                                                        q_seq, win_num)
    # freq_h_new, H_new, gamma_sqr_new = get_h_weighed_mean_multi_slice(total_time, omg_min, omg_max, ele_seq,
    #                                                         q_seq, 2*win_num)
    h_amp_new, h_phase_new = get_amp_pha_from_h(H_new)

    plt.subplot(311)
    plt.semilogx(freq_new, 10 * np.log10(gxx))
    plt.grid()
    plt.title("Gxx")
    plt.gca().set_xlim(omg_min, omg_max)
    plt.grid()
    plt.subplot(312)
    plt.semilogx(freq_new, 10 * np.log10(np.absolute(gxy)))
    plt.grid()
    plt.title("Gxy")
    plt.subplot(313)
    plt.semilogx(freq_new, 10 * np.log10(np.absolute(gyy)))
    plt.grid()
    plt.title("Gyy")

    if enable_total:
        plt.figure(2)
        plt.subplot(121)
        plt.semilogx(freq, h_amp, 'o', freq_new, h_amp_new)
        plt.grid()
        plt.subplot(122)
        plt.semilogx(freq, h_phase, 'o', freq_new, h_phase_new)
        plt.grid()
    else:
        plt.figure(3)
        plt.subplot(311)
        plt.semilogx(freq_new, h_amp_new)
        plt.title("H Amp")
        plt.grid()
        plt.subplot(312)
        plt.semilogx(freq_new, h_phase_new)
        plt.title("H phase")
        plt.grid()
        plt.subplot(313)
        plt.semilogx(freq_new, gamma_sqr_new)
        plt.title("Gamma SQR")
        plt.grid()
    plt.show()


if __name__ == "__main__":
    process_data("../data/sweep_data_2017_10_18_14_07.npy", win_num=32, omg_min=0.1, omg_max=100, enable_total=False)
    # draw_gxx("../data/sweep_data_2017_10_18_14_07.npy")
