from AircraftIden import FreqIdenSIMO
import matplotlib.pyplot as plt
import numpy as np

def siso_freq_iden():
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

if __name__ == "__main__":
    siso_freq_iden()