import matplotlib

matplotlib.use("Qt5Agg")

from AircraftIden import FreqIdenSIMO, TransferFunctionFit
import numpy as np

import matplotlib.pyplot as plt
import math
import pickle
from os.path import basename,splitext

def lat_dyn_freq(fn,show_freq_iden_plots=True):
    name = splitext(basename(fn))[0]
    arr = np.load(fn)
    print("Start analyse case {}".format(name))
    time_seq = arr[:, 0]
    ele_seq = arr[:, 1]
    q_seq = arr[:, 4]
    vvi_seq = arr[:, 6]
    theta_seq = arr[:, 2] / 180 * math.pi
    airspeed_seq = arr[:, 3]
    aoa_seq = arr[:, 5] / 180 * math.pi

    vx_seq = arr[:, 8]
    vy_seq = arr[:, 9]
    vz_seq = arr[:, 10]

    ax_source = arr[:, 11].copy()
    ax_seq = arr[:, 11]  # + np.sin(theta_seq) * 9.8
    # ay_seq = arr[:, 12]
    az_seq = arr[:, 13]
    print("U0 {} W0 {} th0 {}".format(vx_seq[0],vz_seq[0],theta_seq[0]))
    # X = [u,w,q,th]
    # Y = [w,q,th,ax,az]
    # Note ax ay contain gravity acc
    simo_iden = FreqIdenSIMO(time_seq, 1, 20, ele_seq,vz_seq,
                             q_seq,ax_seq, az_seq)
    freqres = simo_iden.get_freqres()
    output_path = "../data/{}_freqres.pkl".format(name)
    with open(output_path, 'wb') as output:
        pickle.dump(freqres, output, pickle.HIGHEST_PROTOCOL)
        print("Saved Freq Res to {}".format(output_path))

    if show_freq_iden_plots:
        fig = plt.figure("source data")
        fig.set_size_inches(18, 10)
        plt.subplot(311)
        # plt.plot(time_seq, aoa_seq * 180 / math.pi, label='aoa')
        plt.plot(time_seq, theta_seq * 180 / math.pi, label='theta')
        # plt.plot(time_seq, az_seq - az_seq[0], label='az - {:2.1f}'.format(az_seq[0]))
        plt.legend()

        plt.subplot(312)
        plt.plot(time_seq, vz_seq, label='vz')
        # plt.plot(time_seq, vx_seq - vx_seq[0], label='vx - {:3.1f}'.format(vx_seq[0]))
        plt.legend()

        plt.subplot(313)
        # plt.plot(time_seq, q_seq, label='q')
        # plt.plot(time_seq, theta_seq*57, label='theta')
        plt.plot(time_seq, ax_seq, label='ax')
        plt.plot(time_seq, - np.sin(theta_seq) * 9.8, label='ax_by_theta')
        plt.plot(time_seq, ax_source, label='axsource')
        plt.plot(time_seq, aoa_seq * 50, label='aoa')
        plt.grid(which='both')
        # plt.plot(time_seq, ele_seq, label="ele")

        plt.legend()
        plt.figure("Ele->U(vx)")
        simo_iden.plt_bode_plot(0)
        plt.figure("Ele->W(vz)")
        simo_iden.plt_bode_plot(1)
        plt.figure("Ele->Q")
        simo_iden.plt_bode_plot(2)
        plt.figure("Ele->Ax")
        simo_iden.plt_bode_plot(3)
        plt.figure("Ele->Az")
        simo_iden.plt_bode_plot(4)

        plt.show()





if __name__ == "__main__":
    import sys
    #around 3100 meter high, full throttle speed 64.2m/s
    fn = "../../XPlaneResearch/data/sweep_data_2017_12_10_19_05.npy"

    if sys.argv.__len__() > 2:
        fn = sys.argv[1]

    lat_dyn_freq(fn,show_freq_iden_plots=True)