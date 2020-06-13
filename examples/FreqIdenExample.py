import matplotlib.pyplot as plt
import numpy as np

def siso_freq_iden(win_num=32):
    #save_data_list = ["running_time", "yoke_pitch", "theta", "airspeed", "q", "aoa", "VVI", "alt"]
    arr = np.load("../data/sweep_data_2017_11_16_11_47.npy")
    time_seq_source = arr[:, 0]
    ele_seq_source = arr[:, 1]
    q_seq_source = arr[:, 4]
    vvi_seq_source = arr[:,6]
    theta_seq_source = arr[:,2]
    airspeed_seq = arr[:, 3]
    simo_iden = FreqIdenSIMO(time_seq_source,0.5, 50, ele_seq_source, q_seq_source,theta_seq_source,airspeed_seq, win_num=win_num)

    # plt.figure(0)
    # simo_iden.plt_bode_plot(0)
    # plt.figure(1)
    # simo_iden.plt_bode_plot(1)
    plt.figure("airspeed num{}".format(win_num))
    simo_iden.plt_bode_plot(2)
    plt.show()


if __name__ == "__main__":
    siso_freq_iden(128)