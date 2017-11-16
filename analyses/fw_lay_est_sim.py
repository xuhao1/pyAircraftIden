from AircraftIden.data_case.GeneralAircraftCase import GeneralAircraftCase, PX4AircraftCase, get_concat_data
from AircraftIden import FreqIdenSIMO, TransferFunctionFit
import numpy as np
import matplotlib.pyplot as plt
import math
from AircraftIden.StateSpaceIden import StateSpaceIdenSIMO, StateSpaceModel
import sympy as sp

def lat_dyn_SIMO(iter, show_freq_iden_plots=False):
    # save_data_list = ["running_time", "yoke_pitch",
    #                  "theta", "airspeed", "q", "aoa", "VVI", "alt", "vx_body", "vy_body", "vz_body"]

    arr = np.load("../data/sweep_data_2017_11_16_11_47.npy")
    time_seq = arr[:, 0]
    ele_seq = arr[:, 1]
    q_seq = arr[:, 4]
    vvi_seq = arr[:, 6]
    theta_seq = arr[:, 2] / 180 * math.pi
    airspeed_seq = arr[:, 3]
    aoa_seq = arr[:, 5] / 180 * math.pi

    vx_seq = arr[:, 10]
    vz_seq = arr[:, 9]
    vy_seq = arr[:, 8]

    # X = [u,w,q,th]
    if show_freq_iden_plots:
        plt.figure("source data")
        plt.subplot(411)
        plt.plot(time_seq, vx_seq, label='vx')
        plt.subplot(412)
        plt.plot(time_seq, vy_seq, label='vy')
        plt.subplot(413)
        plt.plot(time_seq, vz_seq, label='vz')
        plt.subplot(414)
        plt.plot(time_seq, q_seq, label='q')
        plt.plot(time_seq, theta_seq, label='theta')
        plt.legend()
    # plt.show()
    simo_iden = FreqIdenSIMO(time_seq, 0.5, 50, ele_seq, airspeed_seq, vz_seq,
                             q_seq, theta_seq, win_num=32)

    if show_freq_iden_plots:
        plt.figure("Ele->Airspeed")
        simo_iden.plt_bode_plot(0)
        plt.figure("Ele->Vz")
        simo_iden.plt_bode_plot(1)
        plt.figure("Ele->Q")
        simo_iden.plt_bode_plot(2)
        plt.figure("Ele->Th")
        simo_iden.plt_bode_plot(3)
        plt.pause(0.5)

    freq, Hs, coherens = simo_iden.get_all_idens()

    Xwdot, Zwdot, Mwdot = sp.symbols('Xwdot Zwdot Mwdot')

    M = sp.Matrix([[1, -Xwdot, 0, 0],
                   [0, 1 - Zwdot, 0, 0],
                   [0, -Mwdot, 1, 0],
                   [0, 0, 0, 1]])

    g = 9.78
    th0 = theta_seq[0]
    U0 = vx_seq[0]
    W0 = vz_seq[0]
    print("Trim theta {} U0 {} W0 {}".format(th0, U0, W0))
    Xu, Xw, Xq = sp.symbols('Xu Xw Xq')
    Zu, Zw, Zq = sp.symbols('Zu Zw Zq')
    Mu, Mw, Mq = sp.symbols('Mu Mw Mq')

    F = sp.Matrix([[Xu, Xw, Xq - W0, -g * sp.cos(th0)],
                   [Zu, Zw, Zq + U0, -g * sp.sin(th0)],
                   [Mu, Mw, Mq, 0],
                   [0, 0, 1, 0]])

    Xele, Zele, Mele = sp.symbols('Xele,Zele,Mele')
    G = sp.Matrix([[Xele],
                   [Zele],
                   [Mele],
                   [0]])

    # direct using u w q th for y
    H0 = sp.Matrix([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

    H1 = sp.Matrix.zeros(4, 4)

    syms = [Xwdot, Zwdot, Mwdot,
            Xu, Xw, Xq,
            Zu, Zw, Zq,
            Mu, Mw, Mq,
            Xele, Zele, Mele]
    lat_dyn_state_space = StateSpaceModel(M, F, G, H0, H1, syms)

    ssm_iden = StateSpaceIdenSIMO(freq, Hs, coherens, max_sample_time=iter, accept_J=30)
    ssm_iden.estimate(lat_dyn_state_space, syms, constant_defines={})


if __name__ == "__main__":
    lat_dyn_SIMO(10)

