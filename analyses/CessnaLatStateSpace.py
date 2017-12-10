import matplotlib

matplotlib.use("Qt5Agg")

from AircraftIden.data_case.GeneralAircraftCase import GeneralAircraftCase, PX4AircraftCase, get_concat_data
from AircraftIden import FreqIdenSIMO, TransferFunctionFit
import numpy as np

import matplotlib.pyplot as plt
import math
from AircraftIden.StateSpaceIden import StateSpaceIdenSIMO, StateSpaceParamModel
import sympy as sp
import pickle
from AircraftIden.FreqIden import time_seq_preprocess


def lat_dyn_SIMO(freqres,show_freq_iden_plots=False, show_ssm_iden_plot=False):
    # X = [u,w,q,th]
    # Y = [w,q,th,ax,az]
    # Note ax ay contain gravity acc

    Xwdot, Zwdot, Mwdot = sp.symbols('Xwdot Zwdot Mwdot')

    M = sp.Matrix([[1, -Xwdot, 0, 0],
                   [0, 1 - Zwdot, 0, 0],
                   [0, -Mwdot, 1, 0],
                   [0, 0, 0, 1]])

    g = 9.78
    th0 =-0.004783792053474051
    U0 = 64.24
    W0 = -1.1477373864894442

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

    # direct using u w q ax az for y
    H0 = sp.Matrix([
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, W0, - g * math.cos(th0)],  # Our ax is along forward
                    [0, 0, -U0, - g * math.sin(th0)]]  # az is down to earth
                   )

    H1 = sp.Matrix([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ])

    syms = [Xwdot, Zwdot, Mwdot,
            Xu, Xw, Xq,
            Zu, Zw, Zq,
            Mu, Mw, Mq,
            Xele, Zele, Mele]
    lat_dyn_state_space = StateSpaceParamModel(M, F, G, H0, H1, syms)

    ssm_iden = StateSpaceIdenSIMO(freqres, accept_J=20,
                                  enable_debug_plot=show_ssm_iden_plot,
                                  y_names=["w", "q", r"a_x", r"a_z"],)
    J, ssm = ssm_iden.estimate(lat_dyn_state_space, syms, constant_defines={})
    ssm.check_stable()
    ssm_iden.draw_freq_res()
    # print(ssm.
    with open("../data/SIMStateSpaceExample.pkl", 'wb') as output:
        pickle.dump(ssm, output, pickle.HIGHEST_PROTOCOL)


def post_analyse_ssm(pkl_name, show_freq_iden_plots=False):
    #verf data full throttle 3100meter high 62.4m/s
    arr = np.load("../../XPlaneResearch/data/sweep_data_2017_12_10_20_08.npy")
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

    # X = [u,w,q,th]
    # Y = [w,q,th,ax,az]
    # Note ax ay contain gravity acc
    g = 9.78
    th0 = theta_seq[0]
    U0 = vx_seq[0]
    W0 = vz_seq[0]
    ax0 = ax_seq[0]
    az0 = az_seq[0]

    if show_freq_iden_plots:
        fig = plt.figure("source data")
        fig.set_size_inches(18, 10)
        plt.subplot(311)
        plt.plot(time_seq, aoa_seq * 180 / math.pi, label='aoa')
        plt.plot(time_seq, theta_seq * 180 / math.pi, label='theta')
        plt.plot(time_seq, az_seq - az_seq[0], label='az - {:2.1f}'.format(az_seq[0]))
        plt.legend()

        plt.subplot(312)
        plt.plot(time_seq, vz_seq, label='vz')
        plt.plot(time_seq, vx_seq - vx_seq[0], label='vx - {:3.1f}'.format(vx_seq[0]))
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
    time_seq1, ele_seq1 = time_seq_preprocess(time_seq, ele_seq,
                                              remove_drift_and_avg=True,
                                              enable_resample=True)

    with open(pkl_name, 'rb') as input:
        ssm = pickle.load(input)

        # ele_seq = np.zeros(ele_seq.shape)
        t_seq, y_seq,x_out = ssm.response_by_u_seq(t_seq=time_seq1, u_seq=ele_seq1, X0=np.array([0, 0, 0, 0]))

        plt.figure("Elevator->Y")

        plt.subplot(321)
        plt.plot(t_seq, ele_seq1, label="data")
        plt.title("ele")

        # plt.subplot(322)
        # plt.plot(t_seq, y_seq[:, 0] + U0, label="est")
        # plt.plot(time_seq, vx_seq, label="data")
        # plt.legend()
        # plt.title("u")

        plt.subplot(323)
        plt.plot(t_seq, y_seq[:, 0] + W0, label="est")
        plt.plot(time_seq, vz_seq, label="data")
        plt.legend()
        plt.title("w")

        plt.subplot(324)
        plt.plot(t_seq, y_seq[:, 1], label="est")
        plt.plot(time_seq, q_seq, label="data")
        plt.legend()
        plt.title("q")


        plt.subplot(325)
        plt.plot(t_seq, y_seq[:, 2] + ax0, label="est")
        plt.plot(time_seq, ax_seq, label="data")
        plt.legend()
        plt.title(r"a_x")
        plt.subplot(326)
        plt.plot(t_seq, y_seq[:, 3] + az0, label="est")
        plt.plot(time_seq, az_seq, label="data")
        plt.legend()
        plt.title(r"a_z")

        plt.figure("Elevator->x")
        plt.subplot(221)
        plt.plot(t_seq, vx_seq,label="Data")
        plt.plot(t_seq, x_out[:,0] + U0,label="Est")
        plt.legend()
        plt.title("u")

        plt.subplot(222)
        plt.plot(t_seq, vz_seq,label="Data")
        plt.plot(t_seq, x_out[:,1] + W0,label="Est")
        plt.legend()
        plt.title("w")

        plt.subplot(223)
        plt.plot(t_seq, q_seq, label="Data")
        plt.plot(t_seq, x_out[:,2] , label="Est")
        plt.legend()
        plt.title("q")

        plt.subplot(224)
        plt.plot(t_seq, theta_seq, label="Data")
        plt.plot(t_seq, x_out[:,3] + th0, label="Est")
        plt.legend()
        plt.title("th")

        plt.show()


if __name__ == "__main__":
    pkl_name = "../data/sweep_data_2017_12_10_19_05_freqres.pkl"
    with open(pkl_name, 'rb') as inp:
        freqres = pickle.load(inp)
        # lat_dyn_SIMO(freqres,show_freq_iden_plots=True)
        post_analyse_ssm("../data/SIMStateSpaceExample.pkl")
