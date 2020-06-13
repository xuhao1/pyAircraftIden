import sys

sys.path.insert(0, '../')
from AircraftIden import FreqIdenSIMO, TransferFunctionFit
import math
import matplotlib.pyplot as plt
import pickle
import multiprocessing

# X = [u,w,th]
# Y = [u,w]
import sympy as sp
from AircraftIden.StateSpaceIden import StateSpaceIdenSIMO, StateSpaceParamModel

M = sp.Matrix([[1, 0, 0],
               [0, 1, 0],
               [0, 0, 1]])

# Tailsittel
#    0---Vx, W
# --- | ----
# |  |  |
#    |
#  Vz ,-U
# 10m/s

# Vz trim is -6.05,Vz is negative U
# Vx trim is 4.55, Vx is W
g = 9.78

Xu, Xw, Xq = sp.symbols('Xu Xw Xq')
Zu, Zw, Zq = sp.symbols('Zu Zw Zq')
Mu, Mw, Mq = sp.symbols('Mu Mw Mq')


def callback(xk, state):
    print(xk)
    print(state)


def process_ssm(freqres, trims):
    th0 = trims["theta"]
    W0 = trims["W0"]
    U0 = trims["U0"]
    F = sp.Matrix([[Xu, Xw, -g * math.cos(th0)],
                   [Zu, Zw, -g * math.sin(th0)],
                   [0, 0, 0]])
    G = sp.Matrix([[Xq - W0], [Zq + U0], [1]])
    # direct using u w q for y
    # U equal to negative u
    H0 = sp.Matrix([
        [1, 0, 0],
        [0, 1, 0]])
    H1 = sp.Matrix([
        [0, 0, 0],
        [0, 0, 0],
    ])
    syms = [Xu, Xw, Zu, Zw, Xq, Zq]
    LatdynSSPM = StateSpaceParamModel(M, F, G, H0, H1, syms)

    plt.rc('figure', figsize=(10.0, 5.0))
    freqres = freqres.get_freqres(indexs=[0, 1])
    ssm_iden = StateSpaceIdenSIMO(freqres, accept_J=150,
                                  enable_debug_plot=False,
                                  y_names=['U', "w"], reg=0.1, iter_callback=callback, max_sample_times=10)
    J, ssm = ssm_iden.estimate(LatdynSSPM, syms, constant_defines={}, rand_init_max=10)
    ssm.check_stable()
    ssm_iden.draw_freq_res()
    ssm_iden.print_res()

    plt.show()



if __name__ == "__main__":
    multiprocessing.freeze_support()
    pkl_name = "../data/sweep_data_2017_12_10_19_05_freqres.pkl"
    with open(pkl_name, 'rb') as finput:
        freqres = pickle.load(finput)
        process_ssm(freqres,{
            "theta":0,
            "U0":64.24,
            "W0":-1.14
        })