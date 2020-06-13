from AircraftIden.data_case.GeneralAircraftCase import GeneralAircraftCase, PX4AircraftCase, get_concat_data
from AircraftIden import FreqIdenSIMO, TransferFunctionFit
import numpy as np
import matplotlib.pyplot as plt
import math


def process_lat_analyse(test_case: GeneralAircraftCase, time_ranges, win_num=32, omg_min=0.1, omg_max=100):
    # FreqIdenSIMO
    needed_data = ['ele', 'p', 'q', 'r', 'pitchrate_flted', 'thr', 'climb_rate', 'alt','pitch']
    total_time, t_data, datas = get_concat_data(test_case, time_ranges, needed_data)

    win_num_min = 2
    win_num_max = int(total_time * omg_max / (20 * math.pi) - 1)

    if time_ranges.__len__() < 3:
        win_num_min = 5

    plt.figure("sourcedata")
    plt.plot(t_data,datas['ele'],label='ele')
    plt.plot(t_data,datas['pitch'],label='pitch')
    plt.legend()
    plt.pause(0.1)

    print("win num should in {} and {}".format(win_num_min, win_num_max))

    print("Process pitch rate")
    simo_iden = FreqIdenSIMO(t_data, omg_min, omg_max, datas['ele'], datas['r'], datas['climb_rate'], datas['pitch'],
                             win_num=win_num,
                             assit_input=datas['thr'])

    print("Process roc")
    freq, H, gamma2, gxx, gxy, gyy = simo_iden.get_freq_iden(0)
    fitter = TransferFunctionFit(freq, H, gamma2, 1, 2, enable_debug_plot=True)
    fitter.estimate()

    plt.figure('Elevator ->Pitch rate')
    fitter.plot()

    freq, H, gamma2, gxx, gxy, gyy = simo_iden.get_freq_iden(1)
    fitter = TransferFunctionFit(freq, H, gamma2, 2, 4, enable_debug_plot=False)
    fitter.estimate()

    plt.figure('Elevator -> rate of climb rate')
    fitter.plot()

    print("Process pitchangle")
    freq, H, gamma2, gxx, gxy, gyy = simo_iden.get_freq_iden(2)
    fitter = TransferFunctionFit(freq, H, gamma2, 1, 3, enable_debug_plot=True)
    fitter.estimate()

    plt.figure('Elevator -> pitch angle')
    fitter.plot()

    plt.show()


if __name__ == "__main__":
    px4_case = PX4AircraftCase("C:\\Users\\xuhao\\Desktop\\FLYLOG\\2017-10-26\\log002.ulg")
    process_lat_analyse(px4_case, [(72.5, 88.2), (145.2, 160.6), (202, 219)], win_num=16, omg_min=3, omg_max=30)
