from AircraftIden.data_case.GeneralAircraftCase import GeneralAircraftCase, PX4AircraftCase
from AircraftIden import FreqIdenSIMO, TransferFunctionFit
import numpy as np
import matplotlib.pyplot as plt
import math


def process_pitch_analyse(test_case: GeneralAircraftCase, time_ranges, win_num=None, omg_min=0.1, omg_max=100):
    # FreqIdenSIMO
    needed_data = ['ele', 'q','pitch']
    total_time, t_data, datas = test_case.get_concat_data(time_ranges, needed_data)
    print("sample rate is {:3.1f}".format(datas.__len__()/total_time))
    plt.figure("sourcedata")
    plt.plot(t_data, datas['ele'],'r.', label='ele')
    plt.plot(t_data, datas['q'], label='pitch')
    plt.grid(which='both')
    plt.legend()
    plt.pause(0.1)

    print("Process pitch rate")
    simo_iden = FreqIdenSIMO(t_data, omg_min, omg_max, datas['ele'], datas['q'],datas['pitch'],
                             uniform_input=True,win_num=win_num)
    plt.figure("ele->q")
    simo_iden.plt_bode_plot(0)
    plt.figure("ele->pitch")
    simo_iden.plt_bode_plot(1)
    plt.show()

def show_logs(px4_case: PX4AircraftCase):
    needed_data = ['ele', 'q','thr','pitch']
    t_arr,data_list = px4_case.get_data_time_range_list(needed_data)
    plt.figure("Ele")
    plt.grid(which='both')
    plt.plot(t_arr, data_list[0],'r.', label='ele')

    plt.figure("sourcedata")
    print(t_arr)
    print(data_list)
    for i in range(needed_data.__len__()):
        plt.plot(t_arr,data_list[i],label = needed_data[i])
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # path = "/Users/xuhao/Dropbox/FLYLOG/foam-tail-sitter/hover/2018_2_6_tailsitter_hover_sweep_outdoor.ulg"
    path = "/Users/xuhao/Dropbox/FLYLOG/foam-tail-sitter/hover/2018_3_9_tailsitter_hover_outdoot_high_frequency.ulg"
    import sys
    if sys.argv.__len__() > 2:
        print(sys.argv)
        path = sys.argv[2]
    px4_case = PX4AircraftCase(path)
    #show_logs(px4_case)
    process_pitch_analyse(px4_case,[(62,98),(122.2,140)],omg_min=5,omg_max=120)
    #process_pitch_analyse(px4_case, [(62,98),(122.2,140),(155,220),(230,285),(302,360)],omg_min=5, omg_max=120)