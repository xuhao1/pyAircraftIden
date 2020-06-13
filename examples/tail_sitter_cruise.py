# coding=utf-8
import sys

sys.path.insert(0, '../')

from AircraftIden.data_case.GeneralAircraftCase import GeneralAircraftCase, PX4AircraftCase
from AircraftIden import FreqIdenSIMO, TransferFunctionFit
import numpy as np
import matplotlib.pyplot as plt
import math


def process_pitch_analyse(test_case: GeneralAircraftCase, time_ranges, win_num=None, omg_min=0.1, omg_max=100):
    # FreqIdenSIMO
    needed_data = ['ele', 'q', 'pitch']
    total_time, t_data, datas = test_case.get_concat_data(time_ranges, needed_data)
    print("sample rate is {:3.1f}".format(datas.__len__() / total_time))
    plt.figure("sourcedata")
    plt.plot(t_data, datas['ele'], 'r.', label='ele')
    plt.plot(t_data, datas['q'], label='pitch')
    plt.grid(which='both')
    plt.legend()
    plt.pause(0.1)

    print("Process pitch rate")
    simo_iden = FreqIdenSIMO(t_data, omg_min, omg_max, datas['ele'], datas['q'], datas['pitch'],
                             uniform_input=True, win_num=win_num)
    plt.figure("ele->q")
    simo_iden.plt_bode_plot(0)
    plt.figure("ele->pitch")
    simo_iden.plt_bode_plot(1)
    plt.show()


def show_logs(px4_case: PX4AircraftCase):
    print("LOW")
    needed_data = ['ele', 'q', 'thr', 'body_vx', "iden_start_time"]
    t_arr, data_list = px4_case.get_data_time_range_list(needed_data)
    print(t_arr)
    data_list[-1] = data_list[-1] / 10
    plt.figure("Ele")
    plt.grid(which='both')
    plt.plot(t_arr, data_list[0], 'r.', label='ele')

    plt.figure("sourcedata")
    # print(t_arr)
    # print(data_list)
    for i in range(needed_data.__len__()):
        plt.plot(t_arr, data_list[i], label=needed_data[i])
    plt.legend()
    plt.show()


def split_logs(px4_case: PX4AircraftCase):
    needed_data = ["iden_start_time"]
    print("Will start split data with", needed_data)
    t_arr, data_list = px4_case.get_data_time_range_list(needed_data)
    iden_start_time = data_list[-1]
    data_splited = []
    is_in_a_test = False
    for i in range(1, t_arr.__len__() - 1):
        if (iden_start_time[i] > iden_start_time[i + 1] or (
                len(data_splited) > 0 and t_arr[i] - data_splited[-1]["start"] > 20)) \
                and is_in_a_test:
            data_splited[-1]["end"] = t_arr[i]
            print("Data split {}th, {:5.2f}:{:5.2f}  len {:5.2f}".format(
                data_splited.__len__(), data_splited[-1]["start"], data_splited[-1]["end"],
                data_splited[-1]["end"] - data_splited[-1]["start"]
            ))
            is_in_a_test = False

        # if (0 <= iden_start_time[i] < iden_start_time[i-1] and iden_start_time[i] < iden_start_time[i+1]):
        if (iden_start_time[i - 1] <= 0 and 0 < iden_start_time[i]):
            # Is a start
            is_in_a_test = True
            data_splited.append({"start": t_arr[i]})

    return data_splited


def join_data(data_splited, status):
    joined_data_status = {}
    assert data_splited.__len__() == status.__len__(), "Status Length must equal to data_split but {} {}".format(
        data_splited.__len__(), status.__len__())

    for i in range(data_splited.__len__()):
        status_test = status[i]
        if status_test == "-" or status_test == "wrong":
            continue
        if status_test in joined_data_status:
            # Join data
            joined_data_status[status_test].append((data_splited[i]["start"], data_splited[i]["end"]))
        else:
            joined_data_status[status_test] = [(data_splited[i]["start"], data_splited[i]["end"])]
    return joined_data_status


def split_and_join_data(px4_case: PX4AircraftCase, status):
    sp = split_logs(px4_case)
    return join_data(sp, status)


def draw_freq_response_on_fig(figure_name, label, freq, H, gamma2):
    plt.figure(figure_name + ":Amp")
    h_amp, h_phase = FreqIdenSIMO.get_amp_pha_from_h(H)
    plt.semilogx(freq, h_amp, label=label)
    plt.title("H Amp")
    plt.grid(which='both')
    plt.legend()

    plt.figure(figure_name + ":Phase")
    plt.semilogx(freq, h_phase, label=label)
    plt.title("H Phase")
    plt.grid(which='both')
    plt.legend()

    plt.figure(figure_name + ":Coherence")
    plt.semilogx(freq, gamma2, label=label)
    plt.legend()
    plt.title("gamma2")
    plt.grid(which='both')


def process_splited_data(test_case, joined_data_status, omg_min, omg_max, win_num=None):
    needed_data = ['ele', 'q', 'thr', "body_vx", "body_vz", "ax", "az"]
    res = {}
    for key in joined_data_status:
        plt.figure("Data case: {}".format(key))
        total_time, t_data, datas = test_case.get_concat_data(joined_data_status[key], needed_data)
        for i in range(1, needed_data.__len__()):
            plt.plot(t_data, datas[needed_data[i]], label=needed_data[i])
            pass

        plt.grid(which='both')
        plt.legend()
        print("Process pitch rate")
        iden = FreqIdenSIMO(t_data, omg_min, omg_max, datas['ele'], datas['q'], datas["body_vx"], datas["body_vz"],
                            datas["ax"], datas["az"],
                            uniform_input=True, win_num=None)  # ,assit_input=datas["thr"])
        res[key] = iden
        plt.figure("ele_q")
        iden.plt_bode_plot(0, label=key)
        plt.figure("ele_body_vx")
        iden.plt_bode_plot(1, label=key)
        plt.figure("ele_body_vz")
        iden.plt_bode_plot(2, label=key)
        plt.figure("ele_body_ax")
        iden.plt_bode_plot(3, label=key)
        plt.figure("ele_body_az")
        iden.plt_bode_plot(4, label=key)
    return iden

if __name__ == "__main__":
    plt.ion()
    fpath = "/Users/xuhao/Dropbox/FLYLOG/foam-tail-sitter/cruising/log_34_2018-4-10-16-16-04.ulg"
    status = [
            "5m/s", "5m/s", "5m/s", "5m/s", "5m/s", "5m/s", "5m/s", "5m/s",
            "8m/s", "8m/s", "8m/s", "8m/s", "8m/s", "8m/s","8m/s", "8m/s",
            "10m/s", "10m/s", "10m/s", "10m/s", "10m/s", "10m/s","10m/s", "10m/s",
    ]
    #process_file(fpath,status)
    px4_case = PX4AircraftCase(fpath)
    show_logs(px4_case)

    status_low = [
            "1m/s", "1m/s", "1m/s", "1m/s", "1m/s", "1m/s", "1m/s", "1m/s",
            "3m/s", "3m/s", "3m/s", "3m/s", "3m/s", "3m/s","3m/s", "3m/s",
            "-", "-", "-",
    ]

    fpath_low = "/Users/xuhao/Dropbox/FLYLOG/foam-tail-sitter/cruising/log_32_2018-4-10-15-53-08.ulg"
    px4_case_low = PX4AircraftCase(fpath_low)
    show_logs(px4_case_low)

    data_splited = split_logs(px4_case)
    res = join_data(data_splited, status)
    idens = process_splited_data(px4_case, res, 5, 50)

    data_splited_low = split_and_join_data(px4_case_low, status_low)
    idens_low = process_splited_data(px4_case_low, res, 5, 50)
    plt.ioff()
    plt.show()