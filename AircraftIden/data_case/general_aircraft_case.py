import numpy as np
from pyulog.core import ULog
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from AircraftIden import FreqIdenSIMO


class GeneralAircraftCase(object):
    sample_rate = 0
    total_time = 0
    p = np.ndarray([])
    q = np.ndarray([])
    r = np.ndarray([])
    t_seq = np.ndarray([])

    roll = np.ndarray([])
    pitch = np.ndarray([])
    yaw = np.ndarray([])

    ail = np.ndarray([])
    ele = np.ndarray([])
    rud = np.ndarray([])
    thr = np.ndarray([])

    controls = np.ndarray([])

    def __init__(self):
        pass

    def display_log(self):
        plt.plot(self.t_seq, self.r, label='q')
        plt.plot(self.t_seq, self.ele, label='ele')
        plt.plot(self.t_seq,self.thr)
        plt.show()

    def get_data_time_range(self, attr_names, t_min=0, t_max=0):
        assert 0 < t_min < t_max < self.total_time
        ptr_min = int(t_min * self.sample_rate)
        ptr_max = int(t_max * self.sample_rate)

        ress = [self.t_seq[ptr_min:ptr_max]]
        for attr in attr_names:
            arr = getattr(self, attr)
            ress.append(arr[ptr_min:ptr_max])
        return tuple(ress)


class PX4AircraftCase(GeneralAircraftCase):
    def __init__(self, px4_log_name, default_sample_rate=200, log_type="ulog"):
        super().__init__()

        self.sample_rate = default_sample_rate

        self.t_min = 0
        self.t_max = 0

        if log_type == "ulog":
            self.parse_ulog(px4_log_name)
        else:
            raise "Un recognize log type {}".format(log_type)

    def parse_ulog(self, fn):
        open(fn)
        self.ulog = ULog(fn)

        for data_obj in self.ulog.data_list:  # type:ULog.Data
            if data_obj.name == "sensor_gyro":
                # We use gyro to setup time seq
                self.parse_gyro_data(data_obj)
                break

        for data_obj in self.ulog.data_list:  # type:ULog.Data
            if data_obj.name == "vehicle_attitude":
                self.parse_attitude_data(data_obj)
            elif data_obj.name == "actuator_controls_0":
                self.parse_actuator_controls(data_obj)

    def resample_data(self, t, *x_seqs):
        resampled_datas = []
        for x_seq in x_seqs:
            assert len(x_seq) == len(x_seqs[0]), "Length of data seq must be euqal to time seq"

            inte_func = interp1d(t, x_seq, bounds_error=False)
            data = inte_func(self.t_seq)
            # TODO:deal with t< t_min and t > t_max
            # zi = signal.lfilter_zi(b, a)
            # z, _ = signal.lfilter(b, a, data, zi=zi * data[0])
            resampled_datas.append(data)
        return tuple(resampled_datas)

    def parse_gyro_data(self, data: ULog.Data):
        t = data.data['timestamp'] / 1000000 - self.t_min
        p = data.data['x']
        q = data.data['y']
        r = data.data['z']

        self.t_min = t[0]
        self.t_max = t[-1]

        self.total_time = self.t_max - self.t_min
        self.t_seq = np.linspace(0, self.total_time, num=int((self.t_max - self.t_min) * self.sample_rate),
                                 endpoint=True)
        self.p, self.q, self.r = self.resample_data(t - self.t_min, p, q, r)

    def parse_pwm_data(self, data: ULog.Data):
        pass

    def parse_actuator_controls(self, data: ULog.Data):
        t = data.data['timestamp'] / 1000000 - self.t_min
        ail = data.data['control[0]']
        ele = data.data['control[1]']
        rud = data.data['control[2]']
        thr = data.data['control[3]']
        self.ail, self.ele, self.thr, self.rud = self.resample_data(t, ail, ele, thr, rud)

    def parse_attitude_data(self, data):
        pass


def process_lat_analyse(test_case: GeneralAircraftCase, time_ranges,win_num = 32):
    # FreqIdenSIMO
    ele_data = None
    r_data = None
    for t_min,t_max in time_ranges:
        assert 0 < t_min < t_max < test_case.total_time
        t_data_tmp, ele_data_tmp, p_data_tmp, q_data_tmp, r_data_tmp = test_case.get_data_time_range(['ele', 'p', 'q', 'r'], t_min=t_min,
                                                                             t_max=t_max)
        if ele_data is None:
            ele_data = ele_data_tmp
        else:
            ele_data = np.concatenate([ele_data, ele_data_tmp])

        if r_data is None:
            r_data = r_data_tmp
        else:
            r_data = np.concatenate([r_data,r_data_tmp])

    t_data = np.linspace(0,len(ele_data)/test_case.sample_rate,len(ele_data))

    plt.figure(0)
    plt.plot(t_data, ele_data,t_data,r_data)
    plt.grid(which='both',axis='both')

    plt.figure(1)
    simo_iden = FreqIdenSIMO(t_data, 0.1, 100, ele_data, r_data, win_num=win_num)
    simo_iden.plt_bode_plot(0)
    plt.show()


if __name__ == "__main__":

    px4_case = PX4AircraftCase("C:\\Users\\xuhao\\Desktop\\FLYLOG\\2017-10-26\\log002.ulg")
    #px4_case.display_log()

    process_lat_analyse(px4_case, [(72.5,88.2),(145.2,160.6),(202,219)],win_num=32)

    # plt.figure(0)
