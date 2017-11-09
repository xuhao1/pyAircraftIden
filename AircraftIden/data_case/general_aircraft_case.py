import numpy as np
from pyulog.core import ULog
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from AircraftIden import FreqIdenSIMO
from AircraftIden.FreqIden import remove_seq_average_and_drift
import math


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

    pitchrate_flted = np.ndarray([])

    alt = np.ndarray([])
    climb_rate = np.ndarray([])

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
        plt.plot(self.t_seq, self.thr)
        plt.show()

    def get_data_time_range(self, attr_names, t_min=0, t_max=0):
        assert 0 < t_min < t_max < self.total_time
        ptr_min = int(t_min * self.sample_rate)
        ptr_max = int(t_max * self.sample_rate)

        ress = [self.t_seq[ptr_min:ptr_max]]
        for attr in attr_names:
            assert hasattr(self, attr), "Case has no attr {}".format(attr)
            arr = getattr(self, attr)
            ress.append(arr[ptr_min:ptr_max])
        return tuple(ress)


# sensor_accel
# sensor_gyro
# wind_estimate
# vehicle_status
# vehicle_rates_setpoint
# vehicle_local_position
# vehicle_land_detected
# vehicle_gps_position
# vehicle_global_position
# vehicle_attitude_setpoint
# vehicle_attitude
# telemetry_status
# tecs_status
# task_stack_info
# system_power
# sensor_preflight
# sensor_combined
# rc_channels
# position_setpoint_triplet
# input_rc
# estimator_status
# ekf2_innovations
# cpuload
# control_state
# commander_state
# battery_status
# actuator_outputs
# actuator_outputs
# actuator_controls_0
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
            elif data_obj.name == "vehicle_local_position":
                self.parse_local_position_data(data_obj)

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
        if resampled_datas.__len__() > 1:
            return tuple(resampled_datas)
        else:
            return resampled_datas[0]

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
        # dict_keys(['timestamp', 'rollspeed', 'pitchspeed', 'yawspeed', 'q[0]', 'q[1]', 'q[2]', 'q[3]'])
        t = data.data['timestamp'] / 1000000 - self.t_min
        pitchspeed = data.data['pitchspeed']
        self.pitchrate_flted = self.resample_data(t, pitchspeed)
        pass

    def parse_local_position_data(self, data: ULog.Data):
        # ['timestamp', 'ref_timestamp', 'ref_lat', 'ref_lon', 'surface_bottom_timestamp', 'x', 'y', 'z', 'delta_xy[0]',
        #  'delta_xy[1]', 'delta_z', 'vx', 'vy', 'vz', 'z_deriv', 'delta_vxy[0]', 'delta_vxy[1]', 'delta_vz', 'ax', 'ay',
        #  'az', 'yaw', 'ref_alt', 'dist_bottom', 'dist_bottom_rate', 'eph', 'epv', 'evh', 'evv', 'estimator_type',
        #  'xy_valid', 'z_valid', 'v_xy_valid', 'v_z_valid', 'xy_reset_counter', 'z_reset_counter', 'vxy_reset_counter',
        #  'vz_reset_counter', 'xy_global', 'z_global', 'dist_bottom_valid']
        t = data.data['timestamp'] / 1000000 - self.t_min
        roc = data.data['vz']
        alt = data.data['z']
        self.climb_rate, self.alt = self.resample_data(t, roc, - alt)


def get_concat_data(test_case: GeneralAircraftCase, time_ranges, attrs):
    res = dict()
    for attr in attrs:
        attr_data = []
        for t_min, t_max in time_ranges:
            _, piece_data = test_case.get_data_time_range(
                [attr], t_min=t_min,
                t_max=t_max)
            piece_data = remove_seq_average_and_drift(piece_data.copy())
            attr_data.append(piece_data)
        res[attr] = np.concatenate(attr_data)
    return res


def process_lat_analyse(test_case: GeneralAircraftCase, time_ranges, win_num=32, omg_min=0.1, omg_max=100):
    # FreqIdenSIMO
    ele_data = None
    r_data = None

    needed_data = ['ele', 'p', 'q', 'r', 'thr', 'climb_rate', 'alt']
    datas = get_concat_data(test_case, time_ranges, needed_data)
    ele_data = datas['ele']
    p_data = datas['p']
    q_data = datas['q']
    r_data = datas['r']
    thr_data = datas['thr']
    roc_data = datas['climb_rate']
    alt = datas['alt']

    total_time = len(ele_data) / test_case.sample_rate

    t_data = np.linspace(0, total_time, len(ele_data))

    win_num_min = 2
    win_num_max = int(total_time * omg_max / (20 * math.pi) - 1)

    if time_ranges.__len__() < 3:
        win_num_min = 5

    print("win num should in {} and {}".format(win_num_min, win_num_max))

    simo_iden = FreqIdenSIMO(t_data, omg_min, omg_max, ele_data, r_data, roc_data, win_num=win_num,
                             assit_input=thr_data)

    # simo_iden = FreqIdenSIMO(t_data, omg_min, omg_max, ele_data,r_data, roc_data, win_num=win_num)
    # plt.figure(0)
    # plt.subplot(211)
    # plt.plot(simo_iden.time_seq, simo_iden.x_seq, simo_iden.time_seq, simo_iden.y_seqs[1])
    # plt.grid(which='both', axis='both')
    # plt.subplot(212)
    # plt.plot(t_data, ele_data, t_data, alt)
    # plt.grid(which='both', axis='both')

    plt.figure('Elevator ->Pitch rate')

    simo_iden.plt_bode_plot(0)

    plt.figure('Elevator -> rate of climb rate')

    simo_iden.plt_bode_plot(1)

    plt.show()


if __name__ == "__main__":
    px4_case = PX4AircraftCase("C:\\Users\\xuhao\\Desktop\\FLYLOG\\2017-10-26\\log002.ulg")

    process_lat_analyse(px4_case, [(72.5, 88.2), (145.2, 160.6), (202, 219)], win_num=16, omg_min=0.1, omg_max=36)
    #process_lat_analyse(px4_case, [(72.5, 88.2)], win_num=5, omg_min=3, omg_max=36)
