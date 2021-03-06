import numpy as np
from pyulog.core import ULog
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from AircraftIden.FreqIden import remove_seq_average_and_drift

import math
from pymavlink import quaternion
from pymavlink.rotmat import Vector3


class GeneralAircraftCase(object):
    # Todo add inteplote
    sample_rate = 0
    total_time = 0
    p = np.ndarray([])
    q = np.ndarray([])
    r = np.ndarray([])
    t_seq = np.ndarray([])

    roll = np.ndarray([])
    pitch = np.ndarray([])
    yaw = np.ndarray([])

    roll_sp = np.ndarray([])
    pitch_sp = np.ndarray([])
    yaw_sp = np.ndarray([])

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
        if t_max is None or t_max > self.total_time:
            t_max = self.total_time
        if t_min is None:
            t_min = 0
        assert 0 <= t_min < t_max
        ptr_min = int(t_min * self.sample_rate)
        ptr_max = int(t_max * self.sample_rate)

        ress = [self.t_seq[ptr_min:ptr_max]]
        for attr in attr_names:
            assert hasattr(self, attr), "Case has no attr {}".format(attr)
            arr = getattr(self, attr)
            ress.append(arr[ptr_min:ptr_max])
        return tuple(ress)

    def get_data_time_range_list(self, attr_names, t_min=None, t_max=None):
        if t_max is None or t_max > self.total_time:
            t_max = self.total_time
        if t_min is None:
            t_min = 0
        assert 0 <= t_min < t_max

        ptr_min = int(t_min * self.sample_rate)
        ptr_max = int(t_max * self.sample_rate)

        ress = []
        for attr in attr_names:
            assert hasattr(self, attr), "Case has no attr {}".format(attr)
            arr = getattr(self, attr)
            ress.append(arr[ptr_min:ptr_max])
        return self.t_seq[ptr_min:ptr_max], ress

    def get_concat_data(self, time_ranges, attrs, return_trimed = True):
        res = dict()
        count = 0
        sumup = 0
        for attr in attrs:
            attr_data = []
            for t_min, t_max in time_ranges:
                _, piece_data = self.get_data_time_range(
                    [attr], t_min=t_min,
                    t_max=t_max)
                # piece_data = remove_seq_average_and_drift(piece_data.copy())
                if return_trimed:
                    piece_data = piece_data.copy() - np.average(piece_data)
                else:
                    piece_data = piece_data.copy()
                # print("Do not remove drift")
                attr_data.append(piece_data)
            res[attr] = np.concatenate(attr_data)
            datalen = res[attrs[0]].__len__()
            totaltime = datalen / self.sample_rate
            tseq = np.linspace(0, totaltime, datalen)
        return totaltime, tseq, res


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
        try:
            print(f"Opening log file {fn}")
            self.ulog = ULog(fn)
        except Exception as e:
            print("Error while parse ulog")
            print(e)
            exit(-1)
        
        for data_obj in self.ulog.data_list:  # type:ULog.Data
            if data_obj.name == "sensor_gyro":
                # We use gyro to setup time seq
                print("Parse sensor gyro")
                self.parse_gyro_data(data_obj)
                break
                
        for data_obj in self.ulog.data_list:  # type:ULog.Data                
            if data_obj.name == "vehicle_attitude":
                print("Parse vehicle attitude")
                self.parse_attitude_data(data_obj)
                
            elif data_obj.name == "vehicle_attitude_setpoint":
                print("Parse vehicle attitude sp")
                self.parse_attitude_data_sp(data_obj)
            
        for data_obj in self.ulog.data_list:  # type:ULog.Data
            if data_obj.name == "actuator_controls_0":
                print("actuator_controls_0")
                self.parse_actuator_controls(data_obj)
                
            elif data_obj.name == "vehicle_local_position":
                print("vehicle_local_position")
                self.parse_local_position_data(data_obj)
                
            elif data_obj.name == "vehicle_iden_status":
                print("vehicle_iden_status")
                self.parse_vehicle_iden_status(data_obj)
                
            elif data_obj.name == "sensor_accel":
                print("sensor_accel")
                self.parse_sensor_accel(data_obj)


    def resample_data(self, t, *x_seqs):
        resampled_datas = []
        for x_seq in x_seqs:
            func = lambda x: 0 if np.isnan(x) or np.isinf(x) else x
            x_seq = (np.vectorize(func))(x_seq)
            assert len(x_seq) == len(x_seqs[0]), "Length of data seq must be euqal to time seq"
            inte_func = interp1d(t, x_seq, bounds_error=False, kind='linear', fill_value=0)
            data = inte_func(self.t_seq)
            # TODO:deal with t< t_min and t > t_max

            data = (np.vectorize(func))(data)
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

    def parse_sensor_accel(self, data: ULog.Data):
        # (['timestamp', 'integral_dt', 'error_count', 'x', 'y', 'z', 'x_integral', 'y_integral', 'z_integral',
        #            'temperature', 'range_m_s2', 'scaling', 'device_id', 'x_raw', 'y_raw', 'z_raw', 'temperature_raw'])
        t = data.data['timestamp'] / 1000000 - self.t_min
        ax = data.data['x']
        ay = data.data['y']
        az = data.data['z']
        self.ax, self.ay, self.az = self.resample_data(t, ax, ay, az)

    def parse_actuator_controls(self, data: ULog.Data):
        t = data.data['timestamp'] / 1000000 - self.t_min
        ail = data.data['control[0]']
        ele = data.data['control[1]']
        rud = data.data['control[2]']
        thr = data.data['control[3]']

        self.ail, self.ele, self.thr, self.rud = self.resample_data(t, ail, ele, thr, rud)

    def parse_vehicle_iden_status(self, data: ULog.Data):
        # ['timestamp', 'iden_start_time', 'iden_wait_time', 'inject_param1', 'inject_param2', 'inject_param3',
        # 'inject_param4', 'inject_value', 'inject_channel', 'inject_signal_mode']
        # print(data.data.keys())
        t = data.data['timestamp'] / 1000000 - self.t_min
        iden_start_time = data.data["iden_start_time"]
        self.iden_start_time = self.resample_data(t,iden_start_time)

    def parse_attitude_data(self, data):
        # dict_keys(['timestamp', 'rollspeed', 'pitchspeed', 'yawspeed', 'q[0]', 'q[1]', 'q[2]', 'q[3]'])
        t = data.data['timestamp'] / 1000000 - self.t_min
        pitchspeed = data.data['pitchspeed']
        self.pitchrate_flted = self.resample_data(t, pitchspeed)

        q0_arr = data.data['q[0]']
        q1_arr = data.data['q[1]']
        q2_arr = data.data['q[2]']
        q3_arr = data.data['q[3]']

        self.q0, self.q1, self.q2, self.q3 = self.resample_data(t, q0_arr, q1_arr, q2_arr, q3_arr)

        roll_arr = []
        pitch_arr = []
        yaw_arr = []
        for i in range(len(q0_arr)):
            q0 = q0_arr[i]
            q1 = q1_arr[i]
            q2 = q2_arr[i]
            q3 = q3_arr[i]

            quat = quaternion.Quaternion([q0, q1, q2, q3])
            euler = quat.euler
            # print(euler)
            roll_arr.append(euler[0])
            pitch_arr.append(euler[1])
            yaw_arr.append(euler[2])

        self.roll, self.pitch, self.yaw = self.resample_data(t, roll_arr, pitch_arr, yaw_arr)

    def parse_attitude_data_sp(self, data):
        # dict_keys(['timestamp', 'rollspeed', 'pitchspeed', 'yawspeed', 'q[0]', 'q[1]', 'q[2]', 'q[3]'])
        t = data.data['timestamp'] / 1000000 - self.t_min
        print(f"sample rate {len(t) / (t[-1] - t[0])}")
        
        self.roll_sp, self.pitch_sp, self.yaw_sp = self.resample_data(t, data.data["roll_body"], data.data["pitch_body"], data.data["yaw_body"])

    def parse_local_position_data(self, data: ULog.Data):
        # ['timestamp', 'ref_timestamp', 'ref_lat', 'ref_lon', 'surface_bottom_timestamp', 'x', 'y', 'z', 'delta_xy[0]',
        #  'delta_xy[1]', 'delta_z', 'vx', 'vy', 'vz', 'z_deriv', 'delta_vxy[0]', 'delta_vxy[1]', 'delta_vz', 'ax', 'ay',
        #  'az', 'yaw', 'ref_alt', 'dist_bottom', 'dist_bottom_rate', 'eph', 'epv', 'evh', 'evv', 'estimator_type',
        #  'xy_valid', 'z_valid', 'v_xy_valid', 'v_z_valid', 'xy_reset_counter', 'z_reset_counter', 'vxy_reset_counter',
        #  'vz_reset_counter', 'xy_global', 'z_global', 'dist_bottom_valid']
        t = data.data['timestamp'] / 1000000 - self.t_min
        vx = data.data['vx']
        vy = data.data['vy']
        vz = data.data['vz']
        alt = data.data['z']
        self.climb_rate, self.alt = self.resample_data(t, vz, - alt)
        # Setup inteplote for q and analyze vx

        body_vx = []
        body_vy = []
        body_vz = []

        self.vx, self.vy, self.vz = self.resample_data(t, vx, vy, vz)

        print("Try to transform vx vy vz into body frame")
        for i in range(len(self.t_seq)):
            q0, q1, q2, q3 = self.q0[i], self.q1[i], self.q2[i], self.q3[i]
            quat = quaternion.Quaternion([q0, q1, q2, q3])
            try:
                quat.normalize()
            except Exception as e:
                print(e)
                body_vx.append(0)
                body_vy.append(0)
                body_vz.append(0)
                continue
            local_vel = Vector3([self.vx[i], self.vy[i], self.vz[i]])
            if math.isnan(q0) or math.isnan(quat.q.data[0]) or math.isnan(self.vx[i]):
                body_vx.append(0)
                body_vy.append(0)
                body_vz.append(0)
            else:
                body_vel = quat.inversed.transform(local_vel)
                body_vx.append(body_vel.x)
                body_vy.append(body_vel.y)
                body_vz.append(body_vel.z)

        self.body_vx = body_vx
        self.body_vy = body_vy
        self.body_vz = body_vz


