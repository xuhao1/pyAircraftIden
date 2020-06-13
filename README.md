# pyAircraftIden
This project stand for system identification for aircraft or other system.

## Single-input Multi Output Frequency
For frequency identification

```python
simo_iden = FreqIdenSIMO(time_seq_source, omega_min, omega_max, a_seq, data1_seq, data2_seq ..., win_num=None)
simo_iden.plt_bode_plot(0)
```
__time_seq_source__ is the time series, __a_seq__ is your input data, __data1_seq, data2_seq ...__ are your output data
__omega_min, omega_max__ are your min and max angular frequency.

![frequency response example](plots/ele_q_cessna.png)

Also, please take a look at [SIMO example](./examples/FreqIdenExample.ipynb)

## Transfer Function Identification
For transfer Function identification, the just specify your transfer function model and the frequnecy domain to fit.


```python
# Set transfer function model
num = d
den = a*s*s + b*s + c

tfpm = TransferFunctionParamModel(num, den, tau)

#Use freq, H, gamma2 output from SIMO identification
# Nw gives the sample point for optimization
# Iter times is the trial times
# Reg is regularization parameter
fitter = TransferFunctionFit(freq, H, gamma2, tfpm, nw=20, iter_times=50, reg = 0.1)

```
Please check take a look at [SIMO example](./examples/FreqIdenExample.ipynb) also.

![transfer function identification](plots/ele_q_transferfunc.PNG)

Also [Tail-sitter example](./examples/TSCruisingFreqResQFitting.ipynb) gives a more complex example on transfer function fitting with real world experiment data of a Tail-sitter VTOL UAV, the data is collect via Pixhawk.

## State-space Idenification
This project uses frequency approach for state-space identification.
Check [State Space example](./examples/TSCruisingSSM.ipynb) for details

## PX4 ULog data parsing
PX4 records ULog data, read PX4 data is easy with this project

```python
fpath = "data/foam-tail-sitter/log_32_2018-4-10-15-53-08.ulg"
px4_case = PX4AircraftCase(fpath)
needed_data = ['ele', 'q', 'thr', 'body_vx', "iden_start_time"]
t_arr, data_list = px4_case.get_data_time_range_list(needed_data)
```

Check  [Tail-sitter example](./examples/TSCruisingFreqResQFitting.ipynb) for details.

Here shows the Ulog data read from ulog
![px4 data](plots/px4_data.png)


## License
This project uses MIT license