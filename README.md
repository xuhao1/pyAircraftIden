# pyAircraftIden
This project stand for system idenification for aircraft or other system.

## Single-input Multi Output Frequency
For frequency idenification

```python
simo_iden = FreqIdenSIMO(time_seq_source, omega_min, omega_max, a_seq, data1_seq, data2_seq ..., win_num=None)
simo_iden.plt_bode_plot(0)
```
time_seq_source is the time series, a_seq is your input data, data1_seq, data2_seq ..., is your output data
omega_min, omega_max is your min and max angular frequency.

Also, please take a look at 

## License
This project uses MIT license, if you feel happy with my project, you can 请我吃烤肉.