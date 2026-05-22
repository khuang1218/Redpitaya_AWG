#!/usr/bin/env python3

import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import redpitaya_scpi as scpi

IP = "169.254.202.253"       # 'rp-f066c8.local'
rp = scpi.scpi(IP)

wave_form = 'sine'
freq = 1000000
ampl = 1
sample_rate = 125_000_000
buffer_size = 16_384
burst_cycles = 3

# Reset Generation and Acquisition
rp.tx_txt('GEN:RST')
rp.tx_txt('ACQ:RST')

##### Generation #####
rp.tx_txt('SOUR1:FUNC ' + str(wave_form).upper())
rp.tx_txt('SOUR1:FREQ:FIX ' + str(freq))
rp.tx_txt('SOUR1:VOLT ' + str(ampl))

rp.tx_txt('SOUR1:BURS:STAT BURST')        # Mode set to BURST
rp.tx_txt('SOUR1:BURS:NCYC ' + str(burst_cycles))            # 3 periods in each burst

##### Acqusition #####
rp.tx_txt('ACQ:DEC 1')
rp.tx_txt('ACQ:TRig:LEV 0.02')
rp.tx_txt('ACQ:TRig:DLY 8192')

rp.tx_txt('ACQ:START')
time.sleep(1)
rp.tx_txt('ACQ:TRig CH1_PE')
rp.tx_txt('OUTPUT1:STATE ON')
time.sleep(1)

rp.tx_txt('SOUR1:TRig:INT')

# Wait for trigger
while 1:
    rp.tx_txt('ACQ:TRig:STAT?')           # Get Trigger Status
    if rp.rx_txt() == 'TD':               # Triggerd?
        break

## ! OS 2.00 or higher only ! ##
while 1:
    rp.tx_txt('ACQ:TRig:FILL?')
    if rp.rx_txt() == '1':
        break

# Read data and plot
rp.tx_txt('ACQ:SOUR1:DATA?')              # Read full buffer (source 1)
data_string = rp.rx_txt()                 # data into a string

# Remove brackets and empty spaces + string => float
data_string = data_string.strip('{}\n\r').replace("  ", "").split(',')
data = np.array(list(map(float, data_string)))        # transform data into float

time_us = np.arange(len(data)) / sample_rate * 1e6
expected = np.zeros_like(data)
burst_samples = int(round(burst_cycles * sample_rate / freq))
expected[:burst_samples] = ampl * np.sin(2 * np.pi * freq * np.arange(burst_samples) / sample_rate)

expected_zero_mean = expected - np.mean(expected)
data_zero_mean = data - np.mean(data)
corr = np.correlate(data_zero_mean, expected_zero_mean, mode='same')
lag_samples = int(np.argmax(corr) - len(data) // 2)
lag_us = lag_samples / sample_rate * 1e6
aligned_expected = np.roll(expected, lag_samples)

print(f"Estimated input/output lag: {lag_samples} samples ({lag_us:.3f} us)")

plt.figure()
plt.plot(time_us, data, label='Captured output')
plt.plot(time_us, expected, label='Expected input burst')
plt.xlabel('Time (us)')
plt.ylabel('Voltage (V)')
plt.title('Raw AWG-Synced Capture')
plt.grid(True)
plt.legend()

plt.figure()
plt.plot(time_us, data, label='Captured output')
plt.plot(time_us, aligned_expected, '--', label='Expected input aligned by correlation')
plt.xlabel('Time (us)')
plt.ylabel('Voltage (V)')
plt.title('Input vs Output After Estimated Alignment')
plt.grid(True)
plt.legend()

plt.show()
