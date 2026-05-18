#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import redpitaya_scpi as scpi

IP = 'rp-f0a235.local'

dec = 1
trig_lvl = 0.1
data_units = 'volts'
data_format = 'ascii'
acq_trig = 'CH1_PE'

rp = scpi.scpi(IP)

rp.tx_txt('ACQ:RST')

rp.tx_txt(f"ACQ:DEC:Factor {dec}")
rp.tx_txt(f"ACQ:DATA:Units {data_units.upper()}")
rp.tx_txt(f"ACQ:DATA:FORMAT {data_format.upper()}")

rp.tx_txt(f"ACQ:TRig:LEV {trig_lvl}")

rp.tx_txt('ACQ:START')
rp.tx_txt(f"ACQ:TRig {acq_trig}")

while 1:
    rp.tx_txt('ACQ:TRig:STAT?')
    if rp.rx_txt() == 'TD':
        break

## ! OS 2.00 or higher only ! ##
while 1:
    rp.tx_txt('ACQ:TRig:FILL?')
    if rp.rx_txt() == '1':
        break