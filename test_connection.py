import redpitaya_scpi as scpi

IP = "169.254.202.253"       # 'rp-f066c8.local'
rp = scpi.scpi(IP, timeout=5)
rp.tx_txt("GEN:RST")