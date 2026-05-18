import time

import numpy as np
import redpitaya_scpi as scpi


IP = "169.254.77.151"
AMPLITUDE = 0.8
FREQUENCY = 10_000
WAVEFORM_SAMPLES = 16_384
ACQ_READ_SAMPLES = 8_192
ACQ_TIMEOUT_SECONDS = 10
DECIMATION = 1
SAMPLE_RATE = 125_000_000 / DECIMATION


def waveform_csv(values: np.ndarray) -> str:
    """Format an arbitrary waveform for Red Pitaya SCPI."""
    return ",".join(f"{value:.5f}" for value in values)


def sharp_pulse(
    samples: int = WAVEFORM_SAMPLES,
    low: float = -1.0,
    high: float = 1.0,
    start_fraction: float = 0.1,
    width_fraction: float = 0.1,
) -> np.ndarray:
    """Create a waveform with a clean rising edge."""
    waveform = np.full(samples, low)
    start = int(samples * start_fraction)
    stop = start + int(samples * width_fraction)
    waveform[start:stop] = high
    return waveform


def wait_complete(rp: scpi.scpi) -> None:
    """Wait until the Red Pitaya SCPI server reports previous commands complete."""
    rp.txrx_txt("*OPC?")


def wait_for_acquisition(rp: scpi.scpi, timeout_s: float = ACQ_TIMEOUT_SECONDS) -> None:
    """Wait until acquisition has triggered and the capture buffer is full."""
    deadline = time.perf_counter() + timeout_s

    while time.perf_counter() < deadline:
        if rp.txrx_txt("ACQ:TRig:STAT?") == "TD":
            break
        time.sleep(0.001)
    else:
        raise TimeoutError("Acquisition did not trigger before timeout.")

    while time.perf_counter() < deadline:
        if rp.txrx_txt("ACQ:TRig:FILL?") == "1":
            return
        time.sleep(0.001)

    raise TimeoutError("Acquisition buffer did not fill before timeout.")


def read_acquisition_channel(
    rp: scpi.scpi,
    chan: int,
    num_samples: int = ACQ_READ_SAMPLES,
) -> np.ndarray:
    """Read acquisition data immediately after the data query."""
    rp.tx_txt(f"ACQ:SOUR{chan}:DATA:STArt:N? 0,{num_samples}")
    raw = rp.rx_txt()
    values = raw.strip("{}\n\r").replace("  ", "").split(",")
    return np.array(values, dtype=np.float64)


def first_rising_edge_time(signal: np.ndarray, sample_rate: float) -> float:
    """Find the first strong rising edge and return its time in seconds."""
    threshold = (float(np.min(signal)) + float(np.max(signal))) / 2
    crossings = np.flatnonzero((signal[:-1] < threshold) & (signal[1:] >= threshold))
    if len(crossings) == 0:
        raise RuntimeError(
            "No rising edge found. "
            f"threshold={threshold:.4f}, min={np.min(signal):.4f}, max={np.max(signal):.4f}"
        )
    return crossings[0] / sample_rate


def configure_generator(rp: scpi.scpi, chan: int, waveform: np.ndarray) -> None:
    """Load one pulse waveform into one generator channel."""
    rp.tx_txt(f"SOUR{chan}:FUNC ARBITRARY")
    rp.tx_txt(f"SOUR{chan}:TRAC:DATA:DATA " + waveform_csv(waveform))
    rp.tx_txt(f"SOUR{chan}:FREQ:FIX {FREQUENCY}")
    rp.tx_txt(f"SOUR{chan}:VOLT {AMPLITUDE}")
    rp.tx_txt(f"SOUR{chan}:BURS:STAT BURST")
    rp.tx_txt(f"SOUR{chan}:BURS:NCYC 1")
    rp.tx_txt(f"SOUR{chan}:BURS:NOR 1")
    rp.tx_txt(f"SOUR{chan}:TRig:SOUR INT")


def configure_acquisition(rp: scpi.scpi) -> None:
    """Prepare acquisition before sending the two consecutive SCPI triggers."""
    rp.tx_txt("ACQ:RST")
    rp.tx_txt(f"ACQ:DEC:Factor {DECIMATION}")
    rp.tx_txt("ACQ:DATA:Units VOLTS")
    rp.tx_txt("ACQ:DATA:FORMAT ASCII")
    rp.tx_txt("ACQ:START")
    time.sleep(0.01)
    rp.tx_txt("ACQ:TRig NOW")


rp = scpi.scpi(IP, timeout=10)

try:
    # Use identical sharp pulses so both channels have the same crossing shape.
    pulse_1 = sharp_pulse()
    pulse_2 = sharp_pulse()

    rp.tx_txt("GEN:RST")
    configure_generator(rp, 1, pulse_1)
    configure_generator(rp, 2, pulse_2)
    rp.tx_txt("OUTPUT:STATE ON")
    wait_complete(rp)

    configure_acquisition(rp)

    host_t1 = time.perf_counter()
    rp.tx_txt("SOUR1:TRig:INT")
    host_t2 = time.perf_counter()
    rp.tx_txt("SOUR2:TRig:INT")
    host_t3 = time.perf_counter()

    wait_for_acquisition(rp)

    ch1 = read_acquisition_channel(rp, 1)
    ch2 = read_acquisition_channel(rp, 2)

    print(f"CH1 captured min/max: {np.min(ch1):.4f} V / {np.max(ch1):.4f} V")
    print(f"CH2 captured min/max: {np.min(ch2):.4f} V / {np.max(ch2):.4f} V")

    hardware_start_1 = first_rising_edge_time(ch1, SAMPLE_RATE)
    hardware_start_2 = first_rising_edge_time(ch2, SAMPLE_RATE)

    host_trigger_gap = host_t3 - host_t2
    first_trigger_send_time = host_t2 - host_t1
    hardware_delay = hardware_start_2 - hardware_start_1

    print(f"Host time to send first trigger command: {first_trigger_send_time * 1e6:.1f} us")
    print(f"Host gap between consecutive trigger sends: {host_trigger_gap * 1e6:.1f} us")
    print(f"Hardware delay between captured rising edges: {hardware_delay * 1e9:.1f} ns")

finally:
    rp.close()
