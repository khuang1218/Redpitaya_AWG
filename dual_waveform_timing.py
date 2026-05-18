import time

import numpy as np
import redpitaya_scpi as scpi


IP = '169.254.77.151'  # Change this to your Red Pitaya hostname or IP.
AMPLITUDE = 0.8
FREQUENCY = 10_000
SAMPLES = 16_384
# Toggle this only when OUT1/OUT2 are physically looped back to IN1/IN2.
MEASURE_HARDWARE_SIDE = True
ACQ_TIMEOUT_SECONDS = 10
TIMING_UNITS = "ms"


def waveform_csv(values: np.ndarray) -> str:
    """Format an arbitrary waveform for Red Pitaya SCPI."""
    return ",".join(f"{value:.5f}" for value in values)


def first_crossing_time(signal: np.ndarray, sample_rate: float, threshold: float = 0.1) -> float:
    """Return the first positive threshold crossing time in seconds."""
    crossings = np.flatnonzero((signal[:-1] < threshold) & (signal[1:] >= threshold))
    if len(crossings) == 0:
        raise RuntimeError("No rising threshold crossing found. Check wiring/threshold.")
    return crossings[0] / sample_rate


def wait_complete() -> None:
    """Wait until the Red Pitaya SCPI server reports previous commands complete."""
    rp.txrx_txt("*OPC?")


def wait_for_acquisition(timeout_s: float = ACQ_TIMEOUT_SECONDS) -> None:
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


def send_dataset_and_trigger(chan: int, waveform: np.ndarray) -> float:
    """Upload one waveform, trigger it, and return elapsed wall-clock time."""
    start = time.perf_counter()

    rp.tx_txt(f"SOUR{chan}:TRAC:DATA:DATA " + waveform_csv(waveform))
    wait_complete()

    rp.tx_txt(f"SOUR{chan}:TRig:INT")
    wait_complete()

    return time.perf_counter() - start


rp = scpi.scpi(IP, timeout=5)

# Two different arbitrary waveforms, both normalized to about +/-1.
x = np.linspace(0, 2 * np.pi, SAMPLES, endpoint=False)
waveform_1 = np.sin(x)
waveform_2 = np.sign(np.sin(x))

rp.tx_txt("GEN:RST")

# Load the two waveforms on output channel 1 and output channel 2.
rp.tx_txt("SOUR1:FUNC ARBITRARY")
rp.tx_txt("SOUR1:TRAC:DATA:DATA " + waveform_csv(waveform_1))
rp.tx_txt(f"SOUR1:FREQ:FIX {FREQUENCY}")
rp.tx_txt(f"SOUR1:VOLT {AMPLITUDE}")
rp.tx_txt("SOUR1:BURS:STAT BURST")
rp.tx_txt("SOUR1:BURS:NCYC 1")
rp.tx_txt("SOUR1:BURS:NOR 1")
rp.tx_txt("SOUR1:TRig:SOUR INT")

rp.tx_txt("SOUR2:FUNC ARBITRARY")
rp.tx_txt("SOUR2:TRAC:DATA:DATA " + waveform_csv(waveform_2))
rp.tx_txt(f"SOUR2:FREQ:FIX {FREQUENCY}")
rp.tx_txt(f"SOUR2:VOLT {AMPLITUDE}")
rp.tx_txt("SOUR2:BURS:STAT BURST")
rp.tx_txt("SOUR2:BURS:NCYC 1")
rp.tx_txt("SOUR2:BURS:NOR 1")
rp.tx_txt("SOUR2:TRig:SOUR INT")

rp.tx_txt("OUTPUT:STATE ON")
wait_complete()

# Training-loop style timing:
# This includes Python time, TCP/network delay, SCPI parsing, waveform upload,
# and waiting for Red Pitaya command completion.
dataset_1_elapsed = send_dataset_and_trigger(1, waveform_1)
dataset_2_elapsed = send_dataset_and_trigger(2, waveform_2)

print(f"Dataset 1 upload + trigger elapsed: {dataset_1_elapsed * 1e3:.3f} ms")
print(f"Dataset 2 upload + trigger elapsed: {dataset_2_elapsed * 1e3:.3f} ms")
print(f"Time from dataset 1 start to dataset 2 completion: {(dataset_1_elapsed + dataset_2_elapsed) * 1e3:.3f} ms")

if MEASURE_HARDWARE_SIDE:
    # Hardware-side timing:
    # Connect OUT1 -> IN1 and OUT2 -> IN2, then capture both input channels.
    # This estimates the actual time between the two generated signals.s
    decimation = 1
    sample_rate = 125_000_000 / decimation

    rp.tx_txt("ACQ:RST")
    rp.tx_txt(f"ACQ:DEC:Factor {decimation}")
    rp.tx_txt("ACQ:DATA:Units VOLTS")
    rp.tx_txt("ACQ:DATA:FORMAT ASCII")
    rp.tx_txt("ACQ:START")
    time.sleep(0.01)  # Let the acquisition buffer collect fresh samples.

    # Capture immediately, then trigger both outputs while acquisition is running.
    rp.tx_txt("ACQ:TRig NOW")
    rp.tx_txt("SOUR1:TRig:INT")
    rp.tx_txt("SOUR2:TRig:INT")
    wait_for_acquisition()

    ch1 = rp.acq_data(1)
    ch2 = rp.acq_data(2)

    start_1 = first_crossing_time(ch1, sample_rate)
    start_2 = first_crossing_time(ch2, sample_rate)
    print(f"Measured signal start difference: {(start_2 - start_1) * 1e9:.1f} ns")

rp.close()
