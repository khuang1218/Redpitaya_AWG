import time

import matplotlib.pyplot as plt
import numpy as np
import redpitaya_scpi as scpi


IP = "169.254.77.151"
AMPLITUDE = 0.8
FREQUENCY = 7_000
WAVEFORM_SAMPLES = 16_384
ACQ_READ_SAMPLES = 16_384
ACQ_TIMEOUT_SECONDS = 10
DECIMATION = 1
SAMPLE_RATE = 125_000_000 / DECIMATION
PULSE_DURATION = 1 / FREQUENCY
POST_PULSE_SETTLE_SECONDS = 0.005
TRIGGER_LEVEL = 0.1


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


def configure_acquisition(rp: scpi.scpi, chan: int) -> None:
    """Arm acquisition to trigger from the input channel rising edge."""
    rp.tx_txt("ACQ:RST")
    rp.tx_txt(f"ACQ:DEC:Factor {DECIMATION}")
    rp.tx_txt("ACQ:DATA:Units VOLTS")
    rp.tx_txt("ACQ:DATA:FORMAT ASCII")
    rp.tx_txt(f"ACQ:TRig:LEV {TRIGGER_LEVEL}")
    rp.tx_txt("ACQ:TRig:DLY 0")
    rp.tx_txt("ACQ:START")
    rp.tx_txt(f"ACQ:TRig CH{chan}_PE")
    time.sleep(0.01)
    if rp.txrx_txt("ACQ:TRig:STAT?") == "TD":
        raise RuntimeError(
            "Acquisition triggered before the output pulse was sent. "
            "Raise TRIGGER_LEVEL or check input noise/offset."
        )


def run_pulse_cycle(rp: scpi.scpi, chan: int, waveform: np.ndarray) -> tuple[float, np.ndarray]:
    """Upload one pulse, trigger it, let it finish, and read the captured output."""
    configure_acquisition(rp, chan)

    start = time.perf_counter()

    rp.tx_txt(f"SOUR{chan}:TRAC:DATA:DATA " + waveform_csv(waveform))
    wait_complete(rp)

    rp.tx_txt(f"SOUR{chan}:TRig:INT")

    # tx_txt() only sends the trigger command. It does not wait for the analog
    # burst to finish, so we wait for the configured burst duration here.
    time.sleep(PULSE_DURATION + POST_PULSE_SETTLE_SECONDS)

    wait_for_acquisition(rp)
    captured = read_acquisition_channel(rp, chan)

    elapsed = time.perf_counter() - start
    return elapsed, captured


rp = scpi.scpi(IP, timeout=15)

try:
    # Two sharp-edged pulse datasets, both sent consecutively through OUT1.
    pulse_1 = sharp_pulse(width_fraction=0.1)
    pulse_2 = sharp_pulse(width_fraction=0.2)

    rp.tx_txt("GEN:RST")
    configure_generator(rp, 1, pulse_1)
    rp.tx_txt("OUTPUT:STATE ON")
    wait_complete(rp)

    cycle_1_elapsed, ch1 = run_pulse_cycle(rp, 1, pulse_1)
    cycle_2_elapsed, ch2 = run_pulse_cycle(rp, 1, pulse_2)

    print(f"CH1 captured min/max: {np.min(ch1):.4f} V / {np.max(ch1):.4f} V")
    print(f"Second capture min/max: {np.min(ch2):.4f} V / {np.max(ch2):.4f} V")
    print(f"Configured pulse duration: {PULSE_DURATION * 1e6:.1f} us")
    print(f"Cycle 1 upload + trigger + pulse + acquire/read: {cycle_1_elapsed * 1e3:.3f} ms")
    print(f"Cycle 2 upload + trigger + pulse + acquire/read: {cycle_2_elapsed * 1e3:.3f} ms")
    print(f"Total time for two consecutive pulse cycles: {(cycle_1_elapsed + cycle_2_elapsed) * 1e3:.3f} ms")

    waveform_time_us = np.arange(WAVEFORM_SAMPLES) / (WAVEFORM_SAMPLES * FREQUENCY) * 1e6
    capture_time_us = np.arange(len(ch1)) / SAMPLE_RATE * 1e6

    _, axes = plt.subplots(2, 1, sharex=False)

    axes[0].plot(waveform_time_us, pulse_1 * AMPLITUDE, label="Original pulse 1")
    axes[0].plot(waveform_time_us, pulse_2 * AMPLITUDE, label="Original pulse 2")
    axes[0].set_title("Original AWG Waveforms")
    axes[0].set_xlabel("Time (us)")
    axes[0].set_ylabel("Voltage (V)")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(capture_time_us, ch1, label="Cycle 1 capture")
    axes[1].plot(capture_time_us, ch2, label="Cycle 2 capture")
    axes[1].set_title("Captured Output Signals")
    axes[1].set_xlabel("Time (us)")
    axes[1].set_ylabel("Voltage (V)")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.show()

finally:
    rp.close()
