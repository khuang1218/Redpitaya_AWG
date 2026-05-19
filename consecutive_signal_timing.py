import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import redpitaya_scpi as scpi


IP = "169.254.77.151"
AMPLITUDE = 0.8
WAVEFORM_SAMPLES = 16_384
ACQ_READ_SAMPLES = 16_384
ACQ_PRE_POST_SAMPLES = 8_191
ACQ_TIMEOUT_SECONDS = 10
DECIMATION = 1
DAC_SAMPLE_RATE = 125_000_000
ADC_SAMPLE_RATE = 125_000_000
SAMPLE_RATE = ADC_SAMPLE_RATE / DECIMATION
FREQUENCY = DAC_SAMPLE_RATE / WAVEFORM_SAMPLES
PULSE_DURATION = 1 / FREQUENCY
POST_PULSE_SETTLE_SECONDS = 0.005
TRIGGER_LEVEL = 0.3
CAPTURE_DIR = Path("captures")


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
    num_samples: int = ACQ_PRE_POST_SAMPLES,
) -> np.ndarray:
    """Read trigger-aligned acquisition data immediately after the data query."""
    rp.tx_txt(f"ACQ:SOUR{chan}:DATA:TRig? {num_samples},PRE_POST_TRIG")
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


def analyse_capture(name: str, signal: np.ndarray) -> dict[str, float]:
    """Print simple shape diagnostics for one captured signal."""
    low = float(np.percentile(signal, 5))
    high = float(np.percentile(signal, 95))
    threshold = (low + high) / 2
    above = signal >= threshold
    rising = np.flatnonzero((~above[:-1]) & above[1:])
    falling = np.flatnonzero(above[:-1] & (~above[1:]))
    duration_us = len(signal) / SAMPLE_RATE * 1e6

    print(f"{name} analysis:")
    print(f"  min/max: {np.min(signal):.4f} V / {np.max(signal):.4f} V")
    print(f"  p5/p95: {low:.4f} V / {high:.4f} V")
    print(f"  threshold: {threshold:.4f} V")
    print(f"  capture duration: {duration_us:.3f} us")

    result = {
        "low_v": low,
        "high_v": high,
        "threshold_v": threshold,
        "capture_duration_us": duration_us,
    }

    if len(rising) > 0:
        rising_us = rising[0] / SAMPLE_RATE * 1e6
        result["first_rising_edge_us"] = rising_us
        print(f"  first rising edge: {rising_us:.3f} us")
    else:
        print("  first rising edge: not found")

    if len(rising) > 0 and len(falling) > 0:
        later_falls = falling[falling > rising[0]]
        if len(later_falls) > 0:
            width_us = (later_falls[0] - rising[0]) / SAMPLE_RATE * 1e6
            result["first_pulse_width_us"] = width_us
            print(f"  first pulse width: {width_us:.3f} us")
        else:
            print("  first pulse width: high level continues to end of capture")

    return result


def align_capture_to_original(original: np.ndarray, captured: np.ndarray) -> tuple[np.ndarray, int]:
    """Align captured samples to original samples using cross-correlation."""
    original_scaled = original * AMPLITUDE
    if len(captured) < len(original_scaled):
        original_scaled = original_scaled[:len(captured)]

    original_zero_mean = original_scaled - np.mean(original_scaled)
    captured_zero_mean = captured - np.mean(captured)
    correlation = np.correlate(captured_zero_mean, original_zero_mean, mode="valid")
    start = int(np.argmax(correlation))
    aligned = captured[start:start + len(original)]
    return aligned, start


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

    CAPTURE_DIR.mkdir(exist_ok=True)

    waveform_time_us = np.arange(WAVEFORM_SAMPLES) / (WAVEFORM_SAMPLES * FREQUENCY) * 1e6
    capture_time_us = (np.arange(len(ch1)) - ACQ_PRE_POST_SAMPLES) / SAMPLE_RATE * 1e6

    np.savetxt(
        CAPTURE_DIR / "original_waveforms.csv",
        np.column_stack((waveform_time_us, pulse_1 * AMPLITUDE, pulse_2 * AMPLITUDE)),
        delimiter=",",
        header="time_us,pulse_1_v,pulse_2_v",
        comments="",
    )
    np.savetxt(
        CAPTURE_DIR / "captured_waveforms.csv",
        np.column_stack((capture_time_us, ch1, ch2)),
        delimiter=",",
        header="time_us,capture_1_v,capture_2_v",
        comments="",
    )

    aligned_ch1, align_start_1 = align_capture_to_original(pulse_1, ch1)
    aligned_ch2, align_start_2 = align_capture_to_original(pulse_2, ch2)
    aligned_len = min(len(aligned_ch1), len(aligned_ch2), WAVEFORM_SAMPLES)
    aligned_ch1 = aligned_ch1[:aligned_len]
    aligned_ch2 = aligned_ch2[:aligned_len]
    aligned_time_us = np.arange(aligned_len) / DAC_SAMPLE_RATE * 1e6

    np.savetxt(
        CAPTURE_DIR / "aligned_training_pairs.csv",
        np.column_stack((
            aligned_time_us,
            pulse_1[:aligned_len] * AMPLITUDE,
            aligned_ch1,
            pulse_2[:aligned_len] * AMPLITUDE,
            aligned_ch2,
        )),
        delimiter=",",
        header="time_us,input_1_v,measured_output_1_v,input_2_v,measured_output_2_v",
        comments="",
    )

    print(f"CH1 captured min/max: {np.min(ch1):.4f} V / {np.max(ch1):.4f} V")
    print(f"Second capture min/max: {np.min(ch2):.4f} V / {np.max(ch2):.4f} V")
    print(f"Configured pulse duration: {PULSE_DURATION * 1e6:.1f} us")
    print(f"Cycle 1 upload + trigger + pulse + acquire/read: {cycle_1_elapsed * 1e3:.3f} ms")
    print(f"Cycle 2 upload + trigger + pulse + acquire/read: {cycle_2_elapsed * 1e3:.3f} ms")
    print(f"Total time for two consecutive pulse cycles: {(cycle_1_elapsed + cycle_2_elapsed) * 1e3:.3f} ms")
    analyse_capture("Capture 1", ch1)
    analyse_capture("Capture 2", ch2)
    print(f"AWG point interval: {1 / DAC_SAMPLE_RATE * 1e9:.3f} ns")
    print(f"ADC sample interval: {1 / SAMPLE_RATE * 1e9:.3f} ns")
    print(f"Alignment start sample capture 1: {align_start_1}")
    print(f"Alignment start sample capture 2: {align_start_2}")
    print(f"Saved original waveforms to {CAPTURE_DIR / 'original_waveforms.csv'}")
    print(f"Saved captured waveforms to {CAPTURE_DIR / 'captured_waveforms.csv'}")
    print(f"Saved aligned training pairs to {CAPTURE_DIR / 'aligned_training_pairs.csv'}")

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

    axes[1].plot(aligned_time_us, aligned_ch1, "--", label="Aligned cycle 1")
    axes[1].plot(aligned_time_us, aligned_ch2, "--", label="Aligned cycle 2")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(CAPTURE_DIR / "waveform_capture_comparison.png", dpi=150)
    plt.show()

finally:
    rp.close()
