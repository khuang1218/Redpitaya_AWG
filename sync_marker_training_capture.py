import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import redpitaya_scpi as scpi


IP = "169.254.77.151"
CAPTURE_DIR = Path("captures_sync")

DAC_SAMPLE_RATE = 125_000_000
ADC_SAMPLE_RATE = 125_000_000
WAVEFORM_SAMPLES = 16_384
DECIMATION = 1
SAMPLE_RATE = ADC_SAMPLE_RATE / DECIMATION
FREQUENCY = DAC_SAMPLE_RATE / WAVEFORM_SAMPLES

AMPLITUDE = 0.7
TRIGGER_LEVEL = 0.2
SYNC_SAMPLES = 1_024
SYNC_TRIGGER_OFFSET = 16
GUARD_SAMPLES = 512
TRAINING_SAMPLES = WAVEFORM_SAMPLES - SYNC_SAMPLES - GUARD_SAMPLES
ACQ_TIMEOUT_SECONDS = 10
ACQ_TRIGGER_DELAY_SAMPLES = 8_192


def waveform_csv(values: np.ndarray) -> str:
    return ",".join(f"{value:.5f}" for value in values)


def make_sync_marker(samples: int = SYNC_SAMPLES) -> np.ndarray:
    """Make a repeatable bipolar pseudo-random sync marker."""
    rng = np.random.default_rng(12345)
    chips = rng.choice([-1.0, 1.0], size=samples)
    chips[:SYNC_TRIGGER_OFFSET] = -1.0
    chips[SYNC_TRIGGER_OFFSET:SYNC_TRIGGER_OFFSET + 16] = 1.0
    return chips


def make_training_signal(samples: int = TRAINING_SAMPLES) -> np.ndarray:
    """Example training signal. Replace this with your dataset."""
    rng = np.random.default_rng(67890)
    raw = rng.normal(0, 0.35, samples)
    smoothed = np.convolve(raw, np.ones(9) / 9, mode="same")
    return np.clip(smoothed, -1.0, 1.0)


def build_waveform() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sync = make_sync_marker()
    guard = np.zeros(GUARD_SAMPLES)
    training = make_training_signal()
    waveform = np.concatenate((sync, guard, training))
    return waveform, sync, guard, training


def wait_complete(rp: scpi.scpi) -> None:
    rp.txrx_txt("*OPC?")


def wait_for_acquisition(rp: scpi.scpi, timeout_s: float = ACQ_TIMEOUT_SECONDS) -> None:
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


def configure_generator(rp: scpi.scpi, waveform: np.ndarray) -> None:
    rp.tx_txt("GEN:RST")
    rp.tx_txt("SOUR1:FUNC ARBITRARY")
    rp.tx_txt("SOUR1:TRAC:DATA:DATA " + waveform_csv(waveform))
    rp.tx_txt(f"SOUR1:FREQ:FIX {FREQUENCY}")
    rp.tx_txt(f"SOUR1:VOLT {AMPLITUDE}")
    rp.tx_txt("SOUR1:BURS:STAT BURST")
    rp.tx_txt("SOUR1:BURS:NCYC 1")
    rp.tx_txt("SOUR1:BURS:NOR 1")
    rp.tx_txt(f"SOUR1:INITValue {-AMPLITUDE}")
    rp.tx_txt(f"SOUR1:BURS:LASTValue {-AMPLITUDE}")
    rp.tx_txt("SOUR1:TRig:SOUR INT")
    rp.tx_txt("OUTPUT:STATE ON")
    wait_complete(rp)


def configure_acquisition(rp: scpi.scpi) -> None:
    rp.tx_txt("ACQ:RST")
    rp.tx_txt(f"ACQ:DEC:Factor {DECIMATION}")
    rp.tx_txt("ACQ:DATA:Units VOLTS")
    rp.tx_txt("ACQ:DATA:FORMAT ASCII")
    rp.tx_txt(f"ACQ:TRig:LEV {TRIGGER_LEVEL}")
    rp.tx_txt(f"ACQ:TRig:DLY {ACQ_TRIGGER_DELAY_SAMPLES}")
    rp.tx_txt("ACQ:START")
    rp.tx_txt("ACQ:TRig CH1_PE")
    time.sleep(0.01)
    if rp.txrx_txt("ACQ:TRig:STAT?") == "TD":
        raise RuntimeError("Acquisition triggered before the waveform was sent.")


def read_post_trigger_capture(rp: scpi.scpi, samples: int = WAVEFORM_SAMPLES) -> np.ndarray:
    """Read samples after the trigger, rather than half pre-trigger."""
    rp.tx_txt(f"ACQ:SOUR1:DATA:TRig? {samples},POST_TRIG")
    raw = rp.rx_txt()
    values = raw.strip("{}\n\r").replace("  ", "").split(",")
    return np.array(values, dtype=np.float64)


def find_sync_start(captured: np.ndarray, sync: np.ndarray) -> int:
    sync_scaled = sync[SYNC_TRIGGER_OFFSET:] * AMPLITUDE
    sync_zero_mean = sync_scaled - np.mean(sync_scaled)
    captured_zero_mean = captured - np.mean(captured)
    corr = np.correlate(captured_zero_mean, sync_zero_mean, mode="valid")
    return int(np.argmax(corr))


waveform, sync, guard, training = build_waveform()
rp = scpi.scpi(IP, timeout=15)

try:
    configure_generator(rp, waveform)
    configure_acquisition(rp)

    cycle_start = time.perf_counter()
    rp.tx_txt("SOUR1:TRig:INT")
    time.sleep((WAVEFORM_SAMPLES / DAC_SAMPLE_RATE) + 0.005)
    wait_for_acquisition(rp)
    captured = read_post_trigger_capture(rp)
    elapsed = time.perf_counter() - cycle_start

finally:
    rp.close()

sync_start = find_sync_start(captured, sync)
training_start = sync_start + (SYNC_SAMPLES - SYNC_TRIGGER_OFFSET) + GUARD_SAMPLES
training_end = training_start + TRAINING_SAMPLES
aligned_output = captured[training_start:training_end]
aligned_input = training[:len(aligned_output)] * AMPLITUDE

time_us = np.arange(WAVEFORM_SAMPLES) / DAC_SAMPLE_RATE * 1e6
capture_time_us = np.arange(len(captured)) / SAMPLE_RATE * 1e6
training_time_us = np.arange(len(aligned_output)) / DAC_SAMPLE_RATE * 1e6

CAPTURE_DIR.mkdir(exist_ok=True)
np.savetxt(
    CAPTURE_DIR / "original_with_sync.csv",
    np.column_stack((time_us, waveform * AMPLITUDE)),
    delimiter=",",
    header="time_us,input_v",
    comments="",
)
np.savetxt(
    CAPTURE_DIR / "captured_post_trigger.csv",
    np.column_stack((capture_time_us, captured)),
    delimiter=",",
    header="time_us,captured_v",
    comments="",
)
np.savetxt(
    CAPTURE_DIR / "aligned_training_pairs.csv",
    np.column_stack((training_time_us, aligned_input, aligned_output)),
    delimiter=",",
    header="time_us,input_v,measured_output_v",
    comments="",
)

print(f"AWG frequency: {FREQUENCY:.3f} Hz")
print(f"Waveform duration: {WAVEFORM_SAMPLES / DAC_SAMPLE_RATE * 1e6:.3f} us")
print(f"Cycle trigger + capture/read elapsed: {elapsed * 1e3:.3f} ms")
print(f"Captured min/max: {np.min(captured):.4f} V / {np.max(captured):.4f} V")
print(f"Detected sync start in capture: sample {sync_start}")
print(f"Aligned training samples: {len(aligned_output)} / {TRAINING_SAMPLES}")
print(f"Saved captures in {CAPTURE_DIR}")

_, axes = plt.subplots(3, 1, sharex=False)
axes[0].plot(time_us, waveform * AMPLITUDE)
axes[0].axvspan(0, SYNC_SAMPLES / DAC_SAMPLE_RATE * 1e6, alpha=0.2, label="sync")
axes[0].axvspan(
    SYNC_SAMPLES / DAC_SAMPLE_RATE * 1e6,
    (SYNC_SAMPLES + GUARD_SAMPLES) / DAC_SAMPLE_RATE * 1e6,
    alpha=0.2,
    label="guard",
)
axes[0].set_title("Original Waveform: Sync + Guard + Training")
axes[0].set_ylabel("Voltage (V)")
axes[0].grid(True)
axes[0].legend()

axes[1].plot(capture_time_us, captured)
axes[1].axvline(sync_start / SAMPLE_RATE * 1e6, color="tab:red", label="detected sync")
axes[1].set_title("Captured Post-Trigger Signal")
axes[1].set_ylabel("Voltage (V)")
axes[1].grid(True)
axes[1].legend()

axes[2].plot(training_time_us, aligned_input, label="aligned input")
axes[2].plot(training_time_us, aligned_output, label="measured output")
axes[2].set_title("Aligned Training Pair")
axes[2].set_xlabel("Time (us)")
axes[2].set_ylabel("Voltage (V)")
axes[2].grid(True)
axes[2].legend()

plt.tight_layout()
plt.savefig(CAPTURE_DIR / "sync_alignment.png", dpi=150)
plt.show()
