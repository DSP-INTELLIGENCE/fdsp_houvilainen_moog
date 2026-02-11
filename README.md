# fdsp_houvilainen_moog

```python
# fdsp_huovilainen_moog.py
# -----------------------------------------------------------------------------
# Huovilainen Moog Ladder Filter (CSound5-derived) implemented in NumPy + Numba.
#
# Behavior (kept structurally identical to the provided C++ code):
# - 4 cascaded one-pole stages
# - Distributed tanh saturation inside each stage
# - 2x oversampling inside the per-sample loop
# - Feedback resonance (supports self-oscillation when resonance > 1)
# - 0.5 sample phase compensation at output
#
# DSP core rules honored:
# - DSP functions are pure functional: (state, inputs, params...) -> (y, new_state)
# - DSP state is a tuple of NumPy arrays + scalars only (no dict/list/class in DSP)
# - Numba-jitted DSP uses only arrays/scalars/tuples, no dynamic allocation in loop
# -----------------------------------------------------------------------------

from __future__ import annotations

import math
import os
import numpy as np

try:
    from numba import njit
    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False
    njit = None  # type: ignore


MOOG_PI = 3.1415926535897932384626433832795028841971


# =============================================================================
# 1) init
# =============================================================================

def huovilainen_moog_init(sample_rate: float) -> tuple:
    """
    Initialize state.

    Returns state tuple:
      (stage, stage_tanh, delay, sample_rate, thermal, tune, acr, res_quad, resonance, cutoff)
    """
    sr = float(sample_rate)
    thermal = 0.000025  # identical to C++ ctor

    stage = np.zeros((4,), dtype=np.float64)
    stage_tanh = np.zeros((3,), dtype=np.float64)
    delay = np.zeros((6,), dtype=np.float64)

    # Default params (match the C++ constructor)
    cutoff = 1000.0
    resonance = 0.10

    # Compute derived coefficients (same math as SetCutoff + SetResonance)
    state = (stage, stage_tanh, delay, sr, thermal, 0.0, 0.0, 0.0, float(resonance), float(cutoff))
    state = huovilainen_moog_update_state(state, cutoff=cutoff, resonance=resonance)
    return state


# =============================================================================
# 2) update_state
# =============================================================================

def _clamp_params(cutoff: float, resonance: float, sample_rate: float) -> tuple[float, float]:
    # DSP limits (stability + sane bounds)
    sr = float(sample_rate)
    c = float(cutoff)
    r = float(resonance)

    # Keep cutoff below Nyquist; use 0.49*sr to avoid edge singularities
    c_min = 1e-6
    c_max = 0.49 * sr
    if c < c_min:
        c = c_min
    elif c > c_max:
        c = c_max

    # Resonance: allow self-oscillation; cap to avoid runaway in extreme misuse
    r_min = 0.0
    r_max = 4.0
    if r < r_min:
        r = r_min
    elif r > r_max:
        r = r_max

    return c, r


def huovilainen_moog_update_state(state: tuple, cutoff: float, resonance: float) -> tuple:
    """
    Update coefficients exactly like SetCutoff + SetResonance in the C++ code.
    Pure functional: returns a new state tuple (arrays are reused/mutated is allowed externally,
    but we return the updated scalars in the tuple).
    """
    (stage, stage_tanh, delay, sr, thermal, tune, acr, res_quad, res_old, cut_old) = state

    c, r = _clamp_params(cutoff, resonance, sr)

    fc = c / sr
    f = fc * 0.5  # oversampled

    fc2 = fc * fc
    fc3 = fc2 * fc

    fcr = 1.8730 * fc3 + 0.4955 * fc2 - 0.6490 * fc + 0.9988
    acr = -3.9364 * fc2 + 1.8409 * fc + 0.9968

    tune = (1.0 - math.exp(-((2.0 * MOOG_PI) * f * fcr))) / thermal

    res_quad = 4.0 * r * acr

    return (stage, stage_tanh, delay, sr, thermal, float(tune), float(acr), float(res_quad), float(r), float(c))


# =============================================================================
# 3) tick (optional)
# =============================================================================

def huovilainen_moog_tick(state: tuple, x_sample: float) -> tuple[float, tuple]:
    """
    Per-sample processing in pure Python (non-JIT). Useful for correctness checks
    or very small buffers. Uses identical loop structure to C++.
    """
    (stage, stage_tanh, delay, sr, thermal, tune, acr, res_quad, resonance, cutoff) = state

    # force float64 math internally like C++ uses doubles
    xs = float(x_sample)
    for _j in range(2):  # 2x oversampling
        inp = xs - res_quad * delay[5]
        delay[0] = stage[0] = delay[0] + tune * (math.tanh(inp * thermal) - stage_tanh[0])

        for k in (1, 2, 3):
            inp = stage[k - 1]
            # stage_tanh[k-1] gets assigned the tanh(input * thermal)
            stage_tanh[k - 1] = math.tanh(inp * thermal)
            if k != 3:
                sub = stage_tanh[k]
            else:
                sub = math.tanh(delay[k] * thermal)
            stage[k] = delay[k] + tune * (stage_tanh[k - 1] - sub)
            delay[k] = stage[k]

        delay[5] = 0.5 * (stage[3] + delay[4])
        delay[4] = stage[3]

    y = float(delay[5])
    new_state = (stage, stage_tanh, delay, sr, thermal, tune, acr, res_quad, resonance, cutoff)
    return y, new_state


# =============================================================================
# 4) process_block (Numba JIT)
# =============================================================================

if _HAVE_NUMBA:
    @njit(cache=True, fastmath=True)
    def _huovilainen_moog_kernel(
        x: np.ndarray,
        y: np.ndarray,
        stage: np.ndarray,
        stage_tanh: np.ndarray,
        delay: np.ndarray,
        thermal: float,
        tune: float,
        res_quad: float,
    ) -> None:
        n = x.shape[0]
        for s in range(n):
            xs = float(x[s])
            # Oversample
            for _j in range(2):
                inp = xs - res_quad * delay[5]
                # delay[0] = stage[0] = delay[0] + tune*(tanh(inp*thermal) - stageTanh[0])
                t0 = math.tanh(inp * thermal)
                delay[0] = delay[0] + tune * (t0 - stage_tanh[0])
                stage[0] = delay[0]

                # for k=1..3
                for k in range(1, 4):
                    inp2 = stage[k - 1]
                    t_in = math.tanh(inp2 * thermal)
                    stage_tanh[k - 1] = t_in
                    if k != 3:
                        sub = stage_tanh[k]
                    else:
                        sub = math.tanh(delay[k] * thermal)
                    stage_k = delay[k] + tune * (t_in - sub)
                    stage[k] = stage_k
                    delay[k] = stage_k

                delay[5] = 0.5 * (stage[3] + delay[4])
                delay[4] = stage[3]

            y[s] = delay[5]


def huovilainen_moog_process_block(state: tuple, x: np.ndarray) -> tuple[np.ndarray, tuple]:
    """
    Block processing. Uses Numba JIT if available; falls back to tick loop otherwise.
    Pure functional: returns (y, new_state).
    """
    (stage, stage_tanh, delay, sr, thermal, tune, acr, res_quad, resonance, cutoff) = state

    x_arr = np.asarray(x)
    y = np.empty_like(x_arr)

    if _HAVE_NUMBA:
        # Ensure contiguous arrays and float64 state as required
        x_in = np.ascontiguousarray(x_arr)
        y_out = np.ascontiguousarray(y)

        _huovilainen_moog_kernel(
            x_in, y_out,
            stage, stage_tanh, delay,
            float(thermal), float(tune), float(res_quad)
        )

        # preserve dtype of input
        if y_out.dtype != x_arr.dtype:
            y = y_out.astype(x_arr.dtype, copy=False)
        else:
            y = y_out
    else:
        # Fallback: pure python tick per sample
        for i in range(x_arr.shape[0]):
            yi, state = huovilainen_moog_tick(state, float(x_arr[i]))
            y[i] = yi
        # state already updated in loop
        return y, state

    new_state = (stage, stage_tanh, delay, sr, thermal, tune, acr, res_quad, resonance, cutoff)
    return y, new_state


# =============================================================================
# 5) Optional wrapper class (no DSP inside)
# =============================================================================

class HuovilainenMoogWrapper:
    """
    Convenience wrapper (allowed outside DSP core).
    Holds state and exposes a simple API.
    """
    def __init__(self, sample_rate: float):
        self.state = huovilainen_moog_init(sample_rate)

    def set_params(self, cutoff: float, resonance: float) -> None:
        self.state = huovilainen_moog_update_state(self.state, cutoff=cutoff, resonance=resonance)

    def process(self, x: np.ndarray) -> np.ndarray:
        y, self.state = huovilainen_moog_process_block(self.state, x)
        return y


# =============================================================================
# 6) __main__ smoke test + plot + listen tests
# =============================================================================

def _make_saw(f0: float, sr: float, n: int) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / sr
    phase = (f0 * t) % 1.0
    return (2.0 * phase - 1.0).astype(np.float32)


def _smoke_test() -> None:
    sr = 48000.0
    n = int(sr * 2.0)

    filt = HuovilainenMoogWrapper(sr)

    # Input: saw + gentle fade-in
    x = _make_saw(110.0, sr, n)
    x *= np.linspace(0.0, 1.0, n, dtype=np.float32)

    # Parameter sweep (update per block to keep it simple)
    block = 256
    y = np.zeros_like(x)
    idx = 0
    while idx < n:
        b = min(block, n - idx)
        # sweep cutoff from 200 -> 8000 Hz
        pos = idx / max(1, (n - 1))
        cutoff = 200.0 * (8000.0 / 200.0) ** pos
        resonance = 0.9
        filt.set_params(cutoff=cutoff, resonance=resonance)
        y[idx:idx + b] = filt.process(x[idx:idx + b])
        idx += b

    # Plot
    import matplotlib.pyplot as plt
    t = np.arange(n) / sr
    plt.figure(figsize=(12, 6))
    plt.plot(t[:2000], x[:2000], label="input")
    plt.plot(t[:2000], y[:2000], label="output", alpha=0.9)
    plt.title("Huovilainen Moog Ladder (time domain, first ~2000 samples)")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Listen test (sounddevice)
    try:
        import sounddevice as sd
        sd.default.samplerate = int(sr)
        sd.default.channels = 1
        sd.play(y.astype(np.float32, copy=False), int(sr))
        sd.wait()
    except Exception as e:
        print("sounddevice listen test skipped:", repr(e))

    # Listen test with input.wav (soundfile) if present
    wav_path = "input.wav"
    if os.path.exists(wav_path):
        try:
            import soundfile as sf
            xwav, sr_wav = sf.read(wav_path, dtype="float32", always_2d=False)
            if xwav.ndim > 1:
                xwav = xwav[:, 0]
            filt2 = HuovilainenMoogWrapper(float(sr_wav))
            filt2.set_params(cutoff=1200.0, resonance=0.8)
            ywav = filt2.process(np.asarray(xwav))
            sf.write("output_filtered.wav", ywav, int(sr_wav))
            try:
                import sounddevice as sd
                sd.default.samplerate = int(sr_wav)
                sd.default.channels = 1
                sd.play(ywav.astype(np.float32, copy=False), int(sr_wav))
                sd.wait()
            except Exception as e:
                print("sounddevice listen for input.wav skipped:", repr(e))
            print("Wrote output_filtered.wav")
        except Exception as e:
            print("soundfile listen test skipped:", repr(e))
    else:
        print("input.wav not found; skipping soundfile test.")


if __name__ == "__main__":
    _smoke_test()
```
