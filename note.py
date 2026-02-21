import numpy as np
import sounddevice as sd

# Parameters
fs = 44100
duration = 7.5

# C major chord frequencies
frequencies = [261.63, 328.63, 392.00, 261.63*2]

# Time array
t = np.linspace(0, duration, int(fs * duration), endpoint=False)

# Generate chord
audio = sum(np.sin(2 * np.pi * f * t) for f in frequencies)

# Create fade-out envelope (1 → 0)
fade_out = np.linspace(1, 0, len(audio))

# Apply fade-out
audio = 0.2 * audio * fade_out

# Play
sd.play(audio, fs)
sd.wait()