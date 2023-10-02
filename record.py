#!/usr/bin/env python3
# ./record.py <duration in seconds> <output file>
import scipy.io.wavfile
import sounddevice as sd
import numpy as np
import sys

sampling_rate = 48000

if len(sys.argv) < 2:
    print("Usage: ./record.py <duration> <output-filename>")
    exit(1)

duration = float(sys.argv[1])
output = sys.argv[2]

# Record
print("Recording now (%.1f seconds)" % duration)
rec_raw = sd.rec(int(duration * sampling_rate), channels=1, blocking=True, samplerate=sampling_rate)
rec = np.reshape(rec_raw, len(rec_raw))
print("Recording length:", len(rec), "samples,", len(rec)/sampling_rate, "seconds")

scipy.io.wavfile.write(output, sampling_rate, np.int16(2**16 * rec))

