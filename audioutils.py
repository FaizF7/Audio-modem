#!/usr/bin/env python3
import sounddevice as sd
import numpy as np
import scipy.io.wavfile
import wave
import struct

def readwav(filename, enforce_samplerate=48000):
    w = wave.open(filename)
    assert w.getframerate() == enforce_samplerate
    waveform = [x[0] for x in struct.iter_unpack("<h", w.readframes(w.getnframes()))]
    return waveform

def writewav(filename, waveform, samplerate):
    scipy.io.wavfile.write(filename, samplerate, np.int16((2**15 - 2) * np.array(waveform) / max(waveform)))

def record(duration, samplerate):
    rec_raw = sd.rec(int(duration * samplerate), channels=1, blocking=True, samplerate=samplerate)
    rec = np.reshape(rec_raw, len(rec_raw))
    return rec

def normalize(waveform):
    return np.int16((2**15 - 1) * np.array(waveform) / max(waveform))

