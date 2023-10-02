#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

def bandpass(signal, lo, hi, sampling_rate):
    N = len(signal)

    lo_bin = lo * N // sampling_rate
    hi_bin = hi * N // sampling_rate

    signal_dft = np.fft.fft(signal)
    signal_dft[:lo_bin] = 0
    signal_dft[(-lo_bin-1):] = 0
    signal_dft[hi_bin:(-hi_bin-1)] = 0

    return np.real(np.fft.ifft(signal_dft))

def ir_noise_level(ir):
    N = 100
    window_len = len(ir) // N
    level = []

    for i in range(N):
        ir_slice = ir[i*window_len : (i+1)*window_len]
        #level.append(np.std(ir_slice))
        level.append(max(np.abs(ir_slice)))

    return min(level)

def ir_delay(ir):
    nl = ir_noise_level(ir)
    m = max(np.abs(ir))
    thr = max([0.1*m, 5*nl])

    for i in range(len(ir) * 9 // 10, len(ir)):
        if np.abs(ir[i]) - nl > thr:
            return len(ir) - i

    return 0

def find_sync_signal(transmission_waveform, sync_waveform):
    conv = scipy.signal.convolve(transmission_waveform, sync_waveform[::-1])
    peak_height = max(conv)
    peaks = np.array(scipy.signal.find_peaks(conv, height=peak_height*0.75, distance=len(sync_waveform))[0])
    peaks -= len(sync_waveform)
    return peaks

def estimate_channel(transmission_waveform, estimation_waveform, start):
    cropped = transmission_waveform[start : start+len(estimation_waveform)]
    transmission_dft = np.fft.fft(cropped)
    estimation_dft = np.fft.fft(estimation_waveform)
    channel_dft = transmission_dft / estimation_dft
    channel = np.real(np.fft.ifft(channel_dft))
    return channel, channel_dft

def simplify_channel_ir(ir, cutoff=None, min_shift=50, padding=10):
    d = ir_delay(ir)
    shift_amt = max([min_shift, d + padding])
    if cutoff:
        ir_simplified = np.concatenate([ir[-shift_amt:], ir[:cutoff]])
    else:
        ir_simplified = np.concatenate([ir[-shift_amt:], ir[:-shift_amt]])
    return ir_simplified,shift_amt

def deconvolve(transmission, ir, transmission_shift=0):
    t_padded = np.concatenate([np.zeros(transmission_shift), transmission])
    t_padded_dft = np.fft.fft(t_padded)
    t_deconvolved = np.real(np.fft.ifft(t_padded_dft / np.fft.fft(np.pad(ir, (0, len(t_padded)-len(ir)), mode="constant"))))
    return t_deconvolved

