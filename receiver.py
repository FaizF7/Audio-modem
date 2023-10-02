#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import scipy.signal
import struct
import wave
import time
import sys
import configparser
import math
from header import *
from modemutils import *
from audioutils import *
from ofdm import *
import ldpc_jossy.py.ldpc as ldpc_jossy
import ldpc

recording_time = 35
sampling_rate = 48000
data_offset = 48000*2#60000

# OFDM
config = configparser.ConfigParser()
config.read("config.ini")
ofdm_length = int(config["OFDM"]["dft_length"])
ofdm_prefix_length = int(config["OFDM"]["prefix_length"])
ofdm_bins = [0,0]
ofdm_bins[0] = int(config["OFDM"]["first_bin"])
ofdm_bins[1] = int(config["OFDM"]["last_bin"])

# LDPC
code = ldpc_jossy.code(standard=config["LDPC"]["standard"], rate=config["LDPC"]["rate"], z=int(config["LDPC"]["z"]))
code.path_to_shared_lib = "./ldpc_jossy/bin/c_ldpc.so"
code.assign_proto()
codeword_len = len(code.proto[0]) * code.z

sync_period = 1 # Use low values if the clock mismatch is significant

# ========================= Load signal files =========================

# Load sync signal
sync = readwav("resources/doublechirp_1k_10k.wav", enforce_samplerate=sampling_rate)
sync_rev = sync[::-1]

# Load known OFDM signal
known_ofdm = readwav("resources/known_ofdm.wav", enforce_samplerate=sampling_rate)
known_ofdm_dft = np.fft.fft(known_ofdm)

# Load channel estimation signal
est = readwav("resources/doublechirp_1k_10k.wav", enforce_samplerate=sampling_rate)[:48000]
est_dft = np.fft.fft(est)
est_rev = est[::-1]

# Load recording
rec = []
if len(sys.argv) > 1:
    # Load recording
    print("Loading recording from file", sys.argv[1])
    if ".wav" in sys.argv[1]:
        rec = readwav(sys.argv[1], enforce_samplerate=sampling_rate)
    elif ".npy" in sys.argv[1]:
        rec = np.load(sys.argv[1])
else:
    # Record
    print("Recording now (%.1f seconds)" % recording_time)
    rec = record(recording_time, sampling_rate)
    writewav("received.wav", rec, sampling_rate)

rec_dft = np.fft.fft(rec)
rec = normalize(rec)
print("Recording length:", len(rec), "samples,", len(rec)/sampling_rate, "seconds")

# Remove the first 0.2 seconds
rec = rec[int(0.2*sampling_rate):]

# ========================= Synchronize and estimate the channel IR =========================

print()
print("#### Doing coarse synchronization...")

desync_rate = 0

# Synchronization
print("Finding doublechirps... ", flush=True, end="")
sync_locations = find_sync_signal(rec, sync)
data_frames_N = len(sync_locations) - 1
start = sync_locations[0]
print(sync_locations)
print("Finding known OFDM symbols... ", flush=True, end="")
kos_locations = find_sync_signal(rec, known_ofdm)
print(kos_locations)

if len(sync_locations) >= 2:
    data_length = sync_locations[1] - sync_locations[0] - len(sync) - 2 * len(known_ofdm)
    data_seconds = data_length / sampling_rate
    L = ofdm_length + ofdm_prefix_length
    desync = data_length % L
    if desync > (L / 2):
        desync -= L
    desync_rate = desync / data_seconds
    print("OFDM waveform length:", data_length, "samples")
    print("Estimated desync:", desync_rate, "samples per second")
else:
    print("Second sync signal missing!!! Exiting.")
    exit(1)

# Channel estimation
avg_chirp = np.zeros(48000)
avg_chirp += rec[start : start+48000]
avg_chirp += rec[start+48000 : start+96000]
avg_chirp += rec[sync_locations[1] : sync_locations[1]+48000]
avg_chirp += rec[sync_locations[1]+48000 : sync_locations[1]+96000]
avg_chirp /= 4
print("Estimating channel... ", flush=True, end="")
#channel,channel_dft = estimate_channel(rec, est, start + estimation_signal_offset)
channel,channel_dft = estimate_channel(avg_chirp, est, 0)
#channel,channel_dft = estimate_channel(rec, est, start+48000)
print("ok")

# Noise estimation
lo_cutoff = int(sampling_rate * ofdm_bins[0] / ofdm_length)
hi_cutoff = int(sampling_rate * ofdm_bins[1] / ofdm_length)
received_kos = bandpass(rec[sync_locations[0] : sync_locations[0]+48000], lo_cutoff, hi_cutoff, sampling_rate)
received_kos_dft = np.fft.fft(received_kos)
kos_bp_dft = np.fft.fft(bandpass(est, lo_cutoff, hi_cutoff, sampling_rate))
sim_kos = np.real(np.fft.ifft(scipy.signal.resample(channel_dft, 48000) * kos_bp_dft))
noise = received_kos - sim_kos
noise_var = np.var(noise)
received_kos_var = np.var(received_kos)
print("Noise variance estimate: %.8f" % noise_var)
print("SNR estimate: %.3f" % (received_kos_var / noise_var))
"""
lo_cutoff = int(sampling_rate * ofdm_bins[0] / ofdm_length)
hi_cutoff = int(sampling_rate * ofdm_bins[1] / ofdm_length)
received_kos = bandpass(rec[sync_locations[0]+96000 : sync_locations[0]+96000+len(known_ofdm)], lo_cutoff, hi_cutoff, sampling_rate)
received_kos_dft = np.fft.fft(received_kos)
kos_bp_dft = np.fft.fft(bandpass(known_ofdm, lo_cutoff, hi_cutoff, sampling_rate))
sim_kos = np.real(np.fft.ifft(scipy.signal.resample(channel_dft, len(known_ofdm)) * kos_bp_dft))
noise = received_kos - sim_kos
noise_var = np.var(noise)
received_kos_var = np.var(received_kos)
print("Noise variance estimate: %.3f" % noise_var)
print("SNR estimate: %.3f" % (received_kos_var / noise_var))
"""

# Adjust channel response length
channel_dft = scipy.signal.resample(channel_dft, L)
channel = np.real(np.fft.ifft(channel_dft))

# ========================= Demodulate OFDM waveform =========================

L = ofdm_length + ofdm_prefix_length
all_symbols = []
display_symbols = []
net_timeshift = 0
timeshift_log = []

# Bandpass the recording before demodulation
rec_bp = bandpass(rec, lo_cutoff, hi_cutoff, sampling_rate)

print()
print("#### Doing continuous synchronization...")

for f in range(data_frames_N):
    print("**** Frame", f)
    num_of_symbols = math.floor((sync_locations[f+1] - sync_locations[f] - 2 * len(known_ofdm) - len(sync)) / L + 0.5)
    display_interval = max([1, num_of_symbols // 10])
    print("Number of symbols:", num_of_symbols)

    for i in range(num_of_symbols):
        symbol_length = L
        symbol_offset = symbol_length * i
        symbol_start = sync_locations[f] + data_offset + len(known_ofdm) + symbol_offset

        data = rec_bp[symbol_start : symbol_start+symbol_length]
        sym = ofdm_decode(data, ofdm_length, ofdm_prefix_length, channel_dft, first_bin=ofdm_bins[0], last_bin=ofdm_bins[1], num_of_symbols=sync_period)

        # Find best fractional timeshift
        best_ft = None
        best_phase_mse = None
        best_ft_factors = None
        deltas = np.linspace(-1, 1, 7)**3

        if f == 0 and i == 0:
            deltas = np.linspace(-5, 5, 21)

        for delta in deltas:
            ft_factors = fractional_timeshift_factors(net_timeshift + delta, ofdm_length, ofdm_bins[0], ofdm_bins[1])
            ft_factors = np.repeat(ft_factors, len(sym) / len(ft_factors) + 1)[:len(sym)]
            mse = phase_mse(ft_factors * sym)

            if best_phase_mse == None or mse < best_phase_mse:
                best_ft = net_timeshift + delta
                best_phase_mse = mse
                best_ft_factors = ft_factors

        net_timeshift = best_ft
        """
        net_timeshift += desync_rate * symbol_length / sampling_rate
        best_ft_factors = fractional_timeshift_factors(net_timeshift, ofdm_length, ofdm_bins[0], ofdm_bins[1])
        best_ft_factors = np.repeat(best_ft_factors, len(sym) / len(best_ft_factors) + 1)[:len(sym)]
        """

        # Apply fractional timeshift to symbols
        sym *= best_ft_factors

        display_symbols += list(sym)
        all_symbols += list(sym)

        if i % display_interval == 0 and i > 0:
            print("%.0f%% done, timeshift: %.2f, phase score: %.1f, dist score: %.3f" % (100*i/num_of_symbols, net_timeshift, phase_mse(sym), dist_mse(sym)))
            #plt.scatter(np.real(display_symbols), np.imag(display_symbols), s=1)
            #plt.show()
            display_symbols = []

        timeshift_log.append(net_timeshift)

# ========================= Decode =========================

print()
print("#### Decoding")

msg = qpsk_decode(all_symbols)

num_of_codewords = len(msg) // codeword_len
print("Number of codewords:", num_of_codewords)

print("Computing LLRs... ", flush=True, end="")
llr = []
freq_bins = ofdm_bins[1] - ofdm_bins[0] + 1

for i in range(len(all_symbols)):
    s = all_symbols[i]

    this_bin = ofdm_bins[0] + i % freq_bins
    gainsq = channel_dft[this_bin]*np.conj(channel_dft[this_bin])

    llr1 = gainsq * 2**0.5 * np.imag(s) / noise_var
    llr2 = gainsq * 2**0.5 * np.real(s) / noise_var
    llr.append(np.real(llr1))
    llr.append(np.real(llr2))

llr = np.array(llr)
#llr = np.array(10*(0.5-np.array(msg)))
print(llr[:10])

print("Decoding LDPC codewords... ", flush=True, end="")
#bpd = ldpc.bp_decoder(code.pcmat(), max_iter=100, bp_method="ms", channel_probs=err_rate)
bpd = ldpc.bp_decoder(code.pcmat(), error_rate=0.1, max_iter=100, bp_method="ms")
msg_decoded = []
avg_it = 0
for i in range(num_of_codewords):
    percent_done = int(100 * i / num_of_codewords)
    if i % max([1, num_of_codewords // 10]) == 0:
        print("%d%%... " % percent_done, flush=True, end="")
    codeword = msg[i*codeword_len : (i+1)*codeword_len]
    
    """
    data_word,it = code.decode(llr[i*codeword_len : (i+1)*codeword_len], "sumprod2")
    data_word = (1*(data_word < 0))[:code.K]
    """
    errorless_codeword = bpd.decode(np.array(codeword))
    data_word = errorless_codeword[:code.K] # systematic code
    
    avg_it += it
    msg_decoded += list(1*(data_word > 0.5))

print("done")
avg_it /= num_of_codewords
print("Average number of iterations per codeword:", avg_it)

size,name,data = strip_header(bits2bytes(msg_decoded))
print("First 100 decoded bytes:", data[:100])
print("Last 100 decoded bytes:", data[:size][-100:])
print("Size:", size, "bytes")
print("Name:", name)

try:
    open(b"downloads/"+name, "wb").write(data[:size])
    open(b"downloads/"+name+b".raw", "wb").write(bits2bytes(msg))
    print(b"Saving as downloads/"+name)
except:
    open(b"downloads/received.bin", "wb").write(data[:size])
    open(b"downloads/received.bin.raw", "wb").write(bits2bytes(msg))
    print(b"Saving as downloads/received.bin")

# ========================= Visualize =========================

plt.plot(llr[:freq_bins])
plt.show()

fig,ax = plt.subplots(2,2)

ax[0,0].plot(rec)
#ax[0,0].vlines(start+estimation_signal_offset, min(rec), max(rec), colors="red")
for sl in sync_locations:
    ax[0,0].vlines(sl, min(rec), max(rec), colors="red")
    ax[0,0].vlines(sl + 96000, min(rec), max(rec), colors="red")
    ax[0,0].vlines(sl + 96000 + len(known_ofdm), min(rec), max(rec), colors="green")
for kl in kos_locations:
    ax[0,0].vlines(kl, min(rec), max(rec), colors="yellow")
    ax[0,0].vlines(kl+len(known_ofdm), min(rec), max(rec), colors="yellow")
ax[0,0].set_title("Raw recording")
ax[0,0].set_xlabel("Sample number")

#rec_deconvolved = bandpass(rec_deconvolved, 3500, 5000, sampling_rate)
ax[0,1].scatter(np.real(all_symbols), np.imag(all_symbols), s=0.1)
ax[0,1].set_title("Constellation symbol distribution")

data_lo = int(len(channel_dft) * ofdm_bins[0] / ofdm_length)
data_hi = int(len(channel_dft) * ofdm_bins[1] / ofdm_length)
channel_dft_dB = 20*np.log10(np.abs(channel_dft))
ax[1,0].plot(channel_dft_dB)
ax[1,0].vlines(data_lo, min(channel_dft_dB), max(channel_dft_dB), colors="red")
ax[1,0].vlines(data_hi, min(channel_dft_dB), max(channel_dft_dB), colors="red")
ax[1,0].set_title("Channel freq response (dB)")
ax[1,0].set_xlabel("Frequency bin")

"""
ax[1,1].plot(channel)
ax[1,1].set_title("Channel time response estimate")
ax[1,1].set_xlabel("Sample number")
"""
ax[1,1].plot(timeshift_log)
ax[1,1].set_title("Net timeshift over time")
ax[1,1].set_xlabel("OFDM symbol number")

plt.show()

