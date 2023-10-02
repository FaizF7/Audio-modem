#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from audioutils import *
from modemutils import *
from ofdm import *
from header import *
import ldpc_jossy.py.ldpc as ldpc
import sys
import math
import configparser

if len(sys.argv) < 2:
    print("Usage: ./transmitter.py <file-to-transmit>")
    exit(1)

msg_filename = bytes(sys.argv[1], "ascii")
msg = open(msg_filename, "rb").read()
output_file = "output.wav"
sampling_rate = 48000

# OFDM
config = configparser.ConfigParser()
config.read("config.ini")
max_ofdm_symbols = int(config["Layout"]["max_ofdm_symbols_per_frame"])
ofdm_length = int(config["OFDM"]["dft_length"])
prefix_length = int(config["OFDM"]["prefix_length"])
data_bins = [0,0]
data_bins[0] = int(config["OFDM"]["first_bin"])
data_bins[1] = int(config["OFDM"]["last_bin"])

# LDPC
code = ldpc.code(standard=config["LDPC"]["standard"], rate=config["LDPC"]["rate"], z=int(config["LDPC"]["z"]))
code.path_to_shared_lib = "./ldpc_jossy/bin/c_ldpc.so"
code.assign_proto()
code_data_len = len(code.proto) * code.z

clip = True
clip_max = 3
clip_attempts = 4
dist_margin_threshold = 0.8

# ========================= Encode =========================

print("#### Encoding")

# Add header
print("Adding metadata... ", flush=True, end="")
msg = add_header(msg, msg_filename)
print("done")

# Convert to a list of bits
msg = bytes2bits(msg)

# Encode with the LDPC
print("Encoding with LDPC... ", flush=True, end="")
msg_padding_len = math.ceil(len(msg) / code_data_len) * code_data_len - len(msg)
msg = msg + list(np.random.randint(0, 2, msg_padding_len))
msg_codeword_N = len(msg) // code_data_len
msg_encoded = []
for i in range(msg_codeword_N):
    codeword = code.encode(msg[i*code_data_len : (i+1)*code_data_len])
    msg_encoded += list(codeword)
print("done\n")

# Modulate with OFDM
enc = qpsk_encode(msg_encoded)
wf = ofdm_encode(enc, ofdm_length, prefix_length, first_bin=data_bins[0], last_bin=data_bins[1])
data_frames_N = math.ceil(len(wf) / (ofdm_length + prefix_length) / max_ofdm_symbols)
data_frame_samples_N = max_ofdm_symbols * (ofdm_length + prefix_length)

print("#### Modulation")
print("Original message:      ", bits2bytes(msg[:8*100]))
print("Message length:        ", len(msg)//8, "bytes,", len(msg)//4, "QPSK symbols")
print("OFDM waveform length:  ", len(wf), "samples, %.2f seconds" % (len(wf) / sampling_rate))
print("OFDM symbols:          ", len(wf) / (ofdm_length + prefix_length))
print("Data frames:           ", data_frames_N)

# ========================= Apply clipping to improve PAPR =========================
wf0 = wf
fr = [1] * (ofdm_length + prefix_length) # ideal frequency response
if clip:
    print()
    print("#### Clipping")

    amp = 0

    ratio_lower = 1
    ratio_upper = clip_max

    #for ratio in reversed(np.linspace(1, clip_max, clip_attempts)):
    for i in range(clip_attempts):
        ratio = (ratio_lower + ratio_upper) / 2
        amp = max(np.abs(wf)) / ratio

        print("Clipping factor %.2f: verifying data integrity... " % ratio, flush=True, end="")

        wf_clipped = np.clip(wf, -amp, amp)
        dec = ofdm_decode(wf_clipped, ofdm_length, prefix_length, fr, first_bin=data_bins[0], last_bin=data_bins[1])
        dist_margin = distance_margin(dec)
        msg_decoded = qpsk_decode(dec)

        if msg_decoded[:len(msg_encoded)] != msg_encoded or dist_margin < dist_margin_threshold:
            print("ok but too much distortion", end="")
            ratio_upper = ratio
        else:
            print("ok", end="")
            ratio_lower = ratio

        print(" (distance margin %.3f, PAPR %.2f)" % (dist_margin, papr(wf_clipped)))

    dec = ofdm_decode(np.clip(wf, -amp, amp), ofdm_length, prefix_length, fr, first_bin=data_bins[0], last_bin=data_bins[1])
    wf = np.clip(wf, -amp, amp)
    print()
    print("Final choice: %.2f" % ratio)

dec = ofdm_decode(wf, ofdm_length, prefix_length, fr, first_bin=data_bins[0], last_bin=data_bins[1])
plt.scatter(np.real(dec), np.imag(dec), s=0.1, label="After clipping")
plt.scatter(np.real(enc), np.imag(enc), s=0.1, label="Original")
plt.legend()
plt.title("Constellation symbol distribution")
plt.show()

plt.plot(wf0, label = 'original')
plt.plot(wf, label = 'after clipping')
# ========================= Generate combined waveform =========================

# Components
chirp = normalize(readwav("resources/doublechirp_1k_10k.wav", enforce_samplerate=sampling_rate))[:48000]
known_seq = normalize(readwav("resources/known_ofdm.wav", enforce_samplerate=sampling_rate))
wf = normalize(wf)
data_frames = [wf[data_frame_samples_N*i : data_frame_samples_N*(i+1)] for i in range(math.ceil(len(wf) / data_frame_samples_N))]
data_frames_decorated = []
for d in data_frames:
    data_frames_decorated.append(np.concatenate([chirp, known_seq, d, known_seq, chirp]))

# Combined waveform
output = np.concatenate([np.zeros(1 * sampling_rate), chirp] + data_frames_decorated + [chirp, np.zeros(1 * sampling_rate)])
writewav(output_file, output, sampling_rate)
np.save("output.npy", output)
print("Total duration:", len(output) / sampling_rate, "seconds")

#plt.plot(wf, label = 'after clipping')
plt.plot()
plt.title("Modulated waveform")
plt.legend()
plt.show()

