#!/usr/bin/env python3
import scipy.signal
import numpy as np
import math
import random

# Convert list of bytes to bits
def bytes2bits(byte_str):
    assert type(byte_str) == bytes

    bits = []

    for b in byte_str:
        for shift in range(8):
            bits.append((b >> (7 - shift)) & 0b00000001)

    return bits

# Convert list of bits to bytes
def bits2bytes(bits):
    assert type(bits) == list
    #assert len(bits) % 8 == 0

    pow2 = np.array([128,64,32,16,8,4,2,1])
    byte_list = []

    for i in range(len(bits) // 8):
        bit_group = bits[8*i:8*(i+1)]
        byte_list.append(sum(pow2 * bit_group))

    return bytes(byte_list)

# Returns the 4 bt pairs in a byte
def byte_to_bit_pairs(b):
    assert type(b) == int and b >= 0 and b < 256
    bin_str = bin(b)[2:].zfill(8)
    bit_pairs = [bin_str[2*i:2*i+2] for i in range(4)]
    return bit_pairs

# Quadrant 1(++): 00, quadrant 2(-+): 01, quadrant 3(--): 11, quadrant 4(+-): 10
def qpsk_encode(data, a=1):
    assert len(data) % 2 == 0
    assert type(data) == list # list of bits

    cursor = 0
    symbols = []

    bit_pairs = [data[2*i:2*(i+1)] for i in range(len(data) // 2)]

    for bp in bit_pairs:
        if bp == [0,0]:
            symbols.append(complex(a,a))
        elif bp == [0,1]:
            symbols.append(complex(-a,a))
        elif bp == [1,1]:
            symbols.append(complex(-a,-a))
        elif bp == [1,0]:
            symbols.append(complex(a,-a))

    return symbols

# Decode QPSK constellation symbol into a bit pair
def qpsk_decode(symbols):
    decoded_bits = []

    for s in symbols:
        if np.real(s) > 0 and np.imag(s) > 0:
            decoded_bits += [0,0]
        elif np.real(s) < 0 and np.imag(s) > 0:
            decoded_bits += [0,1]
        elif np.real(s) < 0 and np.imag(s) < 0:
            decoded_bits += [1,1]
        elif np.real(s) > 0 and np.imag(s) < 0:
            decoded_bits += [1,0]
        else:
            print("Warning: symbol on edge of decision region, defaulting to 00.")
            decoded_bits += [0,0]

    return decoded_bits

def random_qpsk_symbols(n, a=1):
    syms = []
    for i in range(n):
        syms.append(random.choice([complex(a,a), complex(a,-a), complex(-a,a), complex(-a,-a)]))
    return syms

# Encode a sequence of constellation symbols into an OFDM waveform
def ofdm_encode(symbols, length, prefix_length, first_bin=0, last_bin=None, use_all_bins=False):
    assert length >= 2 * prefix_length
    assert use_all_bins or first_bin >= 1

    if last_bin != None:
        assert last_bin >= 0
        assert last_bin <= length // 2
        assert last_bin >= first_bin
    else:
        last_bin = length // 2

    if use_all_bins:
        first_bin = 0
        last_bin = length - 1

    # Figure out number of OFDM symbols and padding length
    data_bins_N = last_bin - first_bin + 1
    ofdm_symbols_N = math.ceil(len(symbols) / data_bins_N)
    padding_len = ofdm_symbols_N * data_bins_N - len(symbols)

    # Padding
    #padding_symbol = qpsk_encode(b"\x00")[0]
    #symbols += padding_len * [padding_symbol]
    symbols = np.concatenate([symbols, random_qpsk_symbols(padding_len)])

    waveform = []

    # Generate OFDM symbols
    for i in range(ofdm_symbols_N):
        # Chunk of data transmitted in a single OFDM symbol
        subdata = symbols[data_bins_N*i : data_bins_N*(i+1)]

        # Assign data into bins
        empty_bins_lo = max([0, first_bin])
        empty_bins_hi = length // 2 - last_bin - 1
        ofdm_sym_dft = np.concatenate([random_qpsk_symbols(empty_bins_lo), subdata, random_qpsk_symbols(empty_bins_hi)])

        # Append conjugate symbols
        #middle_symbol = sum(a[::2]) - sum(ofdm_sym_dft[1::2])
        middle_symbol = 0
        if not use_all_bins:
            ofdm_sym_dft = np.concatenate([ofdm_sym_dft, [middle_symbol], np.conjugate(np.flip(ofdm_sym_dft))])[:-1]

        # Compute the IDFT
        ofdm_sym = np.fft.ifft(ofdm_sym_dft)
        
        # Add cyclic prefix and append to waveform
        ofdm_sym = list(np.concatenate([ofdm_sym[-prefix_length:], ofdm_sym]))
        waveform += ofdm_sym

    return np.real(waveform)

# Decode OFDM waveform into a sequence of constellation symbols
def ofdm_decode(waveform, length, prefix_length, channel_dft, first_bin=1, last_bin=None, num_of_symbols=None):
    assert length >= 2 * prefix_length
    assert first_bin >= 0
    assert len(channel_dft) == prefix_length + length

    if last_bin != None:
        assert last_bin >= 0
        assert last_bin <= length // 2
        assert last_bin >= first_bin
    else:
        last_bin = length // 2

    # OFDM symbol length and number
    data_bins_N = last_bin - first_bin + 1
    ofdm_symbol_len = length + prefix_length
    #ofdm_symbols_N = math.ceil(len(waveform) / ofdm_symbol_len)
    ofdm_symbols_N = len(waveform) // ofdm_symbol_len

    # Decoded constellation symbols
    symbols = []

    if num_of_symbols == None:
        num_of_symbols = ofdm_symbols_N

    for i in range(num_of_symbols):
        #ofdm_symbol = np.concatenate([[0]*ir_shift, waveform[i*ofdm_symbol_len : (i+1)*ofdm_symbol_len]])
        ofdm_symbol = waveform[i*ofdm_symbol_len : (i+1)*ofdm_symbol_len]
        ofdm_symbol = np.fft.ifft(np.fft.fft(ofdm_symbol) / channel_dft)

        # Remove cyclic prefix
        data_dft = np.fft.fft(ofdm_symbol[prefix_length:prefix_length+length])

        # Extract data bins
        symbols += list(data_dft[first_bin:(last_bin+1)])

    return symbols

def make_ofdm_sync_symbol(length, prefix_length, first_bin, last_bin):
    data_bins_N = last_bin - first_bin + 1
    ones_symbols = data_bins_N * [qpsk_encode(b"\xff")[0]]
    return ofdm_encode(ones_symbols, length, prefix_length, first_bin=first_bin, last_bin=last_bin)

def papr(waveform):
    pwr = np.abs(waveform)**2
    peak = max(pwr)
    avg = np.mean(pwr)
    return peak / avg

def phase_mse(symbols):
    angles = np.angle(symbols) * 180 / np.pi
    angle_errors = []

    for a in angles:
        if a > 0 and a < 90:
            angle_errors.append(np.abs(a - 45))
        elif a > 90 and a < 180:
            angle_errors.append(np.abs(a - 135))
        elif a < 0 and a > -90:
            angle_errors.append(np.abs(a - (-45)))
        elif a < -90 and a > -180:
            angle_errors.append(np.abs(a - (-135)))
        else:
            angle_errors.append(45)

    return sum(np.array(angle_errors)**2) / len(angle_errors)

def dist_mse(symbols):
    sym_tmp = (np.abs(np.real(symbols)) + 1j * np.abs(np.imag(symbols))) / np.mean(np.abs(symbols))
    dist = np.abs(sym_tmp - np.mean(sym_tmp))
    return sum(dist) / len(dist)

def distance_margin(symbols):
    re_norm = np.mean(np.abs(np.real(symbols)))
    im_norm = np.mean(np.abs(np.imag(symbols)))
    re_margin = min(np.abs(np.real(symbols))) / re_norm
    im_margin = min(np.abs(np.imag(symbols))) / im_norm
    return min([re_margin, im_margin])

def ofdm_sync_symbol_score(waveform, length, prefix_length, first_bin, last_bin):
    symbols = ofdm_decode(waveform, length, prefix_length, first_bin=first_bin, last_bin=last_bin, num_of_symbols=1)
    mean_radius = np.mean(np.abs(symbols))
    target = complex(-mean_radius/2**0.5, -mean_radius/2**0.5)
    dist = np.abs(np.array(symbols) - target)
    return sum(np.array(dist)**2) / len(dist)

def insert_sync_symbols(waveform, sync_period, ofdm_length, prefix_length, first_bin, last_bin):
    data_run_length = (sync_period - 1) * (ofdm_length + prefix_length)
    N = len(waveform) // (ofdm_length + prefix_length)
    
    sync_sym = list(make_ofdm_sync_symbol(ofdm_length, prefix_length, first_bin, last_bin))

    new_waveform = []
    for i in range(N):
        data = list(waveform[i*data_run_length : (i+1)*data_run_length])
        new_waveform += sync_sym
        new_waveform += data

    return new_waveform

def find_sync_symbol_delay(waveform, initial_guess, sync_period, ofdm_length, prefix_length, first_bin, last_bin):
    results = []
    #for d in range(-prefix_length, prefix_length):
    for d in [-1,0,1]:
        data = waveform[initial_guess+d : initial_guess+d+(ofdm_length+prefix_length)*sync_period]
        symbols = ofdm_decode(data, ofdm_length, prefix_length, first_bin=first_bin, last_bin=last_bin, num_of_symbols=sync_period)
        score = (0.3*abs(d)+1) * phase_mse(symbols)
        results.append( (d, score) )

    best = sorted(results, key=lambda x:x[1])[0]
    secondbest = sorted(results, key=lambda x:x[1])[1]
    return best[0], best[1], secondbest[1]

def fractional_timeshift_factors(timeshift, ofdm_length, first_bin, last_bin):
    bins = last_bin - first_bin + 1
    k = np.linspace(first_bin, last_bin, bins)
    return np.exp(1j * 2*np.pi * k * timeshift / ofdm_length)

