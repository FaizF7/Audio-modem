#!/usr/bin/env python3
import numpy as np
import struct

def add_header(data, filename):
    assert type(filename) == bytes
    assert type(data) == bytes
    return struct.pack("<I", len(data)) + filename + b"\x00" + data

def strip_header(file):
    size = struct.unpack("<I", file[:4])[0]
    file = file[4:]
    null_terminator_location = file.find(b"\x00")
    filename = file[:null_terminator_location]
    data = file[null_terminator_location+1:]
    return size,filename,data

