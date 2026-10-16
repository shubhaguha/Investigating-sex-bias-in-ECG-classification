"""Check a PhysioNet/CinC 2021 download: count complete recordings and find damaged .mat files.

A recording is complete when its .hea is a WFDB header and its .mat is the MATLAB v4 file the header
describes: every CinC 2021 header says `<record>.mat 16+24 ...` (16-bit samples after a 24-byte MATLAB
header), so the file holds 24 + 2 * signals * samples bytes and its MATLAB header is that of an int16
signals x samples matrix named "val", the only thing scipy.io.loadmat reads before the samples.
Interrupted downloads leave .mat files that are short or hold something else, which preprocessing
then fails to read ("read length must be non-negative", "Mat 4 mopt wrong format").

    python3 scripts/check_download.py <training dir> <database> [<database> ...]
        prints "<database> <complete>" per database
    python3 scripts/check_download.py --list-bad <training dir> <database> [...]
        prints the files to fetch again: every .mat that exists but does not match its header, and
        the .hea as well when the header itself is damaged

Standard library only, so it runs before the Python environment is set up.
"""
import os
import struct
import sys

MAT_HEADER_BYTES = 24  # 5 int32 (type, rows, cols, imagf, name length) + "val\0"


def expected_shape(header_file):
    """(signals, samples) from a CinC 2021 header; raises ValueError if the header is damaged."""
    with open(header_file) as f:
        lines = f.read().split('\n')
    record_line = lines[0].split()
    name = os.path.basename(header_file)[:-4]
    if len(record_line) < 4 or record_line[0] != name:
        raise ValueError('not a WFDB header')
    signals, samples = int(record_line[1]), int(record_line[3])
    signal_lines = [l.split() for l in lines[1:1 + signals]]
    if signals < 1 or samples < 1 or any(l[:2] != [name + '.mat', '16+24'] for l in signal_lines) \
            or len(signal_lines) < signals:
        raise ValueError('not a 16+24 header for this record')
    return signals, samples


def mat_ok(mat_file, signals, samples):
    try:
        if os.path.getsize(mat_file) != MAT_HEADER_BYTES + 2 * signals * samples:
            return False
        with open(mat_file, 'rb') as f:
            mopt, rows, cols, imagf, namelen = struct.unpack('<5i', f.read(20))
            name = f.read(4)
        # mopt 30 / 40: little-endian, full matrix of int16 / uint16
        return mopt in (30, 40) and (rows, cols) == (signals, samples) and imagf == 0 \
            and namelen == 4 and name == b'val\0'

    except (OSError, struct.error):
        return False


def scan(db_dir):
    """Yields (header_file, mat_file, status) per .hea: status is 'ok', 'missing', 'bad' (the .mat)
    or 'bad header'."""
    for root, _, files in os.walk(db_dir):
        for name in files:
            if not name.endswith('.hea'):
                continue
            header_file = os.path.join(root, name)
            mat_file = header_file[:-4] + '.mat'
            try:
                shape = expected_shape(header_file)
            except (OSError, ValueError, UnicodeDecodeError):
                yield header_file, mat_file, 'bad header'  # fetch the whole recording again
                continue
            if not os.path.exists(mat_file):
                yield header_file, mat_file, 'missing'
                continue
            yield header_file, mat_file, 'ok' if mat_ok(mat_file, *shape) else 'bad'


def main():
    args = sys.argv[1:]
    list_bad = args[:1] == ['--list-bad']
    if list_bad:
        args = args[1:]
    if len(args) < 2:
        sys.exit(__doc__)
    training_dir, databases = args[0], args[1:]
    for db in databases:
        results = list(scan(os.path.join(training_dir, db))) if os.path.isdir(os.path.join(training_dir, db)) else []
        if list_bad:
            for header_file, mat_file, status in results:
                if status == 'bad header':
                    print(header_file)
                if status in ('bad', 'bad header') and os.path.exists(mat_file):
                    print(mat_file)
        else:
            print(db, sum(status == 'ok' for _, _, status in results))


if __name__ == '__main__':
    main()
