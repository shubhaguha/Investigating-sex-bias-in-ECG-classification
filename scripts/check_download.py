"""Check a PhysioNet/CinC 2021 download: count complete recordings and find damaged .mat files.

A recording is complete when its .hea exists and its .mat is the MATLAB v4 file the header describes:
every header line says `<record>.mat 16+24 ...` (16-bit samples after a 24-byte MATLAB header), so
the file holds 24 + 2 * signals * samples bytes and its header names a signals x samples matrix.
Interrupted downloads leave .mat files that are short or hold something else, which preprocessing
then fails to read ("read length must be non-negative", "Mat 4 mopt wrong format").

    python3 scripts/check_download.py <training dir> <database> [<database> ...]
        prints "<database> <complete>" per database
    python3 scripts/check_download.py --list-bad <training dir> <database> [...]
        prints the path of every .mat that exists but does not match its header

Standard library only, so it runs before the Python environment is set up.
"""
import os
import struct
import sys

MAT_HEADER_BYTES = 24  # 5 int32 (type, rows, cols, imagf, name length) + "val\0"


def expected_shape(header_file):
    """(signals, samples), or None if the header does not describe the usual 16+24 .mat layout."""
    with open(header_file) as f:
        record_line, signal_line = f.readline().split(), f.readline().split()
    if len(signal_line) < 2 or signal_line[1] != '16+24':
        return None
    return int(record_line[1]), int(record_line[3])


def mat_ok(mat_file, signals, samples):
    try:
        if os.path.getsize(mat_file) != MAT_HEADER_BYTES + 2 * signals * samples:
            return False
        with open(mat_file, 'rb') as f:
            _, rows, cols = struct.unpack('<3i', f.read(12))
        return (rows, cols) == (signals, samples)
    except (OSError, struct.error):
        return False


def scan(db_dir):
    """Yields (mat_file, status) per .hea: status is 'ok', 'missing' or 'bad'."""
    for root, _, files in os.walk(db_dir):
        for name in files:
            if not name.endswith('.hea'):
                continue
            mat_file = os.path.join(root, name[:-4] + '.mat')
            if not os.path.exists(mat_file):
                yield mat_file, 'missing'
                continue
            try:
                shape = expected_shape(os.path.join(root, name))
            except (OSError, ValueError, IndexError):
                yield mat_file, 'bad'  # unreadable header: fetch the recording again
                continue
            if shape is None:
                yield mat_file, 'ok'  # unknown layout: existence is all that can be checked
                continue
            yield mat_file, 'ok' if mat_ok(mat_file, *shape) else 'bad'


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
            for mat_file, status in results:
                if status == 'bad':
                    print(mat_file)
        else:
            print(db, sum(status == 'ok' for _, status in results))


if __name__ == '__main__':
    main()
