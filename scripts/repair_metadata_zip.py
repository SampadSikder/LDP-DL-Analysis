import argparse
import struct
import sys
import zipfile
import zlib

LOCAL_HEADER_SIG = 0x04034B50
LOCAL_HEADER_FMT = '<HHHHHIIIHH'
LOCAL_HEADER_SIZE = struct.calcsize(LOCAL_HEADER_FMT)


def recover_entries(path: str):
    with open(path, 'rb') as f:
        data = f.read()

    n = len(data)
    pos = 0
    entries = []

    while pos + 4 + LOCAL_HEADER_SIZE <= n:
        sig = struct.unpack_from('<I', data, pos)[0]
        if sig != LOCAL_HEADER_SIG:
            break

        (_version, flags, comp, _mtime, _mdate, crc32, comp_size, _uncomp_size,
         fname_len, extra_len) = struct.unpack_from(LOCAL_HEADER_FMT, data, pos + 4)
        header_end = pos + 4 + LOCAL_HEADER_SIZE + fname_len + extra_len

        if flags & 0x08:
            print(f"  [WARN] entry at offset {pos} uses a streaming data descriptor "
                  f"(unexpected for this codebase); stopping recovery here")
            break
        if header_end + comp_size > n:
            print(f"  [WARN] truncated entry at offset {pos} (process likely died mid-write); "
                  f"stopping recovery here")
            break

        fname = data[pos + 4 + LOCAL_HEADER_SIZE: pos + 4 + LOCAL_HEADER_SIZE + fname_len].decode(
            'utf-8', 'replace'
        )
        payload = data[header_end: header_end + comp_size]
        entries.append((fname, comp, payload, crc32))
        pos = header_end + comp_size

    return entries, pos, n


def write_repaired_zip(entries: list, out_path: str) -> int:
    written = 0
    with zipfile.ZipFile(out_path, 'w', compression=zipfile.ZIP_STORED) as zout:
        for fname, comp, payload, crc32 in entries:
            if comp == 0:
                raw = payload
            elif comp == 8:
                raw = zlib.decompress(payload, -15)
            else:
                print(f"  [WARN] {fname}: unsupported compression method {comp}, skipping")
                continue
            if (zlib.crc32(raw) & 0xFFFFFFFF) != crc32:
                print(f"  [WARN] {fname}: CRC32 mismatch, skipping (corrupted entry)")
                continue
            zout.writestr(fname, raw)
            written += 1
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('corrupted_path', help='The unreadable metadata.npz')
    parser.add_argument('output_path', help='Where to write the repaired, valid zip')
    args = parser.parse_args()

    print(f"Scanning {args.corrupted_path} for recoverable entries...")
    entries, stopped_at, total_size = recover_entries(args.corrupted_path)
    print(f"  Found {len(entries):,} local file header entries "
          f"(stopped at byte {stopped_at:,} of {total_size:,})")

    if not entries:
        print("Nothing recoverable.")
        sys.exit(1)

    print(f"Writing repaired archive to {args.output_path}...")
    written = write_repaired_zip(entries, args.output_path)
    print(f"Done: {written:,}/{len(entries):,} entries passed CRC verification and were written.")


if __name__ == '__main__':
    main()
