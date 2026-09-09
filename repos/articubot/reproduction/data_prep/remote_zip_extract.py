#!/usr/bin/env python3
"""Extract selected members from the large official ArticuBot eval ZIP.

The Google Drive object supports HTTP byte ranges.  The ZIP central directory is
kept in a local sparse file, so this utility downloads only each selected local
header and compressed payload instead of the full 32.27 GB archive.
"""

from __future__ import annotations

import argparse
import binascii
import os
import struct
import zlib
import zipfile
from pathlib import Path

import requests


LOCAL_FILE_HEADER = struct.Struct("<4s5H3L2H")


class RangeExtractor:
    def __init__(self, index_path: Path, url: str, timeout: int = 120):
        self.index_path = index_path
        self.url = url
        self.timeout = timeout
        self.session = requests.Session()
        self.archive = zipfile.ZipFile(self.index_path)

    def _range(self, start: int, end: int) -> bytes:
        response = self.session.get(
            self.url,
            headers={"Range": f"bytes={start}-{end}"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.content
        expected = end - start + 1
        if response.status_code != 206 or len(data) != expected:
            raise RuntimeError(
                f"bad range response: status={response.status_code}, "
                f"expected={expected}, got={len(data)}, headers={dict(response.headers)}"
            )
        return data

    def read_member(self, info: zipfile.ZipInfo) -> bytes:
        fixed = self._range(info.header_offset, info.header_offset + LOCAL_FILE_HEADER.size - 1)
        fields = LOCAL_FILE_HEADER.unpack(fixed)
        if fields[0] != b"PK\x03\x04":
            raise RuntimeError(f"bad local header for {info.filename!r}")
        filename_len, extra_len = fields[-2:]
        payload_start = info.header_offset + LOCAL_FILE_HEADER.size + filename_len + extra_len
        if info.compress_size:
            compressed = self._range(payload_start, payload_start + info.compress_size - 1)
        else:
            compressed = b""
        if info.compress_type == zipfile.ZIP_STORED:
            data = compressed
        elif info.compress_type == zipfile.ZIP_DEFLATED:
            data = zlib.decompress(compressed, -zlib.MAX_WBITS)
        else:
            raise NotImplementedError(
                f"compression method {info.compress_type} for {info.filename!r}"
            )
        crc = binascii.crc32(data) & 0xFFFFFFFF
        if len(data) != info.file_size or crc != info.CRC:
            raise RuntimeError(
                f"integrity failure for {info.filename!r}: "
                f"size {len(data)}/{info.file_size}, crc {crc:08x}/{info.CRC:08x}"
            )
        return data

    def extract(self, member: str, output_root: Path) -> Path:
        info = self.archive.getinfo(member)
        destination = output_root / member
        if info.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
            return destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        data = self.read_member(info)
        temporary = destination.with_name(destination.name + ".part")
        temporary.write_bytes(data)
        os.replace(temporary, destination)
        return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("members", nargs="+")
    args = parser.parse_args()
    extractor = RangeExtractor(args.index, args.url)
    for member in args.members:
        destination = extractor.extract(member, args.output_root)
        print(destination)


if __name__ == "__main__":
    main()
