#!/usr/bin/env python3
"""Build a local sparse ZIP index by downloading only its central-directory tail."""

from __future__ import annotations

import argparse
import os
import struct
from pathlib import Path

import requests


EOCD = struct.Struct("<4s4H2LH")
ZIP64_LOCATOR = struct.Struct("<4sLQL")
ZIP64_EOCD = struct.Struct("<4sQ2H2L4Q")


def fetch_range(session: requests.Session, url: str, start: int, end: int) -> bytes:
    response = session.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
    response.raise_for_status()
    if response.status_code != 206 or len(response.content) != end - start + 1:
        raise RuntimeError(f"unexpected range response {response.status_code}: {response.headers}")
    return response.content


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    session = requests.Session()
    probe = session.get(args.url, headers={"Range": "bytes=0-0"}, timeout=120)
    probe.raise_for_status()
    total = int(probe.headers["Content-Range"].rsplit("/", 1)[1])
    tail_size = min(total, 1024 * 1024)
    tail_start = total - tail_size
    tail = fetch_range(session, args.url, tail_start, total - 1)
    eocd_relative = tail.rfind(b"PK\x05\x06")
    if eocd_relative < 0:
        raise RuntimeError("ZIP EOCD not found")
    fields = EOCD.unpack_from(tail, eocd_relative)
    central_size, central_offset = fields[5], fields[6]
    eocd_absolute = tail_start + eocd_relative
    if central_size == 0xFFFFFFFF or central_offset == 0xFFFFFFFF or fields[4] == 0xFFFF:
        locator_absolute = eocd_absolute - ZIP64_LOCATOR.size
        locator = fetch_range(
            session, args.url, locator_absolute, locator_absolute + ZIP64_LOCATOR.size - 1
        )
        signature, _, zip64_offset, _ = ZIP64_LOCATOR.unpack(locator)
        if signature != b"PK\x06\x07":
            raise RuntimeError("ZIP64 locator not found")
        zip64 = fetch_range(
            session, args.url, zip64_offset, zip64_offset + ZIP64_EOCD.size - 1
        )
        values = ZIP64_EOCD.unpack(zip64)
        if values[0] != b"PK\x06\x06":
            raise RuntimeError("ZIP64 EOCD not found")
        central_size, central_offset = values[-2], values[-1]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".part")
    with temporary.open("wb") as handle:
        handle.truncate(total)
        handle.seek(central_offset)
        response = session.get(
            args.url,
            headers={"Range": f"bytes={central_offset}-{total - 1}"},
            stream=True,
            timeout=120,
        )
        response.raise_for_status()
        if response.status_code != 206:
            raise RuntimeError(f"central-directory range returned {response.status_code}")
        copied = 0
        for chunk in response.iter_content(1024 * 1024):
            if chunk:
                handle.write(chunk)
                copied += len(chunk)
        expected = total - central_offset
        if copied != expected:
            raise RuntimeError(f"central-directory tail {copied} != {expected}")
    os.replace(temporary, args.output)
    print(
        f"indexed {args.output}: total={total} central_offset={central_offset} "
        f"central_size={central_size} downloaded={total-central_offset}"
    )


if __name__ == "__main__":
    main()
