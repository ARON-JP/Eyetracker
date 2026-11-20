"""Tiny dependency-free PNG encoder for the Tk preview."""

from __future__ import annotations

import struct
import zlib

import numpy as np


def encode_png(rgb: np.ndarray) -> bytes:
    """Encode HxWx3 uint8 RGB as a PNG byte string (no Pillow needed)."""
    h, w = rgb.shape[:2]
    if rgb.shape[2] == 4:
        rgb = rgb[:, :, :3]
    raw = np.empty((h, w * 3 + 1), dtype=np.uint8)
    raw[:, 0] = 0  # filter: none
    raw[:, 1:] = rgb.reshape(h, w * 3)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw.tobytes(), 3))
        + chunk(b"IEND", b"")
    )
