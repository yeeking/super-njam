"""Compact integer encoding helpers for NJamV3."""

from __future__ import annotations

ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def to_base36(value: int) -> str:
    if value == 0:
        return "0"
    sign = "-" if value < 0 else ""
    n = abs(value)
    digits = []
    while n:
        n, remainder = divmod(n, 36)
        digits.append(ALPHABET[remainder])
    return sign + "".join(reversed(digits))


def from_base36(text: str) -> int:
    assert text, "Base36 text must be non-empty."
    sign = -1 if text.startswith("-") else 1
    body = text[1:] if sign < 0 else text
    assert body, f"Invalid base36 integer: {text!r}"
    value = 0
    for char in body.upper():
        idx = ALPHABET.find(char)
        assert idx >= 0, f"Invalid base36 digit {char!r} in {text!r}"
        value = value * 36 + idx
    return sign * value

