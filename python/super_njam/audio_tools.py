"""Audio rendering helpers.

Soundfont rendering is preferred when a soundfont path is supplied and
``fluidsynth`` is available through pretty_midi. A lightweight sine fallback is
kept for smoke testing and environments without a local soundfont.
"""

from __future__ import annotations

import math
import wave
from array import array
from pathlib import Path
from typing import Optional

import numpy as np
import pretty_midi

from .midi_tools import njam_to_midi
from .njam_v3 import NJamDocument


def _render_sine(document: NJamDocument, sample_rate: int = 22050) -> np.ndarray:
    end_tick = max(
        event.time + getattr(event, "duration", 0)
        for event in document.events
    )
    tempo = float(document.metadata.get("tempo", "120.0"))
    seconds_per_tick = (60.0 / tempo) / document.ppq
    total_seconds = max(1.0, end_tick * seconds_per_tick + 0.5)
    sample_count = int(total_seconds * sample_rate)
    audio = np.zeros(sample_count, dtype=np.float32)
    for event in document.events:
        if not hasattr(event, "duration"):
            continue
        start_sample = int(event.time * seconds_per_tick * sample_rate)
        end_sample = int((event.time + event.duration) * seconds_per_tick * sample_rate)
        freq = 440.0 * (2.0 ** ((event.pitch - 69) / 12.0))
        amp = event.velocity / 127.0 * 0.15
        for idx in range(start_sample, min(end_sample, sample_count)):
            t = idx / sample_rate
            audio[idx] += amp * math.sin(2.0 * math.pi * freq * t)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0:
        audio /= peak
    return audio


def render_document_audio(
    document: NJamDocument,
    output_path: Path,
    soundfont_path: Optional[Path] = None,
    sample_rate: int = 22050,
) -> None:
    midi = njam_to_midi(document)
    if soundfont_path is not None:
        assert soundfont_path.exists(), f"Soundfont does not exist: {soundfont_path}"
        pretty = pretty_midi.PrettyMIDI()
        temp_midi = output_path.with_suffix(".mid")
        midi.save(str(temp_midi))
        pretty = pretty_midi.PrettyMIDI(str(temp_midi))
        audio = pretty.fluidsynth(fs=sample_rate, sf2_path=str(soundfont_path))
    else:
        audio = _render_sine(document, sample_rate=sample_rate)
    pcm = np.asarray(np.clip(audio, -1.0, 1.0) * 32767.0, dtype=np.int16)
    with wave.open(str(output_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(array("h", pcm.tolist()).tobytes())

