"""Weimar database access and NJamV3 conversion."""

from __future__ import annotations

import bisect
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .njam_v3 import (
    ControlChangeEvent,
    DEFAULT_PPQ,
    NJamDocument,
    NoteEvent,
    PitchBendEvent,
    encode_document,
)

LOUD_MED_MIN = -66.2335977453
LOUD_MED_MAX = 80.3675539194


@dataclass(frozen=True)
class BeatRow:
    onset: float
    bar: int
    beat: int
    signature: str
    chord: str
    form: str


@dataclass(frozen=True)
class MelodyRow:
    eventid: int
    onset: float
    pitch: float
    duration: float
    bar: int
    beat: int
    beatdur: float
    f0_mod: str
    f0_range: Optional[float]
    loud_med: Optional[float]
    loud_max: Optional[float]


@dataclass(frozen=True)
class SoloMetadata:
    melid: int
    performer: str
    title: str
    instrument: str
    avgtempo: float
    signature: str
    style: str
    key: str


@dataclass(frozen=True)
class WeimarSolo:
    metadata: SoloMetadata
    beats: List[BeatRow]
    notes: List[MelodyRow]


def _connect(db_path: Path) -> sqlite3.Connection:
    assert db_path.exists(), f"Database does not exist: {db_path}"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def list_melids(db_path: Path, limit: Optional[int] = None) -> List[int]:
    with _connect(db_path) as conn:
        sql = "SELECT DISTINCT melid FROM melody ORDER BY melid"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        return [int(row[0]) for row in conn.execute(sql)]


def load_solo(db_path: Path, melid: int) -> WeimarSolo:
    with _connect(db_path) as conn:
        solo_row = conn.execute(
            """
            SELECT melid, performer, title, instrument, avgtempo, signature, style, key
            FROM solo_info
            WHERE melid = ?
            """,
            (melid,),
        ).fetchone()
        assert solo_row is not None, f"Unknown melid {melid}"
        beat_rows = conn.execute(
            """
            SELECT onset, bar, beat, signature, chord, form
            FROM beats
            WHERE melid = ?
            ORDER BY onset ASC
            """,
            (melid,),
        ).fetchall()
        note_rows = conn.execute(
            """
            SELECT eventid, onset, pitch, duration, bar, beat, beatdur, f0_mod, f0_range, loud_med, loud_max
            FROM melody
            WHERE melid = ?
            ORDER BY eventid ASC
            """,
            (melid,),
        ).fetchall()
    assert beat_rows, f"No beat rows found for melid {melid}"
    assert note_rows, f"No melody rows found for melid {melid}"
    beats = [
        BeatRow(
            onset=float(row["onset"]),
            bar=int(row["bar"]),
            beat=int(row["beat"]),
            signature=row["signature"] or "",
            chord=row["chord"] or "",
            form=row["form"] or "",
        )
        for row in beat_rows
    ]
    notes = [
        MelodyRow(
            eventid=int(row["eventid"]),
            onset=float(row["onset"]),
            pitch=float(row["pitch"]),
            duration=float(row["duration"]),
            bar=int(row["bar"]),
            beat=int(row["beat"]),
            beatdur=float(row["beatdur"]) if row["beatdur"] is not None else 0.0,
            f0_mod=(row["f0_mod"] or "").strip(),
            f0_range=float(row["f0_range"]) if row["f0_range"] is not None else None,
            loud_med=float(row["loud_med"]) if row["loud_med"] is not None else None,
            loud_max=float(row["loud_max"]) if row["loud_max"] is not None else None,
        )
        for row in note_rows
    ]
    metadata = SoloMetadata(
        melid=int(solo_row["melid"]),
        performer=solo_row["performer"] or "",
        title=solo_row["title"] or "",
        instrument=solo_row["instrument"] or "",
        avgtempo=float(solo_row["avgtempo"]) if solo_row["avgtempo"] is not None else 120.0,
        signature=solo_row["signature"] or "4/4",
        style=solo_row["style"] or "",
        key=solo_row["key"] or "",
    )
    return WeimarSolo(metadata=metadata, beats=beats, notes=notes)


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def loudness_to_velocity(value: Optional[float]) -> int:
    if value is None:
        return 96
    ratio = (value - LOUD_MED_MIN) / (LOUD_MED_MAX - LOUD_MED_MIN)
    velocity = int(round(24 + ratio * (127 - 24)))
    return _clamp(velocity, 1, 127)


def f0_range_to_bend(value: Optional[float]) -> int:
    if value is None or value <= 0:
        return 0
    # The dataset contains large outliers, so compress aggressively.
    normalized = min(value / 300.0, 1.0)
    return int(round(normalized * 4096))


def _compute_time_to_tick(beats: Sequence[BeatRow], ppq: int) -> Tuple[List[float], List[int]]:
    beat_onsets = [beat.onset for beat in beats]
    beat_ticks = [idx * ppq for idx in range(len(beats))]
    return beat_onsets, beat_ticks


def seconds_to_ticks(
    time_seconds: float,
    beat_onsets: Sequence[float],
    beat_ticks: Sequence[int],
    ppq: int,
) -> int:
    idx = bisect.bisect_right(beat_onsets, time_seconds) - 1
    if idx < 0:
        idx = 0
    if idx + 1 < len(beat_onsets):
        span = beat_onsets[idx + 1] - beat_onsets[idx]
    elif len(beat_onsets) > 1:
        span = beat_onsets[idx] - beat_onsets[idx - 1]
    else:
        span = 60.0 / 120.0
    span = max(span, 1e-6)
    local_ratio = (time_seconds - beat_onsets[idx]) / span
    return int(round(beat_ticks[idx] + local_ratio * ppq))


def weimar_to_njam(solo: WeimarSolo, ppq: int = DEFAULT_PPQ) -> NJamDocument:
    beat_onsets, beat_ticks = _compute_time_to_tick(solo.beats, ppq)
    events = []
    for row in solo.notes:
        start_tick = max(0, seconds_to_ticks(row.onset, beat_onsets, beat_ticks, ppq))
        duration_ticks = max(1, int(round((row.duration / max(row.beatdur, 1e-6)) * ppq)))
        pitch = _clamp(int(round(row.pitch)), 0, 127)
        velocity = loudness_to_velocity(row.loud_med if row.loud_med is not None else row.loud_max)
        events.append(NoteEvent(time=start_tick, pitch=pitch, velocity=velocity, duration=duration_ticks))

        if row.loud_max is not None:
            events.append(ControlChangeEvent(time=start_tick, control=11, value=loudness_to_velocity(row.loud_max)))

        modulation = row.f0_mod.lower()
        if modulation == "vibrato":
            events.append(ControlChangeEvent(time=start_tick, control=1, value=96))
            events.append(ControlChangeEvent(time=start_tick + duration_ticks, control=1, value=0))
        elif modulation in {"bend", "slide"}:
            bend_value = f0_range_to_bend(row.f0_range)
            if bend_value:
                events.append(PitchBendEvent(time=start_tick, value=bend_value))
                events.append(PitchBendEvent(time=start_tick + duration_ticks, value=0))
        elif modulation == "fall-off":
            bend_value = -max(1024, f0_range_to_bend(row.f0_range))
            release_tick = max(start_tick, start_tick + duration_ticks - max(1, ppq // 8))
            events.append(PitchBendEvent(time=release_tick, value=bend_value))
            events.append(PitchBendEvent(time=start_tick + duration_ticks, value=0))

    metadata = {
        "ppq": str(ppq),
        "melid": str(solo.metadata.melid),
        "tempo": f"{solo.metadata.avgtempo:.3f}",
        "sig": solo.metadata.signature or "4/4",
        "performer": solo.metadata.performer,
        "title": solo.metadata.title,
        "instrument": solo.metadata.instrument,
        "style": solo.metadata.style,
        "key": solo.metadata.key,
    }
    return NJamDocument(metadata=metadata, events=events)


def export_corpus_jsonl(
    db_path: Path,
    output_path: Path,
    limit: Optional[int] = None,
    ppq: int = DEFAULT_PPQ,
) -> int:
    melids = list_melids(db_path, limit=limit)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for melid in melids:
            solo = load_solo(db_path, melid)
            document = weimar_to_njam(solo, ppq=ppq)
            handle.write(
                json.dumps(
                    {
                        "melid": melid,
                        "performer": solo.metadata.performer,
                        "title": solo.metadata.title,
                        "instrument": solo.metadata.instrument,
                        "tempo": solo.metadata.avgtempo,
                        "signature": solo.metadata.signature,
                        "text": encode_document(document),
                    }
                )
                + "\n"
            )
            count += 1
    return count
