Here’s a **coder-facing spec** for the Weimar Jazz Database SQLite file. I’ve written it so you can drop it straight into a coding agent prompt or README.

---

# 🧾 Weimar Jazz Database (`wjazzd.db`) – Developer Specification

## 1. Overview

The database is a **SQLite3 relational dataset** containing annotated jazz solos.

* Each **solo** is represented as a sequence of **note events**
* Events are stored **row-by-row** in a central table
* Additional tables provide **metadata**, **timing**, and **structure**

👉 Core concept:

```
Solo = ordered sequence of note events (linked by melid)
```

---

## 2. Key Tables (what matters in practice)

### 🎵 `melody` (MAIN TABLE — you will use this most)

Each row = one musical event (note or rest)

Typical important fields (names may vary slightly depending on version):

```sql
melid        -- unique melody/solo ID
eventid      -- ordering index within solo
pitch        -- MIDI pitch (integer)
onset        -- onset time (beats or seconds)
duration     -- duration (same unit as onset)
velocity     -- optional (sometimes present)
channel      -- optional
```

👉 This is your **primary data source for reconstructing solos**

---

### 🧑‍🎤 `solo_info`

Metadata about each solo:

```sql
melid
performer
instrument
title
recording_id
```

---

### 💿 `record_info`

Recording-level metadata:

```sql
recording_id
artist
album
year
```

---

### 🎼 `composition_info`

Composition metadata:

```sql
composition_id
title
composer
```

---

### ⏱ `beats`

Timing grid:

```sql
melid
beat_index
time
```

Used for:

* aligning notes to beats
* tempo-aware processing

---

### 🧱 `sections`

Structural annotations:

```sql
melid
section_name   -- e.g. A, B, chorus
start_time
end_time
```

---

## 3. How to Extract a Solo (CORE TASK)

### Step 1: Select a solo

```sql
SELECT DISTINCT melid FROM melody;
```

Pick one `melid`.

---

### Step 2: Get ordered note events

```sql
SELECT *
FROM melody
WHERE melid = ?
ORDER BY eventid ASC;
```

👉 **Ordering is critical** — always sort by `eventid`

---

### Step 3: Interpret rows as notes

Each row → one note:

```python
note = {
    "pitch": row["pitch"],          # MIDI note number
    "start": row["onset"],          # time
    "duration": row["duration"]
}
```

---

## 4. Time Representation (IMPORTANT)

The dataset may use:

* beats (common)
* or seconds (less common depending on export)

👉 You MUST check:

```sql
SELECT onset, duration FROM melody LIMIT 10;
```

If values look like:

* small decimals → likely seconds
* integers / musically aligned → likely beats

---

## 5. Converting to MIDI

### Basic conversion

```python
import pretty_midi

pm = pretty_midi.PrettyMIDI()
inst = pretty_midi.Instrument(program=0)

for row in melody_rows:
    note = pretty_midi.Note(
        velocity=100,
        pitch=row["pitch"],
        start=row["onset"],
        end=row["onset"] + row["duration"]
    )
    inst.notes.append(note)

pm.instruments.append(inst)
pm.write("solo.mid")
```

---

### If times are in beats

You need tempo:

```python
seconds = beats * (60.0 / bpm)
```

👉 BPM can be inferred from:

* `beats` table
* or assumed (e.g. 120 if not critical)

---

## 6. Converting to Your NJAM Format (for your pipeline)

Given your setup (LLM + MIDI → njam):

### Suggested encoding

```python
njam_token = f"p{pitch}_d{duration}_t{delta}"
```

Where:

* `pitch` = MIDI pitch
* `duration` = quantised duration
* `delta` = time since previous note

---

### Example conversion

```python
prev_onset = 0

for row in melody_rows:
    delta = row["onset"] - prev_onset

    token = f"p{row['pitch']}_d{row['duration']}_t{delta}"
    sequence.append(token)

    prev_onset = row["onset"]
```

---

## 7. Handling Edge Cases

### ⚠️ Rests

* Not always explicit
* Must be inferred from gaps between notes

```python
if next_onset > current_end:
    rest = next_onset - current_end
```

---

### ⚠️ Overlapping notes

* Possible in some solos
* Decide whether to:

  * keep polyphony
  * or flatten to monophonic

---

### ⚠️ Missing velocity

* Default to 100 if absent

---

## 8. Useful Queries

### List solos with metadata

```sql
SELECT s.melid, s.performer, s.title
FROM solo_info s
LIMIT 10;
```

---

### Get full solo + metadata

```sql
SELECT m.*, s.performer, s.title
FROM melody m
JOIN solo_info s ON m.melid = s.melid
WHERE m.melid = ?
ORDER BY m.eventid;
```

---

### Count notes in a solo

```sql
SELECT COUNT(*)
FROM melody
WHERE melid = ?;
```

---

## 9. Mental Model for Agent

Think of the dataset as:

```
Database
 └── Solo (melid)
      └── Ordered events (melody rows)
           └── Note = (pitch, onset, duration)
```

---

## 10. Minimal Extraction Algorithm

```python
def extract_solo(db_path, melid):
    import sqlite3

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT pitch, onset, duration
        FROM melody
        WHERE melid = ?
        ORDER BY eventid ASC
    """, (melid,))

    rows = cursor.fetchall()

    notes = []
    for pitch, onset, duration in rows:
        notes.append({
            "pitch": pitch,
            "start": onset,
            "duration": duration
        })

    return notes
```

---

## 🚀 Summary (for your coding agent)

* Database is **SQLite**
* Main table = `melody`
* Each row = **one note event**
* Group by `melid` → one solo
* Sort by `eventid` → correct order
* Convert `(pitch, onset, duration)` → MIDI or tokens

---

If you want next step, I can:

* map this directly into your **LLM training format**
* or optimise the representation so it tokenises efficiently (given your earlier tokenizer constraints)

