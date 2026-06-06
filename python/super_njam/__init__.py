"""Super NJam package."""

from .njam_v3 import (
    DEFAULT_PPQ,
    ControlChangeEvent,
    NJamDocument,
    NoteEvent,
    PitchBendEvent,
    ProgramChangeEvent,
    encode_document,
    parse_document,
)
from .music_language import detect_language, get_language
