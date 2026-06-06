"""Language adapters for NJam text formats."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Protocol

from .njam_v3 import ContinuationRecoveryStats, NJamDocument


class MusicLanguage(Protocol):
    name: str
    header_prefix: str

    def encode_document(self, document: NJamDocument) -> str: ...
    def parse_document(self, text: str) -> NJamDocument: ...
    def body_text(self, text: str) -> str: ...
    def header_text(self, text: str) -> str: ...
    def extract_header_metadata(self, text: str) -> Dict[str, str]: ...
    def recover_continuation_document(self, text: str, metadata: Dict[str, str] | None = None) -> NJamDocument | None: ...
    def analyze_parseable_continuation(self, text: str) -> ContinuationRecoveryStats: ...
    def document_to_midi(self, document: NJamDocument, max_note_seconds: float | None = None): ...
    def midi_to_document(self, path: Path) -> NJamDocument: ...
    def weimar_to_document(self, solo, ppq: int) -> NJamDocument: ...
    def transpose_document(self, document: NJamDocument, semitones: int) -> NJamDocument: ...


@dataclass(frozen=True)
class NJamV3Language:
    name: str = "njam-v3"
    header_prefix: str = "NV3|"

    def encode_document(self, document: NJamDocument) -> str:
        from . import njam_v3

        return njam_v3.encode_document(document)

    def parse_document(self, text: str) -> NJamDocument:
        from . import njam_v3

        return njam_v3.parse_document(text)

    def body_text(self, text: str) -> str:
        stripped = text.strip()
        if not stripped.startswith(self.header_prefix):
            return stripped
        lines = [line.strip() for line in stripped.splitlines() if line.strip()]
        if len(lines) >= 2:
            return "\n".join(lines[1:]).strip()
        import re

        body_match = re.search(r"\sT[-0-9A-Z]+", stripped)
        assert body_match is not None, "NJam text must contain body tokens after the header."
        return stripped[body_match.start() :].strip()

    def header_text(self, text: str) -> str:
        stripped = text.strip()
        if not stripped.startswith(self.header_prefix):
            return ""
        lines = [line.strip() for line in stripped.splitlines() if line.strip()]
        if lines:
            return lines[0]
        import re

        body_match = re.search(r"\sT[-0-9A-Z]+", stripped)
        assert body_match is not None, "NJam text must contain body tokens after the header."
        return stripped[: body_match.start()].strip()

    def extract_header_metadata(self, text: str) -> Dict[str, str]:
        from . import njam_v3

        return njam_v3.extract_header_metadata(text)

    def recover_continuation_document(self, text: str, metadata: Dict[str, str] | None = None) -> NJamDocument | None:
        from . import njam_v3

        return njam_v3.recover_continuation_document(text, metadata=metadata)

    def analyze_parseable_continuation(self, text: str) -> ContinuationRecoveryStats:
        from . import njam_v3

        return njam_v3.analyze_parseable_continuation(text)

    def document_to_midi(self, document: NJamDocument, max_note_seconds: float | None = None):
        from .midi_tools import njam_to_midi

        return njam_to_midi(document, max_note_seconds=max_note_seconds)

    def midi_to_document(self, path: Path) -> NJamDocument:
        from .midi_tools import midi_to_njam

        return midi_to_njam(path, include_program_changes=False)

    def weimar_to_document(self, solo, ppq: int) -> NJamDocument:
        from .weimar_db import weimar_to_njam

        return weimar_to_njam(solo, ppq=ppq)

    def transpose_document(self, document: NJamDocument, semitones: int) -> NJamDocument:
        from .weimar_db import transpose_document

        return transpose_document(document, semitones)


@dataclass(frozen=True)
class NJamV2Language:
    name: str = "njam-v2"
    header_prefix: str = "NV2|"

    def encode_document(self, document: NJamDocument) -> str:
        from . import njam_v2

        return njam_v2.encode_document(document)

    def parse_document(self, text: str) -> NJamDocument:
        from . import njam_v2

        return njam_v2.parse_document(text)

    def body_text(self, text: str) -> str:
        from . import njam_v2

        return njam_v2.body_text(text)

    def header_text(self, text: str) -> str:
        from . import njam_v2

        return njam_v2.header_text(text)

    def extract_header_metadata(self, text: str) -> Dict[str, str]:
        from . import njam_v2

        return njam_v2.extract_header_metadata(text)

    def recover_continuation_document(self, text: str, metadata: Dict[str, str] | None = None) -> NJamDocument | None:
        from . import njam_v2

        return njam_v2.recover_continuation_document(text, metadata=metadata)

    def analyze_parseable_continuation(self, text: str) -> ContinuationRecoveryStats:
        from . import njam_v2

        return njam_v2.analyze_parseable_continuation(text)

    def document_to_midi(self, document: NJamDocument, max_note_seconds: float | None = None):
        from .midi_tools import njam_to_midi

        return njam_to_midi(document, max_note_seconds=max_note_seconds)

    def midi_to_document(self, path: Path) -> NJamDocument:
        from .midi_tools import midi_to_njam

        document = midi_to_njam(path, include_program_changes=True)
        document.metadata.setdefault("ppq", "960")
        return document

    def weimar_to_document(self, solo, ppq: int) -> NJamDocument:
        from .weimar_db import weimar_to_njam

        return weimar_to_njam(solo, ppq=ppq)

    def transpose_document(self, document: NJamDocument, semitones: int) -> NJamDocument:
        from .weimar_db import transpose_document

        return transpose_document(document, semitones)


@dataclass(frozen=True)
class NJamV4Language:
    name: str = "njam-v4"
    header_prefix: str = "NV4|"

    def encode_document(self, document: NJamDocument) -> str:
        from . import njam_v4

        return njam_v4.encode_document(document)

    def parse_document(self, text: str) -> NJamDocument:
        from . import njam_v4

        return njam_v4.parse_document(text)

    def body_text(self, text: str) -> str:
        from . import njam_v4

        return njam_v4.body_text(text)

    def header_text(self, text: str) -> str:
        from . import njam_v4

        return njam_v4.header_text(text)

    def extract_header_metadata(self, text: str) -> Dict[str, str]:
        from . import njam_v4

        return njam_v4.extract_header_metadata(text)

    def recover_continuation_document(self, text: str, metadata: Dict[str, str] | None = None) -> NJamDocument | None:
        from . import njam_v4

        return njam_v4.recover_continuation_document(text, metadata=metadata)

    def analyze_parseable_continuation(self, text: str) -> ContinuationRecoveryStats:
        from . import njam_v4

        return njam_v4.analyze_parseable_continuation(text)

    def document_to_midi(self, document: NJamDocument, max_note_seconds: float | None = None):
        from .midi_tools import njam_to_midi

        return njam_to_midi(document, max_note_seconds=max_note_seconds)

    def midi_to_document(self, path: Path) -> NJamDocument:
        from .midi_tools import midi_to_njam

        document = midi_to_njam(path, include_program_changes=True)
        document.metadata.setdefault("ppq", "96")
        return document

    def weimar_to_document(self, solo, ppq: int) -> NJamDocument:
        from .weimar_db import weimar_to_njam

        return weimar_to_njam(solo, ppq=ppq)

    def transpose_document(self, document: NJamDocument, semitones: int) -> NJamDocument:
        from .weimar_db import transpose_document

        return transpose_document(document, semitones)

    def tokenizer_model_type(self) -> str:
        return "bpe"

    def tokenizer_train_kwargs(self) -> Dict[str, object]:
        return {
            "normalization_rule_name": "identity",
            "character_coverage": 1.0,
            "byte_fallback": False,
            "split_by_whitespace": False,
            "user_defined_symbols": ["\n"],
        }

    def tokenizer_seed_texts(self) -> list[str]:
        from . import njam_v4

        return [njam_v4.base_vocabulary_seed_text(include_controls=True)]

    def loss_mask_token_ids(self, tokenizer) -> set[int]:
        from . import njam_v4

        control_chars = njam_v4.control_token_chars()
        return {
            idx
            for idx in range(tokenizer.vocab_size)
            if any(ch in tokenizer.processor.id_to_piece(idx) for ch in control_chars)
        }


LANGUAGES: Dict[str, MusicLanguage] = {
    "njam-v3": NJamV3Language(),
    "njam-v2": NJamV2Language(),
    "njam-v4": NJamV4Language(),
}


def get_language(name: str = "njam-v3") -> MusicLanguage:
    normalized = name.strip().lower()
    assert normalized in LANGUAGES, f"Unsupported NJam language {name!r}. Expected one of: {', '.join(sorted(LANGUAGES))}"
    return LANGUAGES[normalized]


def detect_language(text: str, default: str | None = "njam-v3") -> MusicLanguage:
    stripped = text.lstrip()
    if stripped.startswith("NV2|"):
        return get_language("njam-v2")
    if stripped.startswith("NV3|"):
        return get_language("njam-v3")
    if stripped.startswith("NV4|"):
        return get_language("njam-v4")
    assert default is not None, "Cannot detect NJam language from text without NV2|, NV3|, or NV4| header."
    return get_language(default)
