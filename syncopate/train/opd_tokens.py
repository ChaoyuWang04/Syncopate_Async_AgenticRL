"""OPD sampled-token identity and ByteLevel alignment (CPU-only).

This deliberately supports the current text-only ByteLevel tokenizer, not arbitrary
HF decoders.  Token IDs are never recovered by encoding decoded text.  The byte
alphabet is the canonical GPT-2/ByteLevel bijection; its set and each used token's
decode are checked against the installed public Tokenizers API.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


def positions_from_mask(mask: tuple[int, ...]) -> tuple[int, ...]:
    """Current GenerationMixin text-only positions: cumsum-1, padding set to 0."""
    if not mask or any(type(x) is not int or x not in (0, 1) for x in mask):
        raise ValueError("attention mask must be a nonempty binary sequence")
    if not any(mask):
        raise ValueError("prompt must contain nonempty unpadded input")
    if tuple(sorted(mask)) != mask:
        raise ValueError("OPD generation requires left padding")
    current = -1
    result = []
    for active in mask:
        current += active
        result.append(current if active else 0)
    return tuple(result)


def _ids(values, name: str) -> tuple[int, ...]:
    values = tuple(values)
    if any(type(x) is not int or x < 0 for x in values):
        raise ValueError(f"{name} must contain nonnegative integer IDs")
    return values


def _response_ids(prompt_ids, generated_ids, eos_token_ids, pad_token_id):
    if generated_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("generate changed its input prefix")
    response = generated_ids[len(prompt_ids):]
    stop = next((i + 1 for i, token in enumerate(response) if token in eos_token_ids), len(response))
    if any(token != pad_token_id for token in response[stop:]):
        raise ValueError("only batch padding may follow the first generated EOS")
    return response[:stop]


def _utf8_units(payload: bytes):
    """Return exact byte ranges per decoded scalar, including replace-error ranges.

    UnicodeDecodeError's start/end delimit exactly the bytes consumed by the
    replacement handler.  A valid character may span multiple original tokens.
    A literal, valid U+FFFD is not confused with an invalid-byte replacement.
    """
    units, invalid = [], []
    offset = 0
    while offset < len(payload):
        tail = payload[offset:]
        try:
            valid = tail.decode("utf-8", errors="strict")
            error = None
        except UnicodeDecodeError as exc:
            valid = tail[:exc.start].decode("utf-8", errors="strict")
            error = exc
        cursor = offset
        for char in valid:
            code = ord(char)
            width = 1 if code < 0x80 else 2 if code < 0x800 else 3 if code < 0x10000 else 4
            units.append((cursor, cursor + width, char, False))
            cursor += width
        if error is None:
            break
        start, end = offset + error.start, offset + error.end
        if end <= start:
            raise ValueError("UTF-8 decoder did not locate its invalid byte range")
        units.append((start, end, "\ufffd", True))
        invalid.append((start, end))
        offset = end
    return units, tuple(invalid)


@dataclass(frozen=True)
class TokenAlignment:
    text: str
    text_with_special_tokens: str
    token_bytes: tuple[str, ...]
    special_tokens_mask: tuple[int, ...]
    byte_spans: tuple[tuple[int, int], ...]
    labels: tuple[str, ...]
    warnings: tuple[str, ...]
    invalid_utf8_spans: tuple[tuple[int, int], ...]
    implicit_think_open: bool


def align_token_bytes(token_bytes: tuple[str, ...], special_tokens_mask: tuple[int, ...], *,
                      implicit_think_open: bool) -> TokenAlignment:
    """Align the existing v15 character labels to *original* token byte spans."""
    from syncopate.train.opd_render import v15_char_labels

    if len(token_bytes) != len(special_tokens_mask):
        raise ValueError("alignment length mismatch")
    if any(type(x) is not int or x not in (0, 1) for x in special_tokens_mask):
        raise ValueError("invalid special-token mask")
    try:
        pieces = [bytes.fromhex(piece) for piece in token_bytes]
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid token byte identity") from exc
    visible = b"".join(piece for piece, special in zip(pieces, special_tokens_mask) if not special)
    text = visible.decode("utf-8", errors="replace")
    full_text = b"".join(pieces).decode("utf-8", errors="replace")
    warnings = []
    try:
        units, invalid = _utf8_units(visible)
        if "".join(unit[2] for unit in units) != text:
            raise ValueError("UTF-8 scalar reconstruction differs from decode")
    except ValueError:
        # Decoding still succeeded, but its exact ranges are not proven.  Never
        # invent labels from the visible replacement string in this situation.
        units, invalid = [], ((0, len(visible)),) if visible else ()
        warnings.append("utf8_alignment_unlocated")
    if invalid:
        warnings.append("invalid_utf8")
    char_labels = v15_char_labels(text, implicit_think_open=implicit_think_open)
    byte_labels = ["format"] * len(visible)
    for (start, end, _char, uncertain), label in zip(units, char_labels):
        byte_labels[start:end] = ["format" if uncertain else label] * (end - start)
    spans, labels, cursor = [], [], 0
    for piece, special in zip(pieces, special_tokens_mask):
        end = cursor if special else cursor + len(piece)
        spans.append((cursor, end))
        span = set(byte_labels[cursor:end])
        if special or not span:
            labels.append("format")
        elif len(span) == 1:
            labels.append(next(iter(span)))
        else:
            # In particular: a token containing both </think> and answer text
            # cannot become a target merely because most characters are prose.
            labels.append("format")
            warnings.append("mixed_label_span")
        cursor = end
    return TokenAlignment(text, full_text, token_bytes, special_tokens_mask,
        tuple(spans), tuple(labels), tuple(warnings), invalid, bool(implicit_think_open))


class ByteLevelTokenMapper:
    """Validated public backend lookup; no private byte_decoder or encode fallback."""

    def __init__(self, tokenizer):
        from tokenizers.pre_tokenizers import ByteLevel

        self.tokenizer = tokenizer
        self.backend = getattr(tokenizer, "backend_tokenizer", None)
        if self.backend is None:
            raise ValueError("OPD requires a fast ByteLevel tokenizer backend")
        decoder = json.loads(self.backend.to_str()).get("decoder")
        if not isinstance(decoder, dict) or decoder.get("type") != "ByteLevel":
            raise ValueError("unsupported OPD decoder: expected ByteLevel")
        kept = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        remaining = [byte for byte in range(256) if byte not in kept]
        self.byte_decoder = dict(zip(
            [chr(byte) for byte in kept] + [chr(256 + i) for i in range(len(remaining))],
            kept + remaining,
        ))
        # alphabet() is unordered: never enumerate it to assign byte values.
        if set(ByteLevel.alphabet()) != set(self.byte_decoder):
            raise ValueError("installed ByteLevel alphabet changed")
        self.added_tokens = self.backend.get_added_tokens_decoder()
        self.special_ids = frozenset(tokenizer.all_special_ids) | frozenset(
            token_id for token_id, token in self.added_tokens.items() if token.special)
        self._cache: dict[int, str] = {}

    def evidence(self) -> dict:
        return {"backend_sha256": hashlib.sha256(self.backend.to_str().encode()).hexdigest(),
            "byte_mapping_sha256": hashlib.sha256(json.dumps(self.byte_decoder, sort_keys=True).encode()).hexdigest(),
            "vocab_size": self.backend.get_vocab_size(with_added_tokens=True),
            "hf_special_ids": sorted(self.tokenizer.all_special_ids),
            "backend_special_ids": sorted(token_id for token_id, token in self.added_tokens.items() if token.special),
            "special_ids_union": sorted(self.special_ids)}

    def token_bytes(self, token_id: int) -> str:
        if token_id in self._cache:
            return self._cache[token_id]
        token = self.backend.id_to_token(token_id)
        if token is None:
            raise ValueError(f"unknown tokenizer ID: {token_id}")
        added = self.added_tokens.get(token_id)
        if added is not None and added.content != token:
            raise ValueError(f"added-token identity mismatch for ID {token_id}")
        try:
            raw = bytes(self.byte_decoder[char] for char in token)
        except KeyError as exc:
            raise ValueError(f"unsupported ByteLevel vocabulary for ID {token_id}") from exc
        if raw.decode("utf-8", errors="replace") != self.backend.decode([token_id], skip_special_tokens=False):
            raise ValueError(f"token byte/decode mismatch for ID {token_id}")
        self._cache[token_id] = raw.hex()
        return self._cache[token_id]

    def align(self, ids: tuple[int, ...], *, implicit_think_open: bool) -> TokenAlignment:
        ids = _ids(ids, "response")
        alignment = align_token_bytes(tuple(self.token_bytes(token_id) for token_id in ids),
            tuple(int(token_id in self.special_ids) for token_id in ids),
            implicit_think_open=implicit_think_open)
        for skip, expected in ((False, alignment.text_with_special_tokens), (True, alignment.text)):
            backend_text = self.backend.decode(list(ids), skip_special_tokens=skip)
            hf_text = self.tokenizer.decode(list(ids), skip_special_tokens=skip,
                clean_up_tokenization_spaces=False)
            if expected != backend_text or expected != hf_text:
                raise ValueError(f"raw ByteLevel/backend/HF decode identity mismatch (skip_special={skip})")
        return alignment


@dataclass(frozen=True)
class GeneratedSample:
    prompt_ids: tuple[int, ...]
    prompt_attention_mask: tuple[int, ...]
    prompt_position_ids: tuple[int, ...]
    generated_ids: tuple[int, ...]
    response_ids: tuple[int, ...]
    eos_token_ids: tuple[int, ...]
    pad_token_id: int
    alignment: TokenAlignment

    @property
    def response_text(self) -> str:
        return self.alignment.text

    @property
    def input_ids(self) -> tuple[int, ...]:
        return self.prompt_ids + self.response_ids

    @property
    def attention_mask(self) -> tuple[int, ...]:
        return self.prompt_attention_mask + (1,) * len(self.response_ids)

    @property
    def position_ids(self) -> tuple[int, ...]:
        last = self.prompt_position_ids[-1]
        return self.prompt_position_ids + tuple(range(last + 1, last + len(self.response_ids) + 1))

    def validate(self, *, mapper: ByteLevelTokenMapper | None = None) -> None:
        for name in ("prompt_ids", "prompt_position_ids", "generated_ids", "response_ids", "eos_token_ids"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or _ids(values, name) != values:
                raise ValueError(f"{name} must be immutable token IDs")
        if type(self.pad_token_id) is not int or self.pad_token_id < 0:
            raise ValueError("invalid padding token ID")
        if not (len(self.prompt_ids) == len(self.prompt_attention_mask) == len(self.prompt_position_ids)):
            raise ValueError("prompt ID/mask/position length mismatch")
        if self.prompt_position_ids != positions_from_mask(self.prompt_attention_mask):
            raise ValueError("prompt position IDs do not match generation attention mask")
        if self.response_ids != _response_ids(self.prompt_ids, self.generated_ids,
                                             self.eos_token_ids, self.pad_token_id):
            raise ValueError("response ID identity mismatch")
        if not isinstance(self.alignment, TokenAlignment):
            raise ValueError("missing token alignment")
        if any(not isinstance(getattr(self.alignment, field), tuple) for field in
               ("token_bytes", "special_tokens_mask", "byte_spans", "labels", "warnings", "invalid_utf8_spans")):
            raise ValueError("alignment fields must be immutable")
        if len(self.response_ids) != len(self.alignment.labels):
            raise ValueError("response/alignment length mismatch")
        expected = align_token_bytes(self.alignment.token_bytes, self.alignment.special_tokens_mask,
            implicit_think_open=self.alignment.implicit_think_open)
        if self.alignment != expected:
            raise ValueError("stored alignment differs from raw byte spans")
        if mapper is not None:
            # Structural self-consistency is not ID identity.  The independently
            # supplied frozen vocabulary must own the bytes and special flags.
            # generated_ids includes prompt, kept response, and post-EOS padding.
            for token_id in (*self.generated_ids, *self.eos_token_ids, self.pad_token_id):
                mapper.token_bytes(token_id)
            actual = mapper.align(self.response_ids,
                                  implicit_think_open=self.alignment.implicit_think_open)
            if self.alignment != actual:
                raise ValueError("stored alignment does not match the frozen tokenizer's actual IDs")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict, *, mapper: ByteLevelTokenMapper) -> "GeneratedSample":
        if not isinstance(mapper, ByteLevelTokenMapper):
            raise ValueError("deserializing a sample requires an external frozen tokenizer mapper")
        payload = dict(payload)
        alignment = dict(payload.pop("alignment"))
        for key in ("token_bytes", "special_tokens_mask", "labels", "warnings"):
            alignment[key] = tuple(alignment[key])
        for key in ("byte_spans", "invalid_utf8_spans"):
            alignment[key] = tuple(tuple(span) for span in alignment[key])
        for key in ("prompt_ids", "prompt_attention_mask", "prompt_position_ids", "generated_ids",
                    "response_ids", "eos_token_ids"):
            payload[key] = tuple(payload[key])
        sample = cls(**payload, alignment=TokenAlignment(**alignment))
        sample.validate(mapper=mapper)
        return sample


def capture_generated_sample(mapper: ByteLevelTokenMapper, *, prompt_ids,
    prompt_attention_mask, prompt_position_ids, generated_ids, eos_token_ids,
    pad_token_id: int, implicit_think_open: bool) -> GeneratedSample:
    prompt_ids = _ids(prompt_ids, "prompt")
    generated_ids = _ids(generated_ids, "generated")
    eos_token_ids = _ids(eos_token_ids, "eos")
    response_ids = _response_ids(prompt_ids, generated_ids, eos_token_ids, pad_token_id)
    # IDs absent from the tokenizer vocabulary must fail, not silently decode to "".
    for token_id in (*generated_ids, *eos_token_ids, pad_token_id):
        mapper.token_bytes(token_id)
    sample = GeneratedSample(prompt_ids, tuple(prompt_attention_mask), tuple(prompt_position_ids),
        generated_ids, response_ids, eos_token_ids, pad_token_id,
        mapper.align(response_ids, implicit_think_open=implicit_think_open))
    sample.validate()
    return sample


class TokenAuditWriter:
    """One append-only file per run/rank; generation and actual KL call evidence."""

    def __init__(self, out_dir: Path, *, run_token: str, rank: int, runtime: dict):
        self.rank = rank
        self.path = Path(out_dir) / "token_audit" / f"{run_token}.rank{rank}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.counts = Counter()
        self.warning_counts = Counter()
        self.route_forwards = Counter()
        self.run_token = run_token
        with self.path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps({"event": "runtime", "run_token": run_token,
                "rank": rank, "runtime": runtime}, ensure_ascii=False) + "\n")

    def _write(self, row):
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"run_token": self.run_token, "rank": self.rank, **row},
                ensure_ascii=False) + "\n")

    def generation(self, sample: GeneratedSample, *, attempt: int, sample_index: int,
                   turn_index: int, family: str, final_turn: bool) -> str:
        sample.validate()
        self.counts["generated_records"] += 1
        self.counts["final_generation_records"] += int(final_turn)
        self.counts["warning_records"] += int(bool(sample.alignment.warnings))
        self.warning_counts.update(sample.alignment.warnings)
        gen_id = f"{self.rank}:{self.counts['generated_records']}"
        self._write({"event": "generation", "generation_id": gen_id, "attempt": attempt,
            "sample_index": sample_index, "turn_index": turn_index, "family": family,
            "final_turn": final_turn, "sample": sample.to_dict()})
        return gen_id

    def kl(self, generation_id: str, *, attempt: int, family: str, zero_mask: bool, audit: dict):
        prefix = "zero" if zero_mask else "kl"
        self.counts[f"{prefix}_records"] += 1
        for call in ("student_forward", "aux_forward", "backward"):
            self.counts[f"{prefix}_{call}"] += audit[call]
        self.route_forwards[family] += audit["aux_forward"]
        self._write({"event": "kl", "generation_id": generation_id, "attempt": attempt,
            "family": family, "zero_mask": zero_mask, "audit": audit})

    def summary(self) -> dict:
        return {"rank": self.rank, "path": str(self.path), **dict(self.counts),
            "warning_counts": dict(self.warning_counts), "route_forwards": dict(self.route_forwards)}
