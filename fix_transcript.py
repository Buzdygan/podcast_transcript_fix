#!/usr/bin/env python3
"""
Correct a raw AI/ASR podcast transcript with Gemini (default: gemini-2.5-flash).

Splits long transcripts into chunks with continuity context from the prior chunk,
then runs an optional merge pass to smooth section boundaries.

Usage:
    python fix_transcript.py episode.txt
    python fix_transcript.py episode.txt -o corrected.txt
    python fix_transcript.py episode.txt --model gemini-2.5-flash --chunk-size 12000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_CHUNK_SIZE = 10_000
DEFAULT_CONTEXT_CHARS = 500
# Single merge call; skip merge above this to avoid huge requests (chunks still help).
MERGE_MAX_CHARS = 280_000

SYSTEM_CORRECT = """\
You are an editor polishing an automatic transcription of a spoken podcast episode.

Goals:
- Add clear sentence breaks, commas, and question marks where natural.
- Start new paragraphs when the topic or speaker focus clearly shifts.
- Fix obvious spelling and grammar mistakes and likely proper-noun errors, using \
context and general knowledge; do not invent facts or change what was said.
- Keep the voice conversational; do not turn it into formal essay prose.
- Do not add labels like "Host:" or "Guest:" unless they already appear in the text.
- Output only the corrected transcript text for this segment — no preamble or markdown.
"""

USER_CORRECT_TEMPLATE = """\
{context_block}--- TRANSCRIPT SEGMENT TO CORRECT ---
{segment}
--- END SEGMENT ---

Return only the corrected text for this segment.{context_note}"""

CONTEXT_BLOCK = """\
For continuity, here is the end of the immediately preceding corrected segment \
(do not repeat it; use it only to keep names, terms, and flow consistent):

\"\"\"
{tail}
\"\"\"

"""

SYSTEM_MERGE = """\
You are editing a podcast transcript that was corrected in separate sections.

Smooth awkward joins between those sections, remove accidental duplicate sentences \
that appear exactly at boundaries, and fix words split across former breaks. \
Preserve all substantive content and meaning; do not summarize or shorten.

Output only the full merged transcript — no commentary or markdown.
"""

USER_MERGE_TEMPLATE = """\
--- MERGED SECTIONS (may have rough joins) ---
{text}
--- END ---

Return the single smoothed transcript."""


def create_client() -> genai.Client:
    import os

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print(
            "Error: No API key found. Set GOOGLE_API_KEY or GEMINI_API_KEY "
            "as an environment variable or in a .env file.",
            file=sys.stderr,
        )
        sys.exit(1)
    return genai.Client(api_key=api_key)


def find_chunk_split_index(text: str, start: int, target_end: int) -> int:
    """Pick an end index <= target_end, preferring paragraph/sentence boundaries."""
    n = len(text)
    max_end = min(target_end, n)
    if max_end >= n:
        return n
    window_start = start
    # Search backward from max_end within a reasonable tail window.
    scan_lo = max(window_start, max_end - 2000)
    for sep in ("\n\n", "\n", ". ", "? ", "! ", " "):
        idx = text.rfind(sep, scan_lo, max_end)
        if idx != -1:
            return idx + len(sep)
    return max_end


def generate_text(
    client: genai.Client,
    model: str,
    system_instruction: str,
    user_text: str,
    temperature: float,
) -> str:
    response = client.models.generate_content(
        model=model,
        contents=user_text,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
        ),
    )
    if not response.text:
        print("Error: Empty response from model.", file=sys.stderr)
        sys.exit(1)
    return response.text.strip()


def correct_segment(
    client: genai.Client,
    model: str,
    segment: str,
    prior_tail: str,
) -> str:
    context_block = ""
    context_note = ""
    if prior_tail.strip():
        context_block = CONTEXT_BLOCK.format(tail=prior_tail.strip())
        context_note = (
            " Match style and terminology with the preceding context where relevant."
        )
    user = USER_CORRECT_TEMPLATE.format(
        context_block=context_block,
        segment=segment,
        context_note=context_note,
    )
    return generate_text(
        client, model, SYSTEM_CORRECT, user, temperature=0.25
    )


def merge_sections(client: genai.Client, model: str, body: str) -> str:
    user = USER_MERGE_TEMPLATE.format(text=body)
    return generate_text(client, model, SYSTEM_MERGE, user, temperature=0.15)


def iter_raw_chunks(text: str, chunk_size: int) -> list[tuple[int, int]]:
    """Non-overlapping [start, end) spans covering text."""
    spans: list[tuple[int, int]] = []
    start = 0
    n = len(text)
    while start < n:
        end = find_chunk_split_index(text, start, start + chunk_size)
        if end <= start:
            end = min(start + chunk_size, n)
        spans.append((start, end))
        start = end
    return spans


def process_transcript(
    client: genai.Client,
    raw: str,
    model: str,
    chunk_size: int,
    context_chars: int,
    do_merge: bool,
) -> str:
    raw = raw.strip()
    if not raw:
        return ""

    spans = iter_raw_chunks(raw, chunk_size)
    corrected_parts: list[str] = []
    prior_tail = ""

    for i, (a, b) in enumerate(spans):
        segment = raw[a:b]
        print(f"  Chunk {i + 1}/{len(spans)} ({len(segment)} chars)...", file=sys.stderr)
        fixed = correct_segment(client, model, segment, prior_tail)
        corrected_parts.append(fixed)
        prior_tail = fixed[-context_chars:] if context_chars and fixed else ""

    merged = "\n\n".join(corrected_parts)

    if do_merge and len(merged) <= MERGE_MAX_CHARS:
        print("  Running merge pass...", file=sys.stderr)
        merged = merge_sections(client, model, merged)
    elif do_merge and len(merged) > MERGE_MAX_CHARS:
        print(
            f"  Skipping merge pass (length {len(merged)} > limit {MERGE_MAX_CHARS}).",
            file=sys.stderr,
        )

    return merged.strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Correct a podcast ASR transcript using Gemini."
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to a transcript file (plain text)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write result here instead of stdout",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Gemini model id (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Target max characters per API chunk (default: {DEFAULT_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--context-chars",
        type=int,
        default=DEFAULT_CONTEXT_CHARS,
        help="Tail of prior corrected text passed as continuity context "
        f"(default: {DEFAULT_CONTEXT_CHARS})",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Skip the final boundary-smoothing pass",
    )
    args = parser.parse_args()

    path = args.input_file
    if not path.is_file():
        print(f"Error: not a file: {path}", file=sys.stderr)
        sys.exit(1)

    raw = path.read_text(encoding="utf-8")
    print(f"Processing {path} ({len(raw)} chars)...", file=sys.stderr)
    client = create_client()
    result = process_transcript(
        client,
        raw,
        args.model,
        args.chunk_size,
        args.context_chars,
        do_merge=not args.no_merge,
    )

    if args.output:
        args.output.write_text(result + ("\n" if result and not result.endswith("\n") else ""), encoding="utf-8")
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(result)
        if result and not result.endswith("\n"):
            sys.stdout.write("\n")


if __name__ == "__main__":
    main()
