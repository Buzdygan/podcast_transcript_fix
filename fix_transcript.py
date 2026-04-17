#!/usr/bin/env python3
"""
Correct a raw AI/ASR podcast transcript with Gemini (default: gemini-2.5-flash).

Splits long transcripts into chunks with continuity context from the prior chunk,
then runs an optional merge pass to smooth section boundaries.

Usage:
    python fix_transcript.py episode.txt
    # writes output_data/episode-fixed.txt by default
    python fix_transcript.py episode.txt -o custom/path.txt
    python fix_transcript.py episode.txt -o -   # stdout
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError

load_dotenv()

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DATA_DIR = SCRIPT_DIR / "output_data"

DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_CHUNK_SIZE = 10_000
DEFAULT_CONTEXT_CHARS = 500
# Single merge call; skip merge above this to avoid huge requests (chunks still help).
MERGE_MAX_CHARS = 280_000

# Same-model retries before switching model or prompting (seconds between tries).
OVERLOAD_BACKOFF_SECS = (2, 5, 10)
# Automatic model chain used with --auto-fallback (order matters).
AUTO_FALLBACK_MODELS = (
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
)
# Max times we give up on a model after backoff (not per transcript chunk).
MAX_OVERLOAD_RECOVERY_ROUNDS = 24

# Exact model IDs users can pass to --model (shown in --help and on API errors).
RECOMMENDED_MODEL_OPTIONS: tuple[tuple[str, str], ...] = (
    (
        "gemini-2.5-flash",
        "Default. Fast general-purpose model; occasionally returns temporary overload (503).",
    ),
    (
        "gemini-2.5-flash-lite",
        "Smaller / cheaper Flash; often still available when gemini-2.5-flash is saturated.",
    ),
    (
        "gemini-2.0-flash",
        "Previous-generation Flash; different capacity pool from the 2.5 family.",
    ),
    (
        "gemini-1.5-flash",
        "Older Flash; use if newer models error or are unavailable for your API key.",
    ),
)

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


@dataclass
class RetryContext:
    """Shared state for overload retries, model switches, and user prompts."""

    model_cell: list[str]
    exhausted_models: set[str]
    interactive: bool
    auto_fallback: bool
    full_instructions_shown: list[bool]
    overload_rounds: list[int]


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


def is_transient_overload(err: Exception) -> bool:
    """True for temporary capacity / rate issues where retry or another model may help."""
    if not isinstance(err, APIError):
        return False
    code = getattr(err, "code", None)
    status = (getattr(err, "status", None) or "") or ""
    message = (getattr(err, "message", None) or "") or ""
    combined = f"{status} {message}".lower()
    if code in (503, 429):
        return True
    if "unavailable" in combined or "overloaded" in combined:
        return True
    if "resource_exhausted" in combined or (
        "rate" in combined and "limit" in combined
    ):
        return True
    if "high demand" in message.lower() or "try again later" in message.lower():
        return True
    return False


def format_model_catalog_text() -> str:
    lines: list[str] = []
    for mid, blurb in RECOMMENDED_MODEL_OPTIONS:
        lines.append(f"  • {mid}")
        lines.append(f"      {blurb}")
    return "\n".join(lines)


def format_full_overload_instructions(failed_model: str) -> str:
    """Printed on stderr when the API is overloaded; no repo README required."""
    catalog = format_model_catalog_text()
    return f"""
================================================================================
GEMINI API: TEMPORARY UNAVAILABLE OR OVERLOADED
================================================================================
The request to model "{failed_model}" failed after automatic retries (HTTP 503 /
429 or similar). This is usually NOT a bug in this script — Google's side is
busy or your free-tier quota for that model is momentarily exhausted.

WHAT YOU CAN DO RIGHT NOW (try in this order)
--------------------------------------------------------------------------------
1) WAIT AND RETRY
   Spikes in demand are often short. Wait 2–15 minutes and run the same command
   again with the same model.

2) USE A DIFFERENT MODEL (exact IDs you can pass to --model)
{catalog}

   Example:
     python fix_transcript.py your.txt --model gemini-2.5-flash-lite

   Or run with automatic fallback (tries other models in a fixed order):
     python fix_transcript.py your.txt --auto-fallback

3) CHECK YOUR API KEY AND ACCOUNT
   • Open Google AI Studio and confirm your key is valid:
     https://aistudio.google.com/apikey
   • In Google AI Studio / Google Cloud, check for quota, billing, or region
     messages. Free tier limits vary; some models may be restricted.

4) CONFIRM MODEL NAMES
   Model IDs change over time. The list above matches current Gemini API names;
   if you see "model not found" (404), pick another ID from the list or check:
     https://ai.google.dev/gemini-api/docs/models

5) NON-INTERACTIVE / AUTOMATION
   If a prompt would block your script, use:
     python fix_transcript.py your.txt --no-interactive --auto-fallback

6) STILL STUCK?
   • Try again at a different time of day (global demand varies).
   • If only one model ever works, your project may be hitting limits — review
     usage in AI Studio / Cloud console.
================================================================================
"""


def _generate_text_once(
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


def _next_auto_fallback_model(current: str, exhausted: set[str]) -> str | None:
    """
    Pick another model to try after `current` is given up on.
    Prefer the fixed chain order, then any remaining recommended ids.
    """
    for mid in AUTO_FALLBACK_MODELS:
        if mid not in exhausted and mid != current:
            return mid
    for mid, _ in RECOMMENDED_MODEL_OPTIONS:
        if mid not in exhausted and mid != current:
            return mid
    return None


def _prompt_pick_alternative_model(
    failed_model: str,
    exhausted: set[str],
    full_instructions_shown: list[bool],
) -> str | None:
    """
    Ask the user which model to use next. Returns new model id, or None to abort.
    full_instructions_shown is a single-element list used as a mutable flag.
    """
    if not full_instructions_shown[0]:
        print(format_full_overload_instructions(failed_model), file=sys.stderr)
        full_instructions_shown[0] = True
    else:
        print(
            f"\nModel {failed_model!r} is still overloaded. Pick another option.\n",
            file=sys.stderr,
        )

    suggested = _next_auto_fallback_model(failed_model, exhausted)
    if suggested:
        print(
            f"Suggested next try: {suggested} (often less busy than {failed_model}).\n",
            file=sys.stderr,
        )

    print("Alternative models (copy any ID exactly as shown):", file=sys.stderr)
    for n, (mid, desc) in enumerate(RECOMMENDED_MODEL_OPTIONS, start=1):
        print(f"  [{n}] {mid}", file=sys.stderr)
        print(f"       {desc}", file=sys.stderr)

    print(
        "\nEnter a number 1–"
        f"{len(RECOMMENDED_MODEL_OPTIONS)}, paste a full model id, "
        "or press Enter for the suggested model (if any).",
        file=sys.stderr,
    )
    print("Type 'q' or 'quit' to exit.\n", file=sys.stderr)

    try:
        raw = input("Choice: ").strip()
    except EOFError:
        print("\nNo input (EOF). Exiting.", file=sys.stderr)
        return None

    if raw.lower() in ("q", "quit", "exit"):
        return None
    if not raw and suggested:
        return suggested
    if not raw:
        print("Nothing entered and no suggestion; exiting.", file=sys.stderr)
        return None

    if raw.isdigit():
        idx = int(raw)
        if 1 <= idx <= len(RECOMMENDED_MODEL_OPTIONS):
            return RECOMMENDED_MODEL_OPTIONS[idx - 1][0]
        print(f"Invalid number: {idx}", file=sys.stderr)
        return None

    return raw


def generate_text(
    client: genai.Client,
    model_cell: list[str],
    system_instruction: str,
    user_text: str,
    temperature: float,
    *,
    interactive: bool,
    auto_fallback: bool,
    exhausted_models: set[str],
    full_instructions_shown: list[bool],
    overload_rounds: list[int],
) -> str:
    """
    Call Gemini with retries on overload, optional auto-fallback chain, or prompt.
    model_cell[0] is the active model id and may be updated in place.
    """
    while True:
        model = model_cell[0]
        last_error: APIError | None = None

        for delay_idx, delay in enumerate([0, *OVERLOAD_BACKOFF_SECS]):
            if delay:
                print(
                    f"  API busy for {model!r} (retry {delay_idx}/"
                    f"{len(OVERLOAD_BACKOFF_SECS)} after {delay}s)...",
                    file=sys.stderr,
                )
                time.sleep(delay)
            try:
                return _generate_text_once(
                    client, model, system_instruction, user_text, temperature
                )
            except APIError as e:
                last_error = e
                if not is_transient_overload(e):
                    print(
                        f"Error: Gemini API request failed ({e.code} {e.status}).",
                        file=sys.stderr,
                    )
                    raise
                if delay_idx < len(OVERLOAD_BACKOFF_SECS):
                    continue
                break

        # Exhausted backoff for this model
        assert last_error is not None
        print(
            f"  Model {model!r} still unavailable after retries: {last_error}",
            file=sys.stderr,
        )

        overload_rounds[0] += 1
        if overload_rounds[0] > MAX_OVERLOAD_RECOVERY_ROUNDS:
            print(
                f"Error: Exceeded {MAX_OVERLOAD_RECOVERY_ROUNDS} overload recovery "
                "rounds. Wait and retry later, or check AI Studio / billing.",
                file=sys.stderr,
            )
            print(format_full_overload_instructions(model_cell[0]), file=sys.stderr)
            sys.exit(1)

        exhausted_models.add(model)
        next_auto = _next_auto_fallback_model(model, exhausted_models)

        if auto_fallback and next_auto is not None:
            print(
                f"  --auto-fallback: switching to {next_auto!r} ...",
                file=sys.stderr,
            )
            model_cell[0] = next_auto
            continue

        if auto_fallback and next_auto is None:
            print(
                "  --auto-fallback: no more models in the built-in chain to try.",
                file=sys.stderr,
            )
            print(format_full_overload_instructions(model), file=sys.stderr)
            sys.exit(1)

        if not interactive:
            print(
                "\nNon-interactive mode: stopping. Re-run with a different model, "
                "or use --auto-fallback.\n",
                file=sys.stderr,
            )
            print(format_full_overload_instructions(model), file=sys.stderr)
            sys.exit(1)

        choice = _prompt_pick_alternative_model(
            model, exhausted_models, full_instructions_shown
        )
        if choice is None:
            print(format_full_overload_instructions(model), file=sys.stderr)
            sys.exit(1)
        print(f"  Continuing with model {choice!r} ...\n", file=sys.stderr)
        model_cell[0] = choice


def call_resilient_generate(
    client: genai.Client,
    ctx: RetryContext,
    system_instruction: str,
    user_text: str,
    temperature: float,
) -> str:
    return generate_text(
        client,
        ctx.model_cell,
        system_instruction,
        user_text,
        temperature,
        interactive=ctx.interactive,
        auto_fallback=ctx.auto_fallback,
        exhausted_models=ctx.exhausted_models,
        full_instructions_shown=ctx.full_instructions_shown,
        overload_rounds=ctx.overload_rounds,
    )


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


def correct_segment(
    client: genai.Client,
    ctx: RetryContext,
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
    return call_resilient_generate(
        client, ctx, SYSTEM_CORRECT, user, temperature=0.25
    )


def merge_sections(client: genai.Client, ctx: RetryContext, body: str) -> str:
    user = USER_MERGE_TEMPLATE.format(text=body)
    return call_resilient_generate(
        client, ctx, SYSTEM_MERGE, user, temperature=0.15
    )


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
    ctx: RetryContext,
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
        fixed = correct_segment(client, ctx, segment, prior_tail)
        corrected_parts.append(fixed)
        prior_tail = fixed[-context_chars:] if context_chars and fixed else ""

    merged = "\n\n".join(corrected_parts)

    if do_merge and len(spans) > 1 and len(merged) <= MERGE_MAX_CHARS:
        print("  Running merge pass...", file=sys.stderr)
        merged = merge_sections(client, ctx, merged)
    elif do_merge and len(spans) == 1:
        pass  # Single chunk: nothing to merge across boundaries.
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
        help=(
            "Output file path, or '-' for stdout. "
            f"Default: {OUTPUT_DATA_DIR.name}/<input-stem>-fixed.txt"
        ),
    )
    model_names = ", ".join(m for m, _ in RECOMMENDED_MODEL_OPTIONS)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            f"Gemini model id (default: {DEFAULT_MODEL}). "
            f"Typical options: {model_names}."
        ),
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
    parser.add_argument(
        "--auto-fallback",
        action="store_true",
        help=(
            "On repeated overload errors, automatically try other models in order: "
            f"{', '.join(AUTO_FALLBACK_MODELS)}, then remaining recommended ids."
        ),
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help=(
            "Never prompt for input on API overload. "
            "Print instructions to stderr and exit unless --auto-fallback can switch models."
        ),
    )
    args = parser.parse_args()

    path = args.input_file
    if not path.is_file():
        print(f"Error: not a file: {path}", file=sys.stderr)
        sys.exit(1)

    raw = path.read_text(encoding="utf-8")
    print(f"Processing {path} ({len(raw)} chars)...", file=sys.stderr)
    client = create_client()
    interactive = not args.no_interactive and sys.stdin.isatty()
    ctx = RetryContext(
        model_cell=[args.model],
        exhausted_models=set(),
        interactive=interactive,
        auto_fallback=args.auto_fallback,
        full_instructions_shown=[False],
        overload_rounds=[0],
    )
    print(f"  Initial model: {ctx.model_cell[0]!r}", file=sys.stderr)
    result = process_transcript(
        client,
        raw,
        ctx,
        args.chunk_size,
        args.context_chars,
        do_merge=not args.no_merge,
    )
    print(f"  Model used: {ctx.model_cell[0]!r}", file=sys.stderr)

    if args.output is not None and str(args.output) == "-":
        sys.stdout.write(result)
        if result and not result.endswith("\n"):
            sys.stdout.write("\n")
    else:
        if args.output is not None:
            out_path = args.output
        else:
            OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
            out_path = OUTPUT_DATA_DIR / f"{path.stem}-fixed.txt"
        text = result + ("\n" if result and not result.endswith("\n") else "")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(f"Wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
