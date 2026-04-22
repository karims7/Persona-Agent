"""Preprocess raw MELD CSVs into clean, analysis-ready CSVs.

Standalone script — run once before any Phase 2 pipeline work.
No dependency on any other project file.

Reads from:  data/original/meld_train_sent_emo.csv
             data/original/meld_test_sent_emo.csv

Writes to:   data/processed/meld_train_processed.csv
             data/processed/meld_test_processed.csv

New columns added:
  scene_key          — Dialogue_ID string (globally unique within each split)
  utterance_position — 0-indexed position within scene, gap-free after numeric sort
  word_count         — number of whitespace-separated words in Utterance
  is_short           — 1 if word_count < 5, else 0

Sr No. column is dropped (sequential row number, no analytical value).

Output column order:
  scene_key, Dialogue_ID, utterance_position, Utterance_ID,
  Season, Episode, Speaker, Emotion, Sentiment, Utterance,
  word_count, is_short, StartTime, EndTime
"""

import csv
import os
from collections import defaultdict

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ORIGINAL_DIR = os.path.join(_SCRIPT_DIR, "original")
_PROCESSED_DIR = os.path.join(_SCRIPT_DIR, "processed")

_OUTPUT_COLUMNS = [
    "scene_key", "Dialogue_ID", "utterance_position", "Utterance_ID",
    "Season", "Episode", "Speaker", "Emotion", "Sentiment", "Utterance",
    "word_count", "is_short", "StartTime", "EndTime",
]

_SHORT_WORD_THRESHOLD = 5


def _load_csv(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _word_count(text: str) -> int:
    return len(text.split())


def _validate_and_report(rows: list[dict], split: str) -> None:
    print(f"\n=== VALIDATION: {split} ===")
    print(f"Total rows: {len(rows)}")
    print(f"Columns: {list(rows[0].keys()) if rows else []}")

    groups: dict[str, list] = defaultdict(list)
    for r in rows:
        groups[r["Dialogue_ID"]].append(r)

    print(f"Unique Dialogue_IDs: {len(groups)}")

    speakers = sorted(set(r["Speaker"] for r in rows))
    print(f"Unique speakers ({len(speakers)}): {speakers}")

    emotions = sorted(set(r["Emotion"].lower() for r in rows))
    print(f"Unique emotion labels: {emotions}")

    # Utterance_ID gaps
    gap_dialogues = []
    for did, d_rows in sorted(groups.items(), key=lambda x: int(x[0])):
        utts = sorted(int(r["Utterance_ID"]) for r in d_rows)
        expected = list(range(len(utts)))
        if utts != expected:
            gap_dialogues.append((did, utts))
    pct_gap = round(100 * len(gap_dialogues) / len(groups), 1) if groups else 0
    print(f"Scenes with Utterance_ID gaps: {len(gap_dialogues)} ({pct_gap}%)")
    for did, utts in gap_dialogues[:5]:
        print(f"  Dialogue_ID={did}: Utterance_IDs={utts}")

    # Single-utterance scenes
    single = sum(1 for d_rows in groups.values() if len(d_rows) == 1)
    pct_single = round(100 * single / len(groups), 1) if groups else 0
    print(f"Single-utterance scenes: {single} ({pct_single}%)")

    # Short utterances
    short = sum(1 for r in rows if _word_count(r["Utterance"]) < _SHORT_WORD_THRESHOLD)
    pct_short = round(100 * short / len(rows), 1) if rows else 0
    print(f"Short utterances (<{_SHORT_WORD_THRESHOLD} words): {short} ({pct_short}%)")

    avg_len = round(sum(len(v) for v in groups.values()) / len(groups), 2) if groups else 0
    print(f"Average scene length: {avg_len} utterances")


def _build_processed_rows(rows: list[dict]) -> list[dict]:
    groups: dict[str, list] = defaultdict(list)
    for r in rows:
        groups[r["Dialogue_ID"]].append(r)

    processed = []
    for did in sorted(groups.keys(), key=int):
        d_rows = sorted(groups[did], key=lambda r: int(r["Utterance_ID"]))
        for pos, row in enumerate(d_rows):
            wc = _word_count(row["Utterance"])
            processed.append({
                "scene_key": did,
                "Dialogue_ID": did,
                "utterance_position": pos,
                "Utterance_ID": row["Utterance_ID"],
                "Season": row["Season"],
                "Episode": row["Episode"],
                "Speaker": row["Speaker"],
                "Emotion": row["Emotion"],
                "Sentiment": row["Sentiment"],
                "Utterance": row["Utterance"],
                "word_count": wc,
                "is_short": 1 if wc < _SHORT_WORD_THRESHOLD else 0,
                "StartTime": row.get("StartTime", ""),
                "EndTime": row.get("EndTime", ""),
            })
    return processed


def _write_csv(rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _verify_no_position_gaps(processed: list[dict]) -> int:
    groups: dict[str, list] = defaultdict(list)
    for r in processed:
        groups[r["scene_key"]].append(int(r["utterance_position"]))
    gap_count = 0
    for did, positions in groups.items():
        positions.sort()
        if positions != list(range(len(positions))):
            gap_count += 1
    return gap_count


def _scene_length_distribution(processed: list[dict]) -> dict[str, int]:
    groups: dict[str, list] = defaultdict(list)
    for r in processed:
        groups[r["scene_key"]].append(r)
    lengths = [len(v) for v in groups.values()]
    total = len(lengths)

    def pct(n): return round(100 * n / total, 1) if total else 0

    b1 = sum(1 for l in lengths if l == 1)
    b25 = sum(1 for l in lengths if 2 <= l <= 5)
    b610 = sum(1 for l in lengths if 6 <= l <= 10)
    b1120 = sum(1 for l in lengths if 11 <= l <= 20)
    b21 = sum(1 for l in lengths if l >= 21)
    return {
        "1": (b1, pct(b1)),
        "2-5": (b25, pct(b25)),
        "6-10": (b610, pct(b610)),
        "11-20": (b1120, pct(b1120)),
        "21+": (b21, pct(b21)),
    }


def _print_report(
    split: str,
    original: list[dict],
    processed: list[dict],
    output_file: str,
) -> None:
    orig_groups: dict[str, list] = defaultdict(list)
    for r in original:
        orig_groups[r["Dialogue_ID"]].append(r)

    proc_groups: dict[str, list] = defaultdict(list)
    for r in processed:
        proc_groups[r["scene_key"]].append(r)

    gap_count = _verify_no_position_gaps(processed)
    dist = _scene_length_distribution(processed)

    speakers = sorted(set(r["Speaker"] for r in original))
    emotion_counts: dict[str, int] = defaultdict(int)
    for r in original:
        emotion_counts[r["Emotion"].lower()] += 1
    short_orig = sum(1 for r in original if _word_count(r["Utterance"]) < _SHORT_WORD_THRESHOLD)
    pct_short_orig = round(100 * short_orig / len(original), 1) if original else 0
    single_orig = sum(1 for v in orig_groups.values() if len(v) == 1)
    pct_single = round(100 * single_orig / len(orig_groups), 1) if orig_groups else 0

    gap_orig_count = 0
    for d_rows in orig_groups.values():
        utts = sorted(int(r["Utterance_ID"]) for r in d_rows)
        if utts != list(range(len(utts))):
            gap_orig_count += 1
    pct_gap_orig = round(100 * gap_orig_count / len(orig_groups), 1) if orig_groups else 0

    avg_len_orig = round(sum(len(v) for v in orig_groups.values()) / len(orig_groups), 2) if orig_groups else 0

    print(f"\n{'='*60}")
    print(f"=== PROCESSING REPORT: {split} ===")
    print(f"{'='*60}")
    print()
    print("ORIGINAL:")
    print(f"  Total rows: {len(original)}")
    print(f"  Total scenes: {len(orig_groups)}")
    print(f"  Unique speakers: {speakers}")
    print(f"  Emotion label counts: {dict(sorted(emotion_counts.items()))}")
    print(f"  Scenes with Utterance_ID gaps: {gap_orig_count} ({pct_gap_orig}%)")
    print(f"  Single-utterance scenes: {single_orig} ({pct_single}%)")
    print(f"  Short utterances (<{_SHORT_WORD_THRESHOLD} words): {short_orig} ({pct_short_orig}%)")
    print(f"  Average scene length: {avg_len_orig} utterances")
    print()
    print("PROCESSED:")
    print(f"  Total rows: {len(processed)} (should equal original: {len(original)})")
    print(f"  Total scenes: {len(proc_groups)} (should equal original: {len(orig_groups)})")
    print(f"  utterance_position gaps: {gap_count} (should always be 0)")
    print(f"  Columns added: scene_key, utterance_position, word_count, is_short")
    print(f"  Output file: {output_file}")
    print()
    print("SCENE LENGTH DISTRIBUTION:")
    for label, (count, pct) in dist.items():
        print(f"  {label:>12} utterances: {count} scenes ({pct}%)")
    print()
    print(f"SAMPLE — first 3 processed scenes (all columns, all rows):")
    seen = []
    for r in processed:
        if r["scene_key"] not in seen:
            seen.append(r["scene_key"])
        if len(seen) > 3:
            break
    for sk in seen[:3]:
        scene_rows = [r for r in processed if r["scene_key"] == sk]
        s = scene_rows[0]["Season"]
        e = scene_rows[0]["Episode"]
        print(f"\n  --- scene_key={sk} | Season={s} Episode={e} ---")
        print(f"  {'pos':>4} {'utt_id':>6} {'speaker':<20} {'emotion':<10} {'wc':>3} {'short':>5}  utterance")
        for r in scene_rows:
            utt = r["Utterance"][:60] + ("..." if len(r["Utterance"]) > 60 else "")
            print(f"  {r['utterance_position']:>4} {r['Utterance_ID']:>6} {r['Speaker']:<20} {r['Emotion']:<10} {r['word_count']:>3} {r['is_short']:>5}  {utt}")
    print()
    print(f"{'='*60}")
    print(f"=== END REPORT: {split} ===")
    print(f"{'='*60}")


def process_split(
    split: str,
    input_filename: str,
    output_filename: str,
) -> None:
    input_path = os.path.join(_ORIGINAL_DIR, input_filename)
    output_path = os.path.join(_PROCESSED_DIR, output_filename)

    print(f"\n>>> Processing {split} split: {input_path}")
    original = _load_csv(input_path)
    _validate_and_report(original, split)

    processed = _build_processed_rows(original)
    _write_csv(processed, output_path)

    _print_report(split, original, processed, output_path)
    print(f"\nSaved to {output_path}")


def main() -> None:
    process_split(
        split="train",
        input_filename="meld_train_sent_emo.csv",
        output_filename="meld_train_processed.csv",
    )
    process_split(
        split="test",
        input_filename="meld_test_sent_emo.csv",
        output_filename="meld_test_processed.csv",
    )


if __name__ == "__main__":
    main()
