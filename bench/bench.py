"""Test bench for the score-building half of the pipeline (hands, rhythm, voices).

Real scores with known hands and rhythm are "performed": a rubato tempo curve,
finger timing scatter, chord spread and slightly late or early hands. The result
goes through transcribe.quantize() with slightly wrong beats, as a beat tracker
would give them, and is compared with the score note by note.

    .venv\\Scripts\\python bench\\bench.py            all pieces, 3 performances each
    .venv\\Scripts\\python bench\\bench.py maple -v   one piece, with the first errors listed

Numbers (fraction of notes):
    hand    written in the right hand / staff
    onset   at exactly the right position in the bar
    grace   turned into a grace note (the score has none)
    length  written with exactly the score's note value (right hand and onset needed)
"""

import math
import sys
import warnings
from fractions import Fraction
from pathlib import Path

import numpy as np
import pretty_midi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import transcribe as tr  # noqa: E402

# name: (corpus path or generator, tempo in quarter notes per minute, bars to use)
PIECES = {
    "chopin":   ("chopin/mazurka06-2.krn", 130, 48),
    "maple":    ("joplin/maple_leaf_rag.mxl", 90, 48),
    "mozart":   ("mozart/k545/movement1_exposition.mxl", 110, 28),
    "polonaise": ("schumann_clara/polonaise_op1n1.mxl", 90, 40),
    "schoen2":  ("schoenberg/opus19/movement2.mxl", 60, 9),
    "schoen6":  ("schoenberg/opus19/movement6.mxl", 50, 9),
    "cross":    ("gen:cross", 80, 16),
    "romantic": ("gen:romantic", 66, 16),
    "runs":     ("gen:runs", 60, 16),
}


TRACK_DRIFT = 0.03    # s, std of the beat tracker's slow error
TRACK_JITTER = 0.015  # s, its per-beat error


# ------------------------------------------------------------------ ground truth

class TruthNote:
    def __init__(self, q, dur, pitch, hand, voice=1):
        self.q, self.dur, self.pitch, self.hand, self.voice = Fraction(q), Fraction(dur), pitch, hand, voice


def from_corpus(path, bars):
    from music21 import corpus, note

    s = corpus.parse(path)
    parts = list(s.parts)[-2:]  # the piano staves are the last two parts
    out = []
    for hand, part in zip("RL", parts):
        part = part.stripTies()
        measures = list(part.getElementsByClass("Measure"))[:bars]
        for m in measures:
            for v_i, voice in enumerate(m.voices or [m], start=1):
                for n in voice.notesAndRests:
                    if isinstance(n, note.Rest) or n.duration.isGrace or n.duration.quarterLength == 0:
                        continue
                    q = Fraction(m.offset).limit_denominator(96) + Fraction(n.offset).limit_denominator(96)
                    d = Fraction(n.duration.quarterLength).limit_denominator(96)
                    for p in n.pitches:
                        out.append(TruthNote(q, d, p.midi, hand, v_i))
    return out


def generate(kind, bars, rng):
    """Textures the corpus lacks: hand crossings, romantic figuration, fast tuplet runs."""
    out = []
    chords = [[0, 4, 7], [9, 0, 4], [5, 9, 0], [7, 11, 2], [2, 5, 9], [4, 7, 11]]
    for b in range(bars):
        ch = chords[rng.integers(len(chords))]
        root = 48 + ch[0]
        base = b * 4
        if kind == "cross":
            # RH keeps 16th arpeggios around middle C; LH alternates bass and a melody note
            # crossing over the right hand (Liszt / Rachmaninov-style)
            arp = sorted({60 + (c - 60) % 12 for c in [60 + x for x in ch]})
            seq = arp + [arp[0] + 12] + arp[::-1][1:]
            for i in range(16):
                out.append(TruthNote(Fraction(base) + Fraction(i, 4), Fraction(1, 4), seq[i % len(seq)], "R"))
            out.append(TruthNote(base, 1, root - 12, "L"))
            out.append(TruthNote(base + 1, 1, 79 + ch[1] % 5, "L"))
            out.append(TruthNote(base + 2, 1, root - 12 + 7, "L"))
            out.append(TruthNote(base + 3, 1, 81 + ch[2] % 5, "L"))
        elif kind == "romantic":
            # LH: wide arpeggio over two octaves in triplet 8ths; RH: melody held over inner 8ths
            lh = [root - 12, root - 5, root, root + 4, root + 7, root + 12]
            for i in range(12):
                out.append(TruthNote(Fraction(base) + Fraction(i, 3), Fraction(1, 3), lh[i % 6] if i % 12 < 6
                                     else lh[5 - i % 6], "L"))
            mel = 72 + ch[rng.integers(3)] % 12
            out.append(TruthNote(base, 2, mel, "R", 1))
            out.append(TruthNote(base + 2, 1, mel + 2, "R", 1))
            out.append(TruthNote(base + 3, 1, mel - 1, "R", 1))
            for i in range(8):
                inner = mel - 3 - (mel - 3 - (60 + ch[i % 3] % 12)) % 12  # a chord tone below the melody
                out.append(TruthNote(Fraction(base) + Fraction(i, 2), Fraction(1, 2), inner, "R", 2))
        elif kind == "runs":
            # runs of 5, 6, 7 notes per beat in the RH over LH quarter chords
            for beat in range(4):
                n = [4, 5, 6, 7][rng.integers(4)]
                start = 72 + rng.integers(-5, 6)
                for i in range(n):
                    out.append(TruthNote(Fraction(base + beat) + Fraction(i, n), Fraction(1, n),
                                         start + (i * 2 if beat % 2 == 0 else -i * 2), "R"))
                for c in ch:
                    out.append(TruthNote(base + beat, 1, 48 + c % 12, "L"))
    return out


# ------------------------------------------------------------------ performance

def perform(truth, bpm, rng, rubato=0.12, scatter=0.018, spread=0.012, hand_lag=0.015):
    """Truth notes -> pretty_midi notes with human timing; returns notes, beat times, note->truth."""
    n_beats = int(max(float(t.q + t.dur) for t in truth)) + 4
    walk = np.convolve(rng.normal(0, 1, n_beats + 20), np.hanning(9), "same")[10:10 + n_beats + 1]
    walk = walk / (np.abs(walk).max() + 1e-9) * rubato
    ibi = 60.0 / bpm * np.exp(walk)
    beat_times = np.concatenate([[1.0], 1.0 + np.cumsum(ibi)])
    at = lambda q: float(np.interp(float(q), np.arange(len(beat_times)), beat_times))
    lag = {"R": 0.0, "L": rng.normal(0, hand_lag)}
    notes, truth_of = [], {}
    chord_jitter = {}
    for t in truth:
        key = (t.hand, t.q)
        if key not in chord_jitter:
            chord_jitter[key] = rng.normal(0, scatter)
        start = at(t.q) + chord_jitter[key] + rng.normal(0, spread) + lag[t.hand]
        end = max(start + 0.04, at(t.q + t.dur) - 0.03 + rng.normal(0, 0.02))
        n = pretty_midi.Note(velocity=int(rng.integers(55, 90)), pitch=t.pitch, start=start, end=end)
        notes.append(n)
        truth_of[id(n)] = t
    # a key is let go (the pedal holds the sound) once its hand has to play out of reach of it
    for hand in "LR":
        mine = sorted((n for n in notes if truth_of[id(n)].hand == hand), key=lambda n: n.start)
        for i, n in enumerate(mine):
            for m in mine[i + 1:]:
                if m.start >= n.end:
                    break
                if m.start > n.start + 0.03 and abs(m.pitch - n.pitch) > tr.HAND_SPAN:
                    n.end = max(n.start + 0.04, m.start - 0.02)
                    break
    notes.sort(key=lambda n: (n.start, n.pitch))
    return notes, beat_times, truth_of


# ------------------------------------------------------------------ scoring

HARD = dict(rubato=0.25, scatter=0.03, spread=0.02, hand_lag=0.025)  # a freer player


def run(name, seed, verbose=False, hard=False):
    src, bpm, bars = PIECES[name]
    rng = np.random.default_rng(seed)
    truth = generate(src[4:], bars, rng) if src.startswith("gen:") else from_corpus(src, bars)
    notes, beat_times, truth_of = perform(truth, bpm, rng, **(HARD if hard else {}))
    # a tracker is never exact: a slowly wandering error plus a small one per beat
    drift = np.convolve(rng.normal(0, 1, len(beat_times) + 8), np.hanning(7), "same")[4:4 + len(beat_times)]
    drift = drift / (drift.std() + 1e-9) * TRACK_DRIFT
    tracked = beat_times + drift + rng.normal(0, TRACK_JITTER, len(beat_times))
    grid, n_pre = tr.extend_beats(tracked, max(n.end for n in notes))
    hands, grid = tr.quantize(notes, grid, n_pre)
    idx = np.arange(len(grid), dtype=float)
    chunks, _ = tr.arrange(hands, lambda t: float(np.interp(t, grid, idx)) - n_pre, 4)
    written = {id(n): c.dur for cs in chunks.values() for c in cs for n in c.notes}

    placed = {}
    for side, events in hands.items():
        for e in events:
            for n in e.notes:
                placed[id(n)] = (side, e.q, e)
    n = len(notes)
    hand_ok = onset_ok = grace = dur_ok = 0
    errors = []
    for note_ in notes:
        t = truth_of[id(note_)]
        if id(note_) not in placed:
            grace += 1
            errors.append(f"  grace  q={t.q} pitch={t.pitch} {t.hand}")
            continue
        side, q, e = placed[id(note_)]
        hand_ok += side == t.hand
        onset_ok += q == t.q
        if side != t.hand:
            errors.append(f"  hand   q={t.q} pitch={t.pitch} {t.hand}->{side}")
        if q != t.q:
            errors.append(f"  onset  q={t.q} -> {q} pitch={t.pitch} {t.hand}")
        elif side == t.hand:
            dur_ok += written[id(note_)] == t.dur
            if written[id(note_)] != t.dur:
                errors.append(f"  dur    q={t.q} pitch={t.pitch} {t.hand} {t.dur} -> {written[id(note_)]}")
    if verbose:
        print("\n".join(sorted(errors, key=lambda s: Fraction(s.split("q=")[1].split()[0]))[:40]))
    return dict(hand=hand_ok / n, onset=onset_ok / n, grace=grace / n, dur=dur_ok / n, notes=n)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    verbose = "-v" in sys.argv
    names = args or list(PIECES)
    seeds = [1] if verbose else [1, 2, 3]
    rows = []
    print(f"{'piece':10} {'notes':>6} {'hand':>7} {'onset':>7} {'grace':>7} {'length':>7}")
    for name in names:
        rs = [run(name, s, verbose, "--hard" in sys.argv) for s in seeds]
        r = {k: float(np.mean([x[k] for x in rs])) for k in rs[0]}
        rows.append(r)
        print(f"{name:10} {r['notes']:6.0f} {r['hand']:7.1%} {r['onset']:7.1%} {r['grace']:7.1%} {r['dur']:7.1%}")
    if len(rows) > 1:
        m = {k: float(np.mean([r[k] for r in rows])) for k in ("hand", "onset", "grace", "dur")}
        print(f"{'MEAN':10} {'':6} {m['hand']:7.1%} {m['onset']:7.1%} {m['grace']:7.1%} {m['dur']:7.1%}")


if __name__ == "__main__":
    main()
