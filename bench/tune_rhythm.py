"""Tunes the rhythm constants of transcribe.py on the bench (coordinate search).

Performances and their hand assignment are computed once; each trial only re-runs
quantization. Tuning uses other random performances than bench.py reports on.

    .venv\\Scripts\\python bench\\tune_rhythm.py
"""

import numpy as np

from bench import HARD, PIECES, from_corpus, generate, perform, tr, TRACK_DRIFT, TRACK_JITTER

TUNE_SEEDS = range(20, 26)


def prepare(seeds):
    """Normal and free (HARD) performances of every piece."""
    data = []
    for name, (src, bpm, bars) in PIECES.items():
        for seed in seeds:
            hard = seed % 2 == 1
            rng = np.random.default_rng(seed)
            truth = generate(src[4:], bars, rng) if src.startswith("gen:") else from_corpus(src, bars)
            notes, beat_times, truth_of = perform(truth, bpm, rng, **(HARD if hard else {}))
            drift = np.convolve(rng.normal(0, 1, len(beat_times) + 8), np.hanning(7), "same")[4:4 + len(beat_times)]
            tracked = beat_times + drift / (drift.std() + 1e-9) * TRACK_DRIFT + rng.normal(0, TRACK_JITTER,
                                                                                            len(beat_times))
            grid, n_pre = tr.extend_beats(tracked, max(n.end for n in notes))
            hand_notes = tr.assign_hands(tr.group_onsets(notes))
            data.append((name + ("-hard" if hard else ""), notes, grid, n_pre, hand_notes, truth_of))
    return data


def score(data):
    per_piece = {}
    for name, notes, grid, n_pre, hand_notes, truth_of in data:
        hands, _ = tr.quantize(notes, grid, n_pre, hand_notes)
        ok = 0
        for evs in hands.values():
            for e in evs:
                ok += sum(truth_of[id(n)].q == e.q for n in e.notes)
        per_piece.setdefault(name, []).append(ok / len(notes))
    return float(np.mean([np.mean(v) for v in per_piece.values()]))  # every piece counts the same


def get(name):
    if name.startswith("sub"):
        k = int(name[3:])
        return dict(tr.SUBDIVISIONS)[k]
    return getattr(tr, name)


def put(name, value):
    if name.startswith("sub"):
        k = int(name[3:])
        tr.SUBDIVISIONS[:] = [(s, value if s == k else p) for s, p in tr.SUBDIVISIONS]
    else:
        setattr(tr, name, value)


PARAMS = ["TIMING_SIGMA", "SWITCH_PENALTY", "COLLISION_PENALTY", "ROLL_SIGMA", "FAST_GRID", "REFIT_WIDTH",
          "REFIT_FIT", "REFIT_MAX",
          "sub2", "sub4", "sub3", "sub8", "sub6", "sub5", "sub7"]


def main():
    data = prepare(TUNE_SEEDS)
    rounds = {}
    for r in range(4):
        tr.REFIT_ROUNDS = r
        rounds[r] = score(data)
    tr.REFIT_ROUNDS = max(rounds, key=rounds.get)
    print("REFIT_ROUNDS:", {r: f"{v:.2%}" for r, v in rounds.items()}, "->", tr.REFIT_ROUNDS)
    best = score(data)
    print(f"start {best:.2%}")
    for rnd in range(3):
        improved = False
        for name in PARAMS:
            v0 = get(name)
            for factor in (0.6, 0.8, 1.25, 1.6):
                put(name, v0 * factor)
                s = score(data)
                if s > best + 0.0005:
                    best, v0, improved = s, v0 * factor, True
                    print(f"  {name} -> {v0:.4g}: {best:.2%}")
            put(name, v0)
        if not improved:
            break
    print("final", f"{best:.2%}")
    for name in PARAMS:
        print(f"  {name} = {get(name):.4g}")


if __name__ == "__main__":
    main()
