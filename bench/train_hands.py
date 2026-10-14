"""Learns transcribe.HAND_WEIGHTS from scores with known hands (averaged structured perceptron).

For each training performance the beam search finds the cheapest assignment with the
current weights; weights then move so the true assignment becomes cheaper than the
one found. Training uses other random performances than bench.py reports on.

    .venv\\Scripts\\python bench\\train_hands.py [epochs]
"""

import sys
from collections import defaultdict

import numpy as np

import bench
from bench import PIECES, from_corpus, generate, perform, tr

TRAIN_SEEDS = range(10, 14)


def examples():
    out = []
    for name, (src, bpm, bars) in PIECES.items():
        for seed in TRAIN_SEEDS:
            rng = np.random.default_rng(seed)
            truth = generate(src[4:], bars, rng) if src.startswith("gen:") else from_corpus(src, bars)
            notes, _, truth_of = perform(truth, bpm, rng)
            groups = tr.group_onsets(notes)
            gold = []
            for g in groups:
                g = sorted(g, key=lambda n: n.pitch)
                gold.append(([n for n in g if truth_of[id(n)].hand == "L"],
                             [n for n in g if truth_of[id(n)].hand == "R"]))
            out.append((name, groups, gold, truth_of))
    return out


def accuracy(groups, assignment, truth_of):
    ok = n = 0
    for lh, rh in assignment:
        for side, hn in (("L", lh), ("R", rh)):
            for note in hn:
                ok += truth_of[id(note)].hand == side
                n += 1
    return ok / n


def main():
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    data = examples()
    gold_f = [tr.hand_features(g, gold) for _, g, gold, _ in data]
    w = dict(tr.HAND_WEIGHTS)
    total = defaultdict(float)
    steps = 0
    rng = np.random.default_rng(0)
    for epoch in range(epochs):
        accs = []
        for i in rng.permutation(len(data)):
            name, groups, gold, truth_of = data[i]
            tr.HAND_WEIGHTS.clear()
            tr.HAND_WEIGHTS.update(w)
            found = tr.assign_hands(groups)
            accs.append(accuracy(groups, found, truth_of))
            ff = tr.hand_features(groups, found)
            n = max(1, len(groups))
            for k in set(ff) | set(gold_f[i]):
                # per-group feature difference keeps the step size independent of length
                w[k] = max(0.0, w.get(k, 0.0) + 0.5 * (ff[k] - gold_f[i][k]) / n * 10)
            steps += 1
            for k, v in w.items():
                total[k] += v
        print(f"epoch {epoch + 1}: train hand accuracy {np.mean(accs):.1%}")
    avg = {k: round(total[k] / steps, 3) for k in w}
    tr.HAND_WEIGHTS.clear()
    tr.HAND_WEIGHTS.update(avg)
    print("HAND_WEIGHTS =", avg)
    print("test (bench seeds):")
    for name in PIECES:
        rs = [bench.run(name, s) for s in (1, 2, 3)]
        print(f"  {name:10} hand {np.mean([r['hand'] for r in rs]):.1%}")
    print(f"  MEAN       hand {np.mean([np.mean([bench.run(n, s)['hand'] for s in (1, 2, 3)]) for n in PIECES]):.1%}")


if __name__ == "__main__":
    main()
