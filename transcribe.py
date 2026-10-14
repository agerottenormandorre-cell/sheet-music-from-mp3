"""Piano recording (MP3/WAV/FLAC) -> editable two-staff score.

Pipeline:
    1. Transkun            audio -> performance MIDI (notes, velocities, sustain pedal)
    2. Beat This!          audio -> frame-wise beat / downbeat probabilities
    3. tempo-following DP  probabilities -> one consistent beat per quarter note,
                           following rubato but never jumping to half/double tempo
    4. quantization        per hand and per beat (so 3-against-4 between hands survives);
                           runs too fast for the grid become grace notes
    5. engraving           key-aware spelling, pedal marks, dynamics, tempo changes
    6. export              MusicXML (for MuseScore/Dorico), quantized MIDI, PDF preview

Usage:
    .venv\\Scripts\\python transcribe.py file.mp3 [more ...] [--start 30 --duration 20]
    .venv\\Scripts\\python transcribe.py                       (all audio files in this folder)

Output per recording, in out/<name>/:
    performance.mid   exactly what was played (listen to check the note recognition)
    score.musicxml    the score: open in MuseScore / Dorico / Sibelius to edit
    score.mid         the quantized score (listen to check the rhythm)
    score.pdf         printable preview
"""

import argparse
import bisect
import math
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np
import pretty_midi

ROOT = Path(__file__).resolve().parent
AUDIO_EXT = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}

BEAT_FPS = 50          # Beat This! frame rate
DP_FPS = 100           # resolution of the beat DP
CHORD_WINDOW = 0.035   # s: onsets closer than this are one chord
HAND_SPAN = 14         # semitones one hand covers comfortably
# hand separation (beam search): feature weights learned by bench/train_hands.py
HAND_WEIGHTS = {"span": 25.4, "finger": 20.0, "move_fast": 0.202, "move": 0.255, "move_any": 0.378,
                "jump": 1.868, "cross_enter": 0.106, "crossed": 0.611, "reg_left": 0.347, "reg_right": 0.234,
                "touching": 0.015, "stream_step": 0.183, "stream_near": 0.178, "switch": 0.317, "rhythm": 0.663}
MIN_DT = 0.03
HELD_MARGIN = 0.06     # a key released this soon after an onset no longer binds the hand
OCTAVE_TIME = 0.15     # s: the fastest leap is about an octave in this time
STREAM_DT = 0.3        # s: consecutive onsets closer than this form a line
BEAM = 48
REFIT_ROUNDS = 2       # quantize, move the beats to the notes, quantize again
REFIT_WIDTH = 1.125    # beats: how far around a beat its notes count
REFIT_FIT = 0.2        # beats: only onsets this close to their grid position count
REFIT_MAX = 0.15       # of a beat: largest correction per round
SYNC_WINDOW = 0.05     # s: onsets of both hands closer than this were played together
# Quantization is a likelihood: onsets scatter around the grid with TIMING_SIGMA
# (measured on real playing: ~0.05 beat), finer grids pay a prior penalty.
TIMING_SIGMA = 0.078
SUBDIVISIONS = [(1, 0.0), (2, 0.5), (4, 1.5), (3, 0.96), (8, 7.0), (6, 3.3), (5, 3.2), (7, 4.5)]
FAST_GRID = 1.013      # a grid of n >= 5 per beat only where the hand plays faster than FAST_GRID / n
SWITCH_PENALTY = 0.72        # switching between duple (16ths) and triple (triplets) grids
COLLISION_PENALTY = 6.25     # two onsets forced onto one grid slot (-> grace note)
ROLL_SIGMA = 0.044           # s: typical spread of a rolled chord / melody played after its chord
UNCERTAIN = 0.09             # beats off the grid: such notes are colored for checking by ear
UNCERTAIN_COLOR = "#C2185B"
LEGATO_GAP = Fraction(1, 2)  # rests shorter than this (in beats) are absorbed into the note
VOICE_HOLD = Fraction(1, 2)  # a key held this long (beats) after the hand's next onset is a voice ...
VOICE_MIN = Fraction(1)      # ... if it lasts at least this long
# bar lengths (beats) the meter tracker may choose, with their cost per bar; 2/4 is
# mostly written as 4/4 (2/4 bars are too short to read at these tempi)
METERS = {4: 0.0, 3: 0.3, 2: 1.5}
METER_CHANGE = 6.0     # cost of a change of meter (log probability units)
# MIDI velocity upper bounds for dynamics
DYNAMICS = [(45, "pp"), (56, "p"), (66, "mp"), (78, "mf"), (90, "f"), (128, "ff")]


# ------------------------------------------------------------------ audio / models

def decode(src, dst, start, duration):
    import soundfile as sf

    audio, sr = sf.read(str(src), always_2d=True)
    audio = audio.mean(axis=1)
    a = int(start * sr)
    b = a + int(duration * sr) if duration else len(audio)
    sf.write(str(dst), audio[a:b], sr)


_models = {}


def load_transkun(device):
    """Transkun is called in-process: its CLI needs pydub/ffmpeg and ncls (no Windows wheel)."""
    if "transkun" not in _models:
        import moduleconf
        import torch
        import transkun

        pre = Path(transkun.__file__).parent / "pretrained"
        conf = moduleconf.parseFromFile(str(pre / "2.0.conf"))
        model = conf["Model"].module.TransKun(conf=conf["Model"].config).to(device)
        ckpt = torch.load(str(pre / "2.0.pt"), map_location=device)
        model.load_state_dict(ckpt.get("best_state_dict", ckpt.get("state_dict")), strict=False)
        model.eval()
        _models["transkun"] = model
    return _models["transkun"]


def transcribe(wav, mid, device):
    import soundfile as sf
    import soxr
    import torch
    from transkun.Data import writeMidi

    model = load_transkun(device)
    audio, sr = sf.read(str(wav), dtype="float32", always_2d=True)
    if sr != model.fs:
        audio = soxr.resample(audio, sr, model.fs)
    with torch.no_grad():
        notes = model.transcribe(torch.from_numpy(audio).to(device), discardSecondHalf=False)
    writeMidi(notes).write(str(mid))


def beat_activations(wav, device):
    import soundfile as sf
    import torch
    from beat_this.inference import Audio2Frames

    if "beat_this" not in _models:
        _models["beat_this"] = Audio2Frames(checkpoint_path="final0", device=device)
    audio, sr = sf.read(str(wav), dtype="float32", always_2d=True)
    with torch.no_grad():
        beat, down = _models["beat_this"](audio, sr)
    return torch.sigmoid(beat).cpu().numpy(), torch.sigmoid(down).cpu().numpy()


# ------------------------------------------------------------------ beat tracking

def global_period(act, fps):
    """Dominant beat period, restricted to 60-160 bpm with a prior around 100 bpm."""
    x = act - act.mean()
    ac = np.correlate(x, x, "full")[len(x) - 1:]
    lags = np.arange(len(ac)) / fps
    ok = (lags >= 0.375) & (lags <= 1.0)
    prior = np.exp(-0.5 * (np.log2(lags[ok] / 0.6) / 0.9) ** 2)
    return lags[ok][np.argmax(ac[ok] * prior)]


def dp_beats(act, fps_in=BEAT_FPS, fps=DP_FPS, beta=60.0, gamma=1.5):
    """Viterbi over (time, period): rewards beat probability, penalizes tempo jumps
    between consecutive beats (beta) and drift from the global tempo (gamma).
    The period range 0.6..1.6x excludes half/double-tempo switches."""
    n = int(len(act) * fps / fps_in)
    x = np.interp(np.arange(n) * fps_in / fps, np.arange(len(act)), act)
    T = global_period(x, fps)
    periods = np.unique(np.round(T * fps * np.exp(np.linspace(np.log(0.6), np.log(1.6), 41))).astype(int))
    K = len(periods)
    lp = np.log(periods)
    trans = beta * (lp[:, None] - lp[None, :]) ** 2          # [j_prev, k]
    drift = gamma * (lp - np.log(T * fps)) ** 2

    S = np.full((n, K), -np.inf)
    back = np.full((n, K), -1, dtype=np.int32)
    for t in range(n):
        tp = t - periods
        valid = tp >= 0
        base = x[t] - drift
        S[t] = base
        if valid.any():
            ks = np.nonzero(valid)[0]
            cand = S[tp[ks]] - trans[:, ks].T                  # [k, j]
            j = np.argmax(cand, axis=1)
            best = cand[np.arange(len(ks)), j]
            take = best > 0
            S[t, ks[take]] += best[take]
            back[t, ks[take]] = j[take]

    tail = slice(max(0, n - periods.max()), n)
    t_end, k = np.unravel_index(np.argmax(S[tail]), S[tail].shape)
    t = t_end + tail.start
    beats = []
    while k >= 0 and t >= 0:
        beats.append(t)
        prev_k = back[t, k]
        t -= periods[k]
        k = prev_k
    return np.array(beats[::-1]) / fps, T


def downbeat_strength(beats, down):
    """Downbeat probability at each beat (strongest frame within 40 ms)."""
    out = []
    for t in beats:
        i = int(round(t * BEAT_FPS))
        out.append(float(down[max(0, i - 2):i + 3].max()) if len(down) else 0.0)
    return np.array(out)


def track_bars(beats, down):
    """Where the bars start and how many beats each has (2, 3 or 4), over the whole piece.

    Viterbi over (bar length, beat in bar): a bar line where the downbeat probability is
    high; a change of bar length costs METER_CHANGE, so the meter only changes where the
    downbeats keep saying so. Returns (index of the first downbeat, list of bar lengths
    from there to the last beat)."""
    d = downbeat_strength(beats, down)
    on, off = np.log(d + 0.05), np.log(1.05 - d)
    states = [(m, k) for m in METERS for k in range(m)]
    S = len(states)
    score = np.array([on[0] if k == 0 else off[0] for m, k in states]) - [METERS[m] * (k == 0) for m, k in states]
    back = []
    for i in range(1, len(beats)):
        new = np.full(S, -np.inf)
        bp = np.zeros(S, dtype=int)
        for j, (m, k) in enumerate(states):
            if k > 0:
                src = states.index((m, k - 1))
                new[j], bp[j] = score[src], src
            else:  # a bar starts: from the last beat of any bar
                for m0 in METERS:
                    src = states.index((m0, m0 - 1))
                    v = score[src] - METERS[m] - (METER_CHANGE if m0 != m else 0)
                    if v > new[j]:
                        new[j], bp[j] = v, src
            new[j] += on[i] if k == 0 else off[i]
        score = new
        back.append(bp)
    j = int(np.argmax(score))
    path = [j]
    for bp in reversed(back):
        j = int(bp[j])
        path.append(j)
    path.reverse()
    starts = [i for i, j in enumerate(path) if states[j][1] == 0]
    if not starts:
        return 0, [4]
    lengths = [states[path[i]][0] for i in starts]
    return starts[0], lengths


class Bars:
    """The bar lines of the score: start (in beats from the score's start) and length of each bar."""

    def __init__(self, lengths):
        self.lengths = list(lengths) or [4]
        self.starts = [Fraction(0)]
        for m in self.lengths[:-1]:
            self.starts.append(self.starts[-1] + m)

    def extend_to(self, end):
        while self.end < end:
            self.starts.append(self.end)
            self.lengths.append(self.lengths[-1])

    @property
    def end(self):
        return self.starts[-1] + self.lengths[-1]

    def index(self, q):
        """Bar containing score position q (positions before the first bar map to bar 0)."""
        return max(0, bisect.bisect_right(self.starts, q) - 1)

    def __len__(self):
        return len(self.lengths)

    def signature(self):
        names = []
        for m in self.lengths:
            if not names or names[-1] != f"{m}/4":
                names.append(f"{m}/4")
        return ", ".join(dict.fromkeys(names))


# ------------------------------------------------------------------ quantization

def extend_beats(beats, t_end):
    """Extrapolate the beat grid so every note (incl. before the first beat) maps."""
    ibi = float(np.median(np.diff(beats)))
    pre = []
    t = beats[0] - ibi
    while t > -2 * ibi:
        pre.insert(0, t)
        t -= ibi
    post = []
    t = beats[-1] + ibi
    while t < t_end + 2 * ibi:
        post.append(t)
        t += ibi
    return np.concatenate([pre, beats, post]), len(pre)


def group_onsets(notes):
    groups, cur = [], [notes[0]]
    for n in notes[1:]:
        if n.start - cur[0].start < CHORD_WINDOW:
            cur.append(n)
        else:
            groups.append(cur)
            cur = [n]
    groups.append(cur)
    return groups


def split_hands(pitches, centers):
    """Split a sorted chord into (left, right) minimizing hand movement and stretch."""
    c_left, c_right = centers
    best, best_cost = None, math.inf
    for k in range(len(pitches) + 1):
        lh, rh = pitches[:k], pitches[k:]
        cost = 0.0
        for hand, c in ((lh, c_left), (rh, c_right)):
            if not hand:
                continue
            cost += abs(np.mean(hand) - c)
            cost += 3 * max(0, hand[-1] - hand[0] - HAND_SPAN)
            cost += 10 * max(0, len(hand) - 5)
        if cost < best_cost:
            best, best_cost = (lh, rh), cost
    lh, rh = best
    if lh:
        c_left = 0.6 * c_left + 0.4 * np.mean(lh)
    if rh:
        c_right = 0.6 * c_right + 0.4 * np.mean(rh)
    return lh, rh, (c_left, c_right)


class Event:
    """One onset of one hand: a note or chord at a beat position."""

    def __init__(self, pos, time, notes, end_pos):
        self.pos = pos            # position in beats (float)
        self.time = time          # seconds
        self.notes = notes        # pretty_midi notes
        self.end_pos = end_pos    # latest note end, in beats
        self.q = None             # quantized position (Fraction)
        self.sub = 2              # grid of its beat
        self.graces = []          # list of pitch lists, played just before
        self.element = None       # its first written note or chord


def merge_cost(main, other):
    """Cost of writing `other` as part of `main`'s chord. Only for notes a third or
    more apart from the chord (a step apart means a run, which stays melodic)."""
    gap = min(abs(a.pitch - b.pitch) for a in main.notes for b in other.notes)
    if gap < 3:
        return math.inf
    return (abs(main.time - other.time) / ROLL_SIGMA) ** 2 / 2


def assign_slots(evs, fr, sub):
    """Place a beat's onsets (in time order) on grid slots 0..sub, keeping their order.
    Each onset either takes a later slot than the previous one (paying its timing
    error) or shares the previous slot (paying the chord-roll / grace-note cost).
    This keeps an evenly played but slightly early/late stream on consecutive slots."""
    n, S = len(evs), sub + 1
    err = np.array([[((f - s / sub) / TIMING_SIGMA) ** 2 / 2 for s in range(S)] for f in fr])
    dp = np.full((n, S), np.inf)
    bp = np.zeros((n, S), dtype=int)
    dp[0] = err[0]
    for i in range(1, n):
        share = min(COLLISION_PENALTY, merge_cost(evs[i - 1], evs[i]))
        run_min, run_arg = np.inf, -1
        for s in range(S):
            if s > 0 and dp[i - 1, s - 1] < run_min:
                run_min, run_arg = dp[i - 1, s - 1], s - 1
            take, stay = run_min + err[i, s], dp[i - 1, s] + share
            dp[i, s], bp[i, s] = (take, run_arg) if take <= stay else (stay, s)
    s = int(np.argmin(dp[-1]))
    cost = dp[-1, s]
    slots = [s]
    for i in range(n - 1, 0, -1):
        s = int(bp[i, s])
        slots.append(s)
    return slots[::-1], cost


def quantize_hand(events):
    """Choose a grid per beat for one hand, then resolve grid collisions:
    rolled chords merge, faster runs become grace notes."""
    events.sort(key=lambda e: e.time)
    by_beat = defaultdict(list)
    for i, e in enumerate(events):
        e.ioi = events[i + 1].pos - e.pos if i + 1 < len(events) else math.inf
        e.ioi_in = e.pos - events[i - 1].pos if i > 0 else math.inf
        by_beat[int(math.floor(e.pos + 1 / 16))].append(e)
    subs = [s for s, _ in SUBDIVISIONS]
    b0, b1 = min(by_beat), max(by_beat)
    costs = np.full((b1 - b0 + 1, len(subs)), 0.0)
    slots_for = {}
    for b, evs in by_beat.items():
        fr = [e.pos - b for e in evs]
        # finer than 16ths only where the hand really plays faster than 16ths
        fastest = min(min(e.ioi for e in evs), getattr(evs[0], "ioi_in", math.inf))
        for si, (sub, penalty) in enumerate(SUBDIVISIONS):
            # grids of 5+ only for fast notes; odd tuplets only when the beat has (nearly) that many notes
            if (sub >= 5 and fastest > FAST_GRID / sub) or (sub in (5, 7) and len(evs) < sub - 1):
                costs[b - b0, si] = np.inf
                continue
            slots, cost = assign_slots(evs, fr, sub)
            costs[b - b0, si] = cost + penalty
            slots_for[b, si] = slots

    # Viterbi over beats: triplets only where they fit clearly better for a while
    triple = np.array([s % 3 == 0 for s in subs])
    trans = SWITCH_PENALTY * (triple[:, None] != triple[None, :])
    acc, back = costs[0].copy(), []
    for row in costs[1:]:
        m = acc[:, None] + trans
        bp = m.argmin(axis=0)
        back.append(bp)
        acc = m[bp, np.arange(len(subs))] + row
    path = [int(acc.argmin())]
    for bp in reversed(back):
        path.append(int(bp[path[-1]]))
    path.reverse()

    for b, evs in by_beat.items():
        si = path[b - b0]
        for e, s in zip(evs, slots_for[b, si]):
            e.q = b + Fraction(s, subs[si])
            e.sub = subs[si]

    events.sort(key=lambda e: e.time)
    out, pending = [], []
    i = 0
    while i < len(events):
        j = i
        while j + 1 < len(events) and events[j + 1].q == events[i].q:
            j += 1
        same = events[i:j + 1]
        main = min(same, key=lambda e: abs(e.pos - float(e.q)))
        for e in same:
            if e is main:
                continue
            if merge_cost(main, e) < COLLISION_PENALTY:
                main.notes += e.notes
                main.end_pos = max(main.end_pos, e.end_pos)
            elif e.time < main.time:
                pending.append(sorted(n.pitch for n in e.notes))
            else:
                # played after the beat but too fast for the grid: lead into the next note
                pending_after = sorted(n.pitch for n in e.notes)
                main.graces_after = getattr(main, "graces_after", []) + [pending_after]
        main.graces = pending
        pending = getattr(main, "graces_after", [])
        out.append(main)
        i = j + 1
    return out


class HandState:
    """Where one hand is: centre and time of its last onset, keys it still holds down,
    time between its last two onsets."""
    __slots__ = ("c", "t", "held", "ioi")

    def __init__(self, c, t, held, ioi=None):
        self.c, self.t, self.held, self.ioi = c, t, held, ioi


class Terms(dict):
    def __missing__(self, key):
        return 0.0


def hand_step(state, notes, t, f):
    """One hand plays `notes` (pretty_midi) at time t: adds its features to f, returns its next state."""
    pitches = [n.pitch for n in notes]
    held = [n for n in state.held if n.end > t + HELD_MARGIN]
    keys = pitches + [n.pitch for n in held]
    f["span"] += max(0, max(keys) - min(keys) - HAND_SPAN) ** 2
    f["finger"] += max(0, len(set(keys)) - 5)
    c = float(np.mean(pitches))
    ioi = None
    if state.t is not None:
        dt = max(t - state.t, MIN_DT)
        octaves = abs(c - state.c) / 12
        f["move_fast"] += octaves * min(1.0, 0.15 / dt)
        f["move"] += octaves * min(1.0, 0.5 / dt)
        f["move_any"] += octaves
        f["jump"] += max(0.0, octaves - dt / OCTAVE_TIME) ** 2
        ioi = dt
        if state.ioi is not None and max(dt, state.ioi) < 1.0:
            # a hand's figuration tends to keep its pace
            f["rhythm"] += min(2.0, abs(math.log2(dt / state.ioi)))
    return HandState(c, t, held + list(notes), ioi)


def hands_step(L, R, last, t, lh, rh, f):
    """Both hands at one onset group: adds the features of this step to f.
    `last` is (time, [(pitch, hand)], hand of a single-hand group or None)."""
    nl = hand_step(L, lh, t, f) if lh else L
    nr = hand_step(R, rh, t, f) if rh else R
    crossed = nl.c > nr.c + 1
    f["cross_enter"] += crossed and not (L.c > R.c + 1)
    if lh:  # register and crossing count where the hand plays, not while it waits
        f["crossed"] += crossed
        f["reg_left"] += max(0, nl.c - 64) / 12
    if rh:
        f["reg_right"] += max(0, 57 - nr.c) / 12
    if lh and rh:  # both hands on neighbouring keys at once
        f["touching"] += max(0, 3 - min(abs(a.pitch - b.pitch) for a in lh for b in rh))
    played = [(n.pitch, "L") for n in lh] + [(n.pitch, "R") for n in rh]
    single = None if lh and rh else "L" if lh else "R"
    if last and t - last[0] < STREAM_DT:
        # a fast line stays in one hand: each note continues the nearest note before it
        for pitch, side in played:
            p0, s0 = min(last[1], key=lambda x: abs(x[0] - pitch))
            if s0 != side:
                d = abs(p0 - pitch)
                f["stream_step"] += d <= 2
                f["stream_near"] += 2 < d <= 5
        f["switch"] += single is not None and last[2] is not None and single != last[2]
    return nl, nr, (t, played, single)


def step_cost(f):
    return sum(HAND_WEIGHTS.get(k, 0.0) * v for k, v in f.items())


def hand_options(g):
    """A chord splits into a lower and an upper block; the left hand takes the lower
    one, or the upper one when the hands are crossed."""
    options = set()
    for k in range(len(g) + 1):
        options.add((k, False))
        if 0 < k < len(g):
            options.add((k, True))
    for k, crossed in sorted(options):
        low, high = g[:k], g[k:]
        yield (high, low) if crossed else (low, high)


def assign_hands(groups):
    """Beam search over chord groups for the cheapest way to play them with two hands.
    The cost is a weighted sum of playing features (HAND_WEIGHTS, learned on scores
    by bench/train_hands.py). Returns per group (left notes, right notes)."""
    beam = [(0.0, HandState(48.0, None, []), HandState(67.0, None, []), None, None)]
    history = []
    for g in groups:
        g = sorted(g, key=lambda n: n.pitch)
        t = float(np.mean([n.start for n in g]))
        nxt = {}
        for bi, (cost0, L, R, _, last) in enumerate(beam):
            for lh, rh in hand_options(g):
                f = Terms()
                nl, nr, played = hands_step(L, R, last, t, lh, rh, f)
                cost = cost0 + step_cost(f)
                key = (round(nl.c), round(nr.c), len(lh), played[2])
                if key not in nxt or cost < nxt[key][0]:
                    nxt[key] = (cost, nl, nr, (bi, lh, rh), played)
        beam = sorted(nxt.values(), key=lambda s: s[0])[:BEAM]
        history.append(beam)
    out = []
    i = 0
    for step in range(len(history) - 1, -1, -1):
        bi, lh, rh = history[step][i][3]
        out.append((lh, rh))
        i = bi
    return out[::-1]


def hand_features(groups, assignment):
    """Summed features of a given assignment (for learning the weights)."""
    f = Terms()
    L, R, last = HandState(48.0, None, []), HandState(67.0, None, []), None
    for g, (lh, rh) in zip(groups, assignment):
        t = float(np.mean([n.start for n in g]))
        L, R, last = hands_step(L, R, last, t, lh, rh, f)
    return f


def hand_events(hand_notes, beat_of):
    """Per hand a time-ordered list of Events from the (left, right) note lists per group."""
    hands = {"L": [], "R": []}
    for lh, rh in hand_notes:
        for side, hn in (("L", lh), ("R", rh)):
            if hn:
                t0 = float(np.mean([n.start for n in hn]))
                hands[side].append(Event(beat_of(t0), t0, list(hn), max(beat_of(n.end) for n in hn)))
    return hands


def refit_beats(hands, grid, origin):
    """Move each beat to where the quantized notes around it say it is.

    Every placed onset is a (score position, time) pair; around each beat a weighted
    straight line through these pairs gives the beat's time. Only onsets on beats and
    half beats that fitted well count, and the correction is limited, so a wrong
    quantization cannot pull the beats after it."""
    # anchors: onsets on beats and half beats that fitted the grid well
    pts = np.array([(float(e.q) + origin, e.time) for evs in hands.values() for e in evs
                    if abs(e.pos - float(e.q)) < REFIT_FIT and (e.q * 2).denominator == 1])
    if len(pts) < 4:
        return grid
    new = grid.copy()
    ibi = np.diff(grid)
    for i in range(len(grid)):
        w = np.exp(-0.5 * ((pts[:, 0] - i) / REFIT_WIDTH) ** 2)
        near = w > 0.1
        if w.sum() < 3 or not near.any() or np.ptp(pts[near, 0]) < 0.5:
            continue
        x, y = pts[:, 0] - i, pts[:, 1]
        sw, sx, sy = w.sum(), (w * x).sum(), (w * y).sum()
        sxx, sxy = (w * x * x).sum(), (w * x * y).sum()
        det = sw * sxx - sx * sx
        if det <= 1e-9:
            continue
        a = (sxx * sy - sx * sxy) / det  # fitted time at the beat
        step = ibi[min(i, len(ibi) - 1)]
        new[i] = grid[i] + np.clip(a - grid[i], -REFIT_MAX * step, REFIT_MAX * step)
    # keep the beats in order
    for i in range(1, len(new)):
        new[i] = max(new[i], new[i - 1] + 0.1 * ibi[min(i - 1, len(ibi) - 1)])
    return new


def align_hands(hands):
    """Onsets the two hands played together get the same score position.

    Of two such onsets the one that sits worse on its grid moves to the other's
    position, if that keeps its own hand in order. Returns how many moved."""
    moved = 0
    left, right = hands["L"], hands["R"]
    j = 0
    for e in left:
        while j < len(right) and right[j].time < e.time - SYNC_WINDOW:
            j += 1
        k = j
        while k < len(right) and right[k].time <= e.time + SYNC_WINDOW:
            r = right[k]
            if r.q != e.q:
                # the one whose own grid has the other's position moves; if both do,
                # the worse fitted one
                fits_e = (r.q * e.sub).denominator == 1
                fits_r = (e.q * r.sub).denominator == 1
                if fits_e != fits_r:
                    e_moves = fits_e
                else:
                    e_moves = abs(e.pos - float(e.q)) > abs(r.pos - float(r.q))
                a, b, own = (e, r, left) if e_moves else (r, e, right)
                i = own.index(a)
                lo = own[i - 1].q if i > 0 else None
                hi = own[i + 1].q if i + 1 < len(own) else None
                if (lo is None or lo < b.q) and (hi is None or b.q < hi):
                    a.q, a.sub = b.q, b.sub
                    moved += 1
            k += 1
    return moved


def quantize(notes, grid, origin, hand_notes=None):
    """Performance notes + beat times -> ({"L"/"R": quantized Events}, refined beat times).

    grid[i] is the time of beat i; score position 0 is beat `origin`. Hands are
    assigned once; quantization and beat refitting then alternate REFIT_ROUNDS times."""
    if hand_notes is None:
        hand_notes = assign_hands(group_onsets(notes))
    for rnd in range(REFIT_ROUNDS + 1):
        idx = np.arange(len(grid), dtype=float)
        beat_of = lambda t, g=grid: float(np.interp(t, g, idx)) - origin
        hands = {side: quantize_hand(evs) if evs else []
                 for side, evs in hand_events(hand_notes, beat_of).items()}
        align_hands(hands)
        if rnd < REFIT_ROUNDS:
            grid = refit_beats(hands, grid, origin)
    return hands, grid


# ------------------------------------------------------------------ voices and note values

class Chunk:
    """The part of one Event written in one voice of its staff."""

    def __init__(self, event, voice, notes, end_pos, held=None):
        self.event, self.voice, self.notes = event, voice, notes
        self.q = event.q
        self.end_pos = end_pos   # release of its notes, in beats
        self.held = held         # for a held note: its written end
        self.dur = None
        self.element = None


def split_voices(events, beat_of):
    """One hand's events -> Chunks in voice 1 (stems up) or 2 (stems down).

    A note the finger holds while the same hand plays on is a voice of its own:
    held on top it is a melody (voice 1) over the moving notes (voice 2), held at
    the bottom a bass (voice 2) under them. Without held notes everything is voice 1."""
    chunks = []
    busy = {1: Fraction(-1), 2: Fraction(-1)}  # until when a held note occupies the voice
    for i, e in enumerate(events):
        later = [x.q for x in events[i + 1:]]
        ends = {id(n): beat_of(n.end) for n in e.notes}
        first_next = e.q + Fraction(1, e.sub)

        def held_to(n):
            end = Fraction(round(ends[id(n)] * 2), 2)
            for q2 in later:
                if q2 >= end:
                    break
                if q2 >= first_next and end - q2 >= VOICE_HOLD and end - e.q >= VOICE_MIN:
                    return end
            return None

        held = {id(n): held_to(n) for n in e.notes}
        S = [n for n in e.notes if held[id(n)] is not None]
        M = [n for n in e.notes if held[id(n)] is None]
        if S and M and not (min(n.pitch for n in S) > max(n.pitch for n in M)
                            or max(n.pitch for n in S) < min(n.pitch for n in M)):
            S, M = [], list(e.notes)  # held notes in the middle of a chord: keep the chord
        if S:
            until = max(held[id(n)] for n in S)
            if M:
                upper = min(n.pitch for n in S) > max(n.pitch for n in M)
            else:  # a single held note: above or below what the hand plays meanwhile?
                during = [n.pitch for x in events[i + 1:] if x.q < until for n in x.notes]
                upper = not during or np.median(during) < np.mean([n.pitch for n in S])
            sv, mv = (1, 2) if upper else (2, 1)
            chunks.append(Chunk(e, sv, S, max(ends[id(n)] for n in S), until))
            busy[sv] = until
            if M:
                chunks.append(Chunk(e, mv, M, max(ends[id(n)] for n in M)))
        else:
            free = [v for v in (1, 2) if busy[v] <= e.q]
            v = 1 if len(free) != 1 else free[0]
            if not free:
                v = min((1, 2), key=lambda x: busy[x])
            chunks.append(Chunk(e, v, list(e.notes), max(ends.values())))
    return chunks


def note_values(chunks, total):
    """Written length of every chunk. Within its voice a note runs on to the next
    onset unless a real rest (LEGATO_GAP or longer) follows; then its release is
    rounded to the half beat. A held note keeps its held length."""
    for v in (1, 2):
        cs = sorted((c for c in chunks if c.voice == v), key=lambda c: c.q)
        for i, c in enumerate(cs):
            nxt = cs[i + 1].q if i + 1 < len(cs) else total
            sub = c.event.sub
            if c.held is not None:
                end = max(c.held, c.q + Fraction(1, sub))
            else:
                end = Fraction(round(c.end_pos * sub), sub)
            if nxt - end < LEGATO_GAP:
                end = nxt
            elif c.held is None:
                end = min(max(Fraction(math.ceil(c.end_pos * 2), 2), c.q + Fraction(1, sub)), nxt)
            else:
                end = min(end, nxt)
            # a value running from a triplet into a duple beat (or back) would need a
            # 12-tuplet: end it at the beat line and let a rest take the remainder
            if (c.q.denominator % 3 == 0) != (end.denominator % 3 == 0) and math.floor(end) > c.q:
                end = Fraction(math.floor(end))
            c.dur = end - c.q


def arrange(hands, beat_of, bars):
    """Quantized events of both hands -> ({"L"/"R": Chunks with note values}, total length).
    `bars` is cut or extended to end with the bar holding the last note."""
    chunks = {side: split_voices(evs, beat_of) for side, evs in hands.items()}
    last_end = max((c.q + max(Fraction(round(c.end_pos * 2), 2) - c.q, Fraction(1, c.event.sub))
                    for cs in chunks.values() for c in cs), default=Fraction(1))
    bars.extend_to(last_end)
    n = bars.index(last_end - Fraction(1, 96)) + 1
    del bars.lengths[n:], bars.starts[n:]
    total = bars.end
    for cs in chunks.values():
        note_values(cs, total)
    return chunks, total


# ------------------------------------------------------------------ engraving

def spelling_table(k):
    """Pitch-class -> note name for this key, incl. raised 6th/7th in minor."""
    from music21 import pitch

    names = {p.pitchClass: p.name for p in k.pitches}
    if k.mode == "minor":
        for iv in ("M6", "M7"):
            p = k.tonic.transpose(iv)
            names[p.pitchClass] = p.name
    flat = k.sharps < 0
    default = {1: "C#", 3: "E-", 6: "F#", 8: "G#", 10: "B-"}
    for pc in range(12):
        if pc in names:
            continue
        p = pitch.Pitch(pc)
        black_key = p.accidental is not None and p.accidental.alter != 0
        if k.sharps != 0 and black_key:
            want = "-" if flat else "#"
            if want not in p.name:
                p = p.getEnharmonic()
            names[pc] = p.name
        else:
            names[pc] = default.get(pc, p.name)
    return names


def make_pitch(midi, names):
    from music21 import pitch

    p = pitch.Pitch(names[midi % 12])
    p.octave = 4
    p.octave += round((midi - p.ps) / 12)
    return p


def make_element(pitches, names, ql=None, grace_type=None):
    from music21 import chord, note

    ps = [make_pitch(m, names) for m in sorted(set(pitches))]
    el = note.Note(ps[0]) if len(ps) == 1 else chord.Chord(ps)
    if grace_type:
        el.duration.type = grace_type
        el = el.getGrace()
    else:
        el.quarterLength = ql
    return el


def build_score(mid_path, act, down, out_xml, out_mid, mark_uncertain=True):
    from music21 import clef, dynamics, expressions, key, layout, meter, stream, tempo
    from music21 import note as m21note

    pm = pretty_midi.PrettyMIDI(str(mid_path))
    inst = pm.instruments[0]
    notes = sorted(inst.notes, key=lambda n: (n.start, n.pitch))
    if not notes:
        raise RuntimeError("no notes transcribed")
    t_end = max(n.end for n in notes)

    beats, T = dp_beats(act)
    first_down, lengths = track_bars(beats, down)
    grid, n_pre = extend_beats(beats, t_end)
    idx = np.arange(len(grid), dtype=float)
    to_beat = lambda t: float(np.interp(t, grid, idx))

    # the score starts on the last downbeat at or before the first note
    first = to_beat(notes[0].start)
    origin = n_pre + first_down
    while origin > first + 0.25:
        origin -= lengths[0]
        lengths.insert(0, lengths[0])
    while len(lengths) > 1 and origin + lengths[0] <= first + 0.25:
        origin += lengths.pop(0)
    bars = Bars(lengths)
    beat_of = lambda t: to_beat(t) - origin

    hands, grid = quantize(notes, grid, origin)
    to_beat = lambda t: float(np.interp(t, grid, idx))
    beat_of = lambda t: to_beat(t) - origin

    chunks, total = arrange(hands, beat_of, bars)

    # key from the performance itself, then spell every pitch in it
    probe = stream.Stream()
    for n in notes:
        probe.append(m21note.Note(n.pitch, quarterLength=max(0.25, n.end - n.start)))
    k = probe.analyze("key")
    names = spelling_table(k)

    score = stream.Score()
    parts = {}
    n_uncertain = 0
    for side in ("R", "L"):
        part = stream.PartStaff(id=side)  # two PartStaffs export as one piano part
        part.append(clef.TrebleClef() if side == "R" else clef.BassClef())
        part.append(key.KeySignature(k.sharps))
        part.append(meter.TimeSignature(f"{bars.lengths[0]}/4"))
        for b in range(1, len(bars)):
            if bars.lengths[b] != bars.lengths[b - 1]:
                part.insert(bars.starts[b], meter.TimeSignature(f"{bars.lengths[b]}/4"))
        used = sorted({c.voice for c in chunks[side]}) or [1]
        for v in used:
            line = stream.Voice(id=v) if len(used) > 1 else part
            cursor = Fraction(0)
            for c in sorted((c for c in chunks[side] if c.voice == v), key=lambda c: c.q):
                e = c.event
                if c.q > cursor:
                    line.append(m21note.Rest(quarterLength=c.q - cursor))
                if e.element is None:  # grace notes go before the event's first chunk
                    gtype = "16th" if len(e.graces) > 1 else "eighth"
                    for gp in e.graces:
                        g = make_element(gp, names, grace_type=gtype)
                        if mark_uncertain:
                            g.style.color = UNCERTAIN_COLOR
                        line.append(g)
                el = make_element([n.pitch for n in c.notes], names, ql=c.dur)
                if mark_uncertain and abs(e.pos - float(e.q)) > UNCERTAIN:
                    el.style.color = UNCERTAIN_COLOR
                    n_uncertain += e.element is None
                c.element = el
                if e.element is None:
                    e.element = el
                line.append(el)
                cursor = c.q + c.dur
            if total > cursor:
                line.append(m21note.Rest(quarterLength=total - cursor))
            if line is not part:
                part.insert(0, line)
        parts[side] = part

    rh, lh = parts["R"], parts["L"]
    n_bars = len(bars)

    # clef changes: a hand that stays out of its staff's range for a bar or more
    # switches clef (with hysteresis), as an engraver would
    for side, part in (("R", rh), ("L", lh)):
        pitches_by_bar = defaultdict(list)
        for e in hands[side]:
            pitches_by_bar[bars.index(e.q)] += [n.pitch for n in e.notes]
        treble = side == "R"
        for b in range(n_bars):
            ps = pitches_by_bar.get(b)
            if not ps:
                continue
            med = float(np.median(ps))
            if (treble and med < 55) or (not treble and med > 64):
                treble = not treble
                if b == 0:
                    part.remove(part.getElementsByClass(clef.Clef).first())
                part.insert(bars.starts[b], clef.TrebleClef() if treble else clef.BassClef())

    # tempo: 3-bar median of the tracked tempo; a new mark only for a lasting change > 15%
    beat_t = lambda b: float(np.interp(b + origin, idx, grid))
    tracked_end = len(beats) - 1 + n_pre - origin
    bar_bpm = [60.0 * m / (beat_t(float(b0 + m)) - beat_t(float(b0)))
               for b0, m in zip(bars.starts, bars.lengths) if b0 + m <= tracked_end]
    printed = None
    for i in range(len(bar_bpm)):
        smooth = np.median(bar_bpm[max(0, i - 1):i + 2])
        nxt = np.median(bar_bpm[i:i + 3])
        if printed is None or (abs(smooth / printed - 1) > 0.15 and abs(nxt / printed - 1) > 0.15):
            printed = int(round(smooth))
            rh.insert(bars.starts[i], tempo.MetronomeMark(number=printed))
    if printed is None:
        rh.insert(0, tempo.MetronomeMark(number=round(60 / T)))

    # dynamics: 3-bar median velocity, with hysteresis so the level does not flicker
    vel_by_bar = defaultdict(list)
    for n in notes:
        vel_by_bar[bars.index(beat_of(n.start))].append(n.velocity)
    bounds = [0] + [hi for hi, _ in DYNAMICS]
    level = None
    for b in range(n_bars):
        around = [v for j in (b - 1, b, b + 1) for v in vel_by_bar.get(j, [])]
        if not vel_by_bar.get(b) or not around:
            continue
        v = float(np.median(around))
        if level is not None and bounds[level] - 3 <= v < bounds[level + 1] + 3:
            continue
        new = next(i for i, (hi, _) in enumerate(DYNAMICS) if v < hi)
        if new != level:
            level = new
            rh.insert(bars.starts[b], dynamics.Dynamic(DYNAMICS[new][1]))

    # sustain pedal: Ped. ... * spanning the left-hand notes it holds
    downs = sorted((c.time, c.value) for c in inst.control_changes if c.number == 64)
    pedal_on = None
    for t, v in downs:
        if v >= 64 and pedal_on is None:
            pedal_on = t
        elif v < 64 and pedal_on is not None:
            a, b = beat_of(pedal_on) - 0.1, beat_of(t)
            held = [e.element for e in hands["L"] if a <= float(e.q) < b] or \
                   [e.element for e in hands["R"] if a <= float(e.q) < b]
            if held and b - a >= 0.6:  # shorter presses are flutter, not pedaling
                pm_ = expressions.PedalMark(*held)
                pm_.pedalType = expressions.PedalType.Sustain
                pm_.pedalForm = expressions.PedalForm.Symbol
                (lh if held[0] in lh else rh).insert(0, pm_)
            pedal_on = None

    score.insert(0, rh)
    score.insert(0, lh)
    rh.partName = "Piano"
    score.insert(0, layout.StaffGroup([rh, lh], name="Piano", symbol="brace", barTogether=True))
    score.makeNotation(inPlace=True)
    for part in (rh, lh):  # bars where a hand has one voice are written without voices
        for m in part.getElementsByClass(stream.Measure):
            if m.voices and sum(1 for v in m.voices if v.notes) <= 1:
                voices = list(m.voices)
                keep = next((v for v in voices if v.notes), voices[0])
                for v in voices:
                    if v is not keep:
                        m.remove(v)
                m.flattenUnnecessaryVoices(force=True, inPlace=True)
            for i, v in enumerate(m.voices):  # makeNotation numbers voices from 0, MusicXML from 1
                v.id = i + 1
                if i:  # the second voice's rests only clutter a piano texture
                    for r in v.getElementsByClass(m21note.Rest):
                        r.style.hideObjectOnPrint = True
    score.write("musicxml", fp=str(out_xml))
    score.write("midi", fp=str(out_mid))
    ibi = np.diff(beats)
    return dict(notes=len(notes), bpm=round(60 / T), tempo_range=f"{60 / ibi.max():.0f}-{60 / ibi.min():.0f}",
                meter=bars.signature(), key=str(k), bars=n_bars, uncertain=n_uncertain)


# ------------------------------------------------------------------ PDF preview

def find_browser():
    for p in (r"C:\Program Files\Google\Chrome\Application\chrome.exe",
              r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"):
        if Path(p).exists():
            return p
    return shutil.which("chrome") or shutil.which("msedge")


def render_pdf(xml, pdf, title):
    """Engrave with Verovio, print to PDF with a headless browser."""
    import verovio

    tk = verovio.toolkit()
    tk.setOptions({"pageWidth": 2100, "pageHeight": 2970, "scale": 40, "adjustPageHeight": False,
                   "breaks": "auto", "footer": "none", "header": "none",
                   "pageMarginTop": 80, "pageMarginBottom": 80, "pageMarginLeft": 80, "pageMarginRight": 80})
    tk.loadData(Path(xml).read_text(encoding="utf-8"))  # loadFile fails on non-ASCII paths
    pages = [tk.renderToSVG(i) for i in range(1, tk.getPageCount() + 1)]
    html = ["<!doctype html><meta charset='utf-8'><title>%s</title>" % title,
            "<style>@page{size:A4;margin:0}body{margin:0}div{page-break-after:always;width:210mm;height:297mm}"
            "svg{width:100%;height:100%}</style>"]
    html += [f"<div>{svg}</div>" for svg in pages]
    browser = find_browser()
    if not browser:
        return None
    with tempfile.TemporaryDirectory() as td:
        page = Path(td) / "score.html"
        page.write_text("\n".join(html), encoding="utf-8")
        subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--user-data-dir={td}\\profile", f"--print-to-pdf={pdf}", page.as_uri()],
                       check=True, capture_output=True, timeout=180)
    return len(pages)


# ------------------------------------------------------------------ main

STEPS = {"decode": "decoding", "notes": "transcribing notes", "beats": "tracking beats",
         "score": "building score", "pdf": "rendering PDF"}


def process(src, out_root, device, start=0, duration=0, rescore=False, marks=True, pdf=True, step=None):
    """One recording -> out_root/<name>/. step(key) is called as each stage of STEPS begins."""
    step = step or (lambda key: print(f"[{src.name}] {STEPS[key]}", flush=True))
    out = Path(out_root) / src.stem
    out.mkdir(parents=True, exist_ok=True)
    wav, mid, act_file = out / "audio.wav", out / "performance.mid", out / "beats.npz"
    if rescore and mid.exists() and act_file.exists():
        data = np.load(act_file)
        act, down = data["act"], data["down"]
    else:
        step("decode")
        decode(src, wav, start, duration)
        step("notes")
        transcribe(wav, mid, device)
        step("beats")
        act, down = beat_activations(wav, device)
        np.savez_compressed(act_file, act=act, down=down)
        wav.unlink(missing_ok=True)
    step("score")
    info = build_score(mid, act, down, out / "score.musicxml", out / "score.mid", mark_uncertain=marks)
    if pdf:
        step("pdf")
        info["pdf_pages"] = render_pdf(out / "score.musicxml", out / "score.pdf", src.stem)
    return out, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="*", help="audio files (default: all in this folder)")
    ap.add_argument("--start", type=float, default=0, help="seconds to skip")
    ap.add_argument("--duration", type=float, default=0, help="seconds to keep (0 = all)")
    ap.add_argument("--out", default=str(ROOT / "out"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-pdf", action="store_true")
    ap.add_argument("--no-marks", action="store_true", help="do not color doubtful notes")
    ap.add_argument("--rescore", action="store_true",
                    help="reuse performance.mid and beats.npz, only rebuild the score")
    args = ap.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    inputs = [Path(p) for p in args.inputs] or sorted(p for p in ROOT.iterdir() if p.suffix.lower() in AUDIO_EXT)
    if not inputs:
        sys.exit("no audio files")

    failed = []
    for src in inputs:
        try:
            out, info = process(src, args.out, device, args.start, args.duration, args.rescore,
                                marks=not args.no_marks, pdf=not args.no_pdf)
            print(f"[{src.name}] done -> {out}  {info}", flush=True)
        except Exception as ex:  # keep going with the other files
            import traceback
            traceback.print_exc()
            failed.append(src.name)
            print(f"[{src.name}] FAILED: {ex}", flush=True)
    if failed:
        sys.exit(f"failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
