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
# Quantization is a likelihood: onsets scatter around the grid with TIMING_SIGMA
# (measured on real playing: ~0.05 beat), finer grids pay a prior penalty.
TIMING_SIGMA = 0.05
SUBDIVISIONS = [(1, 0.0), (2, 0.5), (4, 1.2), (3, 2.0), (8, 3.5), (6, 3.5)]
SWITCH_PENALTY = 2.0         # switching between duple (16ths) and triple (triplets) grids
COLLISION_PENALTY = 4.0      # two onsets forced onto one grid slot (-> grace note)
ROLL_SIGMA = 0.035           # s: typical spread of a rolled chord / melody played after its chord
UNCERTAIN = 0.09             # beats off the grid: such notes are colored for checking by ear
UNCERTAIN_COLOR = "#C2185B"
LEGATO_GAP = Fraction(1, 2)  # rests shorter than this (in beats) are absorbed into the note
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


def meter_and_phase(beats, down):
    """Beats per bar and which beat is the downbeat, from downbeat probabilities.
    Duple meters are written as 4/4 (2/4 bars are too short to read at these tempi)."""
    idx = np.clip(np.round(beats * BEAT_FPS).astype(int), 0, len(down) - 1)
    d = down[idx]

    def contrast(m, p):
        on = d[p::m]
        off = np.delete(d, np.arange(p, len(d), m))
        return on.mean() - off.mean() if len(on) and len(off) else -np.inf

    duple = max(contrast(2, p) for p in range(2))
    triple = max(contrast(3, p) for p in range(3))
    m = 3 if triple > duple + 0.05 else 4
    return m, int(np.argmax([contrast(m, p) for p in range(m)]))


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
        self.dur = None


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
            if (sub == 8 and fastest > 0.17) or (sub == 6 and fastest > 0.20):
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
    bar, phase = meter_and_phase(beats, down)
    grid, n_pre = extend_beats(beats, t_end)
    idx = np.arange(len(grid), dtype=float)
    to_beat = lambda t: float(np.interp(t, grid, idx))

    # the score starts on the last downbeat at or before the first note
    first = to_beat(notes[0].start)
    origin = n_pre + phase
    while origin > first + 0.25:
        origin -= bar
    beat_of = lambda t: to_beat(t) - origin

    # split chords between hands, build per-hand events
    hands = {"L": [], "R": []}
    centers = (50.0, 67.0)
    for g in group_onsets(notes):
        g = sorted(g, key=lambda n: n.pitch)
        lh, rh, centers = split_hands([n.pitch for n in g], centers)
        t0 = float(np.mean([n.start for n in g]))
        for side, ps in (("L", lh), ("R", rh)):
            hn = [n for n in g if n.pitch in ps]
            if hn:
                hands[side].append(Event(beat_of(t0), t0, hn, max(beat_of(n.end) for n in hn)))

    for side in hands:
        hands[side] = quantize_hand(hands[side]) if hands[side] else []

    last_end = max(max(e.q + max(Fraction(round(e.end_pos * 2), 2) - e.q, Fraction(1, e.sub)) for e in h)
                   for h in hands.values() if h)
    total = Fraction(max(1, math.ceil(last_end / bar)) * bar)

    # Piano writing: a note runs on to the hand's next onset unless a real rest
    # (an eighth or longer) follows; then its release is rounded to the half beat.
    for events in hands.values():
        for i, e in enumerate(events):
            nxt = events[i + 1].q if i + 1 < len(events) else total
            end = Fraction(round(e.end_pos * e.sub), e.sub)
            if nxt - end < LEGATO_GAP:
                end = nxt
            else:
                end = min(max(Fraction(math.ceil(e.end_pos * 2), 2), e.q + Fraction(1, e.sub)), nxt)
            # a value running from a triplet into a duple beat (or back) would need a
            # 12-tuplet: end it at the beat line and let a rest take the remainder
            if (e.q.denominator % 3 == 0) != (end.denominator % 3 == 0) and math.floor(end) > e.q:
                end = Fraction(math.floor(end))
            e.dur = end - e.q

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
        part.append(meter.TimeSignature(f"{bar}/4"))
        cursor = Fraction(0)
        for e in hands[side]:
            if e.q > cursor:
                part.append(m21note.Rest(quarterLength=e.q - cursor))
            gtype = "16th" if len(e.graces) > 1 else "eighth"
            for gp in e.graces:
                g = make_element(gp, names, grace_type=gtype)
                if mark_uncertain:
                    g.style.color = UNCERTAIN_COLOR
                part.append(g)
            el = make_element([n.pitch for n in e.notes], names, ql=e.dur)
            if mark_uncertain and abs(e.pos - float(e.q)) > UNCERTAIN:
                el.style.color = UNCERTAIN_COLOR
                n_uncertain += 1
            e.element = el
            part.append(el)
            cursor = e.q + e.dur
        if total > cursor:
            part.append(m21note.Rest(quarterLength=total - cursor))
        parts[side] = part

    rh, lh = parts["R"], parts["L"]
    n_bars = int(total / bar)

    # clef changes: a hand that stays out of its staff's range for a bar or more
    # switches clef (with hysteresis), as an engraver would
    for side, part in (("R", rh), ("L", lh)):
        pitches_by_bar = defaultdict(list)
        for e in hands[side]:
            pitches_by_bar[int(e.q // bar)] += [n.pitch for n in e.notes]
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
                part.insert(b * bar, clef.TrebleClef() if treble else clef.BassClef())

    # tempo: 3-bar median of the tracked tempo; a new mark only for a lasting change > 15%
    beat_t = lambda b: float(np.interp(b + origin, idx, grid))
    tracked_end = len(beats) - 1 + n_pre - origin
    bar_bpm = [60.0 * bar / (beat_t((i + 1) * bar) - beat_t(i * bar)) for i in range(n_bars)
               if (i + 1) * bar <= tracked_end]
    printed = None
    for i in range(len(bar_bpm)):
        smooth = np.median(bar_bpm[max(0, i - 1):i + 2])
        nxt = np.median(bar_bpm[i:i + 3])
        if printed is None or (abs(smooth / printed - 1) > 0.15 and abs(nxt / printed - 1) > 0.15):
            printed = int(round(smooth))
            rh.insert(i * bar, tempo.MetronomeMark(number=printed))
    if printed is None:
        rh.insert(0, tempo.MetronomeMark(number=round(60 / T)))

    # dynamics: 3-bar median velocity, with hysteresis so the level does not flicker
    vel_by_bar = defaultdict(list)
    for n in notes:
        vel_by_bar[int(beat_of(n.start) // bar)].append(n.velocity)
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
            rh.insert(b * bar, dynamics.Dynamic(DYNAMICS[new][1]))

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
    score.write("musicxml", fp=str(out_xml))
    score.write("midi", fp=str(out_mid))
    ibi = np.diff(beats)
    return dict(notes=len(notes), bpm=round(60 / T), tempo_range=f"{60 / ibi.max():.0f}-{60 / ibi.min():.0f}",
                meter=f"{bar}/4", key=str(k), bars=n_bars, uncertain=n_uncertain)


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
    tk.loadFile(str(xml))
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
        out = Path(args.out) / src.stem
        out.mkdir(parents=True, exist_ok=True)
        wav, mid, act_file = out / "audio.wav", out / "performance.mid", out / "beats.npz"
        try:
            if args.rescore and mid.exists() and act_file.exists():
                data = np.load(act_file)
                act, down = data["act"], data["down"]
            else:
                print(f"[{src.name}] decoding", flush=True)
                decode(src, wav, args.start, args.duration)
                print(f"[{src.name}] transcribing notes ({device})", flush=True)
                transcribe(wav, mid, device)
                print(f"[{src.name}] tracking beats", flush=True)
                act, down = beat_activations(wav, device)
                np.savez_compressed(act_file, act=act, down=down)
                wav.unlink(missing_ok=True)
            print(f"[{src.name}] building score", flush=True)
            info = build_score(mid, act, down, out / "score.musicxml", out / "score.mid",
                               mark_uncertain=not args.no_marks)
            if not args.no_pdf:
                pages = render_pdf(out / "score.musicxml", out / "score.pdf", src.stem)
                info["pdf_pages"] = pages
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
