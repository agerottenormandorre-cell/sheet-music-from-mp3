"""Window for correcting the beats and bar lines of a finished recording.

The notes as played (performance.mid) are drawn as a piano roll under the beat grid
the score was built on (beats.json). Beats can be dragged, added and deleted, bar
lines set, the meter changed from a bar on; the recording can be played with a click
on every beat. "Применить" saves beats_edit.json; rebuilding the score then uses it.
"""

import json
import tempfile
import threading
import time
import tkinter as tk
import winsound
from pathlib import Path
from tkinter import ttk

import numpy as np

from trim_editor import BG, END_C, FONT, GRID, INK, MUTED, PANEL, PLAY_C, START_C, fmt, time_step

BEAT_C, BAR_C, SEL_C, NOTE_C = "#6E6887", "#8B7BE8", "#FDE047", "#E9A23B"
RULER = 30
NEAR = 7          # px: how close a click must be to grab a beat
MIN_GAP = 0.05    # s between beats
MIN_VIEW = 1.0    # s: deepest zoom
CLICK_HZ = (1000, 1600)  # beat, first beat of a bar


def load_beats(out_dir):
    """The edited beats if there are any, otherwise those the score was built on."""
    for name in ("beats_edit.json", "beats.json"):
        f = Path(out_dir) / name
        if f.exists():
            d = json.loads(f.read_text(encoding="utf-8"))
            return [float(t) for t in d["beats"]], set(int(i) for i in d["bars"])
    raise FileNotFoundError("beats.json")


def auto_beats(out_dir):
    """Beats and bar lines as the program finds them by itself."""
    import transcribe as tr

    d = np.load(Path(out_dir) / "beats.npz")
    beats, _ = tr.dp_beats(d["act"])
    first, lengths = tr.track_bars(beats, d["down"])
    bars, i = set(), first
    for m in lengths:
        if i < len(beats):
            bars.add(i)
        i += m
    return [float(t) for t in beats], bars


class BeatEditor(tk.Toplevel):
    """on_apply() is called after beats_edit.json was written."""

    def __init__(self, master, src, out_dir, offset, on_apply, button):
        super().__init__(master, bg=BG)
        self.src, self.out, self.offset, self.on_apply, self.button = src, Path(out_dir), offset, on_apply, button
        self.title(f"Доли и такты — {src.name}")
        self.geometry(f"{min(1280, self.winfo_screenwidth() - 60)}x{min(760, self.winfo_screenheight() - 80)}")
        self.minsize(900, 560)
        self.transient(master)

        import pretty_midi
        pm = pretty_midi.PrettyMIDI(str(self.out / "performance.mid"))
        self.notes = [(n.start, n.end, n.pitch, n.velocity) for n in pm.instruments[0].notes]
        self.beats, self.bars = load_beats(self.out)
        self.dur = max([n[1] for n in self.notes] + self.beats) + 1.0
        lo, hi = min(n[2] for n in self.notes), max(n[2] for n in self.notes)
        self.p_lo, self.p_hi = lo - 2, hi + 2
        self.v0, self.v1 = 0.0, min(self.dur, 20.0)
        self.sel, self.dragging, self.pan_from, self.cursor = None, False, None, None
        self.changed = False
        self.audio = None  # loaded in the background for playback
        self.play = None
        self.tmp = Path(tempfile.gettempdir()) / "play-me-mus-beats"
        self.tmp_n = 0

        self.build()
        self.protocol("WM_DELETE_WINDOW", self.close)
        for key, d in (("<Left>", -0.01), ("<Right>", 0.01), ("<Shift-Left>", -0.001), ("<Shift-Right>", 0.001)):
            self.bind(key, lambda e, d=d: self.nudge(d))
        self.bind("<Delete>", lambda e: self.delete_beat(self.sel) if self.sel is not None else None)
        self.bind("<space>", lambda e: self.toggle_play())
        self.bind("<Escape>", lambda e: self.close())
        threading.Thread(target=self.load_audio, daemon=True).start()
        self.after(150, lambda: (self.lift(), self.focus_force()))
        self.grab_set()

    # ---------------------------------------------------------------- layout

    def build(self):
        b = self.button
        head = tk.Frame(self, bg=BG)
        head.pack(fill="x", padx=18, pady=(14, 4))
        tk.Label(head, text=self.src.name, bg=BG, fg=INK, font=(FONT, 13, "bold")).pack(side="left")
        self.info = tk.Label(head, text="", bg=BG, fg=MUTED, font=(FONT, 10))
        self.info.pack(side="left", padx=12)
        tk.Label(self, text="Тянуть долю — сдвинуть · двойной щелчок — добавить долю · ПКМ по доле — удалить · "
                            "Ctrl+щелчок по доле — здесь начинается такт · ←/→ сдвиг выбранной доли на 0.01 с · "
                            "колесо — масштаб · Shift+колесо — прокрутка · Пробел — слушать с метрономом",
                 bg=BG, fg=MUTED, font=(FONT, 8), anchor="w", justify="left", wraplength=1200).pack(fill="x", padx=18)

        self.canvas = tk.Canvas(self, bg=PANEL, highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True, padx=18, pady=(6, 0))
        self.scroll = ttk.Scrollbar(self, orient="horizontal", command=self.xview)
        self.scroll.pack(fill="x", padx=18)

        tools = tk.Frame(self, bg=BG)
        tools.pack(fill="x", padx=18, pady=(10, 0))
        b(tools, "Начало такта (Ctrl+щелчок)", self.toggle_bar, small=True).pack(side="left")
        tk.Label(tools, text="  Размер с выбранной доли:", bg=BG, fg=INK, font=(FONT, 10)).pack(side="left")
        for n in (2, 3, 4, 5, 6):
            b(tools, f"{n}/4", lambda n=n: self.set_meter(n), small=True).pack(side="left", padx=(4, 0))
        b(tools, "Доли ×2", self.double, small=True).pack(side="left", padx=(18, 4))
        b(tools, "Доли ÷2", self.halve, small=True).pack(side="left")
        tk.Label(tools, text="(на видимом участке)", bg=BG, fg=MUTED, font=(FONT, 8)).pack(side="left", padx=4)
        b(tools, "Как было", self.reset, small=True).pack(side="right")

        bottom = tk.Frame(self, bg=BG)
        bottom.pack(fill="x", padx=18, pady=(12, 16))
        self.play_btn = b(bottom, "▶ Слушать с метрономом", self.toggle_play, small=True)
        self.play_btn.pack(side="left")
        b(bottom, "Показать всё", lambda: self.set_view(0, self.dur), small=True).pack(side="left", padx=8)
        self.status = tk.Label(bottom, text="", bg=BG, fg=MUTED, font=(FONT, 9))
        self.status.pack(side="left", padx=8)
        b(bottom, "Применить и пересобрать ноты", self.apply, primary=True).pack(side="right")
        b(bottom, "Закрыть", self.close).pack(side="right", padx=8)

        c = self.canvas
        c.bind("<Configure>", lambda e: self.redraw())
        c.bind("<ButtonPress-1>", self.press)
        c.bind("<Control-ButtonPress-1>", self.ctrl_press)
        c.bind("<B1-Motion>", self.drag)
        c.bind("<ButtonRelease-1>", lambda e: setattr(self, "dragging", False))
        c.bind("<Double-Button-1>", self.add_beat)
        c.bind("<ButtonPress-3>", self.right_press)
        c.bind("<ButtonPress-2>", lambda e: setattr(self, "pan_from", (e.x, self.v0)))
        c.bind("<B2-Motion>", self.pan)
        c.bind("<MouseWheel>", self.wheel)
        c.bind("<Shift-MouseWheel>", self.shift_wheel)
        c.bind("<Motion>", self.hover)

    # ---------------------------------------------------------------- geometry

    def size(self):
        return max(self.canvas.winfo_width(), 10), max(self.canvas.winfo_height(), RULER + 40)

    def x_of(self, t):
        return (t - self.v0) / (self.v1 - self.v0) * self.size()[0]

    def t_of(self, x):
        return self.v0 + x / self.size()[0] * (self.v1 - self.v0)

    def y_of(self, pitch):
        h = self.size()[1]
        return RULER + 6 + (self.p_hi - pitch) / (self.p_hi - self.p_lo) * (h - RULER - 12)

    def set_view(self, v0, span):
        span = min(max(span, MIN_VIEW), self.dur)
        v0 = min(max(v0, 0.0), self.dur - span)
        self.v0, self.v1 = v0, v0 + span
        self.redraw()

    def near_beat(self, x):
        if not self.beats:
            return None
        xs = np.array([self.x_of(t) for t in self.beats])
        i = int(np.argmin(np.abs(xs - x)))
        return i if abs(xs[i] - x) <= NEAR else None

    # ---------------------------------------------------------------- drawing

    def redraw(self):
        c = self.canvas
        w, h = self.size()
        c.delete("all")
        # octave lines and the notes
        for p in range(self.p_lo, self.p_hi + 1):
            if p % 12 == 0:
                y = self.y_of(p)
                c.create_line(0, y, w, y, fill=GRID)
                c.create_text(4, y - 2, text=f"C{p // 12 - 1}", anchor="sw", fill=MUTED, font=(FONT, 7))
        nh = max(2.0, (h - RULER - 12) / (self.p_hi - self.p_lo) - 1)
        for s, e, p, v in self.notes:
            if e < self.v0 or s > self.v1:
                continue
            y = self.y_of(p)
            shade = int(90 + v)  # louder notes brighter
            col = f"#{min(255, shade + 80):02x}{min(255, shade + 20):02x}{max(0, shade - 60):02x}"
            c.create_rectangle(self.x_of(s), y - nh / 2, max(self.x_of(e), self.x_of(s) + 2), y + nh / 2,
                               fill=col, outline="")
        # ruler
        c.create_rectangle(0, 0, w, RULER, fill=BG, outline="")
        span = self.v1 - self.v0
        step = time_step(span, w, 90)
        t = (self.v0 // step + 1) * step
        while t < self.v1:
            x = self.x_of(t)
            c.create_line(x, RULER - 6, x, RULER, fill=MUTED)
            c.create_text(x + 3, 3, text=fmt(t, 2 if step < 1 else 0), anchor="nw", fill=MUTED, font=(FONT, 8))
            t += step
        # beats and bar lines
        bar_no = {i: n + 1 for n, i in enumerate(sorted(self.bars))}
        for i, t in enumerate(self.beats):
            if not self.v0 - 1 <= t <= self.v1 + 1:
                continue
            x = self.x_of(t)
            is_bar = i in self.bars
            col = SEL_C if i == self.sel else BAR_C if is_bar else BEAT_C
            c.create_line(x, RULER, x, h, fill=col, width=3 if is_bar else 1, dash=() if is_bar else (3, 3))
            if is_bar:
                c.create_text(x + 3, RULER + 2, text=str(bar_no[i]), anchor="nw", fill=INK, font=(FONT, 9, "bold"))
        if self.cursor is not None:
            x = self.x_of(self.cursor)
            c.create_line(x, RULER, x, h, fill=START_C, width=1)
        self.scroll.set(self.v0 / self.dur, self.v1 / self.dur)
        self.update_info()

    def update_info(self):
        n_bars = len(self.bars)
        firsts = sorted(self.bars)
        lengths = [b - a for a, b in zip(firsts, firsts[1:])]
        meters = sorted(set(lengths)) or [4]
        bpm = 60 / np.median(np.diff(self.beats)) if len(self.beats) > 1 else 0
        self.info.config(text=f"долей: {len(self.beats)} · тактов: {n_bars} · размеры: "
                              f"{', '.join(f'{m}/4' for m in meters)} · ♩≈{bpm:.0f}"
                              + (" · изменено" if self.changed else ""))
        if self.sel is not None and self.sel < len(self.beats):
            self.status.config(text=f"выбрана доля {self.sel + 1}: {self.beats[self.sel]:.3f} с")
        else:
            self.status.config(text="")

    # ---------------------------------------------------------------- editing

    def edited(self):
        self.changed = True
        self.redraw()

    def press(self, e):
        self.canvas.focus_set()
        i = self.near_beat(e.x)
        self.sel = i
        self.dragging = i is not None
        if i is None:
            self.cursor = self.t_of(e.x)  # playback starts here
        self.redraw()

    def ctrl_press(self, e):
        i = self.near_beat(e.x)
        if i is not None:
            self.sel = i
            self.toggle_bar()
        return "break"

    def drag(self, e):
        if not self.dragging or self.sel is None:
            return
        i = self.sel
        lo = self.beats[i - 1] + MIN_GAP if i > 0 else 0.0
        hi = self.beats[i + 1] - MIN_GAP if i + 1 < len(self.beats) else self.dur
        self.beats[i] = min(max(self.t_of(e.x), lo), hi)
        self.edited()

    def nudge(self, d):
        if self.sel is None or isinstance(self.focus_get(), tk.Entry):
            return
        i = self.sel
        lo = self.beats[i - 1] + MIN_GAP if i > 0 else 0.0
        hi = self.beats[i + 1] - MIN_GAP if i + 1 < len(self.beats) else self.dur
        self.beats[i] = min(max(self.beats[i] + d, lo), hi)
        self.edited()

    def insert_beat(self, t):
        i = int(np.searchsorted(self.beats, t))
        if (i > 0 and t - self.beats[i - 1] < MIN_GAP) or (i < len(self.beats) and self.beats[i] - t < MIN_GAP):
            return None
        self.beats.insert(i, t)
        self.bars = {b + 1 if b >= i else b for b in self.bars}
        return i

    def add_beat(self, e):
        if self.near_beat(e.x) is None:
            i = self.insert_beat(self.t_of(e.x))
            if i is not None:
                self.sel = i
                self.edited()

    def delete_beat(self, i):
        if i is None or len(self.beats) < 3:
            return
        was_bar = i in self.bars
        del self.beats[i]
        self.bars = {b - 1 if b > i else b for b in self.bars if b != i}
        if was_bar and i < len(self.beats):
            self.bars.add(i)  # the bar now starts on the next beat
        self.sel = None
        self.edited()

    def right_press(self, e):
        self.delete_beat(self.near_beat(e.x))

    def toggle_bar(self):
        if self.sel is None:
            return
        if self.sel in self.bars:
            if len(self.bars) > 1:
                self.bars.discard(self.sel)
        else:
            self.bars.add(self.sel)
        self.edited()

    def set_meter(self, n):
        """From the selected beat on, a bar line every n beats."""
        if self.sel is None:
            self.status.config(text="Сначала выберите долю, с которой начинается такт")
            return
        s = self.sel
        self.bars = {b for b in self.bars if b < s} | set(range(s, len(self.beats), n))
        self.edited()

    def visible(self):
        return [i for i, t in enumerate(self.beats) if self.v0 <= t <= self.v1]

    def double(self):
        vis = self.visible()
        for i in reversed(vis[:-1]):
            self.insert_beat((self.beats[i] + self.beats[i + 1]) / 2)
        self.sel = None
        self.edited()

    def halve(self):
        vis = self.visible()
        if len(vis) < 3:
            return
        for i in reversed(vis[1::2]):
            if i not in self.bars:
                self.delete_beat(i)
        self.edited()

    def reset(self):
        self.beats, self.bars = auto_beats(self.out)
        self.sel = None
        self.edited()

    # ---------------------------------------------------------------- view

    def pan(self, e):
        if self.pan_from:
            x0, v0 = self.pan_from
            self.set_view(v0 - (e.x - x0) / self.size()[0] * (self.v1 - self.v0), self.v1 - self.v0)

    def wheel(self, e):
        t, span = self.t_of(e.x), (self.v1 - self.v0) * (0.8 if e.delta > 0 else 1.25)
        span = min(max(span, MIN_VIEW), self.dur)
        self.set_view(t - e.x / self.size()[0] * span, span)
        return "break"

    def shift_wheel(self, e):
        span = self.v1 - self.v0
        self.set_view(self.v0 - span * 0.15 * (1 if e.delta > 0 else -1), span)
        return "break"

    def xview(self, *args):
        span = self.v1 - self.v0
        if args[0] == "moveto":
            self.set_view(float(args[1]) * self.dur, span)
        elif args[0] == "scroll":
            self.set_view(self.v0 + int(args[1]) * span * (0.9 if args[2] == "pages" else 0.1), span)

    def hover(self, e):
        near = self.near_beat(e.x) is not None
        self.canvas.config(cursor="sb_h_double_arrow" if near or self.dragging else "crosshair")

    # ---------------------------------------------------------------- playback with clicks

    def load_audio(self):
        try:
            import soundfile as sf

            audio, sr = sf.read(str(self.src), dtype="float32", always_2d=True)
            a = int(self.offset * sr)
            self.audio = (audio[a:], sr)
        except Exception as ex:
            self.audio = ex

    def toggle_play(self):
        if self.play:
            self.stop()
            return
        if self.audio is None:
            self.status.config(text="запись ещё загружается…")
            return
        if isinstance(self.audio, Exception):
            self.status.config(text=f"не удалось открыть запись: {self.audio}")
            return
        import soundfile as sf

        audio, sr = self.audio
        t0 = self.cursor if self.cursor is not None and self.v0 <= self.cursor <= self.v1 else self.v0
        t1 = min(self.dur, t0 + 60)
        seg = audio[int(t0 * sr):int(t1 * sr)].copy() * 0.8
        click_len = int(0.03 * sr)
        env = np.exp(-np.linspace(0, 6, click_len)).astype(np.float32)
        for i, t in enumerate(self.beats):
            if t0 <= t < t1:
                k = int((t - t0) * sr)
                hz = CLICK_HZ[1] if i in self.bars else CLICK_HZ[0]
                tone = 0.5 * np.sin(2 * np.pi * hz * np.arange(click_len) / sr).astype(np.float32) * env
                n = min(click_len, len(seg) - k)
                if n > 0:
                    seg[k:k + n] += tone[:n, None]
        np.clip(seg, -1, 1, out=seg)
        self.tmp_n ^= 1
        path = f"{self.tmp}{self.tmp_n}.wav"
        sf.write(path, seg, sr, subtype="PCM_16")
        winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT)
        self.play = (t0, t1, time.perf_counter())
        self.play_btn.config(text="■ Стоп")
        self.tick()

    def tick(self):
        if not self.play or not self.winfo_exists():
            return
        t0, t1, began = self.play
        t = t0 + time.perf_counter() - began
        self.canvas.delete("playhead")
        if t >= t1:
            self.stop()
            return
        if t > self.v1:  # follow the music
            self.set_view(t - (self.v1 - self.v0) * 0.1, self.v1 - self.v0)
        x = self.x_of(t)
        self.canvas.create_line(x, RULER, x, self.size()[1], fill=PLAY_C, width=2, tags="playhead")
        self.after(30, self.tick)

    def stop(self):
        if self.play:
            winsound.PlaySound(None, 0)
            self.play = None
            self.canvas.delete("playhead")
            self.play_btn.config(text="▶ Слушать с метрономом")

    # ---------------------------------------------------------------- closing

    def apply(self):
        data = {"beats": [round(t, 4) for t in self.beats], "bars": sorted(self.bars)}
        (self.out / "beats_edit.json").write_text(json.dumps(data), encoding="utf-8")
        self.close()
        self.on_apply()

    def close(self):
        self.stop()
        self.grab_release()
        self.destroy()
