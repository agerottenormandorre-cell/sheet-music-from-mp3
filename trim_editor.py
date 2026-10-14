"""Window for choosing where the music starts and ends.

Shows the spectrogram (log frequency) and loudness of a recording with two markers;
markers move in 0.01 s steps, the view zooms down to a fraction of a second, and the
selection can be played back (winsound, no extra libraries).
"""

import math
import tempfile
import time
import tkinter as tk
import winsound
from pathlib import Path
from tkinter import ttk

import numpy as np
from PIL import Image, ImageTk

BG, PANEL, INK, MUTED, GRID = "#16131F", "#221D33", "#ECE8F7", "#9A93B5", "#3A3352"
START_C, END_C, PLAY_C, WAVE_C = "#34D399", "#F472B6", "#FDE047", "#8B7BE8"
FONT = "Segoe UI"
RULER, WAVE = 28, 80     # px: time ruler on top, loudness strip at the bottom
STEP = 0.01              # s: marker precision
MIN_GAP = 0.05           # s: shortest selection
MIN_VIEW = 0.2           # s: deepest zoom
FMIN, FMAX = 30.0, 12000.0
TICKS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600]
MAGMA = [(0, 0, 4), (28, 16, 68), (79, 18, 123), (129, 37, 129), (181, 54, 122),
         (229, 80, 100), (251, 135, 97), (254, 194, 135), (252, 253, 191)]


def colormap():
    stops = np.array(MAGMA, float)
    x, t = np.linspace(0, 1, len(stops)), np.linspace(0, 1, 256)
    return np.stack([np.interp(t, x, stops[:, c]) for c in range(3)], 1).astype(np.uint8)


def fmt(t, decimals=2):
    m, s = divmod(max(t, 0.0), 60)
    return f"{int(m)}:{s:0{3 + decimals if decimals else 2}.{decimals}f}"


class Analysis:
    """Everything the editor draws, computed once per recording (in a background thread)."""

    def __init__(self, path):
        import soundfile as sf

        self.audio, self.sr = sf.read(str(path), dtype="float32", always_2d=True)
        self.dur = len(self.audio) / self.sr
        mono = self.audio.mean(axis=1)
        # one column per 5 ms (coarser only for recordings over 5 minutes)
        hop = max(1, round(self.sr * max(0.005, self.dur / 60000)))
        self.hop_s = hop / self.sr
        n = len(mono) // hop + 1
        blocks = np.zeros(n * hop, np.float32)
        blocks[:len(mono)] = mono
        blocks = blocks.reshape(n, hop)
        self.peak = np.abs(blocks).max(axis=1)
        power = np.convolve((blocks ** 2).mean(axis=1), np.ones(10) / 10, mode="same")  # 50 ms smoothing
        self.level_db = 10 * np.log10(power + 1e-12)
        self.fmax = min(FMAX, self.sr / 2)
        self.spec = self.spectrogram(mono, hop, n)
        self.auto = self.auto_bounds()

    def spectrogram(self, mono, hop, n, rows=200):
        """Power on a log-frequency axis, as 0..255 with high frequencies in row 0."""
        n_fft = 2048 if self.sr <= 48000 else 4096
        buf = np.zeros((n - 1) * hop + hop // 2 + n_fft, np.float32)
        buf[n_fft // 2:n_fft // 2 + len(mono)] = mono  # column i is centred on its 5 ms block
        win = np.hanning(n_fft).astype(np.float32)
        freqs = np.fft.rfftfreq(n_fft, 1 / self.sr)
        edges = np.geomspace(FMIN, self.fmax, rows + 1)
        fold = np.zeros((len(freqs), rows), np.float32)
        for r in range(rows):
            sel = (freqs >= edges[r]) & (freqs < edges[r + 1])
            if not sel.any():  # low rows are narrower than one FFT bin
                sel = np.abs(freqs - math.sqrt(edges[r] * edges[r + 1])) == \
                      np.abs(freqs - math.sqrt(edges[r] * edges[r + 1])).min()
            fold[sel, r] = 1 / sel.sum()
        out = np.empty((n, rows), np.float32)
        offs = np.arange(n_fft)
        for a in range(0, n, 1024):
            idx = (np.arange(a, min(a + 1024, n)) * hop + hop // 2)[:, None] + offs
            out[a:a + len(idx)] = (np.abs(np.fft.rfft(buf[idx] * win, axis=1)) ** 2) @ fold
        db = 10 * np.log10(out + 1e-12)
        top = np.percentile(db, 99.7)
        img = np.clip((db - (top - 75)) / 75, 0, 1)
        return (img[:, ::-1].T * 255).astype(np.uint8)

    def auto_bounds(self):
        """First and last moment louder than 45 dB below the loudest passage."""
        loud = np.flatnonzero(self.level_db > max(self.level_db.max() - 45, -70))
        if not len(loud):
            return 0.0, round(self.dur, 2)
        start = max(0.0, loud[0] * self.hop_s - 0.05)
        end = min(self.dur, (loud[-1] + 1) * self.hop_s + 0.15)
        return math.floor(start * 100) / 100, math.floor(end * 100) / 100


def clamp_marker(key, t, start, end, dur):
    """New (start, end) after moving one marker to t: 0.01 s grid, inside the file, in order."""
    t = round(min(max(t, 0.0), dur) / STEP) * STEP
    if key == "start":
        return round(max(0.0, min(t, end - MIN_GAP)), 2), end
    return start, round(min(dur, max(t, start + MIN_GAP)), 2)


def draw_loudness(c, a, c0, c1, w, top, height, tags=()):
    """Loudness of columns c0..c1 as a mirrored dB envelope, so quiet entries are visible."""
    idx = np.clip(np.linspace(c0, c1, w + 1).astype(int), 0, len(a.peak) - 1)
    peak = np.maximum.reduceat(a.peak, idx[:-1]) if idx[-1] > idx[0] else a.peak[idx[:-1]]
    v = np.clip((20 * np.log10(peak + 1e-9) + 60) / 60, 0, 1)
    mid, half = top + height / 2, height / 2 - 4
    xs = np.arange(w)
    pts = np.concatenate([np.stack([xs, mid - v * half], 1), np.stack([xs, mid + v * half], 1)[::-1]])
    c.create_polygon(*pts.ravel().tolist(), fill=WAVE_C, outline="", tags=tags)


def time_step(span, w, min_px):
    return next((s for s in TICKS if s / span * w >= min_px), TICKS[-1])


class TrimEditor(tk.Toplevel):
    """Full view of one recording; "Подтвердить" calls on_done(start, end), "Закрыть" changes nothing."""

    def __init__(self, master, src, analysis, bounds, on_done, button):
        super().__init__(master, bg=BG)
        self.src, self.on_done, self.bounds0 = src, on_done, bounds
        self.button = button
        self.title(f"Границы — {src.name}")
        self.geometry(f"{min(1200, self.winfo_screenwidth() - 60)}x{min(720, self.winfo_screenheight() - 80)}")
        self.minsize(860, 520)
        self.transient(master)
        self.a = None
        self.v0 = self.v1 = 0.0
        self.start = self.end = 0.0
        self.active, self.dragging, self.pan_from, self.play = "start", None, None, None
        self.tmp = Path(tempfile.gettempdir()) / "play-me-mus-preview"
        self.tmp_n = 0
        self.vars, self.mmss = {}, {}
        self.build()
        self.protocol("WM_DELETE_WINDOW", self.close)
        for key, d in (("<Left>", -STEP), ("<Right>", STEP), ("<Shift-Left>", -0.1), ("<Shift-Right>", 0.1)):
            self.bind(key, lambda e, d=d: self.key_nudge(d))
        self.bind("<space>", self.key_play)
        self.bind("<Escape>", lambda e: self.close())
        self.bind("<Return>", lambda e: self.confirm())
        self.after(1, lambda: self.loaded(analysis))
        self.grab_set()
        self.after(150, lambda: (self.lift(), self.focus_force()))  # keys work without a click first

    # ---------------------------------------------------------------- layout

    def build(self):
        b = self.button
        head = tk.Frame(self, bg=BG)
        head.pack(fill="x", padx=18, pady=(14, 6))
        tk.Label(head, text=self.src.name, bg=BG, fg=INK, font=(FONT, 13, "bold")).pack(side="left")
        self.info = tk.Label(head, text="", bg=BG, fg=MUTED, font=(FONT, 10))
        self.info.pack(side="left", padx=12)
        hint = tk.Frame(self, bg=BG)
        hint.pack(fill="x", padx=18)
        tk.Label(hint, text="ЛКМ — начало · ПКМ — конец · маркеры можно тянуть · колесо — масштаб · "
                            "Shift+колесо или средняя кнопка — прокрутка · ←/→ сдвиг на 0.01 с (Shift — 0.1 с) · "
                            "Пробел — слушать", bg=BG, fg=MUTED, font=(FONT, 8), anchor="w").pack(side="left")
        self.readout = tk.Label(hint, text="", bg=BG, fg=INK, font=("Consolas", 10))
        self.readout.pack(side="right")

        self.canvas = tk.Canvas(self, bg=PANEL, highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True, padx=18, pady=(6, 0))
        self.scroll = ttk.Scrollbar(self, orient="horizontal", command=self.xview)
        self.scroll.pack(fill="x", padx=18)

        markers = tk.Frame(self, bg=BG)
        markers.pack(fill="x", padx=18, pady=(10, 0))
        self.marker_box(markers, "start", "Начало", START_C).grid(row=0, column=0, sticky="w")
        self.marker_box(markers, "end", "Конец", END_C).grid(row=1, column=0, sticky="w", pady=(6, 0))
        markers.columnconfigure(1, weight=1)
        tools = tk.Frame(markers, bg=BG)
        tools.grid(row=0, column=2, sticky="e")
        b(tools, "Найти автоматически", self.use_auto, small=True).pack(side="left", padx=(0, 6))
        b(tools, "Весь файл", self.use_all, small=True).pack(side="left", padx=(0, 6))
        b(tools, "Показать всё", self.zoom_all, small=True).pack(side="left")
        self.selection = tk.Label(markers, text="", bg=BG, fg=INK, font=(FONT, 11, "bold"))
        self.selection.grid(row=1, column=2, sticky="e", pady=(6, 0))

        bottom = tk.Frame(self, bg=BG)
        bottom.pack(fill="x", padx=18, pady=(14, 16))
        for text, cmd in (("▶ Выбранное", lambda: self.play_range(self.start, self.end)),
                          ("▶ Первые 3 с", lambda: self.play_range(self.start, self.start + 3)),
                          ("▶ Последние 3 с", lambda: self.play_range(self.end - 3, self.end)),
                          ("■ Стоп", self.stop)):
            b(bottom, text, cmd, small=True).pack(side="left", padx=(0, 6))
        b(bottom, "Подтвердить", self.confirm, primary=True).pack(side="right")
        b(bottom, "Закрыть", self.close).pack(side="right", padx=8)

        c = self.canvas
        c.bind("<Configure>", lambda e: self.redraw())
        c.bind("<ButtonPress-1>", lambda e: self.press(e, "start"))
        c.bind("<ButtonPress-3>", lambda e: self.press(e, "end"))
        c.bind("<B1-Motion>", self.drag)
        c.bind("<B3-Motion>", self.drag)
        c.bind("<ButtonRelease-1>", self.release)
        c.bind("<ButtonRelease-3>", self.release)
        c.bind("<ButtonPress-2>", lambda e: setattr(self, "pan_from", (e.x, self.v0)))
        c.bind("<B2-Motion>", self.pan)
        c.bind("<MouseWheel>", self.wheel)
        c.bind("<Shift-MouseWheel>", self.shift_wheel)
        c.bind("<Motion>", self.hover)
        c.bind("<Leave>", lambda e: (c.delete("hover"), self.readout.config(text="")))

    def marker_box(self, parent, key, title, color):
        b = self.button
        f = tk.Frame(parent, bg=BG)
        tk.Label(f, text="▌", bg=BG, fg=color, font=(FONT, 12)).pack(side="left")
        tk.Label(f, text=title, bg=BG, fg=INK, font=(FONT, 10, "bold"), width=7, anchor="w").pack(side="left")
        var = tk.StringVar()
        e = tk.Entry(f, textvariable=var, width=8, font=("Consolas", 12), justify="right", bg=PANEL, fg=INK,
                     insertbackground=INK, relief="flat", highlightthickness=1, highlightbackground=GRID,
                     highlightcolor=color)
        e.pack(side="left", ipady=3)
        e.bind("<Return>", lambda ev: (self.entry_set(key), "break")[1])
        e.bind("<FocusOut>", lambda ev: self.entry_set(key))
        e.bind("<Up>", lambda ev: (self.nudge(key, STEP), "break")[1])
        e.bind("<Down>", lambda ev: (self.nudge(key, -STEP), "break")[1])
        tk.Label(f, text="с", bg=BG, fg=MUTED, font=(FONT, 10)).pack(side="left", padx=(4, 4))
        self.mmss[key] = tk.Label(f, text="", bg=BG, fg=MUTED, font=("Consolas", 9), width=8, anchor="w")
        self.mmss[key].pack(side="left")
        for d, text in ((-0.1, "−0.1"), (-STEP, "−0.01"), (STEP, "+0.01"), (0.1, "+0.1")):
            b(f, text, lambda d=d: self.nudge(key, d), small=True).pack(side="left", padx=(4, 0))
        b(f, "🔍", lambda: self.focus_on(key), small=True).pack(side="left", padx=(8, 0))
        self.vars[key] = var
        return f

    # ---------------------------------------------------------------- loading

    def loaded(self, analysis):
        self.update_idletasks()
        self.a = analysis
        self.full = Image.fromarray(colormap()[self.a.spec])
        self.v0, self.v1 = 0.0, self.a.dur
        self.start, self.end = self.bounds0
        self.info.config(text=f"длительность {fmt(self.a.dur)} ({self.a.dur:.2f} с)")
        self.redraw()
        self.update_markers()

    # ---------------------------------------------------------------- geometry

    def size(self):
        return max(self.canvas.winfo_width(), 10), max(self.canvas.winfo_height(), RULER + WAVE + 20)

    def x_of(self, t):
        return (t - self.v0) / (self.v1 - self.v0) * self.size()[0]

    def t_of(self, x):
        return self.v0 + x / self.size()[0] * (self.v1 - self.v0)

    def set_view(self, v0, span):
        span = min(max(span, MIN_VIEW), self.a.dur)
        v0 = min(max(v0, 0.0), self.a.dur - span)
        self.v0, self.v1 = v0, v0 + span
        self.redraw()

    # ---------------------------------------------------------------- drawing

    def redraw(self):
        if not self.a:
            return
        c, a = self.canvas, self.a
        w, h = self.size()
        c.delete("all")
        top, bottom = RULER, h - WAVE
        # spectrogram
        c0, c1 = self.v0 / a.hop_s, self.v1 / a.hop_s
        img = self.full.resize((w, bottom - top), Image.NEAREST if c1 - c0 < w else Image.BOX,
                               box=(c0, 0, min(c1, a.spec.shape[1]), a.spec.shape[0]))
        self.photo = ImageTk.PhotoImage(img)
        c.create_image(0, top, image=self.photo, anchor="nw")
        for f, label in ((100, "100 Гц"), (440, "440"), (1000, "1 кГц"), (4000, "4 кГц")):
            y = top + (bottom - top) * (1 - math.log(f / FMIN) / math.log(a.fmax / FMIN))
            c.create_line(0, y, 8, y, fill=INK)
            box = c.bbox(c.create_text(11, y, text=label, anchor="w", fill=INK, font=(FONT, 7), tags="flabel"))
            c.create_rectangle(box[0] - 2, box[1], box[2] + 2, box[3], fill=BG, outline="", tags="fbox")
        c.tag_raise("flabel")
        # loudness strip, dB scale so quiet entries are visible
        c.create_rectangle(0, bottom, w, h, fill=BG, outline="")
        draw_loudness(c, a, c0, c1, w, bottom, WAVE)
        c.create_text(w - 6, bottom + 4, text="громкость, дБ", anchor="ne", fill=MUTED, font=(FONT, 7))
        # time ruler
        c.create_rectangle(0, 0, w, top, fill=BG, outline="")
        span = self.v1 - self.v0
        step = time_step(span, w, 90)
        decimals = 2 if step < 0.1 else 1 if step < 1 else 0
        t = math.ceil(self.v0 / step) * step
        while t <= self.v1:
            x = self.x_of(t)
            c.create_line(x, top - 8, x, top, fill=MUTED)
            c.create_line(x, top, x, h, fill=GRID, dash=(1, 4))
            c.create_text(x + 3, 3, text=fmt(t, decimals), anchor="nw", fill=INK, font=(FONT, 8))
            for k in range(1, 5):
                xm = self.x_of(t + step * k / 5)
                c.create_line(xm, top - 4, xm, top, fill=GRID)
            t += step
        self.scroll.set(self.v0 / a.dur, self.v1 / a.dur)
        self.draw_markers()

    def draw_markers(self):
        c = self.canvas
        c.delete("marker")
        w, h = self.size()
        xs, xe = self.x_of(self.start), self.x_of(self.end)
        for x0, x1 in ((0, xs), (xe, w)):
            if x1 > x0:
                c.create_rectangle(max(x0, 0), RULER, min(x1, w), h, fill="#000000", stipple="gray50",
                                   outline="", tags="marker")
        for key, x, color, label in (("start", xs, START_C, "Н"), ("end", xe, END_C, "К")):
            if -2 <= x <= w + 2:
                c.create_line(x, RULER, x, h, fill=color, width=3 if key == self.active else 2, tags="marker")
                t = self.start if key == "start" else self.end
                anchor = "nw" if key == "start" else "ne"
                tx = x + 4 if key == "start" else x - 4
                c.create_rectangle(*c.bbox(c.create_text(tx, RULER + 4, text=f"{label} {t:.2f}", anchor=anchor,
                                                         font=("Consolas", 9, "bold"), tags=("marker", "lbl"))),
                                   fill=color, outline=color, tags=("marker", "box"))
        c.tag_raise("lbl")
        c.itemconfigure("lbl", fill=BG)
        c.tag_raise("hover")
        c.tag_raise("playhead")

    def update_markers(self):
        for key in ("start", "end"):
            t = getattr(self, key)
            self.vars[key].set(f"{t:.2f}")
            self.mmss[key].config(text=fmt(t))
        n = self.end - self.start
        self.selection.config(text=f"Выбрано: {fmt(n)}  ({n:.2f} с)")
        self.draw_markers()

    # ---------------------------------------------------------------- markers

    def set_marker(self, key, t):
        if not self.a:
            return
        self.start, self.end = clamp_marker(key, t, self.start, self.end, self.a.dur)
        self.active = key
        self.update_markers()

    def nudge(self, key, d):
        self.set_marker(key, getattr(self, key) + d)
        x = self.x_of(getattr(self, key))
        if not 0 <= x <= self.size()[0]:  # keep the moved marker in view
            self.focus_on(key, keep_zoom=True)

    def entry_set(self, key):
        try:
            text = self.vars[key].get().strip().replace(",", ".")
            if ":" in text:
                m, s = text.split(":")
                t = int(m) * 60 + float(s)
            else:
                t = float(text)
        except ValueError:
            self.update_markers()
            return
        self.set_marker(key, t)

    def focus_on(self, key, keep_zoom=False):
        if not self.a:
            return
        span = self.v1 - self.v0 if keep_zoom else min(4.0, self.a.dur)
        self.set_view(getattr(self, key) - span / 2, span)

    def use_auto(self):
        if self.a:
            self.start, self.end = self.a.auto
            self.update_markers()

    def use_all(self):
        if self.a:
            self.start, self.end = 0.0, round(self.a.dur, 2)
            self.update_markers()

    def zoom_all(self):
        if self.a:
            self.set_view(0, self.a.dur)

    # ---------------------------------------------------------------- mouse / keys

    def press(self, e, default):
        if not self.a:
            return
        self.canvas.focus_set()
        near = [(abs(self.x_of(getattr(self, k)) - e.x), k) for k in ("start", "end")]
        dist, key = min(near)
        self.dragging = key if dist <= 8 else default
        self.set_marker(self.dragging, self.t_of(e.x))

    def drag(self, e):
        if self.dragging:
            w = self.size()[0]
            if e.x < 0 or e.x > w:  # dragging past the edge scrolls
                span = self.v1 - self.v0
                self.set_view(self.v0 + (span * 0.03 if e.x > w else -span * 0.03), span)
            self.set_marker(self.dragging, self.t_of(min(max(e.x, 0), w)))
            self.hover(e)

    def release(self, e):
        self.dragging = None

    def pan(self, e):
        if self.pan_from and self.a:
            x0, v0 = self.pan_from
            self.set_view(v0 - (e.x - x0) / self.size()[0] * (self.v1 - self.v0), self.v1 - self.v0)

    def wheel(self, e):
        if self.a:
            t, span = self.t_of(e.x), (self.v1 - self.v0) * (0.8 if e.delta > 0 else 1.25)
            span = min(max(span, MIN_VIEW), self.a.dur)
            self.set_view(t - e.x / self.size()[0] * span, span)
        return "break"

    def shift_wheel(self, e):
        if self.a:
            span = self.v1 - self.v0
            self.set_view(self.v0 - span * 0.15 * (1 if e.delta > 0 else -1), span)
        return "break"

    def xview(self, *args):
        if not self.a:
            return
        span = self.v1 - self.v0
        if args[0] == "moveto":
            self.set_view(float(args[1]) * self.a.dur, span)
        elif args[0] == "scroll":
            step = span * (0.9 if args[2] == "pages" else 0.1)
            self.set_view(self.v0 + int(args[1]) * step, span)

    def hover(self, e):
        if not self.a:
            return
        c = self.canvas
        c.delete("hover")
        w, h = self.size()
        if 0 <= e.x <= w:
            c.create_line(e.x, RULER, e.x, h, fill="#FFFFFF", dash=(2, 3), tags="hover")
            self.readout.config(text=f"курсор: {self.t_of(e.x):.2f} с  ({fmt(self.t_of(e.x))})")
        near = min(abs(self.x_of(getattr(self, k)) - e.x) for k in ("start", "end"))
        c.config(cursor="sb_h_double_arrow" if near <= 8 or self.dragging else "crosshair")

    def key_nudge(self, d):
        if isinstance(self.focus_get(), tk.Entry):
            return
        self.nudge(self.active, d)

    def key_play(self, e):
        if isinstance(self.focus_get(), tk.Entry):
            return
        if self.play:
            self.stop()
        else:
            self.play_range(self.start, self.end)

    # ---------------------------------------------------------------- playback

    def play_range(self, t0, t1):
        if not self.a:
            return
        import soundfile as sf

        self.stop()
        t0, t1 = max(0.0, t0), min(self.a.dur, t1)
        if t1 - t0 < MIN_GAP:
            return
        self.tmp_n ^= 1  # alternate files: the previous one may still be held by the player
        path = f"{self.tmp}{self.tmp_n}.wav"
        sf.write(path, self.a.audio[int(t0 * self.a.sr):int(t1 * self.a.sr)], self.a.sr, subtype="PCM_16")
        winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT)
        self.play = (t0, t1, time.perf_counter())
        self.tick_play()

    def tick_play(self):
        if not self.play or not self.winfo_exists():
            return
        t0, t1, began = self.play
        t = t0 + time.perf_counter() - began
        self.canvas.delete("playhead")
        if t >= t1:
            self.play = None
            return
        x = self.x_of(t)
        self.canvas.create_line(x, RULER, x, self.size()[1], fill=PLAY_C, width=2, tags="playhead")
        self.after(30, self.tick_play)

    def stop(self):
        if self.play:
            winsound.PlaySound(None, 0)
            self.play = None
            self.canvas.delete("playhead")

    # ---------------------------------------------------------------- closing

    def confirm(self):
        if not self.a:
            return
        for key in ("start", "end"):
            self.entry_set(key)
        start, end = self.start, self.end
        self.close()
        self.on_done(start, end)

    def close(self):
        self.stop()
        self.grab_release()
        self.destroy()


class MiniTrim(tk.Canvas):
    """Simple view for the main window: loudness of the whole recording with the two markers.

    Left click sets the start, right click the end, markers can be dragged.
    on_change(start, end) is called after every move.
    """

    HEIGHT = 64

    def __init__(self, parent, analysis, bounds, on_change):
        super().__init__(parent, height=self.HEIGHT, bg=BG, highlightthickness=0, cursor="crosshair")
        self.a, self.on_change, self.dragging = analysis, on_change, None
        self.start, self.end = bounds
        self.bind("<Configure>", lambda e: self.redraw())
        self.bind("<ButtonPress-1>", lambda e: self.press(e, "start"))
        self.bind("<ButtonPress-3>", lambda e: self.press(e, "end"))
        self.bind("<B1-Motion>", self.drag)
        self.bind("<B3-Motion>", self.drag)
        self.bind("<ButtonRelease-1>", lambda e: setattr(self, "dragging", None))
        self.bind("<ButtonRelease-3>", lambda e: setattr(self, "dragging", None))
        self.bind("<Motion>", self.hover)

    def x_of(self, t):
        return t / self.a.dur * self.winfo_width()

    def t_of(self, x):
        return x / max(self.winfo_width(), 1) * self.a.dur

    def set_bounds(self, start, end):
        self.start, self.end = start, end
        self.draw_markers()

    def redraw(self):
        self.delete("all")
        w, h = self.winfo_width(), self.HEIGHT
        if w < 10:
            return
        draw_loudness(self, self.a, 0, self.a.dur / self.a.hop_s, w, 0, h)
        step = time_step(self.a.dur, w, 70)
        t = step
        while t < self.a.dur:
            x = self.x_of(t)
            self.create_line(x, 0, x, h, fill=GRID, dash=(2, 4))
            t += step
        self.create_text(w - 6, 3, text="громкость, дБ", anchor="ne", fill=MUTED, font=(FONT, 7))
        self.draw_markers()

    def draw_markers(self):
        self.delete("marker")
        w, h = self.winfo_width(), self.HEIGHT
        xs, xe = self.x_of(self.start), self.x_of(self.end)
        for x0, x1 in ((0, xs), (xe, w)):
            if x1 > x0:
                self.create_rectangle(x0, 0, x1, h, fill="#000000", stipple="gray50", outline="", tags="marker")
        for x, color, text, anchor in ((xs, START_C, f"Н {fmt(self.start)}", "nw"),
                                       (xe, END_C, f"К {fmt(self.end)}", "ne")):
            self.create_line(x, 0, x, h, fill=color, width=2, tags="marker")
            tx = min(max(x + 4, 2), w - 120) if anchor == "nw" else max(min(x - 4, w - 2), 120)
            label = self.create_text(tx, h - 3, text=text, anchor="s" + anchor[1], fill=BG,
                                     font=("Consolas", 8, "bold"), tags="marker")
            box = self.bbox(label)
            self.tag_lower(self.create_rectangle(box[0] - 2, box[1], box[2] + 2, box[3], fill=color, outline=color,
                                                 tags="marker"), label)

    def press(self, e, default):
        near = min((abs(self.x_of(getattr(self, k)) - e.x), k) for k in ("start", "end"))
        self.dragging = near[1] if near[0] <= 6 else default
        self.move(e.x)

    def drag(self, e):
        if self.dragging:
            self.move(min(max(e.x, 0), self.winfo_width()))

    def move(self, x):
        self.start, self.end = clamp_marker(self.dragging, self.t_of(x), self.start, self.end, self.a.dur)
        self.draw_markers()
        self.on_change(self.start, self.end)

    def hover(self, e):
        near = min(abs(self.x_of(getattr(self, k)) - e.x) for k in ("start", "end"))
        self.config(cursor="sb_h_double_arrow" if near <= 6 or self.dragging else "crosshair")
