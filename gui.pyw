"""Window for transcribe.py: drop recordings or folders, get scores.

Started by "Ноты из записи.exe" (or: .venv\\Scripts\\pythonw gui.pyw [files ...]).
Recordings are processed one by one in a background thread; the models stay loaded
between them. Results go to out\\<name>\\ as with transcribe.bat.
"""

import ctypes
import math
import os
import queue
import sys
import threading
import time
import tkinter as tk
import traceback
from ctypes import wintypes
from pathlib import Path
from tkinter import filedialog, ttk

from trim_editor import STEP, Analysis, MiniTrim, TrimEditor, fmt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
LOG_FILE = ROOT / "gui_log.txt"
AUDIO_EXT = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}
TITLE = "Ноты из записи"
AUTHOR = "Астрамикс"
AUTHOR_EMAIL = "astramixofficial@gmail.com"

# stage -> (share of the progress bar where it starts, where it ends, typical seconds, label)
STAGES = {
    "trim":   (0.00, 0.00, 1, "выбор границ"),
    "queued": (0.00, 0.00, 1, "в очереди"),
    "decode": (0.00, 0.05, 1, "читаю запись"),
    "notes":  (0.05, 0.65, 8, "распознаю ноты"),
    "beats":  (0.65, 0.78, 3, "ищу доли такта"),
    "score":  (0.78, 0.90, 3, "расставляю ноты по тактам"),
    "pdf":    (0.90, 1.00, 3, "готовлю PDF"),
}

BG, CARD, LINE = "#F5F3FA", "#FFFFFF", "#E2DDF0"
INK, MUTED = "#1F1A33", "#6E6887"
ACCENT, ACCENT_DARK, ACCENT_SOFT = "#5B3CE0", "#3F25B0", "#EEE9FF"
OK, ERR = "#1E8E5A", "#C2185B"
FONT = "Segoe UI"


# ------------------------------------------------------------------ Windows glue

def windows_setup():
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    try:  # own taskbar button with our icon instead of Python's
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("play-me.mus.notes")
    except Exception:
        pass


user32, shell32 = ctypes.windll.user32, ctypes.windll.shell32
WNDPROC = ctypes.WINFUNCTYPE(ctypes.c_ssize_t, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)
user32.SetWindowLongPtrW.restype = ctypes.c_void_p
user32.SetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_void_p]
user32.CallWindowProcW.restype = ctypes.c_ssize_t
user32.CallWindowProcW.argtypes = [ctypes.c_void_p, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
user32.GetParent.restype = wintypes.HWND
shell32.DragQueryFileW.argtypes = [ctypes.c_void_p, wintypes.UINT, ctypes.c_wchar_p, wintypes.UINT]
shell32.DragFinish.argtypes = [ctypes.c_void_p]
shell32.DragAcceptFiles.argtypes = [wintypes.HWND, wintypes.BOOL]
WM_DROPFILES, GWLP_WNDPROC = 0x0233, -4


def accept_drops(root, on_drop):
    """Files dragged from Explorer onto the window -> on_drop(list of paths).

    Plain Tk has no drag and drop; the window procedure is subclassed to catch WM_DROPFILES.
    on_drop runs inside the window procedure, so it must only queue the paths.
    """
    root.update_idletasks()  # the outer window only exists once Tk has mapped it
    hwnd = user32.GetParent(root.winfo_id())

    def proc(h, msg, wp, lp):
        if msg == WM_DROPFILES:
            n = shell32.DragQueryFileW(wp, 0xFFFFFFFF, None, 0)
            paths = []
            for i in range(n):
                buf = ctypes.create_unicode_buffer(shell32.DragQueryFileW(wp, i, None, 0) + 1)
                shell32.DragQueryFileW(wp, i, buf, len(buf))
                paths.append(buf.value)
            shell32.DragFinish(wp)
            on_drop(paths)
            return 0
        return user32.CallWindowProcW(old, h, msg, wp, lp)

    root._drop_proc = WNDPROC(proc)  # keep a reference or the callback is garbage collected
    old = user32.SetWindowLongPtrW(hwnd, GWLP_WNDPROC, ctypes.cast(root._drop_proc, ctypes.c_void_p))
    shell32.DragAcceptFiles(hwnd, True)


def collect_audio(paths):
    found = []
    for p in map(Path, paths):
        if p.is_dir():
            found += sorted(f for f in p.rglob("*") if f.suffix.lower() in AUDIO_EXT and OUT not in f.parents)
        elif p.suffix.lower() in AUDIO_EXT:
            found.append(p)
    return found


def key_name(k):
    """music21 'e- minor' -> 'ми-бемоль минор'."""
    tonic, _, mode = k.partition(" ")
    name = dict(zip("cdefgab", ("до", "ре", "ми", "фа", "соль", "ля", "си"))).get(tonic[:1].lower(), tonic)
    name += {"-": "-бемоль", "#": "-диез"}.get(tonic[1:2], "")
    return f"{name} {dict(major='мажор', minor='минор').get(mode, mode)}"


def open_path(p):
    try:
        os.startfile(str(p))
        return True
    except OSError:
        return False


# ------------------------------------------------------------------ worker

class Log:
    """Library output (pythonw has no console) -> log file and the log panel."""

    def __init__(self, events):
        self.events = events
        self.file = open(LOG_FILE, "a", encoding="utf-8", buffering=1)
        self.file.write(f"\n==== {time.strftime('%Y-%m-%d %H:%M:%S')} ====\n")

    def write(self, s):
        self.file.write(s)
        self.events.put(("log", s))
        return len(s)

    def flush(self):
        self.file.flush()


class Worker(threading.Thread):
    def __init__(self, events):
        super().__init__(daemon=True)
        self.events, self.jobs = events, queue.Queue()

    def run(self):
        try:
            import torch
            import transcribe
            self.engine = transcribe
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            name = torch.cuda.get_device_name(0) if self.device == "cuda" else "процессор (медленнее)"
            self.events.put(("ready", name))
        except Exception:
            traceback.print_exc()
            self.events.put(("broken", "Не удалось загрузить библиотеки — запустите setup.bat"))
            return
        while True:
            item, ticket = self.jobs.get()
            if item.stage != "queued" or item.ticket != ticket:  # removed, or re-queued with new bounds
                continue
            opts = item.opts  # read now: bounds may change while queued
            try:
                out, info = self.engine.process(
                    item.src, OUT, self.device, opts["start"], opts["duration"], opts["rescore"],
                    marks=opts["marks"], pdf=opts["pdf"],
                    step=lambda key, item=item: self.events.put(("stage", item, key)))
                print(f"[{item.src.name}] done -> {out}  {info}")
                self.events.put(("done", item, out, info))
            except Exception as ex:
                traceback.print_exc()
                self.events.put(("fail", item, str(ex) or type(ex).__name__))


# ------------------------------------------------------------------ widgets

class Button(tk.Label):
    """Flat button; tk.Button cannot be styled on Windows."""

    def __init__(self, parent, text, command, primary=False, small=False, **kw):
        self.colors = (ACCENT, ACCENT_DARK, "#FFFFFF") if primary else (ACCENT_SOFT, "#DED5FF", ACCENT_DARK)
        super().__init__(parent, text=text, bg=self.colors[0], fg=self.colors[2], cursor="hand2",
                         font=(FONT, 9 if small else 10, "bold" if primary else "normal"),
                         padx=10 if small else 14, pady=3 if small else 6, **kw)
        self.command = command
        self.bind("<Enter>", lambda e: self.config(bg=self.colors[1]))
        self.bind("<Leave>", lambda e: self.config(bg=self.colors[0]))
        self.bind("<Button-1>", lambda e: self.command())


class DropZone(tk.Canvas):
    def __init__(self, parent, on_files, on_folder):
        super().__init__(parent, height=170, bg=BG, highlightthickness=0, cursor="hand2")
        self.on_files, self.on_folder, self.hover = on_files, on_folder, False
        self.bind("<Configure>", lambda e: self.draw())
        self.bind("<Enter>", lambda e: self.set_hover(True))
        self.bind("<Leave>", lambda e: self.set_hover(False))
        self.bind("<Button-1>", self.click)

    def set_hover(self, on):
        self.hover = on
        self.draw()

    def flash(self):
        self.hover = True
        self.draw()
        self.after(350, lambda: self.set_hover(False))

    def draw(self):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        fill = ACCENT_SOFT if self.hover else CARD
        r, x0, y0, x1, y1 = 18, 2, 2, w - 3, h - 3
        pts = [x0 + r, y0, x1 - r, y0, x1, y0, x1, y0 + r, x1, y1 - r, x1, y1, x1 - r, y1,
               x0 + r, y1, x0, y1, x0, y1 - r, x0, y0 + r, x0, y0]
        self.create_polygon(pts, smooth=True, fill=fill, outline=ACCENT if self.hover else "#B9AEE6",
                            width=2, dash=(6, 4))
        self.create_text(w / 2, h / 2 - 34, text="♫", font=(FONT, 28), fill=ACCENT)
        self.create_text(w / 2, h / 2 + 8, text="Перетащите сюда записи или папку",
                         font=(FONT, 14, "bold"), fill=INK)
        self.create_text(w / 2 - 6, h / 2 + 40, text="или нажмите, чтобы выбрать файлы  ·", anchor="e",
                         font=(FONT, 10), fill=MUTED)
        self.create_text(w / 2 - 2, h / 2 + 40, text="выбрать папку", anchor="w",
                         font=(FONT, 10, "underline"), fill=ACCENT, tags="folder")
        self.create_text(w / 2, h - 18, text="MP3 · WAV · FLAC · OGG", font=(FONT, 8), fill=MUTED)

    def click(self, e):
        near = self.find_overlapping(e.x - 3, e.y - 3, e.x + 3, e.y + 3)
        if any("folder" in self.gettags(i) for i in near):
            self.on_folder()
        else:
            self.on_files()


class Item(tk.Frame):
    """One recording in the list. opts are its own processing options, bounds included."""

    def __init__(self, parent, app, src, opts):
        super().__init__(parent, bg=CARD, highlightbackground=LINE, highlightthickness=1)
        self.app, self.src, self.opts, self.out = app, src, opts, None
        self.stage, self.since, self.ticket = "new", time.time(), 0
        self.analysis, self.bounds, self.mini, self.previous = None, None, None, None
        self.columnconfigure(0, weight=1)
        top = tk.Frame(self, bg=CARD)
        top.grid(row=0, column=0, sticky="ew", padx=14, pady=(10, 4))
        top.columnconfigure(1, weight=1)
        tk.Label(top, text="♪", bg=CARD, fg=ACCENT, font=(FONT, 14)).grid(row=0, column=0, rowspan=2, padx=(0, 10))
        tk.Label(top, text=src.name, bg=CARD, fg=INK, font=(FONT, 11, "bold"), anchor="w").grid(
            row=0, column=1, sticky="ew")
        self.status = tk.Label(top, text="", bg=CARD, fg=MUTED, font=(FONT, 9), anchor="w")
        self.status.grid(row=1, column=1, columnspan=2, sticky="ew")
        self.actions = tk.Frame(top, bg=CARD)
        self.actions.grid(row=0, column=2, sticky="e")
        self.bar = ttk.Progressbar(self, style="Notes.Horizontal.TProgressbar", maximum=1.0)
        self.bar.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 12))

    def show_actions(self, buttons):
        for w in self.actions.winfo_children():
            w.destroy()
        for label, cmd, primary in buttons:
            Button(self.actions, label, cmd, small=True, primary=primary).pack(side="left", padx=(6, 0))

    def bounds_text(self):
        start, duration = self.opts["start"], self.opts["duration"]
        if not start and not duration:
            return ""
        return f" · фрагмент {fmt(start)}–{fmt(start + duration) if duration else 'конец'}"

    def wait_trim(self):
        """Simple loudness view with start/end markers; recognition starts with "Распознать"."""
        self.stage = "trim"
        self.bar.grid_remove()
        if self.analysis:
            self.show_mini()
            return
        self.status.config(text="строю график громкости…", fg=MUTED)
        self.show_actions([self.cancel_button()])
        threading.Thread(target=self.analyze, daemon=True).start()

    def analyze(self):
        try:
            result = Analysis(self.src)
        except Exception as ex:
            traceback.print_exc()
            result = ex
        self.app.events.put(("analysis", self, result))

    def analysis_ready(self, result):
        if self.stage != "trim":
            return
        if isinstance(result, Exception):
            self.status.config(text=f"не удалось открыть запись: {result}", fg=ERR)
            return
        self.analysis = result
        self.show_mini()

    def show_mini(self):
        a, o = self.analysis, self.opts
        if self.previous and (o["start"] or o["duration"]):
            self.bounds = (o["start"], o["start"] + o["duration"] if o["duration"] else round(a.dur, 2))
        elif self.previous:
            self.bounds = (0.0, round(a.dur, 2))
        else:
            self.bounds = a.auto
        if self.mini:
            self.mini.destroy()
        self.mini = MiniTrim(self, a, self.bounds, self.set_bounds)
        self.mini.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 12))
        self.show_actions([("Полный", self.full, False), ("Распознать", self.recognize, True), self.cancel_button()])
        self.show_bounds()

    def cancel_button(self):
        return ("Отмена", self.cancel_edit, False) if self.previous else ("✕", self.drop, False)

    def show_bounds(self):
        s, e = self.bounds
        self.status.config(text=f"начало {s:.2f} с · конец {e:.2f} с · выбрано {fmt(e - s)}   "
                                "(ЛКМ — начало, ПКМ — конец, маркеры можно тянуть)", fg=MUTED)

    def set_bounds(self, start, end):
        self.bounds = (start, end)
        if self.mini:
            self.mini.set_bounds(start, end)
        self.show_bounds()

    def full(self):
        TrimEditor(self.app.root, self.src, self.analysis, self.bounds, self.set_bounds, Button)

    def recognize(self):
        if self.stage != "trim" or not self.analysis:
            return
        s, e = self.bounds
        new = dict(start=s, duration=0.0 if e >= self.analysis.dur - STEP else round(e - s, 2))
        if any(self.opts[k] != v for k, v in new.items()):
            self.opts.update(new, rescore=False)  # other bounds need new recognition
        self.analysis, self.previous = None, None  # frees the decoded audio
        self.app.enqueue(self)

    def cancel_edit(self):
        previous, self.previous, self.analysis = self.previous, None, None
        if previous == "done":
            self.finish(self.out, self.info)
        elif previous == "failed":
            self.fail(self.error)
        else:
            self.app.enqueue(self)
        self.app.update_state()

    def edit(self):
        if self.stage in ("queued", "done", "failed"):
            self.previous = self.stage
            self.wait_trim()
            self.app.update_state()

    def hide_mini(self):
        if self.mini:
            self.mini.destroy()
            self.mini = None

    def queue(self):
        self.hide_mini()
        self.stage, self.out = "queued", None
        self.bar["value"] = 0
        self.bar.grid()
        self.status.config(text="в очереди" + self.bounds_text(), fg=MUTED)
        self.show_actions([("Границы…", self.edit, False), ("✕", self.drop, False)])

    def drop(self):
        self.app.remove(self)

    def set_stage(self, key):
        self.stage, self.since = key, time.time()
        self.status.config(text=STAGES[key][3] + "…", fg=ACCENT)
        self.show_actions([])

    def tick(self):
        """Within a stage the bar creeps towards the stage's end so the window never looks stuck."""
        if self.stage not in STAGES or self.stage in ("queued", "trim"):
            return
        a, b, tau, _ = STAGES[self.stage]
        self.bar["value"] = a + (b - a) * 0.95 * (1 - math.exp(-(time.time() - self.since) / tau))

    def finish(self, out, info):
        self.stage, self.out, self.info = "done", out, info
        self.hide_mini()
        self.bar.grid_remove()
        text = f"готово · тактов: {info.get('bars')} · {info.get('meter')} · ♩≈{info.get('bpm')}"
        if info.get("key"):
            text += " · " + key_name(info["key"])
        if info.get("uncertain"):
            text += f" · проверить на слух: {info['uncertain']}"
        self.status.config(text=text + self.bounds_text(), fg=OK)
        buttons = [(label, lambda n=name: self.open(n), name == "score.pdf")
                   for label, name in (("Ноты PDF", "score.pdf"), ("Открыть в редакторе", "score.musicxml"),
                                       ("Послушать", "performance.mid")) if (out / name).exists()]
        self.show_actions(buttons + [("Папка", lambda: open_path(out), False),
                                     ("Границы…", self.edit, False)])

    def open(self, name):
        if not open_path(self.out / name) and name == "score.musicxml":
            self.app.toast("Нет программы для MusicXML. Установите MuseScore 4 (musescore.org, бесплатно).")

    def fail(self, msg):
        self.stage, self.error = "failed", msg
        self.hide_mini()
        self.bar["value"] = 0
        self.bar.grid()
        self.status.config(text="ошибка: " + msg[:120], fg=ERR)
        self.show_actions([("Журнал", self.app.show_log, False), ("Границы…", self.edit, False),
                           ("✕", self.drop, False)])


# ------------------------------------------------------------------ app

class App:
    def __init__(self, root, initial):
        self.root, self.items, self.events = root, [], queue.Queue()
        self.dropped = queue.Queue()
        sys.stdout = sys.stderr = Log(self.events)
        self.worker = Worker(self.events)

        root.title(TITLE)
        root.configure(bg=BG)
        root.geometry("920x800")
        root.minsize(620, 480)
        try:
            root.iconbitmap(default=str(ROOT / "app.ico"))
        except tk.TclError:
            pass
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("Notes.Horizontal.TProgressbar", troughcolor=ACCENT_SOFT, background=ACCENT,
                        bordercolor=ACCENT_SOFT, lightcolor=ACCENT, darkcolor=ACCENT, thickness=6)
        style.configure("Notes.Vertical.TScrollbar", troughcolor=BG, background=LINE, bordercolor=BG,
                        arrowcolor=MUTED, lightcolor=LINE, darkcolor=LINE)

        self.build()
        accept_drops(root, self.dropped.put)
        self.worker.start()
        if initial:
            self.add(initial)
        self.poll()

    def build(self):
        root = self.root
        head = tk.Frame(root, bg=BG)
        head.pack(fill="x", padx=24, pady=(20, 10))
        try:
            self.logo = tk.PhotoImage(file=str(ROOT / "app.png")).subsample(8)
            tk.Label(head, image=self.logo, bg=BG).pack(side="left", padx=(0, 12))
        except tk.TclError:
            pass
        titles = tk.Frame(head, bg=BG)
        titles.pack(side="left")
        tk.Label(titles, text=TITLE, bg=BG, fg=INK, font=(FONT, 18, "bold")).pack(anchor="w")
        tk.Label(titles, text="Запись фортепиано → ноты для просмотра и правки", bg=BG, fg=MUTED,
                 font=(FONT, 10)).pack(anchor="w")

        self.zone = DropZone(root, self.pick_files, self.pick_folder)
        self.zone.pack(fill="x", padx=24)

        opts = tk.Frame(root, bg=BG)
        opts.pack(fill="x", padx=24, pady=(10, 4))
        self.marks, self.pdf, self.rescore = tk.BooleanVar(value=True), tk.BooleanVar(value=True), tk.BooleanVar()
        self.trim = tk.BooleanVar(value=True)
        opts2 = tk.Frame(root, bg=BG)
        opts2.pack(fill="x", padx=24)
        for frame, text, var in ((opts, "Выбирать начало и конец перед распознаванием", self.trim),
                                 (opts, "Подсвечивать сомнительные ноты", self.marks),
                                 (opts2, "Делать PDF", self.pdf),
                                 (opts2, "Не распознавать заново, только пересобрать ноты", self.rescore)):
            tk.Checkbutton(frame, text=text, variable=var, bg=BG, fg=INK, activebackground=BG,
                           selectcolor=CARD, font=(FONT, 9), cursor="hand2").pack(side="left", padx=(0, 14))

        credits = tk.Frame(root, bg=BG)
        credits.pack(side="bottom", fill="x", padx=24, pady=(0, 12))
        tk.Label(credits, text=f"Автор программы: {AUTHOR} ·", bg=BG, fg=MUTED, font=(FONT, 8)).pack(side="left")
        mail = tk.Label(credits, text=AUTHOR_EMAIL, bg=BG, fg=ACCENT, font=(FONT, 8, "underline"), cursor="hand2")
        mail.pack(side="left", padx=4)
        mail.bind("<Button-1>", lambda e: open_path("mailto:" + AUTHOR_EMAIL))
        tk.Label(credits, text="— по любым вопросам и предложениям", bg=BG, fg=MUTED, font=(FONT, 8)).pack(side="left")

        foot = tk.Frame(root, bg=BG)
        foot.pack(side="bottom", fill="x", padx=24, pady=(6, 6))
        self.state = tk.Label(foot, text="Загружаю библиотеки…", bg=BG, fg=MUTED, font=(FONT, 9))
        self.state.pack(side="left")
        Button(foot, "Все результаты", lambda: (OUT.mkdir(exist_ok=True), open_path(OUT))).pack(side="right")
        Button(foot, "Журнал", self.toggle_log).pack(side="right", padx=8)

        self.log = tk.Text(root, height=8, bg="#1F1A33", fg="#D9D3F0", font=("Consolas", 9), relief="flat",
                           padx=10, pady=8, wrap="word")
        self.log_visible = False

        qhead = tk.Frame(root, bg=BG)
        qhead.pack(fill="x", padx=24, pady=(14, 4))
        tk.Label(qhead, text="Очередь", bg=BG, fg=MUTED, font=(FONT, 9, "bold")).pack(side="left")
        Button(qhead, "Распознать все", self.recognize_all, small=True).pack(side="right")
        box = tk.Frame(root, bg=BG)
        box.pack(fill="both", expand=True, padx=24)
        self.canvas = tk.Canvas(box, bg=BG, highlightthickness=0)
        sb = ttk.Scrollbar(box, orient="vertical", command=self.canvas.yview, style="Notes.Vertical.TScrollbar")
        self.list = tk.Frame(self.canvas, bg=BG)
        self.list.bind("<Configure>", lambda e: self.canvas.config(scrollregion=self.canvas.bbox("all")))
        win = self.canvas.create_window(0, 0, window=self.list, anchor="nw")
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(win, width=e.width))
        self.canvas.config(yscrollcommand=sb.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        root.bind_all("<MouseWheel>", lambda e: self.canvas.yview_scroll(-e.delta // 120, "units"))
        self.empty = tk.Label(self.list, text="Пока пусто. Результаты появятся здесь и в папке out\\",
                              bg=BG, fg=MUTED, font=(FONT, 10))
        self.empty.pack(pady=30)
        self.toast_label = tk.Label(root, bg=INK, fg="#FFFFFF", font=(FONT, 10), padx=14, pady=8)

    # ---------------------------------------------------------------- actions

    def pick_files(self):
        files = filedialog.askopenfilenames(title="Выберите записи", filetypes=[
            ("Аудио", " ".join("*" + e for e in sorted(AUDIO_EXT))), ("Все файлы", "*.*")])
        self.add(files)

    def pick_folder(self):
        folder = filedialog.askdirectory(title="Выберите папку с записями")
        if folder:
            self.add([folder])

    def options(self):
        return dict(start=0.0, duration=0.0, marks=self.marks.get(), pdf=self.pdf.get(), rescore=self.rescore.get())

    def add(self, paths):
        files = collect_audio(paths)
        if paths and not files:
            self.toast("Здесь нет записей (MP3, WAV, FLAC, OGG)")
            return
        if files:
            self.empty.pack_forget()
            self.zone.flash()
        for src in files:
            item = Item(self.list, self, src, self.options())
            item.pack(fill="x", pady=(0, 8))
            self.items.append(item)
            if self.trim.get():
                item.wait_trim()
            else:
                self.enqueue(item)
        self.update_state()

    def enqueue(self, item):
        item.queue()
        item.ticket += 1
        self.worker.jobs.put((item, item.ticket))
        self.update_state()

    def recognize_all(self):
        """Every recording still waiting for its bounds goes with the bounds shown."""
        waiting = [i for i in self.items if i.stage == "trim" and i.analysis]
        for item in waiting:
            item.recognize()
        if not waiting:
            self.toast("Нет записей, ожидающих распознавания")

    def remove(self, item):
        item.stage = "removed"
        item.destroy()
        self.items.remove(item)
        if not self.items:
            self.empty.pack(pady=30)
        self.update_state()

    def toast(self, text, ms=4000):
        self.toast_label.config(text=text)
        self.toast_label.place(relx=0.5, rely=1.0, y=-60, anchor="s")
        self.toast_label.lift()
        if hasattr(self, "_toast_job"):
            self.root.after_cancel(self._toast_job)
        self._toast_job = self.root.after(ms, self.toast_label.place_forget)

    def toggle_log(self):
        self.log_visible = not self.log_visible
        if self.log_visible:
            self.log.pack(side="bottom", fill="x", padx=24, pady=(6, 0))
            self.log.see("end")
        else:
            self.log.pack_forget()

    def show_log(self):
        if not self.log_visible:
            self.toggle_log()

    def update_state(self):
        if not hasattr(self, "device_name"):
            return
        busy = sum(i.stage not in ("done", "failed", "removed", "trim") for i in self.items)
        waiting = sum(i.stage == "trim" for i in self.items)
        parts = ([f"в работе: {busy}"] if busy else []) + ([f"ждут «Распознать»: {waiting}"] if waiting else [])
        self.state.config(text=" · ".join(parts or ["Готово к работе"]) + f" · {self.device_name}")

    # ---------------------------------------------------------------- event loop

    def poll(self):
        while not self.dropped.empty():
            self.add(self.dropped.get())
        while not self.events.empty():
            ev = self.events.get()
            kind, args = ev[0], ev[1:]
            if kind == "log":
                self.log.insert("end", args[0])
                self.log.see("end")
            elif kind == "ready":
                self.device_name = args[0]
                self.update_state()
            elif kind == "broken":
                self.state.config(text=args[0], fg=ERR)
                self.show_log()
            elif args[0].winfo_exists() and args[0].stage != "removed":
                getattr(args[0], {"stage": "set_stage", "done": "finish", "fail": "fail", "analysis": "analysis_ready"}[kind])(*args[1:])
                self.update_state()
        for item in self.items:
            item.tick()
        self.root.after(100, self.poll)


def main():
    windows_setup()
    root = tk.Tk()
    App(root, sys.argv[1:])
    root.mainloop()


if __name__ == "__main__":
    main()
